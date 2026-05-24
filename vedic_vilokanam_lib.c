#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef enum {
    KATAPAYADI_DEVANAGARI = 0,
    KATAPAYADI_BENGALI   = 1,
    KATAPAYADI_TAMIL     = 2,
    KATAPAYADI_LATIN     = 3,
    KATAPAYADI_NUMERIC   = 4,
    KATAPAYADI_PUNCT     = 5
} KatapayadiClass;

static KatapayadiClass vilokanam_detect_class(const float* emb) {
    float32x4_t s0 = vld1q_f32(emb);
    float32x4_t s1 = vld1q_f32(emb + 4);
    float32x4_t sum = vaddq_f32(s0, s1);
    float mean8 = (vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1)
                + vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3)) / 8.0f;
    float32x4_t diff0 = vsubq_f32(s0, vdupq_n_f32(mean8));
    float32x4_t diff1 = vsubq_f32(s1, vdupq_n_f32(mean8));
    float var = (vgetq_lane_f32(vmulq_f32(diff0, diff0), 0)
              + vgetq_lane_f32(vmulq_f32(diff0, diff0), 1)
              + vgetq_lane_f32(vmulq_f32(diff0, diff0), 2)
              + vgetq_lane_f32(vmulq_f32(diff0, diff0), 3)
              + vgetq_lane_f32(vmulq_f32(diff1, diff1), 0)
              + vgetq_lane_f32(vmulq_f32(diff1, diff1), 1)
              + vgetq_lane_f32(vmulq_f32(diff1, diff1), 2)
              + vgetq_lane_f32(vmulq_f32(diff1, diff1), 3)) / 8.0f;
    float magnitude = mean8 * mean8 + var;
    if (mean8 < -0.15f) return KATAPAYADI_PUNCT;
    if (var < 0.02f && mean8 > 0.0f && mean8 < 0.3f) return KATAPAYADI_NUMERIC;
    if (magnitude > 1.5f)  return KATAPAYADI_DEVANAGARI;
    if (magnitude > 0.9f)  return KATAPAYADI_BENGALI;
    if (magnitude > 0.4f)  return KATAPAYADI_TAMIL;
    return KATAPAYADI_LATIN;
}

static int vilokanam_build_mask(
    KatapayadiClass detected_class,
    const int* token_classes,
    int vocab_size,
    int* mask_out
) {
    int count = 0;
    int include_mask = 0;
    include_mask |= (1 << KATAPAYADI_PUNCT);
    include_mask |= (1 << KATAPAYADI_NUMERIC);
    include_mask |= (1 << detected_class);
    include_mask |= (1 << KATAPAYADI_LATIN);
    for (int i = 0; i < vocab_size; i++) {
        int included = (include_mask & (1 << token_classes[i])) ? 1 : 0;
        mask_out[i] = included;
        count += included;
    }
    return count;
}

static void vilokanam_sparse_projection(
    const float* hidden_state,
    const float* lm_head,
    const int* mask,
    int dim,
    int vocab_size,
    float* logits_out
) {
    float neg_large = -1e30f;
    float32x4_t vneg = vdupq_n_f32(neg_large);
    int v;
    for (v = 0; v <= vocab_size - 4; v += 4) {
        vst1q_f32(logits_out + v, vneg);
    }
    for (; v < vocab_size; v++) {
        logits_out[v] = neg_large;
    }
    for (v = 0; v < vocab_size; v++) {
        if (!mask[v]) continue;
        float dot = 0.0f;
        const float* weight_row = lm_head + v * dim;
        int d;
        for (d = 0; d <= dim - 16; d += 16) {
            float32x4_t hs0 = vld1q_f32(hidden_state + d);
            float32x4_t hs1 = vld1q_f32(hidden_state + d + 4);
            float32x4_t hs2 = vld1q_f32(hidden_state + d + 8);
            float32x4_t hs3 = vld1q_f32(hidden_state + d + 12);
            float32x4_t wr0 = vld1q_f32(weight_row + d);
            float32x4_t wr1 = vld1q_f32(weight_row + d + 4);
            float32x4_t wr2 = vld1q_f32(weight_row + d + 8);
            float32x4_t wr3 = vld1q_f32(weight_row + d + 12);
            float32x4_t p0 = vmulq_f32(hs0, wr0);
            float32x4_t p1 = vmulq_f32(hs1, wr1);
            float32x4_t p2 = vmulq_f32(hs2, wr2);
            float32x4_t p3 = vmulq_f32(hs3, wr3);
            float32x4_t s01 = vaddq_f32(p0, p1);
            float32x4_t s23 = vaddq_f32(p2, p3);
            float32x4_t s = vaddq_f32(s01, s23);
            dot += vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1)
                + vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);
        }
        for (; d < dim; d++) {
            dot += hidden_state[d] * weight_row[d];
        }
        logits_out[v] = dot;
    }
}

void vilokanam_output_projection(
    const float* hidden_state,
    const float* lm_head,
    const int* token_classes,
    int dim,
    int vocab_size,
    float* logits_out
) {
    KatapayadiClass detected = vilokanam_detect_class(hidden_state);
    int* mask = (int*)calloc(vocab_size, sizeof(int));
    if (!mask) {
        float neg_large = -1e30f;
        for (int v = 0; v < vocab_size; v++) {
            float dot = 0.0f;
            for (int d = 0; d < dim; d++) {
                dot += hidden_state[d] * lm_head[v * dim + d];
            }
            logits_out[v] = dot;
        }
        return;
    }
    vilokanam_build_mask(detected, token_classes, vocab_size, mask);
    vilokanam_sparse_projection(hidden_state, lm_head, mask, dim, vocab_size, logits_out);
    free(mask);
}

void vilokanam_build_token_classes(
    const float* lm_head,
    int vocab_size,
    int dim,
    int* token_classes_out
) {
    for (int v = 0; v < vocab_size; v++) {
        token_classes_out[v] = vilokanam_detect_class(lm_head + v * dim);
    }
}
