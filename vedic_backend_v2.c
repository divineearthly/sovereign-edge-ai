#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Import from other Vedic modules
extern void nikhilam_sparsha_attention_normalize(
    const float* scores, float* weights, int seq_len, float temperature);

extern void vilokanam_output_projection(
    const float* hidden_state, const float* lm_head,
    const int* token_classes, int dim, int vocab_size, float* logits_out);

extern void vilokanam_build_token_classes(
    const float* lm_head, int vocab_size, int dim, int* token_classes_out);

// Triguna Qutrit activation
static void triguna_qutrit_activate(float* x, int n) {
    for (int i = 0; i < n; i++) {
        if (fabsf(x[i]) > 0.01f) {
            x[i] = sinf(x[i] * 1.570796f) * cosf(x[i] * 0.785398f);
        } else {
            x[i] = 0.0f;
        }
    }
}

// Shunyata pruning
static void shunyata_prune_weights(float* weights, int n, float threshold) {
    float32x4_t vthresh = vdupq_n_f32(threshold);
    float32x4_t vzero = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t w = vld1q_f32(weights + i);
        uint32x4_t mask = vcgtq_f32(vabsq_f32(w), vthresh);
        vst1q_f32(weights + i, vbslq_f32(mask, w, vzero));
    }
    for (; i < n; i++) {
        if (fabsf(weights[i]) < threshold) weights[i] = 0.0f;
    }
}

// AUM attention scores
static void aum_attention_scores(
    const float* q, const float* k, float* scores,
    int seq_len, int head_dim, float temperature
) {
    for (int i = 0; i < seq_len; i++) {
        float q_norm = 0.0f, k_norm = 0.0f, dot = 0.0f;
        for (int d = 0; d <= head_dim - 4; d += 4) {
            float32x4_t qv = vld1q_f32(q + i * head_dim + d);
            float32x4_t kv = vld1q_f32(k + i * head_dim + d);
            float32x4_t q2 = vmulq_f32(qv, qv);
            float32x4_t k2 = vmulq_f32(kv, kv);
            float32x4_t qk = vmulq_f32(qv, kv);
            q_norm += vgetq_lane_f32(q2, 0) + vgetq_lane_f32(q2, 1)
                    + vgetq_lane_f32(q2, 2) + vgetq_lane_f32(q2, 3);
            k_norm += vgetq_lane_f32(k2, 0) + vgetq_lane_f32(k2, 1)
                    + vgetq_lane_f32(k2, 2) + vgetq_lane_f32(k2, 3);
            dot += vgetq_lane_f32(qk, 0) + vgetq_lane_f32(qk, 1)
                 + vgetq_lane_f32(qk, 2) + vgetq_lane_f32(qk, 3);
        }
        for (int d = (head_dim / 4) * 4; d < head_dim; d++) {
            q_norm += q[i * head_dim + d] * q[i * head_dim + d];
            k_norm += k[i * head_dim + d] * k[i * head_dim + d];
            dot += q[i * head_dim + d] * k[i * head_dim + d];
        }
        scores[i] = (dot - (q_norm + k_norm) * 0.5f) * temperature;
    }
}

// Export: full Vedic attention
__attribute__((visibility("default")))
void vedic_attention_full(
    const float* q, const float* k, const float* v,
    float* output, int seq_len, int head_dim, float temperature
) {
    float* scores = NULL;
    float* weights = NULL;
    if (posix_memalign((void**)&scores, 16, seq_len * sizeof(float)) != 0) goto cleanup;
    if (posix_memalign((void**)&weights, 16, seq_len * sizeof(float)) != 0) goto cleanup;
    aum_attention_scores(q, k, scores, seq_len, head_dim, temperature);
    nikhilam_sparsha_attention_normalize(scores, weights, seq_len, 1.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int d = 0; d < head_dim; d++) {
            output[i * head_dim + d] = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                output[i * head_dim + d] += weights[j] * v[j * head_dim + d];
            }
        }
    }
cleanup:
    free(scores);
    free(weights);
}

__attribute__((visibility("default")))
const char* vedic_backend_version(void) {
    return "Vedic-Native-SLM-Backend v0.2.0 (+Vilokanam)";
}

__attribute__((visibility("default")))
int vedic_backend_init(void) { return 0; }

__attribute__((visibility("default")))
void vedic_backend_softmax(float* scores, float* weights, int n, float temp) {
    nikhilam_sparsha_attention_normalize(scores, weights, n, temp);
}

__attribute__((visibility("default")))
void vedic_backend_activate(float* x, int n) {
    triguna_qutrit_activate(x, n);
}

__attribute__((visibility("default")))
void vedic_backend_prune(float* w, int n, float threshold) {
    shunyata_prune_weights(w, n, threshold);
}

__attribute__((visibility("default")))
void vedic_backend_attention(
    const float* q, const float* k, const float* v,
    float* output, int seq_len, int head_dim, float temp
) {
    vedic_attention_full(q, k, v, output, seq_len, head_dim, temp);
}
