#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

static inline float32x4_t neon_exp2f(float32x4_t x) {
    float32x4_t vmin = vdupq_n_f32(-16.0f);
    float32x4_t vmax = vdupq_n_f32(16.0f);
    x = vminq_f32(vmaxq_f32(x, vmin), vmax);
    int32x4_t xi = vcvtq_s32_f32(x);
    float32x4_t xf = vcvtq_f32_s32(xi);
    float32x4_t frac = vsubq_f32(x, xf);
    float32x4_t frac_poly = vdupq_n_f32(1.0f);
    frac_poly = vmlaq_n_f32(frac_poly, frac, 0.693147f);
    frac_poly = vmlaq_n_f32(frac_poly, frac, 0.240226f);
    frac_poly = vmlaq_n_f32(frac_poly, frac, 0.055504f);
    float32x4_t pow2_int;
    int32x4_t exp_bias = vdupq_n_s32(127);
    int32x4_t shift_amt = vdupq_n_s32(23);
    pow2_int = vreinterpretq_f32_s32(
        vshlq_s32(vaddq_s32(xi, exp_bias), shift_amt)
    );
    return vmulq_f32(pow2_int, frac_poly);
}

void nikhilam_sparsha_attention_normalize(
    const float* scores,
    float* weights,
    int seq_len,
    float temperature
) {
    if (seq_len <= 0) return;
    float* scaled = NULL;
    if (posix_memalign((void**)&scaled, 16, seq_len * sizeof(float)) != 0) {
        return;
    }
    float32x4_t vtemp = vdupq_n_f32(temperature);
    int i;
    for (i = 0; i <= seq_len - 4; i += 4) {
        float32x4_t s = vld1q_f32(scores + i);
        vst1q_f32(scaled + i, vmulq_f32(s, vtemp));
    }
    for (; i < seq_len; i++) {
        scaled[i] = scores[i] * temperature;
    }
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    for (i = 0; i <= seq_len - 16; i += 16) {
        float32x4_t e0 = neon_exp2f(vld1q_f32(scaled + i));
        float32x4_t e1 = neon_exp2f(vld1q_f32(scaled + i + 4));
        float32x4_t e2 = neon_exp2f(vld1q_f32(scaled + i + 8));
        float32x4_t e3 = neon_exp2f(vld1q_f32(scaled + i + 12));
        sum0 = vaddq_f32(sum0, e0);
        sum1 = vaddq_f32(sum1, e1);
        sum2 = vaddq_f32(sum2, e2);
        sum3 = vaddq_f32(sum3, e3);
        vst1q_f32(weights + i, e0);
        vst1q_f32(weights + i + 4, e1);
        vst1q_f32(weights + i + 8, e2);
        vst1q_f32(weights + i + 12, e3);
    }
    float sum_exp = vgetq_lane_f32(sum0, 0) + vgetq_lane_f32(sum0, 1)
                  + vgetq_lane_f32(sum0, 2) + vgetq_lane_f32(sum0, 3)
                  + vgetq_lane_f32(sum1, 0) + vgetq_lane_f32(sum1, 1)
                  + vgetq_lane_f32(sum1, 2) + vgetq_lane_f32(sum1, 3)
                  + vgetq_lane_f32(sum2, 0) + vgetq_lane_f32(sum2, 1)
                  + vgetq_lane_f32(sum2, 2) + vgetq_lane_f32(sum2, 3)
                  + vgetq_lane_f32(sum3, 0) + vgetq_lane_f32(sum3, 1)
                  + vgetq_lane_f32(sum3, 2) + vgetq_lane_f32(sum3, 3);
    for (; i < seq_len; i++) {
        float e = exp2f(scaled[i]);
        weights[i] = e;
        sum_exp += e;
    }
    float inv_sum = 1.0f / sum_exp;
    float32x4_t vinv = vdupq_n_f32(inv_sum);
    for (i = 0; i <= seq_len - 4; i += 4) {
        float32x4_t w = vld1q_f32(weights + i);
        vst1q_f32(weights + i, vmulq_f32(w, vinv));
    }
    for (; i < seq_len; i++) {
        weights[i] *= inv_sum;
    }
    free(scaled);
}

int main() {
    const int SEQ = 128;
    const int TRIALS = 10000;
    const float TEMP = 0.125f;
    float scores[SEQ];
    float vedic_w[SEQ];
    float std_w[SEQ];
    int i;
    srand(42);
    for (i = 0; i < SEQ; i++) {
        scores[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }
    nikhilam_sparsha_attention_normalize(scores, vedic_w, SEQ, TEMP);
    float max_s = scores[0];
    for (i = 1; i < SEQ; i++) {
        if (scores[i] > max_s) max_s = scores[i];
    }
    float sum_exp = 0.0f;
    for (i = 0; i < SEQ; i++) {
        std_w[i] = expf((scores[i] - max_s) * TEMP);
        sum_exp += std_w[i];
    }
    for (i = 0; i < SEQ; i++) {
        std_w[i] /= sum_exp;
    }
    float max_err = 0.0f;
    float vedic_sum = 0.0f;
    int worst_idx = 0;
    for (i = 0; i < SEQ; i++) {
        float err = fabsf(vedic_w[i] - std_w[i]);
        if (err > max_err) {
            max_err = err;
            worst_idx = i;
        }
        vedic_sum += vedic_w[i];
    }
    printf("=== NIKHILAM SPARSHA (NEON exp2f) ===\n");
    printf("Max error: %.6f (idx %d)\n", max_err, worst_idx);
    printf("  Vedic[%d] = %.6f\n", worst_idx, vedic_w[worst_idx]);
    printf("  Std[%d]   = %.6f\n", worst_idx, std_w[worst_idx]);
    printf("Vedic sum: %.6f\n", vedic_sum);
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (i = 0; i < TRIALS; i++) {
        nikhilam_sparsha_attention_normalize(scores, vedic_w, SEQ, TEMP);
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double vedic_ns = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (i = 0; i < TRIALS; i++) {
        float ms = scores[0];
        int k;
        for (k = 1; k < SEQ; k++) {
            if (scores[k] > ms) ms = scores[k];
        }
        float se = 0.0f;
        for (k = 0; k < SEQ; k++) {
            std_w[k] = expf((scores[k] - ms) * TEMP);
            se += std_w[k];
        }
        for (k = 0; k < SEQ; k++) {
            std_w[k] /= se;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double std_ns = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    printf("Vedic:   %.2f us/call\n", vedic_ns / TRIALS / 1000.0);
    printf("Standard: %.2f us/call\n", std_ns / TRIALS / 1000.0);
    printf("Speedup: %.2fx\n", std_ns / vedic_ns);
    return 0;
}
