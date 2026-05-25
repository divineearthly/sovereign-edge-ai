#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

// Anurupya hook: intercepts K,V pairs before cache insertion
// Returns 1 if this token should be STORED (phase change)
// Returns 0 if this token should be SKIPPED (proportional to previous)

typedef struct {
    float* last_k;       // [head_dim] most recent stored key
    float* last_v;       // [head_dim] most recent stored value
    float  alpha_threshold;
    float  norm_tolerance;
    int    head_dim;
    int    tokens_seen;
    int    tokens_stored;
    int    tokens_skipped;
} AnurupyaState;

void anurupya_init(AnurupyaState* s, int head_dim) {
    s->head_dim = head_dim;
    s->alpha_threshold = 0.05f;
    s->norm_tolerance = 0.1f;
    s->tokens_seen = 0;
    s->tokens_stored = 0;
    s->tokens_skipped = 0;
    posix_memalign((void**)&s->last_k, 16, head_dim * sizeof(float));
    posix_memalign((void**)&s->last_v, 16, head_dim * sizeof(float));
    memset(s->last_k, 0, head_dim * sizeof(float));
    memset(s->last_v, 0, head_dim * sizeof(float));
}

void anurupya_free(AnurupyaState* s) {
    free(s->last_k);
    free(s->last_v);
}

// NEON dot product
static float neon_dot(const float* a, const float* b, int n) {
    float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0), s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
    int i;
    for (i = 0; i <= n - 16; i += 16) {
        s0 = vmlaq_f32(s0, vld1q_f32(a+i), vld1q_f32(b+i));
        s1 = vmlaq_f32(s1, vld1q_f32(a+i+4), vld1q_f32(b+i+4));
        s2 = vmlaq_f32(s2, vld1q_f32(a+i+8), vld1q_f32(b+i+8));
        s3 = vmlaq_f32(s3, vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    }
    float32x4_t s01 = vaddq_f32(s0, s1), s23 = vaddq_f32(s2, s3), s = vaddq_f32(s01, s23);
    float r = vgetq_lane_f32(s,0)+vgetq_lane_f32(s,1)+vgetq_lane_f32(s,2)+vgetq_lane_f32(s,3);
    for (; i < n; i++) r += a[i] * b[i];
    return r;
}

// Returns 1 if should STORE this K,V pair (phase change or first token)
// Returns 0 if should SKIP (proportional to last stored pair)
int anurupya_should_store(AnurupyaState* s, const float* k_new, const float* v_new) {
    s->tokens_seen++;
    
    // Always store first token
    if (s->tokens_stored == 0) {
        memcpy(s->last_k, k_new, s->head_dim * sizeof(float));
        memcpy(s->last_v, v_new, s->head_dim * sizeof(float));
        s->tokens_stored++;
        return 1;
    }
    
    // Check proportionality with last stored key
    float dot_new_last = neon_dot(k_new, s->last_k, s->head_dim);
    float dot_last_last = neon_dot(s->last_k, s->last_k, s->head_dim);
    
    if (dot_last_last < 1e-8f) {
        memcpy(s->last_k, k_new, s->head_dim * sizeof(float));
        memcpy(s->last_v, v_new, s->head_dim * sizeof(float));
        s->tokens_stored++;
        return 1;
    }
    
    float alpha = dot_new_last / dot_last_last;
    float deviation = fabsf(alpha - 1.0f);
    
    if (deviation < s->alpha_threshold) {
        // Proportional — compute L2 error
        float l2_err = 0;
        for (int i = 0; i < s->head_dim; i++) {
            float diff = k_new[i] - alpha * s->last_k[i];
            l2_err += diff * diff;
        }
        float norm_err = l2_err / (dot_new_last + 1e-8f);
        
        if (norm_err < s->norm_tolerance) {
            // SKIP: proportional enough, update last to latest
            memcpy(s->last_k, k_new, s->head_dim * sizeof(float));
            memcpy(s->last_v, v_new, s->head_dim * sizeof(float));
            s->tokens_skipped++;
            return 0;
        }
    }
    
    // STORE: phase change detected
    memcpy(s->last_k, k_new, s->head_dim * sizeof(float));
    memcpy(s->last_v, v_new, s->head_dim * sizeof(float));
    s->tokens_stored++;
    return 1;
}

void anurupya_stats(AnurupyaState* s) {
    float ratio = s->tokens_seen > 0 ? (float)s->tokens_stored / s->tokens_seen : 1.0f;
    printf("Anurupya: seen=%d stored=%d skipped=%d (%.1f%% stored, %.1f:1 compression)\n",
           s->tokens_seen, s->tokens_stored, s->tokens_skipped,
           100.0f * ratio, 1.0f / ratio);
}

#ifdef ANURUPYA_HOOK_TEST
int main() {
    const int HD = 128;
    const int SEQ = 2048;
    AnurupyaState s;
    anurupya_init(&s, HD);
    
    float* k = malloc(HD * sizeof(float));
    float* v = malloc(HD * sizeof(float));
    float* k_base = malloc(HD * sizeof(float));
    
    srand(42);
    for (int i = 0; i < HD; i++) k_base[i] = ((float)rand()/RAND_MAX)*2-1;
    
    int stored = 0;
    int phase_changes = 0;
    
    for (int pos = 0; pos < SEQ; pos++) {
        if (pos > 0 && pos % 400 == 0) {
            phase_changes++;
            for (int i = 0; i < HD; i++) k_base[i] = ((float)rand()/RAND_MAX)*2-1;
        }
        for (int i = 0; i < HD; i++) {
            float noise = ((float)rand()/RAND_MAX - 0.5f) * 0.02f;
            k[i] = k_base[i] + noise;
            v[i] = k[i] * 0.5f;
        }
        if (anurupya_should_store(&s, k, v)) stored++;
    }
    
    printf("=== ANURUPYA KV HOOK TEST ===\n");
    printf("Sequence: %d, Phase changes: %d\n", SEQ, phase_changes);
    printf("Stored: %d / %d (%.1f%%)\n", stored, SEQ, 100.0*stored/SEQ);
    printf("Compression: %.1f:1\n", (float)SEQ/stored);
    anurupya_stats(&s);
    
    free(k); free(v); free(k_base);
    anurupya_free(&s);
    return 0;
}
#endif
