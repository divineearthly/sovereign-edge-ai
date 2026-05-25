#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

typedef struct {
    float* W1;
    float* W2;
    float* x_mean;
    float  deficiency_threshold;
    int    dim;
    int    hidden;
    int*   active_map;
    int    num_active;
    int    total_calls;
    int    total_skipped;
} YavadunamFFN;

void yavadunam_init(YavadunamFFN* ffn, int dim, int hidden, float threshold) {
    ffn->dim = dim;
    ffn->hidden = hidden;
    ffn->deficiency_threshold = threshold;
    ffn->num_active = 0;
    ffn->total_calls = 0;
    ffn->total_skipped = 0;
    posix_memalign((void**)&ffn->W1, 16, dim * hidden * sizeof(float));
    posix_memalign((void**)&ffn->W2, 16, hidden * dim * sizeof(float));
    ffn->x_mean = (float*)calloc(dim, sizeof(float));
    ffn->active_map = (int*)malloc(hidden * sizeof(int));
    for (int i = 0; i < dim * hidden; i++) {
        ffn->W1[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
        ffn->W2[i] = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.1f;
    }
}

void yavadunam_free(YavadunamFFN* ffn) {
    free(ffn->W1); free(ffn->W2); free(ffn->x_mean); free(ffn->active_map);
}

static void neon_deficiency(const float* x, const float* x_mean, float* def, int n) {
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t xv = vld1q_f32(x + i);
        float32x4_t mv = vld1q_f32(x_mean + i);
        vst1q_f32(def + i, vabsq_f32(vsubq_f32(xv, mv)));
    }
    for (; i < n; i++) def[i] = fabsf(x[i] - x_mean[i]);
}

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

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

void standard_ffn(YavadunamFFN* ffn, const float* x, float* out) {
    int d = ffn->dim, h = ffn->hidden;
    float* hs = malloc(h * sizeof(float));
    for (int j = 0; j < h; j++) hs[j] = silu(neon_dot(x, ffn->W1 + j*d, d));
    for (int i = 0; i < d; i++) {
        float s = 0;
        for (int j = 0; j < h; j++) s += hs[j] * ffn->W2[j*d + i];
        out[i] = s;
    }
    free(hs);
}

void yavadunam_ffn(YavadunamFFN* ffn, const float* x, float* out) {
    int d = ffn->dim, h = ffn->hidden;
    float th = ffn->deficiency_threshold;
    ffn->total_calls++;
    float* def = malloc(d * sizeof(float));
    neon_deficiency(x, ffn->x_mean, def, d);
    ffn->num_active = 0;
    int* am = calloc(h, sizeof(int));
    for (int j = 0; j < h; j++) {
        float md = 0;
        for (int s = 0; s < 8; s++) {
            int idx = (j * 7 + s * 13) % d;
            if (def[idx] > md) md = def[idx];
        }
        if (md > th) { am[j] = 1; ffn->active_map[ffn->num_active++] = j; }
    }
    ffn->total_skipped += (h - ffn->num_active);
    float* hs = malloc(h * sizeof(float));
    memset(hs, 0, h * sizeof(float));
    for (int a = 0; a < ffn->num_active; a++) {
        int j = ffn->active_map[a];
        hs[j] = silu(neon_dot(x, ffn->W1 + j*d, d));
    }
    for (int i = 0; i < d; i++) {
        float s = 0;
        for (int a = 0; a < ffn->num_active; a++) {
            int j = ffn->active_map[a];
            s += hs[j] * ffn->W2[j*d + i];
        }
        out[i] = s;
    }
    float alpha = 0.01f;
    for (int i = 0; i < d; i++) ffn->x_mean[i] = 0.99f * ffn->x_mean[i] + 0.01f * x[i];
    free(def); free(am); free(hs);
}

#ifdef YAVADUNAM_TEST
int main() {
    const int D = 2048, H = 8192, T = 20;
    const float TH = 0.1f;
    printf("=== YAVADUNAM FFN TEST ===\nDim: %d, Hidden: %d\n", D, H);
    YavadunamFFN f;
    yavadunam_init(&f, D, H, TH);
    float* x = NULL, *os = NULL, *ov = NULL;
    posix_memalign((void**)&x, 16, D*sizeof(float));
    posix_memalign((void**)&os, 16, D*sizeof(float));
    posix_memalign((void**)&ov, 16, D*sizeof(float));
    srand(42);
    for (int i = 0; i < D; i++) {
        float u1 = (float)rand()/RAND_MAX, u2 = (float)rand()/RAND_MAX;
        x[i] = sqrtf(-2*logf(u1+1e-10f)) * cosf(6.283f*u2);
    }
    yavadunam_ffn(&f, x, ov);
    standard_ffn(&f, x, os);
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int t = 0; t < T; t++) yavadunam_ffn(&f, x, ov);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double vns = (t2.tv_sec-t1.tv_sec)*1e9+(t2.tv_nsec-t1.tv_nsec);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    for (int t = 0; t < T; t++) standard_ffn(&f, x, os);
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double sns = (t2.tv_sec-t1.tv_sec)*1e9+(t2.tv_nsec-t1.tv_nsec);
    float me = 0, se = 0;
    for (int i = 0; i < D; i++) { float e = fabsf(ov[i]-os[i]); if(e>me)me=e; se+=e; }
    float ac = (float)(H*f.total_calls-f.total_skipped)/f.total_calls;
    printf("Active: %.0f/%d (%.0f%% sparse)\n", ac, H, 100*(1-ac/H));
    printf("Max err: %.4f  Mean err: %.6f\n", me, se/D);
    printf("Yavadunam: %.1f us  Standard: %.1f us  Speedup: %.2fx\n", vns/T/1000, sns/T/1000, sns/vns);
    free(x); free(os); free(ov); yavadunam_free(&f);
    return 0;
}
#endif
