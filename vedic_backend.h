#ifndef VEDIC_BACKEND_H
#define VEDIC_BACKEND_H

#ifdef __cplusplus
extern "C" {
#endif

const char* vedic_backend_version(void);
int vedic_backend_init(void);
void vedic_backend_softmax(float* scores, float* weights, int n, float temp);
void vedic_backend_activate(float* x, int n);
void vedic_backend_prune(float* w, int n, float threshold);
void vedic_backend_attention(
    const float* q, const float* k, const float* v,
    float* output, int seq_len, int head_dim, float temp
);

#ifdef __cplusplus
}
#endif

#endif
