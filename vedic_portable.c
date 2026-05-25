#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void nikhilam_sparsha_attention_normalize(
    const float* scores, float* weights, int seq_len, float temperature
) {
    if (seq_len <= 0) return;
    float* scaled = (float*)malloc(seq_len * sizeof(float));
    if (!scaled) return;
    for (int i = 0; i < seq_len; i++) {
        scaled[i] = scores[i] * temperature;
    }
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        float c = scaled[i];
        if (c < -16.0f) c = -16.0f;
        if (c > 16.0f) c = 16.0f;
        sum += exp2f(c);
    }
    float inv = 1.0f / sum;
    for (int i = 0; i < seq_len; i++) {
        float c = scaled[i];
        if (c < -16.0f) c = -16.0f;
        if (c > 16.0f) c = 16.0f;
        weights[i] = exp2f(c) * inv;
    }
    free(scaled);
}
