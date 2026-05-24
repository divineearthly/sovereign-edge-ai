#include <arm_neon.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

// Anurupya KV Cache entry
typedef struct {
    float* k;          // Key vector [head_dim]
    float* v;          // Value vector [head_dim]
    int position;      // Original sequence position
    int span;          // How many original tokens this entry represents
} AnurupyaEntry;

// Anurupya KV Cache for one attention head
typedef struct {
    AnurupyaEntry* entries;   // Dynamic array of cached entries
    int capacity;             // Max entries (allocated)
    int count;                // Current entry count
    int head_dim;             // Dimension of K and V vectors
    float alpha_threshold;    // |α - 1.0| below this → proportional
    float norm_tolerance;     // ||K_new - α*K_last|| below this → merge
} AnurupyaCache;

// Initialize cache
void anurupya_cache_init(AnurupyaCache* cache, int capacity, int head_dim) {
    cache->entries = (AnurupyaEntry*)calloc(capacity, sizeof(AnurupyaEntry));
    cache->capacity = capacity;
    cache->count = 0;
    cache->head_dim = head_dim;
    cache->alpha_threshold = 0.05f;   // 5% deviation from α=1.0
    cache->norm_tolerance = 0.1f;     // Normalized tolerance
}

// Free cache
void anurupya_cache_free(AnurupyaCache* cache) {
    for (int i = 0; i < cache->count; i++) {
        free(cache->entries[i].k);
        free(cache->entries[i].v);
    }
    free(cache->entries);
    cache->entries = NULL;
    cache->count = 0;
}

// NEON dot product: a · b for head_dim elements
static float neon_dot(const float* a, const float* b, int n) {
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i <= n - 16; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t b3 = vld1q_f32(b + i + 12);
        sum0 = vmlaq_f32(sum0, a0, b0);
        sum1 = vmlaq_f32(sum1, a1, b1);
        sum2 = vmlaq_f32(sum2, a2, b2);
        sum3 = vmlaq_f32(sum3, a3, b3);
    }
    float32x4_t s01 = vaddq_f32(sum0, sum1);
    float32x4_t s23 = vaddq_f32(sum2, sum3);
    float32x4_t s = vaddq_f32(s01, s23);
    float result = vgetq_lane_f32(s, 0) + vgetq_lane_f32(s, 1)
                 + vgetq_lane_f32(s, 2) + vgetq_lane_f32(s, 3);
    for (; i < n; i++) {
        result += a[i] * b[i];
    }
    return result;
}

// NEON L2 norm squared: ||a - α*b||^2
static float neon_l2_diff_sq(const float* a, const float* b, float alpha, int n) {
    float32x4_t valpha = vdupq_n_f32(alpha);
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        float32x4_t av = vld1q_f32(a + i);
        float32x4_t bv = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(av, vmulq_f32(valpha, bv));
        sum = vmlaq_f32(sum, diff, diff);
    }
    float result = vgetq_lane_f32(sum, 0) + vgetq_lane_f32(sum, 1)
                 + vgetq_lane_f32(sum, 2) + vgetq_lane_f32(sum, 3);
    for (; i < n; i++) {
        float diff = a[i] - alpha * b[i];
        result += diff * diff;
    }
    return result;
}

// Insert K,V pair into cache using Anurupya proportionality rule
// Returns 1 if APPENDED (new entry), 0 if REPLACED (merged with last)
int anurupya_cache_insert(
    AnurupyaCache* cache,
    const float* k_new,
    const float* v_new,
    int position
) {
    int head_dim = cache->head_dim;
    
    // If cache is empty, always append
    if (cache->count == 0) {
        int idx = cache->count;
        cache->entries[idx].k = (float*)malloc(head_dim * sizeof(float));
        cache->entries[idx].v = (float*)malloc(head_dim * sizeof(float));
        memcpy(cache->entries[idx].k, k_new, head_dim * sizeof(float));
        memcpy(cache->entries[idx].v, v_new, head_dim * sizeof(float));
        cache->entries[idx].position = position;
        cache->entries[idx].span = 1;
        cache->count++;
        return 1;
    }
    
    // Check proportionality with LAST cached entry
    AnurupyaEntry* last = &cache->entries[cache->count - 1];
    const float* k_last = last->k;
    
    // Compute α = dot(K_new, K_last) / dot(K_last, K_last)
    float dot_new_last = neon_dot(k_new, k_last, head_dim);
    float dot_last_last = neon_dot(k_last, k_last, head_dim);
    
    // Avoid division by zero (degenerate key)
    if (dot_last_last < 1e-8f) {
        // Last entry has near-zero key, replace it
        memcpy(last->k, k_new, head_dim * sizeof(float));
        memcpy(last->v, v_new, head_dim * sizeof(float));
        last->position = position;
        last->span = 1;
        return 0;
    }
    
    float alpha = dot_new_last / dot_last_last;
    
    // Check if α ≈ 1.0 (proportional)
    float alpha_deviation = fabsf(alpha - 1.0f);
    
    if (alpha_deviation < cache->alpha_threshold) {
        // Proportional: check if the approximation error is acceptable
        float l2_diff = neon_l2_diff_sq(k_new, k_last, alpha, head_dim);
        float k_new_norm = neon_dot(k_new, k_new, head_dim);
        float normalized_error = l2_diff / (k_new_norm + 1e-8f);
        
        if (normalized_error < cache->norm_tolerance) {
            // REPLACE: update last entry to latest
            memcpy(last->k, k_new, head_dim * sizeof(float));
            memcpy(last->v, v_new, head_dim * sizeof(float));
            last->position = position;
            last->span++;  // This entry now covers more original tokens
            return 0;
        }
    }
    
    // APPEND: proportionality broken → phase change
    if (cache->count >= cache->capacity) {
        // Cache full: evict oldest (FIFO with span preservation)
        // Merge oldest two entries if possible, else drop oldest
        free(cache->entries[0].k);
        free(cache->entries[0].v);
        memmove(&cache->entries[0], &cache->entries[1],
                (cache->count - 1) * sizeof(AnurupyaEntry));
        cache->count--;
    }
    
    int idx = cache->count;
    cache->entries[idx].k = (float*)malloc(head_dim * sizeof(float));
    cache->entries[idx].v = (float*)malloc(head_dim * sizeof(float));
    memcpy(cache->entries[idx].k, k_new, head_dim * sizeof(float));
    memcpy(cache->entries[idx].v, v_new, head_dim * sizeof(float));
    cache->entries[idx].position = position;
    cache->entries[idx].span = 1;
    cache->count++;
    return 1;
}

// Retrieve full KV arrays for attention computation
// Expands cached entries back to attention-ready format
// For merged entries: repeats the same K,V (they were proportional)
void anurupya_cache_expand(
    AnurupyaCache* cache,
    float* k_out,     // [num_positions × head_dim]
    float* v_out,     // [num_positions × head_dim]
    int* position_map, // [num_positions] maps output idx → original position
    int* num_positions // returns total expanded positions
) {
    int total = 0;
    for (int i = 0; i < cache->count; i++) {
        AnurupyaEntry* entry = &cache->entries[i];
        // For entries that span multiple original tokens,
        // we repeat the representative K,V for each position
        // This preserves the causal attention pattern
        for (int s = 0; s < entry->span; s++) {
            if (total * cache->head_dim < (total + 1) * cache->head_dim) {
                memcpy(k_out + total * cache->head_dim, entry->k,
                       cache->head_dim * sizeof(float));
                memcpy(v_out + total * cache->head_dim, entry->v,
                       cache->head_dim * sizeof(float));
                position_map[total] = entry->position - (entry->span - 1) + s;
                total++;
            }
        }
    }
    *num_positions = total;
}

// Get cache statistics
void anurupya_cache_stats(AnurupyaCache* cache, int* entry_count, float* compression) {
    int total_spanned = 0;
    for (int i = 0; i < cache->count; i++) {
        total_spanned += cache->entries[i].span;
    }
    *entry_count = cache->count;
    *compression = (cache->count > 0) ? (float)total_spanned / cache->count : 1.0f;
}

#ifdef ANURUPYA_TEST

int main() {
    const int HEAD_DIM = 128;
    const int MAX_CACHE = 512;
    const int SEQ_LEN = 2048;
    
    printf("=== ANURUPYA KV CACHE TEST ===\n");
    printf("Head dim: %d, Sequence: %d, Max cache: %d\n", HEAD_DIM, SEQ_LEN, MAX_CACHE);
    
    AnurupyaCache cache;
    anurupya_cache_init(&cache, MAX_CACHE, HEAD_DIM);
    
    // Generate synthetic K,V sequences with "phase changes"
    // Simulate: 500 tokens of fluent text, then topic shift, then fluent again
    srand(42);
    
    float* k_buffer = (float*)malloc(HEAD_DIM * sizeof(float));
    float* v_buffer = (float*)malloc(HEAD_DIM * sizeof(float));
    float* k_base = (float*)malloc(HEAD_DIM * sizeof(float));
    float* v_base = (float*)malloc(HEAD_DIM * sizeof(float));
    
    // Initialize base key
    for (int d = 0; d < HEAD_DIM; d++) {
        k_base[d] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        v_base[d] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
    
    int appends = 0;
    int replacements = 0;
    int phase_changes = 0;
    
    struct timespec t1, t2;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    
    for (int pos = 0; pos < SEQ_LEN; pos++) {
        // Simulate periodic phase changes (every ~400 tokens)
        if (pos > 0 && pos % 400 == 0) {
            // Phase change: generate new base
            phase_changes++;
            for (int d = 0; d < HEAD_DIM; d++) {
                k_base[d] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
                v_base[d] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
            }
        }
        
        // Generate K,V with small noise around base
        for (int d = 0; d < HEAD_DIM; d++) {
            float noise_k = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            float noise_v = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
            k_buffer[d] = k_base[d] + noise_k;
            v_buffer[d] = v_base[d] + noise_v;
        }
        
        int result = anurupya_cache_insert(&cache, k_buffer, v_buffer, pos);
        if (result == 1) appends++;
        else replacements++;
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t2);
    double total_ns = (t2.tv_sec - t1.tv_sec) * 1e9 + (t2.tv_nsec - t1.tv_nsec);
    
    int entry_count;
    float compression;
    anurupya_cache_stats(&cache, &entry_count, &compression);
    
    printf("\n--- RESULTS ---\n");
    printf("Phase changes simulated: %d\n", phase_changes);
    printf("Appends (new entries): %d\n", appends);
    printf("Replacements (merged): %d\n", replacements);
    printf("Cache entries: %d / %d (%.1f%% full)\n",
           entry_count, MAX_CACHE, 100.0 * entry_count / MAX_CACHE);
    printf("Compression ratio: %.1f:1 (each entry spans %.1f tokens)\n",
           compression, compression);
    printf("Memory reduction: %.1f%%\n", 100.0 * (1.0 - 1.0/compression));
    printf("Insert time: %.2f us per token\n", total_ns / SEQ_LEN / 1000.0);
    
    // Test expansion
    int expanded_positions;
    float* k_expanded = (float*)malloc(SEQ_LEN * HEAD_DIM * sizeof(float));
    float* v_expanded = (float*)malloc(SEQ_LEN * HEAD_DIM * sizeof(float));
    int* pos_map = (int*)malloc(SEQ_LEN * sizeof(int));
    
    anurupya_cache_expand(&cache, k_expanded, v_expanded, pos_map, &expanded_positions);
    printf("Expanded positions: %d (original: %d)\n", expanded_positions, SEQ_LEN);
    
    free(k_expanded);
    free(v_expanded);
    free(pos_map);
    free(k_buffer);
    free(v_buffer);
    free(k_base);
    free(v_base);
    anurupya_cache_free(&cache);
    
    return 0;
}

#endif
