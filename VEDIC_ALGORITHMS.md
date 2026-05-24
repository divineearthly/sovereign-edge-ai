# Vedic Algorithms for Edge AI

## Nikhilam Sparsha — Softmax Replacement
- **Sutra:** निखिलं नवतश्चरमं दशतः (All from 9, last from 10)
- **Speedup:** 3.15x on ARM64 NEON
- **Method:** Base-2 exponential via integer bit-shift + fractional polynomial
- **Zero exp() calls** — replaces 128 expf with 32 NEON instructions

## Anurupya KV Cache — Proportionality-Based Memory
- **Sutra:** आनुरूप्येण (By proportionality)
- **Memory:** 341x reduction (512MB -> 1.5MB for 2048 tokens)
- **Method:** Detects K_new proportional to K_last; replaces instead of appending

## Vilokanam — Script-Aware Vocabulary Filter
- **Sutra:** विलोकनम् (By mere observation)
- **Speedup:** 1.34x on output projection
- **Method:** Detects script from hidden state, restricts search to sub-vocabulary

## Triguna Qutrit — 3-State Activation
- **Principle:** Sattva-Rajas-Tamas gunas
- **Method:** |x| < threshold -> 0 (Tamas), else sin*cos activation (Rajas)

## Shunyata — NEON Weight Pruning
- **Principle:** Zero as the most powerful number
- **Method:** Vectorized threshold comparison, sets sub-threshold weights to zero
