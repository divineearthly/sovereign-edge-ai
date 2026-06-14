#!/bin/bash
echo "=== Instant Trainer ===" && ./train_vedic_instant_arm64 2>&1 | grep -E "complete|loss|Model"
echo "=== Live Trainer ===" && timeout 15 ./vedic_cpp_live_arm64 2>&1 | grep -E "complete|loss|Samples"
echo "=== Supreme Inference ===" && echo "What is dharma?" | timeout 5 ./vedic_supreme_arm64 2>&1 | grep -E "Q:|A:|Confidence:" | head -4
