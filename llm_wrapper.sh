#!/bin/bash
# Wrapper that uses vedic_qa_arm64 for inference
PROMPT="$1"
cd ~/sovereign-edge-ai
echo "$PROMPT" | timeout 10 ./vedic_qa_arm64 2>&1 | grep "A:" | head -1 | sed 's/A: //'
