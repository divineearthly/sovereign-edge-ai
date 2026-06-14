#!/bin/bash
# Persistent Vedic Inference Server - keeps model loaded
cd ~/sovereign-edge-ai
echo "🕉 Vedic Inference Server Ready"
while IFS= read -r query; do
    if [ -n "$query" ]; then
        echo "$query" | timeout 5 ./vedic_qa_arm64 2>&1 | grep "A:" | head -1 | sed 's/A: //'
        echo "---READY---"
    fi
done
