# Sovereign Edge AI - Deployment Guide

## Requirements
- Android phone with 4GB+ RAM
- Termux from F-Droid (not Play Store)

## Quick Install
  pkg update && pkg upgrade
  pkg install clang cmake ninja python git wget
  git clone https://github.com/divineearthly/sovereign-edge-ai
  cd sovereign-edge-ai

## Build Vedic Kernels
  clang -O3 -march=armv8-a+simd -fPIC -shared -o libvedic_full.so vedic_nikhilam_sparsha.c vedic_vilokanam_lib.c vedic_anurupya_cache.c -lm

## Start Server
  python3 server.py
  Open http://localhost:8000
