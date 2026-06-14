# 🕉 Sovereign Edge AI — API Reference
**Base URL:** http://localhost:8000
**Platform:** ARM64 Android/Termux

## GET /health
→ `{"status":"ok","vedic":true,"model":true}`

## POST /infer
← `{"prompt":"What is Brahman?","lang":"en"}`
→ `{"text":"Brahman is the ultimate reality..."}`

## POST /kavach
← `{"text":"message to scan"}`
→ `{"verdict":"SAFE/THREAT","score":-1.5}`

## GET /farm?season=kharif
→ `{"crops":["Rice","Jute"],"pests":[...]}`
