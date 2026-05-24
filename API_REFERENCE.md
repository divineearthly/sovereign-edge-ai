# API Reference

Base: `http://localhost:8000`

## GET /health
System status. Returns: `{"status":"ok","vedic":true,"model":true}`

## POST /infer
Inference with Vedic SLM.
- Body: `{"prompt":"...","lang":"en","max_tokens":128,"temperature":0.7}`
- Returns: `{"text":"...","cached":false}`

## GET /farm?season=kharif
Farm advisory (Silchar, Assam).
- Seasons: kharif, rabi, zaid
- Returns: `{"season":"...","crops":[...],"pests":[...],"soil":"..."}`

## POST /kavach
Safety check.
- Body: `{"text":"..."}`
- Returns: `{"verdict":"SAFE|SUSPICIOUS|THREAT","score":0}`

## GET /cache
Offline cache status.
