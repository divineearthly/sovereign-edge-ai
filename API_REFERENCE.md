# API Reference

Base: http://localhost:8000

## GET /health
System status

## POST /infer
Body: {"prompt":"...","lang":"en","max_tokens":128}
Returns: {"text":"...","cached":false}

## GET /farm?season=kharif
Farm advisory for Silchar, Assam

## POST /kavach
Body: {"text":"..."}
Returns: {"verdict":"SAFE|SUSPICIOUS|THREAT"}

## GET /cache
Offline cache status
