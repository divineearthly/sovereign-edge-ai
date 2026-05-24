# API Reference

Base URL: http://localhost:8000

## GET /health
Returns system status with Vedic/model/llama/cache flags

## POST /infer
Body: {"prompt":"...","lang":"en","max_tokens":128}
Returns: {"text":"...","cached":false}

## GET /farm?season=kharif
Farm advisory for Silchar region. Seasons: kharif, rabi, zaid

## POST /kavach
Body: {"text":"..."}
Returns: {"verdict":"SAFE|SUSPICIOUS|THREAT","score":0}

## GET /cache
Offline data cache status with last sync timestamp
