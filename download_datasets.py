"""
DOWNLOAD REAL DATASETS FROM HUGGINGFACE
For Vedic AI training
"""
import json, os

datasets = []

print("📡 Downloading real datasets from HuggingFace...")

# 1. Wikitext — Real Wikipedia articles
try:
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item['text'].strip()
        if len(t) > 100:
            texts.append(t)
        if len(texts) >= 500: break
    with open("dataset_wiki.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ Wikitext: {len(texts)} articles")
except Exception as e:
    print(f"  ⚠ Wikitext: {e}")

# 2. BookCorpus — Free books
try:
    ds = load_dataset("bookcorpus", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item['text'].strip()
        if len(t) > 100:
            texts.append(t[:500])
        if len(texts) >= 500: break
    with open("dataset_books.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ BookCorpus: {len(texts)} books excerpts")
except Exception as e:
    print(f"  ⚠ BookCorpus: {e}")

# 3. News — Real news articles
try:
    ds = load_dataset("ag_news", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item['text'].strip()
        if len(t) > 50:
            texts.append(t)
        if len(texts) >= 500: break
    with open("dataset_news.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ AG News: {len(texts)} articles")
except Exception as e:
    print(f"  ⚠ AG News: {e}")

# 4. OpenWebText — Web content
try:
    ds = load_dataset("openwebtext", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item['text'].strip()
        if len(t) > 100:
            texts.append(t[:500])
        if len(texts) >= 500: break
    with open("dataset_web.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ OpenWebText: {len(texts)} documents")
except Exception as e:
    print(f"  ⚠ OpenWebText: {e}")

# 5. C4 — Colossal Clean Crawled Corpus
try:
    ds = load_dataset("c4", "en", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item['text'].strip()
        if len(t) > 100:
            texts.append(t[:500])
        if len(texts) >= 500: break
    with open("dataset_c4.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ C4: {len(texts)} documents")
except Exception as e:
    print(f"  ⚠ C4: {e}")

# 6. Wikipedia — Full articles
try:
    ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item['text'].strip()
        if len(t) > 100:
            texts.append(t[:500])
        if len(texts) >= 500: break
    with open("dataset_wikipedia.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ Wikipedia: {len(texts)} articles")
except Exception as e:
    print(f"  ⚠ Wikipedia: {e}")

# 7. ArXiv — Scientific papers
try:
    ds = load_dataset("arxiv_dataset", split="train", streaming=True)
    texts = []
    for item in ds:
        t = item.get('abstract', '').strip()
        if len(t) > 100:
            texts.append(t)
        if len(texts) >= 300: break
    with open("dataset_arxiv.json", "w") as f:
        json.dump(texts, f)
    print(f"  ✅ ArXiv: {len(texts)} papers")
except Exception as e:
    print(f"  ⚠ ArXiv: {e}")

print(f"\n✅ Datasets downloaded!")
print("Files: dataset_wiki.json, dataset_books.json, dataset_news.json, dataset_web.json, dataset_c4.json, dataset_wikipedia.json, dataset_arxiv.json")
