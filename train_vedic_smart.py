"""CONTINUE TRAINING 10M VEDIC TRANSFORMER ON LIVE DATA"""
import torch, torch.nn as nn, time, random, json, urllib.request, re
from vedic_transformer import VedicTransformer
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

print("🕉 SMART VEDIC TRAINING — Live Data on Trained Model")

# Load trained model
tokenizer = Tokenizer.from_file("vedic_tokenizer.json")
vocab = tokenizer.get_vocab_size()
model = VedicTransformer(vocab_size=vocab, dim=512, num_layers=6, num_heads=8)
model.load_state_dict(torch.load("vedic_bpe_10m.pt", map_location="cpu"))
print(f"✅ Loaded 10M Vedic Transformer (pre-trained, loss 0.29)")

# Fetch fresh live data
texts = []
print("📡 Fetching live knowledge...")

# Wikipedia
for title in ["Artificial_intelligence", "Machine_learning", "Deep_learning",
              "India", "Agriculture", "Philosophy", "Consciousness",
              "Vedas", "Upanishads", "Bhagavad_Gita", "Yoga", "Ayurveda",
              "Organic_farming", "Neural_network", "Mathematics"]:
    try:
        url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&exlimit=1&explaintext=1&format=json"
        req = urllib.request.Request(url, headers={'User-Agent': 'VedicAI/1.0'})
        with urllib.request.urlopen(req, timeout=5) as f:
            data = json.loads(f.read().decode())
            for page in data.get('query',{}).get('pages',{}).values():
                for s in re.split(r'[.!?]+', page.get('extract','')):
                    s = s.strip()
                    if 50 < len(s) < 500: texts.append(s)
    except: pass

# Vedic + Universal wisdom
wisdom = [
    "Brahman is ultimate reality infinite eternal consciousness beyond all attributes",
    "Atman is identical with Brahman the cosmic self tat tvam asi",
    "Yoga is cessation of mind fluctuations for complete inner liberation",
    "Dharma is righteous action sustaining cosmic order",
    "Consciousness is fundamental substrate of all existence",
    "The universe exists in eternal cycles without beginning or end",
    "Love is the fundamental force binding the universe together",
]
texts.extend(wisdom * 200)
random.shuffle(texts)
texts = texts[:3000]

# Encode
def encode(t, mx=64):
    ids = tokenizer.encode(t).ids[:mx]
    return torch.tensor(ids + [0]*(mx-len(ids)), dtype=torch.long)

data = torch.stack([encode(t) for t in texts])
loader = DataLoader(data, batch_size=8, shuffle=True)
print(f"✅ {len(texts)} texts | {len(loader)} batches | Training...\n")

# Train with low LR (fine-tuning)
opt = torch.optim.AdamW(model.parameters(), lr=1e-5)  # Very low LR for fine-tuning
crit = nn.CrossEntropyLoss(ignore_index=0)
best = float('inf')

for ep in range(3):
    model.train()
    total = 0
    t0 = time.time()
    for i, batch in enumerate(loader):
        logits = model(batch)
        loss = crit(logits[:,:-1,:].reshape(-1,vocab), batch[:,1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item()
        if i % 30 == 0:
            print(f"  Ep {ep+1} | Bat {i:3d}/{len(loader)} | Loss {loss.item():.4f} | {time.time()-t0:.0f}s")
    avg = total/len(loader)
    print(f"✅ Epoch {ep+1}: Loss = {avg:.4f}")
    if avg < best:
        best = avg
        torch.save(model.state_dict(), "vedic_smart_10m.pt")
        print(f"   📦 Saved!")

print(f"\n🎉 SMART TRAINING DONE! Best loss: {best:.4f}")
