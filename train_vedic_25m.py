"""25M VEDIC TRANSFORMER — 10K Multilingual Texts"""
import torch, torch.nn as nn, random
from vedic_transformer import VedicTransformer
from torch.utils.data import DataLoader

print("🕉 25M VEDIC TRANSFORMER TRAINING")
print("=" * 40)

# Simple tokenizer
class Tokenizer:
    def __init__(self, vocab_size=16000):
        self.vocab_size = vocab_size
        self.char_map = {}
    def encode(self, text, max_len=128):
        tokens = []
        for c in text:
            code = ord(c)
            if code not in self.char_map:
                self.char_map[code] = len(self.char_map) + 2
            tid = self.char_map[code] % (self.vocab_size - 2)
            tokens.append(tid + 2 if tid > 0 else 2)
        tokens = tokens[:max_len]
        tokens += [1] * (max_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long)

tokenizer = Tokenizer(16000)
vocab_size = tokenizer.vocab_size

# Load data
with open("vedic_training_data.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(texts)} texts")

# Generate data first
print("Generating training data...")
exec(open("generate_vedic_data.py").read())

# Tokenize
tokenized = [tokenizer.encode(t, 128) for t in all_texts[:8000]]
data = torch.stack(tokenized)
print(f"Data: {data.shape}")

class D(torch.utils.data.Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return {"input_ids": self.d[i], "labels": self.d[i]}

loader = DataLoader(D(data), batch_size=8, shuffle=True)
print(f"Batches: {len(loader)}")

# 25M Model
model = VedicTransformer(vocab_size=vocab_size, dim=768, num_layers=12, num_heads=12)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params:,} params | {vocab_size} vocab | 12 layers | 72 Vedic ops\n")

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
best_loss = float('inf')

for epoch in range(10):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        x, y = batch["input_ids"], batch["labels"]
        logits = model(x)
        loss = criterion(logits[:, :-1, :].reshape(-1, vocab_size), y[:, 1:].reshape(-1))
        if torch.isnan(loss) or torch.isinf(loss): continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()
        if i % 50 == 0:
            print(f"  Epoch {epoch} | Batch {i:4d} | Loss {loss.item():.4f} | {params//1e6:.0f}M params")
    avg = total_loss / len(loader)
    print(f"✅ Epoch {epoch}: Loss = {avg:.4f}")
    if avg < best_loss:
        best_loss = avg
        torch.save(model.state_dict(), "vedic_25m_best.pt")
        print(f"   📦 Saved best model (loss: {avg:.4f})")

print(f"\n🎉 25M VEDIC TRANSFORMER TRAINED!")
print(f"Best loss: {best_loss:.4f}")
