"""VEDIC TRANSFORMER — REAL DATA TRAINING (Fixed)"""
import torch, torch.nn as nn, math
from vedic_transformer import VedicTransformer
from torch.utils.data import DataLoader

print("🕉 VEDIC TRANSFORMER — MULTILINGUAL TRAINING")
print("=" * 40)

# Simple tokenizer that works
class SimpleTokenizer:
    def __init__(self, vocab_size=8000):
        self.vocab_size = vocab_size
    def __call__(self, text, max_length=128):
        tokens = []
        for c in text[:max_length * 4]:
            tokens.append(ord(c) % (self.vocab_size - 4) + 2)
        if len(tokens) < max_length:
            tokens += [1] * (max_length - len(tokens))
        return torch.tensor(tokens[:max_length], dtype=torch.long)

tokenizer = SimpleTokenizer(8000)
vocab_size = tokenizer.vocab_size

# Real multilingual training data
texts = [
    # Hindi
    "किसान खेत में काम कर रहा है। धान की फसल अच्छी है। जैविक खेती से मिट्टी की उर्वरता बढ़ती है। आज मौसम बहुत अच्छा है। बारिश होने की संभावना है। भारत एक कृषि प्रधान देश है। सरकार किसानों की मदद कर रही है। फसल बीमा योजना से किसानों को लाभ मिलता है।",
    "सूरजमुखी की खेती में अच्छा मुनाफा है। गेहूं की फसल तैयार है। मंडी में भाव अच्छे हैं।",
    # Bengali
    "কৃষক মাঠে কাজ করছে। ধানের ফসল ভালো হয়েছে। জৈব চাষ মাটির উর্বরতা বাড়ায়।",
    "আবহাওয়া খুব ভালো। বৃষ্টি হওয়ার সম্ভাবনা আছে। কৃষকরা খুব খুশি।",
    # Assamese
    "কৃষকে পথাৰত কাম কৰি আছে। ধানৰ ফচল ভাল হৈছে।",
    # Tamil
    "விவசாயி வயலில் வேலை செய்கிறார். நெல் பயிர் நன்றாக உள்ளது.",
    # English
    "Organic farming improves soil health and biodiversity. Farmers use natural pesticides.",
    "The weather forecast predicts rain tomorrow. Farmers are preparing their fields.",
] * 60  # 540 samples

# Tokenize
print(f"Tokenizing {len(texts)} texts...")
tokenized = []
for t in texts:
    tokens = tokenizer(t, max_length=64)
    tokenized.append(tokens)
data = torch.stack(tokenized)
print(f"Data: {data.shape}")

class D(torch.utils.data.Dataset):
    def __init__(self, d): self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return {"input_ids": self.d[i], "labels": self.d[i]}

loader = DataLoader(D(data), batch_size=4, shuffle=True)
print(f"Batches: {len(loader)}")

# Model — 10M params
model = VedicTransformer(vocab_size=vocab_size, dim=512, num_layers=6, num_heads=8)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params | {vocab_size} vocab")
print("72 Vedic algorithms | 6 layers | 5 languages\n")

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    total_loss = 0
    for i, batch in enumerate(loader):
        x = batch["input_ids"]
        y = batch["labels"]
        logits = model(x)
        loss = criterion(logits[:, :-1, :].reshape(-1, vocab_size), y[:, 1:].reshape(-1))
        if torch.isnan(loss) or torch.isinf(loss): continue
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        if i % 30 == 0:
            print(f"  Epoch {epoch} | Batch {i:3d} | Loss {loss.item():.4f}")
    avg = total_loss / len(loader)
    print(f"✅ Epoch {epoch}: Loss = {avg:.4f}")
    torch.save(model.state_dict(), f"vedic_real_epoch{epoch}.pt")

print("\n🎉 VEDIC TRANSFORMER TRAINED ON MULTILINGUAL DATA!")
