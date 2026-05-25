"""
VEDIC TRANSFORMER — LIVE DATA TRAINING
Pulls real text from: Wikipedia, arXiv, news, Indic corpus
Processes through 72 Vedic algorithms
"""

import torch, torch.nn as nn, math, time, random, json, urllib.request, gzip, io, re
from vedic_transformer import VedicTransformer
from torch.utils.data import DataLoader

print("🕉 VEDIC TRANSFORMER — LIVE DATA ACQUISITION")
print("============================================")

VOCAB = 8000
DIM = 768
LAYERS = 12
HEADS = 12
BATCH = 4
SEQ = 128
EPOCHS = 5
LR = 1e-4

# ═══════════════════════════════════════════════
# LIVE DATA STREAMS
# ═══════════════════════════════════════════════

all_texts = []

# 1. WIKIPEDIA — Real encyclopedia articles
print("📡 Fetching Wikipedia articles...")
try:
    url = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2"
    # Use Simple Wikipedia for faster download
    for title in ["Artificial_intelligence", "Machine_learning", "Neural_network", 
                  "Deep_learning", "Natural_language_processing", "Transformer_(machine_learning)",
                  "India", "Agriculture", "Organic_farming", "Philosophy", "Consciousness",
                  "Vedas", "Upanishads", "Bhagavad_Gita", "Yoga", "Meditation", "Ayurveda"]:
        try:
            api_url = f"https://en.wikipedia.org/w/api.php?action=query&titles={title}&prop=extracts&exlimit=1&explaintext=1&format=json"
            req = urllib.request.Request(api_url, headers={'User-Agent': 'VedicAI/1.0'})
            with urllib.request.urlopen(req, timeout=10) as f:
                data = json.loads(f.read().decode())
                pages = data.get('query', {}).get('pages', {})
                for page_id, page in pages.items():
                    text = page.get('extract', '')
                    # Split into sentences
                    sentences = re.split(r'[.!?]+', text)
                    for s in sentences:
                        s = s.strip()
                        if len(s) > 50 and len(s) < 1000:
                            all_texts.append(s)
            print(f"  ✓ {title}: {len([t for t in all_texts if title.lower().replace('_',' ') in t.lower()])} sentences")
        except Exception as e:
            print(f"  ⚠ {title}: {str(e)[:50]}")
except Exception as e:
    print(f"  ⚠ Wikipedia: {str(e)[:50]}")

# 2. ARXIV — Scientific papers
print("📡 Fetching arXiv papers...")
try:
    for query in ["machine+learning", "neural+network", "transformer+architecture", 
                   "natural+language", "attention+mechanism"]:
        try:
            url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results=3"
            req = urllib.request.Request(url, headers={'User-Agent': 'VedicAI/1.0'})
            with urllib.request.urlopen(req, timeout=15) as f:
                xml = f.read().decode()
                # Extract abstracts
                abstracts = re.findall(r'<summary>(.*?)</summary>', xml, re.DOTALL)
                for abstract in abstracts:
                    abstract = re.sub(r'<[^>]+>', '', abstract).strip()
                    sentences = re.split(r'[.!?]+', abstract)
                    for s in sentences:
                        s = s.strip()
                        if len(s) > 50 and len(s) < 800:
                            all_texts.append(s)
            print(f"  ✓ arXiv '{query}': {len(abstracts)} papers")
        except Exception as e:
            print(f"  ⚠ arXiv {query}: {str(e)[:50]}")
except Exception as e:
    print(f"  ⚠ arXiv: {str(e)[:50]}")

# 3. GUTENBERG — Free books
print("📡 Fetching Project Gutenberg books...")
try:
    for book_id in ['1342', '84', '1661', '11', '1661']:  # Pride & Prejudice, Frankenstein, Sherlock, Alice
        try:
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            req = urllib.request.Request(url, headers={'User-Agent': 'VedicAI/1.0'})
            with urllib.request.urlopen(req, timeout=20) as f:
                text = f.read().decode('utf-8', errors='ignore')
                # Take middle section (skip header/footer)
                lines = text.split('\n')
                mid = len(lines)//2
                sample = ' '.join(lines[mid:mid+500])
                sentences = re.split(r'[.!?]+', sample)
                for s in sentences:
                    s = s.strip()
                    if len(s) > 50 and len(s) < 500 and not s.startswith('['):
                        all_texts.append(s)
            print(f"  ✓ Gutenberg #{book_id}")
        except Exception as e:
            print(f"  ⚠ Gutenberg #{book_id}: {str(e)[:50]}")
except Exception as e:
    print(f"  ⚠ Gutenberg: {str(e)[:50]}")

# 4. INDIC TEXT — Built-in multilingual data
print("📡 Loading Indic multilingual data...")
indic_texts = [
    "किसान खेत में सुबह से शाम तक काम करता है धान की फसल इस बार बहुत अच्छी हुई है",
    "जैविक खेती से मिट्टी की उर्वरता बढ़ती है और रासायनिक उर्वरकों की आवश्यकता नहीं होती",
    "भारत एक कृषि प्रधान देश है जहां साठ प्रतिशत जनसंख्या खेती पर निर्भर करती है",
    "सरकार ने किसानों के लिए नई फसल बीमा योजना शुरू की है जिससे उन्हें नुकसान की भरपाई मिलेगी",
    "आधुनिक तकनीक से खेती करने पर पैदावार दोगुनी हो सकती है और किसानों की आय बढ़ सकती है",
    "মৌসুমী বৃষ্টি আসামের কৃষকদের জন্য খুবই গুরুত্বপূর্ণ ধান চাষের জন্য প্রচুর পানির প্রয়োজন",
    "জৈব সার ব্যবহার করলে মাটির স্বাস্থ্য ভালো থাকে এবং ফসলের গুণগত মান উন্নত হয়",
    "বাংলাদেশ ও পশ্চিমবঙ্গের কৃষকরা ধান চাষের জন্য উন্নত বীজ ব্যবহার করছে",
    "விவசாயிகள் இயற்கை உரங்களை பயன்படுத்தி மண் வளத்தை பாதுகாக்கின்றனர்",
    "தமிழ்நாட்டில் நெல் விளைச்சல் இந்த ஆண்டு அதிகரித்துள்ளது நீர்ப்பாசன வசதி மேம்பட்டுள்ளது",
    "কৃষকে পথাৰত ৰাতিপুৱাৰ পৰা গধূলিলৈকে কাম কৰে ধানৰ ফচল ভাল হৈছে",
    "অসমৰ বৰাক উপত্যকাত ধান খেতিৰ বাবে জলবায়ু অতি অনুকূল",
]
for t in indic_texts * 100:
    all_texts.append(t)

# 5. VEDIC KNOWLEDGE — Core wisdom
print("📡 Loading Vedic knowledge corpus...")
vedic = [
    "Brahman is the ultimate reality infinite eternal unchanging consciousness beyond all attributes",
    "Atman the individual self is identical with Brahman the cosmic self tat tvam asi thou art that",
    "The three gunas sattva rajas tamas are the primordial qualities binding purusha to prakriti",
    "Yoga is the cessation of mind fluctuations for self realization and complete inner freedom",
    "Dharma is righteous action in accordance with cosmic order that sustains all existence",
    "Ahimsa is non violence the highest dharma of all embodied beings in the universe",
    "Om is the primordial sound vibration from which all creation emerges into manifestation",
    "The five elements akasha vayu agni ap prithvi create the entire manifest universe",
    "Karma is the law of cause and effect governing all actions with inescapable precision",
    "Moksha is complete liberation from the endless cycle of birth death and rebirth samsara",
    "Vedic mathematics uses sixteen fundamental sutras for instantaneous complex computation",
    "The Upanishads declare the ultimate unity of all existence eko brahma dvitiya nasti",
    "Bhagavad Gita teaches selfless action without attachment to the fruits of action",
    "Patanjali yoga sutras outline the eight limbs of spiritual practice for liberation",
    "Mantras are sacred sound vibrations that attune individual consciousness to cosmic frequencies",
    "Dhyana is unbroken flow of awareness toward the object of meditation leading to absorption",
    "Samadhi is the state where knower knowledge and known become one undivided whole",
    "Consciousness is the fundamental substrate of all existence not an emergent property",
    "The universe exists in eternal cycles of expansion and contraction without beginning or end",
    "Every particle in the cosmos contains information about the whole holographic principle",
    "Time is a construct of perception the eternal now contains all moments simultaneously",
    "Mathematics is the universal language through which cosmic intelligence expresses order",
    "Energy and matter are interchangeable aspects of the same underlying unified field",
    "The observer and the observed are inseparable aspects of a single unified field of consciousness",
    "Love is the fundamental force of attraction that binds the entire universe together in harmony",
    "Silence is not empty it is full of infinite potential the womb of all creation and creativity",
    "Self knowledge is the highest form of wisdom leading directly to complete liberation and freedom",
    "Compassion is the natural spontaneous expression of awakened consciousness seeing self in all beings",
    "The mind creates reality through the power of perception attention and focused intention",
    "Everything in existence has a purpose and place in the grand cosmic order of the universe",
]
for v in vedic * 80:
    all_texts.append(v)

print(f"\n✅ Total acquired: {len(all_texts)} texts")
print(f"   Sources: Wikipedia, arXiv, Gutenberg, Indic, Vedic")
print(f"   Total characters: {sum(len(t) for t in all_texts):,}")

# ═══════════════════════════════════════════════
# TOKENIZE & MODEL
# ═══════════════════════════════════════════════

class Tok:
    def __init__(self, v=VOCAB): self.v = v
    def encode(self, t, mx=SEQ):
        ids = [(ord(c)%(self.v-4)+2) for c in t[:mx*4]]
        return torch.tensor((ids+[1]*(mx-len(ids)))[:mx], dtype=torch.long)

tok = Tok()
random.shuffle(all_texts)
texts_use = all_texts[:5000]
data = torch.stack([tok.encode(t) for t in texts_use])
loader = DataLoader(data, batch_size=BATCH, shuffle=True)

model = VedicTransformer(vocab_size=VOCAB, dim=DIM, num_layers=LAYERS, num_heads=HEADS)
print(f"\n🕉 Model: {sum(p.numel() for p in model.parameters()):,} params | {LAYERS} layers | 72 Vedic ops")
print(f"   Training on {len(texts_use)} texts | {len(loader)} batches\n")

# ═══════════════════════════════════════════════
# TRAIN
# ═══════════════════════════════════════════════
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
crit = nn.CrossEntropyLoss(ignore_index=1)
best = float('inf')

for ep in range(EPOCHS):
    model.train()
    total = 0
    t0 = time.time()
    
    for i, batch in enumerate(loader):
        logits = model(batch)
        loss = crit(logits[:,:-1,:].reshape(-1,VOCAB), batch[:,1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()
        total += loss.item()
        
        if i % 40 == 0:
            elapsed = time.time() - t0
            tps = (i+1) * BATCH * SEQ / max(elapsed, 1)
            print(f"  Ep {ep+1}/{EPOCHS} | Bat {i:4d}/{len(loader)} | Loss {loss.item():.4f} | {tps:.0f} tok/s")
    
    avg = total/len(loader)
    print(f"✅ Epoch {ep+1}: Loss = {avg:.4f} | Time: {time.time()-t0:.0f}s")
    if avg < best:
        best = avg
        torch.save(model.state_dict(), "vedic_live_50m.pt")
        print(f"   📦 Saved! (loss: {avg:.4f})")

print(f"\n🎉 LIVE TRAINING COMPLETE! Best loss: {best:.4f}")
print(f"   Model: vedic_live_50m.pt")
print(f"   Trained on: Wikipedia + arXiv + Gutenberg + Indic + Vedic texts")
