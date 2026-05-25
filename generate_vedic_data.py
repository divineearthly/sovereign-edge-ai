"""Generate 10K multilingual training texts for Vedic Transformer"""
import random

languages = {
    'hi': [
        "किसान खेत में काम कर रहा है। धान की फसल अच्छी है।",
        "जैविक खेती से मिट्टी की उर्वरता बढ़ती है। प्राकृतिक कीटनाशक का उपयोग करें।",
        "आज मौसम बहुत अच्छा है। बारिश होने की संभावना है। किसान खुश हैं।",
        "सरकार ने किसानों के लिए नई योजना शुरू की है। फसल बीमा मिलेगा।",
        "गेहूं की फसल तैयार है। मंडी में भाव अच्छे हैं। किसानों को लाभ होगा।",
        "सब्जियों की खेती में अच्छा मुनाफा है। टमाटर और आलू के भाव बढ़े हैं।",
        "डेयरी फार्मिंग से किसानों की आय बढ़ रही है। दूध का अच्छा दाम मिल रहा है।",
        "मधुमक्खी पालन एक अच्छा व्यवसाय है। शहद की मांग बढ़ रही है।",
        "वर्मीकम्पोस्ट बनाकर किसान अतिरिक्त आय कमा सकते हैं। जैविक खाद की मांग है।",
        "सिंचाई के लिए ड्रिप सिस्टम अपनाएं। पानी की बचत होगी और फसल अच्छी होगी।",
    ],
    'bn': [
        "কৃষক মাঠে কাজ করছে। ধানের ফসল ভালো হয়েছে। জৈব চাষ মাটির উর্বরতা বাড়ায়।",
        "আজ আবহাওয়া খুব ভালো। বৃষ্টি হওয়ার সম্ভাবনা আছে। কৃষকরা খুব খুশি।",
        "সরকার কৃষকদের জন্য নতুন প্রকল্প চালু করেছে। ফসল বিমা পাওয়া যাবে।",
        "সবজি চাষে ভালো লাভ হয়। টমেটো ও আলুর দাম বেড়েছে।",
        "মাছ চাষ করে কৃষকরা ভালো আয় করছেন। পুকুরে মাছ চাষ একটি ভালো ব্যবসা।",
        "নারিকেল চাষ বাংলার কৃষকদের জন্য লাভজনক। নারিকেলের বহুবিধ ব্যবহার আছে।",
    ],
    'as': [
        "কৃষকে পথাৰত কাম কৰি আছে। ধানৰ ফচল ভাল হৈছে। জৈৱিক কৃষি মাটিৰ উৰ্ৱৰতা বঢ়ায়।",
        "আজি বতৰ খুব ভাল। বৰষুণ হোৱাৰ সম্ভাৱনা আছে। কৃষকসকল খুব সুখী।",
        "চৰকাৰে কৃষকৰ বাবে নতুন আঁচনি আৰম্ভ কৰিছে। শস্য বীমা পোৱা যাব।",
        "শাক-পাচলিৰ খেতিত ভাল লাভ হয়। বিলাহী আৰু আলুৰ দাম বাঢ়িছে।",
    ],
    'ta': [
        "விவசாயி வயலில் வேலை செய்கிறார். நெல் பயிர் நன்றாக உள்ளது.",
        "இன்று வானிலை மிகவும் நன்றாக உள்ளது. மழை பெய்ய வாய்ப்பு உள்ளது.",
        "அரசு விவசாயிகளுக்கு புதிய திட்டம் தொடங்கியுள்ளது. பயிர் காப்பீடு கிடைக்கும்.",
        "காய்கறி சாகுபடியில் நல்ல லாபம். தக்காளி மற்றும் உருளைக்கிழங்கு விலை உயர்ந்துள்ளது.",
    ],
    'en': [
        "Organic farming improves soil health and biodiversity naturally.",
        "Farmers are using natural pesticides instead of chemical ones.",
        "The weather forecast predicts good rainfall this monsoon season.",
        "Crop rotation helps maintain soil fertility and reduce pests.",
        "Drip irrigation saves water and increases crop yield significantly.",
        "Vermicompost is an excellent organic fertilizer for vegetable farming.",
        "Honey bee farming provides additional income for small farmers.",
        "The government launched a new crop insurance scheme for farmers.",
        "Sustainable agriculture practices help combat climate change effects.",
        "Direct marketing helps farmers get better prices for their produce.",
    ]
}

templates = [
    "{0} {1} {2}",
    "{0}। {1}। {2}।",
    "{0} {1}। {2}",
    "प्रश्न: {0}? उत्तर: {1}। {2}",
    "Question: {0}? Answer: {1}. {2}",
]

all_texts = []
for lang, sentences in languages.items():
    for _ in range(2000 // len(languages)):
        s = random.sample(sentences, min(3, len(sentences)))
        template = random.choice(templates)
        text = template.format(*s) if len(s) == 3 else template.format(s[0], s[1] if len(s)>1 else "", s[2] if len(s)>2 else "")
        all_texts.append(text)

random.shuffle(all_texts)
all_texts = all_texts[:10000]

print(f"Generated {len(all_texts)} multilingual training texts")
print(f"Languages: {', '.join(languages.keys())}")
print(f"Total characters: {sum(len(t) for t in all_texts):,}")

# Save to file
with open("vedic_training_data.txt", "w", encoding="utf-8") as f:
    for text in all_texts:
        f.write(text + "\n")
print("Saved: vedic_training_data.txt")
