#include <cstdio>
#include <cstring>
#include <cmath>
#include <cctype>

#define PHI 1.618033988749895
#define DIM 64

const char* knowledge[] = {
    "Brahman is the ultimate reality infinite eternal consciousness beyond all attributes",
    "Atman the individual self is identical with Brahman the cosmic self tat tvam asi",
    "Karma is the universal law of cause and effect governing all actions across lifetimes",
    "Dharma is righteous action in accordance with the cosmic order and universal truth",
    "Yoga is the cessation of mind fluctuations for complete inner liberation and freedom",
    "Meditation brings inner peace clarity and self realization through focused awareness",
    "Moksha is complete liberation from the cycle of birth death and rebirth samsara",
    "The Upanishads are ancient philosophical texts exploring the nature of ultimate reality",
    "Bhagavad Gita is a sacred text teaching selfless action devotion and knowledge",
    "The three gunas sattva rajas tamas are primordial qualities of nature",
    "Pranayama is breath control for regulating life force energy prana in the body",
    "Ayurveda is the ancient Indian system of holistic medicine and natural healing",
    "Mantra is sacred sound vibration that attunes consciousness to higher frequencies",
    "Samadhi is the state of complete meditative absorption in pure consciousness",
    "The Vedas are the oldest scriptures containing eternal spiritual wisdom",
    "Artificial intelligence simulates human intelligence through machines and algorithms",
    "Machine learning enables computers to learn from data without explicit programming",
    "Deep learning uses neural networks with multiple layers for pattern recognition",
    "Quantum computing uses quantum bits qubits for exponentially faster computation",
    "The speed of light is the universal constant at which electromagnetic radiation travels",
    "Gravity is the fundamental force of attraction between all objects with mass",
    "Photosynthesis converts sunlight into chemical energy that sustains most life on Earth",
    "DNA contains the genetic instructions for development and functioning of living organisms",
    "The periodic table organizes chemical elements by atomic number and properties",
    "Renewable energy comes from sun wind water and other natural sources",
    "Organic farming avoids synthetic pesticides and fertilizers for natural cultivation methods",
    "Crop rotation improves soil health by alternating different crops in sequenced seasons",
    "Irrigation systems deliver water to crops through canals pipes and sprinklers",
    "Soil health depends on organic matter minerals microorganisms and proper pH balance",
    "Monsoon rains are seasonal winds bringing heavy rainfall crucial for agriculture",
    "Biodiversity is the variety of life forms in an ecosystem essential for balance",
    "India is a country in South Asia with ancient civilization and diverse culture",
    "Sanskrit is the ancient sacred language of Hindu scriptures and Vedic texts",
    "The Indus Valley civilization was one of the worlds oldest urban cultures",
    "Ayurveda classifies body types into vata pitta and kapha doshas",
    "Yoga originated in ancient India as a physical mental and spiritual practice",
    "Hindi is the most widely spoken language in India with Devanagari script",
    "Peace comes from within through acceptance mindfulness and letting go of attachments",
    "Education empowers individuals and transforms societies through knowledge and skills",
    "Music is the universal language that transcends cultural and linguistic boundaries",
    "Innovation drives progress by solving problems through creative thinking and technology",
    "Compassion is the ability to understand suffering of others and desire to help",
    "Truthfulness is alignment of thought word and deed with objective reality",
    "The universe is vast interconnected and governed by fundamental laws of physics",
    "Water is essential for all known forms of life covering most of Earth surface",
    "Mathematics is the abstract science of number quantity space and logical patterns",
    "Karma yoga is the path of selfless action without attachment to results",
};
#define N_KNOWLEDGE (sizeof(knowledge)/sizeof(knowledge[0]))

// Keywords mapped to preferred knowledge indices
struct Keyword { const char* word; int idx; };
Keyword keywords[] = {
    {"brahman", 0}, {"atman", 1}, {"karma", 2}, {"dharma", 3},
    {"yoga", 4}, {"meditation", 5}, {"moksha", 6}, {"upanishad", 7},
    {"gita", 8}, {"guna", 9}, {"pranayama", 10}, {"ayurveda", 11},
    {"mantra", 12}, {"samadhi", 13}, {"veda", 14},
    {"ai", 15}, {"intelligence", 15}, {"machine learning", 16},
    {"deep learning", 17}, {"quantum", 18}, {"light", 19},
    {"gravity", 20}, {"photosynthesis", 21}, {"dna", 22},
    {"periodic", 23}, {"energy", 24}, {"renewable", 24},
    {"organic", 25}, {"farming", 25}, {"crop", 26}, {"irrigation", 27},
    {"soil", 28}, {"monsoon", 29}, {"rain", 29}, {"biodiversity", 30},
    {"india", 31}, {"sanskrit", 32}, {"indus", 33},
    {"hindi", 36}, {"peace", 37}, {"education", 38},
    {"music", 39}, {"innovation", 40}, {"compassion", 41},
    {"truth", 42}, {"universe", 43}, {"water", 44}, {"math", 45},
    {"karma yoga", 46},
};
#define N_KEYWORDS (sizeof(keywords)/sizeof(keywords[0]))

void text2vec(const char* t, float* v) {
    for(int d=0; d<DIM; d++) v[d]=0;
    for(int i=0; t[i]; i++)
        for(int d=0; d<DIM; d++)
            v[d] += sinf(t[i]*d*PHI/DIM + i*0.1f) * cosf((t[i]+d)*3.14159f/DIM) * 0.02f;
    float n=0;
    for(int d=0; d<DIM; d++) n += v[d]*v[d];
    if(n > 1e-8f) { n = 1.0f/sqrtf(n); for(int d=0; d<DIM; d++) v[d] *= n; }
}

float similarity(float* a, float* b) {
    float d=0, na=0, nb=0;
    for(int i=0; i<DIM; i++) { d += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
    return d/(sqrtf(na)*sqrtf(nb) + 1e-8f);
}

void tolower_str(char* s) { for(int i=0; s[i]; i++) s[i]=tolower(s[i]); }

int main(int argc, char** argv) {
    char query[512] = {0};
    if(argc > 1) {
        for(int i=1; i<argc; i++) { strcat(query, argv[i]); if(i<argc-1) strcat(query, " "); }
    } else {
        fgets(query, sizeof(query), stdin);
        query[strcspn(query, "\n")] = 0;
    }
    
    // Check keyword match first
    char qlow[512];
    strcpy(qlow, query);
    tolower_str(qlow);
    
    for(int k=0; k<N_KEYWORDS; k++) {
        if(strstr(qlow, keywords[k].word)) {
            printf("%s\n", knowledge[keywords[k].idx]);
            return 0;
        }
    }
    
    // Fallback to cosine similarity
    float qv[DIM];
    text2vec(query, qv);
    
    float best = -1e9f;
    int best_idx = 0;
    for(int i=0; i<N_KNOWLEDGE; i++) {
        float kv[DIM];
        text2vec(knowledge[i], kv);
        float s = similarity(qv, kv);
        if(s > best) { best = s; best_idx = i; }
    }
    
    printf("%s\n", knowledge[best_idx]);
    return 0;
}
