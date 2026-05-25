#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <ctime>

#define PHI 1.618033988749895f
#define DIM 128
#define MAX_QA 500

// Vedic embedding with keyword boosting
void qa_embed(const char* text, float* vec, bool is_question) {
    for(int d=0; d<DIM; d++) vec[d]=0;
    
    // Extract keywords (capitalized or long words get higher weight)
    char txt[512]; strcpy(txt, text);
    char* word = strtok(txt, " ");
    while(word) {
        float weight = 1.0f;
        if(isupper(word[0])) weight = 3.0f;  // Proper nouns
        if(strlen(word) > 8) weight = 2.0f;  // Technical terms
        if(strstr("what how why when where who explain tell define", word)) weight = 5.0f;
        
        for(int d=0; d<DIM; d++)
            vec[d] += sinf(word[0] * (d+1) * PHI / DIM) * weight * 0.05f;
        word = strtok(NULL, " ");
    }
    
    // Add full text harmonics for context
    for(int i=0; text[i]; i++)
        for(int d=0; d<DIM; d++)
            vec[d] += sinf(text[i] * d * PHI / DIM + i * 0.1f) * 0.01f;
    
    // Normalize
    float n=0; for(int d=0; d<DIM; d++) n+=vec[d]*vec[d];
    if(n>1e-8f){n=1.0f/sqrtf(n); for(int d=0; d<DIM; d++) vec[d]*=n;}
}

float sim(float*a, float*b) {
    float d=0,na=0,nb=0;
    for(int i=0; i<DIM; i++){d+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i];}
    return d/(sqrtf(na)*sqrtf(nb)+1e-8f);
}

int main() {
    printf("🕉 VEDIC Q&A EMBEDDING ENGINE\n=============================\n\n");
    
    // Knowledge base with Q&A pairs
    const char* qa[][2] = {
        {"what is artificial intelligence", "Artificial intelligence simulates human intelligence using machines and computer systems"},
        {"what is machine learning", "Machine learning enables computers to learn from data without explicit programming"},
        {"what is deep learning", "Deep learning uses multi-layered neural networks to learn hierarchical patterns"},
        {"what is quantum computing", "Quantum computing harnesses quantum mechanics to solve problems beyond classical computers"},
        {"what is blockchain", "Blockchain creates secure decentralized ledgers for transparent transactions"},
        {"what is climate change", "Climate change is the long-term shift in global weather from human activity"},
        {"what is renewable energy", "Renewable energy comes from naturally replenished sources like solar and wind"},
        {"what is brahman", "Brahman is the ultimate reality infinite eternal consciousness beyond attributes"},
        {"what is yoga", "Yoga is the cessation of mental fluctuations for complete self-realization"},
        {"what is meditation", "Meditation is focused awareness practice for inner peace and mental clarity"},
        {"what is karma", "Karma is the universal law of cause and effect governing all actions"},
        {"what is dharma", "Dharma is righteous living in harmony with cosmic order and moral principles"},
        {"what is moksha", "Moksha is liberation from the cycle of birth death and rebirth"},
        {"how to find peace", "Peace comes from within through meditation self-awareness and letting go"},
        {"what is consciousness", "Consciousness is the fundamental awareness of existence and experience"},
        {"what is love", "Love is the fundamental force of connection and harmony between all beings"},
        {"what is organic farming", "Organic farming uses natural methods without synthetic pesticides or fertilizers"},
        {"what is the internet", "The Internet connects billions of devices globally through standard protocols"},
        {"what is dna", "DNA carries genetic instructions for development and functioning of all life"},
        {"what is gravity", "Gravity is the fundamental force of attraction between all objects with mass"},
        {"what is democracy", "Democracy is government by the people through elected representatives"},
        {"what is india", "India is a diverse South Asian country with ancient civilization and modern democracy"},
        {"what is mathematics", "Mathematics is the abstract science of number quantity pattern and structure"},
        {"what is philosophy", "Philosophy examines fundamental questions about existence knowledge and ethics"},
        {"what is ayurveda", "Ayurveda is the ancient Indian system of holistic medicine and natural healing"},
    };
    
    int qa_count = sizeof(qa)/sizeof(qa[0]);
    float q_emb[MAX_QA][DIM], a_emb[MAX_QA][DIM];
    
    // Pre-compute all embeddings
    for(int i=0; i<qa_count; i++) {
        qa_embed(qa[i][0], q_emb[i], true);
        qa_embed(qa[i][1], a_emb[i], false);
    }
    
    printf("📚 %d Q&A pairs loaded\n\n", qa_count);
    
    // Test queries
    const char* tests[] = {
        "what is artificial intelligence",
        "tell me about brahman",
        "how does deep learning work",
        "what is organic farming",
        "explain quantum computing",
        "i am feeling very sad today",
        "how to find inner peace",
        "what is yoga",
        "tell me about india",
        "what is climate change",
        "what is the meaning of life",
        "how does karma work",
    };
    
    for(int t=0; t<12; t++) {
        float qv[DIM]; qa_embed(tests[t], qv, true);
        
        // Find best match
        int best=-1; float bs=-1;
        for(int i=0; i<qa_count; i++) {
            float s = sim(qv, q_emb[i]);
            if(s > bs) { bs=s; best=i; }
        }
        
        // Sentiment check
        float sent=0;
        const char* pos[]={"peace","love","joy","happy","good","great",NULL};
        const char* neg[]={"sad","angry","fear","pain","bad","terrible",NULL};
        char tl[512]; for(int i=0;tests[t][i];i++) tl[i]=tolower(tests[t][i]); tl[strlen(tests[t])]=0;
        for(int i=0;pos[i];i++) if(strstr(tl,pos[i])) sent+=0.5f;
        for(int i=0;neg[i];i++) if(strstr(tl,neg[i])) sent-=0.5f;
        
        printf("Q: %s\n", tests[t]);
        if(sent < -0.2f) printf("   💙 Detected distress — responding with compassion\n");
        if(best >= 0) {
            printf("A: %s (confidence: %.0f%%)\n", qa[best][1], bs*100);
        }
        printf("\n");
    }
    
    return 0;
}
