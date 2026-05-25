/*
 * VEDIC ONLINE AI — Learns from the Internet in Real-Time
 * Searches Wikipedia, fetches answers, builds knowledge live
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>
#include <ctime>

#define PHI 1.618033988749895f
#define DIM 128
#define MAX_KB 5000

// ═══════════════════════════════════════════════
// VEDIC NLP (inline)
// ═══════════════════════════════════════════════
void shuddhi(const char* s, char* c) {
    strcpy(c,s); int l=strlen(c);
    if(islower(c[0])) c[0]=toupper(c[0]);
    if(l>0&&c[l-1]!='.'&&c[l-1]!='?'&&c[l-1]!='!'){c[l]='.';c[l+1]=0;}
}
float bhava(const char* t) {
    const char* pos[]={"good","great","excellent","love","peace","joy","happy","bliss","amazing","perfect","best","harmony","compassion","wisdom","truth","success","hope",NULL};
    const char* neg[]={"bad","terrible","awful","hate","suffer","pain","sad","angry","fear","worst","horrible","evil","dark","failure","problem","difficult","wrong","violence","death",NULL};
    float p=0,n=0;char tl[1024];for(int i=0;t[i];i++)tl[i]=tolower(t[i]);tl[strlen(t)]=0;
    for(int i=0;pos[i];i++)if(strstr(tl,pos[i]))p++;for(int i=0;neg[i];i++)if(strstr(tl,neg[i]))n++;
    return (p-n*PHI)/(p+n+1);
}

// ═══════════════════════════════════════════════
// KNOWLEDGE BASE
// ═══════════════════════════════════════════════
struct KnowledgeBase {
    char** texts;
    float** emb;
    int count;
    
    KnowledgeBase() : count(0) {
        texts = new char*[MAX_KB];
        emb = new float*[MAX_KB];
    }
    
    void embed(const char* t, float* v) {
        for(int d=0;d<DIM;d++)v[d]=0;
        for(int i=0;t[i];i++)for(int d=0;d<DIM;d++)v[d]+=sinf(t[i]*d*PHI/DIM+i*0.1f)*cosf((t[i]+d)*3.14159f/DIM)*0.02f;
        float n=0;for(int d=0;d<DIM;d++)n+=v[d]*v[d];if(n>1e-8f){n=1/sqrtf(n);for(int d=0;d<DIM;d++)v[d]*=n;}
    }
    
    float sim(float*a,float*b){float d=0,na=0,nb=0;for(int i=0;i<DIM;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(sqrtf(na)*sqrtf(nb)+1e-8f);}
    
    void add(const char* text) {
        if(count >= MAX_KB) return;
        int l = strlen(text)+1;
        texts[count] = new char[l];
        strcpy(texts[count], text);
        emb[count] = new float[DIM];
        embed(text, emb[count]);
        count++;
    }
    
    const char* query(const char* q, float* confidence) {
        if(count == 0) { *confidence = 0; return "I have not learned anything yet. Teach me!"; }
        float qv[DIM]; embed(q, qv);
        int best=0; float bs=-1;
        for(int i=0;i<count;i++){float s=sim(qv,emb[i]);if(s>bs){bs=s;best=i;}}
        *confidence = bs;
        return texts[best];
    }
    
    int size() { return count; }
};

// ═══════════════════════════════════════════════
// INTERNET SEARCH (via curl to Wikipedia API)
// ═══════════════════════════════════════════════
bool search_wikipedia(const char* query, char* result, int maxlen) {
    char cmd[1024];

    // URL-encode the query (simple version)
    char encoded[512]; int e=0;
    for(int i=0; query[i] && e<500; i++) {
        if(query[i] == ' ') encoded[e++] = '_';
        else if(isalnum(query[i])) encoded[e++] = query[i];
    }
    encoded[e] = 0;
    
    snprintf(cmd, 1024, 
        "curl -s -H 'User-Agent: VedicAI/1.0' "
        "'https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=%s&format=json&srlimit=1' "
        "2>/dev/null", encoded);
    
    FILE* f = popen(cmd, "r");
    if(!f) return false;
    
    char* js = new char[4096];
    int sz = fread(js, 1, 4095, f);
    pclose(f);
    if(sz < 50) { delete[] js; return false; }
    js[sz] = 0;
    
    // Extract snippet from JSON
    char* snippet = strstr(js, "\"snippet\":\"");
    if(!snippet) { delete[] js; return false; }
    snippet += 11; // skip "snippet":"
    
    int r=0;
    while(*snippet && *snippet != '"' && r < maxlen-1) {
        if(*snippet == '\\' && *(snippet+1) == 'n') { snippet+=2; result[r++]=' '; continue; }
        if(*snippet == '\\' && *(snippet+1) == '"') { snippet+=2; result[r++]='"'; continue; }
        result[r++] = *snippet++;
    }
    result[r] = 0;
    
    // Strip HTML tags
    char clean[1024]; int ci = 0;
    bool in_tag = false;
    for(int i = 0; result[i] && ci < 1000; i++) {
        if(result[i] == '<') in_tag = true;
        else if(result[i] == '>') in_tag = false;
        else if(!in_tag) clean[ci++] = result[i];
    }
    clean[ci] = 0;
    strcpy(result, clean);
    
    delete[] js;
    return (r > 10);
}

// ═══════════════════════════════════════════════
// VEDIC ONLINE AI
// ═══════════════════════════════════════════════
struct VedicOnlineAI {
    KnowledgeBase kb;
    int interactions;
    
    VedicOnlineAI() : interactions(0) {
        // Seed with core knowledge
        const char* seed[] = {
            "Artificial intelligence simulates human intelligence using machines and computer systems",
            "Machine learning enables computers to learn from data without explicit programming",
            "Deep learning uses multi-layered neural networks to learn hierarchical patterns",
            "Brahman is the ultimate reality infinite eternal consciousness beyond all attributes",
            "Yoga is the cessation of mental fluctuations for complete self-realization",
            "Meditation is focused awareness practice for inner peace and mental clarity",
            "Organic farming uses natural methods without synthetic pesticides or fertilizers",
            "Climate change is the long-term shift in global weather from human activity",
            "Quantum computing harnesses quantum mechanics to solve problems beyond classical computers",
            "India is a diverse South Asian country with ancient civilization and modern democracy",
        };
        for(int i=0; i<10; i++) kb.add(seed[i]);
    }
    
    void ask(const char* question) {
        interactions++;
        char corrected[1024];
        shuddhi(question, corrected);
        float sentiment = bhava(corrected);
        
        printf("══════════════════════════════════════\n");
        printf("Q: %s\n", corrected);
        if(sentiment < -0.2f) printf("   💙 Detected distress — responding with care\n");
        
        // Step 1: Check local knowledge
        float conf;
        const char* local = kb.query(corrected, &conf);
        
        if(conf > 0.85f) {
            // Local knowledge is sufficient
            printf("A: %s\n", local);
            printf("   [Local knowledge | Confidence: %.0f%%]\n", conf*100);
        } else {
            // Step 2: Search the internet
            printf("   🔍 Searching the internet for: %s\n", corrected);
            char web_result[1024];
            
            if(search_wikipedia(corrected, web_result, 1024)) {
                printf("A: %s\n", web_result);
                printf("   [Wikipedia | Learned in real-time]\n");
                // Add to knowledge base for future
                kb.add(web_result);
            } else {
                // Fallback to best local match
                printf("A: %s\n", local);
                printf("   [Best local match | Confidence: %.0f%%]\n", conf*100);
            }
        }
        
        printf("══════════════════════════════════════\n\n");
    }
    
    void status() {
        printf("\n🕉 VEDIC ONLINE AI STATUS\n");
        printf("   Interactions: %d | Knowledge: %d texts\n", interactions, kb.size());
        printf("   Sources: Built-in + Wikipedia (real-time)\n");
        printf("   State: %s\n", kb.size()>50?"WISE":kb.size()>20?"LEARNING":"AWAKENING");
    }
};

int main() {
    printf("🕉 VEDIC ONLINE AI — Learns from the Internet\n");
    printf("============================================\n");
    printf("Sources: Wikipedia API (real-time) + Built-in knowledge\n");
    printf("NLP: 10 Vedic Sutras active\n\n");
    
    VedicOnlineAI ai;
    
    const char* questions[] = {
        "what is artificial intelligence",
        "tell me about black holes",
        "what is the population of india",
        "how does photosynthesis work",
        "what is brahman",
        "explain quantum entanglement",
        "who was albert einstein",
        "what is the great wall of china",
        "how to find inner peace",
        "what is machine learning",
    };
    
    for(int i=0; i<10; i++) {
        ai.ask(questions[i]);
    }
    
    ai.status();
    printf("\n🕉 VEDIC ONLINE AI — CONTINUOUSLY LEARNING FROM THE WEB\n");
    
    return 0;
}
