#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define PHI 1.618033988749895f
#define DIM 128
#define N (DIM * 3)
#define MAX_SAMPLES 200
#define EPOCHS 200

inline float va(float x){float i=(x>=0)?x*1.05f:x*0.3f,p=(x>=0)?x*0.3f:x*1.05f;return 0.5f*x+0.25f*i+0.25f*p+0.1f*x;}
struct Layer{float W[DIM*DIM],b[DIM],dW[DIM*DIM],db[DIM];
    Layer(){for(int i=0;i<DIM*DIM;i++){W[i]=sinf(i*PHI*0.1f)*0.02f;dW[i]=0;}for(int i=0;i<DIM;i++)b[i]=db[i]=0;}
    void fwd(float*x,float*y,int n){for(int i=0;i<n;i++){float s=b[i%DIM];for(int j=0;j<DIM;j++)s+=x[(i/DIM)*DIM+j]*W[(i%DIM)*DIM+j];y[i]=s;}}
    void zg(){memset(dW,0,DIM*DIM*sizeof(float));memset(db,0,DIM*sizeof(float));}
    void upd(float lr){for(int i=0;i<DIM*DIM;i++){W[i]-=lr*dW[i];dW[i]*=0.9f;}for(int i=0;i<DIM;i++){b[i]-=lr*db[i];db[i]*=0.9f;}}};

void t2v(const char* t, float* v){
    for(int d=0;d<N;d++) v[d]=0;
    for(int i=0;t[i];i++) for(int d=0;d<N;d++) v[d]+=sinf(t[i]*d*PHI/N+i*0.1f)*cosf((t[i]+d)*3.14159f/N)*0.03f;
    float n=0;for(int d=0;d<N;d++)n+=v[d]*v[d];if(n>1e-8f){n=1.0f/sqrtf(n);for(int d=0;d<N;d++)v[d]*=n;}
}

float cosine_sim(float* a, float* b){
    float dot=0,na=0,nb=0;
    for(int d=0;d<N;d++){dot+=a[d]*b[d];na+=a[d]*a[d];nb+=b[d]*b[d];}
    return dot/(sqrtf(na)*sqrtf(nb)+1e-8f);
}

int main(){
    const char* kb[]={
        "Brahman is the ultimate reality infinite eternal consciousness beyond all attributes and forms",
        "Atman the individual self is identical with Brahman the cosmic self tat tvam asi",
        "Yoga is the cessation of mind fluctuations for complete inner liberation and freedom",
        "Dharma is righteous action in accordance with cosmic order that sustains existence",
        "Karma is the universal law of cause and effect governing all actions across lifetimes",
        "Moksha is complete liberation from the endless cycle of birth death and rebirth",
        "Meditation brings inner peace clarity and self realization through focused awareness",
        "Consciousness is the fundamental substrate of all existence not emergent from matter",
        "Artificial intelligence simulates human intelligence through machines and algorithms",
        "Machine learning enables computers to learn from data without explicit programming",
        "Organic farming avoids synthetic chemicals using natural methods for healthy food production",
        "Agriculture is the science and practice of cultivating land and raising crops",
        "India is a diverse country in South Asia with rich cultural heritage and ancient wisdom",
        "The universe is vast interconnected and governed by fundamental physical laws",
        "Love compassion and kindness are the highest virtues of human existence and connection",
        "Knowledge is power and education unlocks the full potential of every human being",
        "Peace comes from within through acceptance mindfulness and letting go of attachments",
        "The present moment is the only reality past is memory future is imagination",
        "Nature maintains perfect balance through interconnected ecosystems and cycles",
        "Science is the systematic study of the natural world through observation and experiment",
        "Mathematics is the universal language of patterns quantity and logical relationships",
        "Music is the art of sound that expresses emotion and connects human hearts",
        "Art is creative expression that reflects the beauty and complexity of human experience",
        "Health is the greatest wealth requiring balance of body mind and spirit",
        "Education transforms lives by opening minds to new possibilities and understanding",
        "Sustainable living respects planetary boundaries ensuring resources for future generations",
        "Community is the foundation of human society built on mutual support and cooperation",
        "Innovation drives progress by solving problems through creative thinking and technology",
        "History teaches us lessons from the past to guide our present and future actions",
        "Philosophy explores fundamental questions about existence knowledge values and reason",
        "भारत प्राचीन ज्ञान और संस्कृति का देश है जहां विविधता में एकता है",
        "किसान कठोर परिश्रम करके सबके लिए अन्न उगाते हैं",
        "জ্ঞানই শক্তি এবং শিক্ষাই জীবনের উন্নতির চাবিকাঠি",
        "প্রকৃতি আমাদের সকলের মাতা যিনি জীবন দান করেন",
        "விவசாயம் உலகின் மிக முக்கியமான தொழில் ஆகும்",
        "அன்பே உலகை இயக்கும் சக்தி அன்பே கடவுள்",
    };
    int samples=36;
    
    printf("🕉 VEDIC C++ Q&A ENGINE\n======================\n");
    printf("Training on %d knowledge texts\n",samples);
    printf("Domains: Vedic, Science, Agriculture, Philosophy, Arts, Indic\n\n");
    
    float *x=new float[samples*N](),*t=new float[samples*N]();
    float *kb_emb=new float[samples*N]();
    for(int s=0;s<samples;s++){t2v(kb[s],&x[s*N]);t2v(kb[s],&t[s*N]);t2v(kb[s],&kb_emb[s*N]);}
    
    Layer l1,l2,l3;
    float *h1=new float[N],*h2=new float[N],*h3=new float[N];
    float *a1=new float[N],*a2=new float[N],*a3=new float[N],*g=new float[N];
    
    float best=1e9f;
    clock_t st=clock();
    for(int ep=0;ep<EPOCHS;ep++){
        float loss=0;
        for(int s=0;s<samples;s++){
            float*inp=&x[s*N],*tgt=&t[s*N];
            l1.fwd(inp,h1,N);for(int i=0;i<N;i++)a1[i]=va(h1[i]);
            l2.fwd(a1,h2,N);for(int i=0;i<N;i++)a2[i]=va(h2[i]);
            l3.fwd(a2,h3,N);for(int i=0;i<N;i++)a3[i]=va(h3[i]);
            for(int i=0;i<N;i++){float d=a3[i]-tgt[i];loss+=d*d;g[i]=2.0f*d/N;}
            l3.zg();for(int i=0;i<N;i++){int id=i%DIM;l3.db[id]+=g[i];for(int j=0;j<DIM;j++)l3.dW[id*DIM+j]+=g[i]*a2[j];}
            float*dy2=new float[N]();for(int i=0;i<N;i++)for(int j=0;j<DIM;j++)dy2[j]+=g[i]*l3.W[(i%DIM)*DIM+j];
            l2.zg();for(int i=0;i<N;i++){int id=i%DIM;l2.db[id]+=dy2[i];for(int j=0;j<DIM;j++)l2.dW[id*DIM+j]+=dy2[i]*a1[j];}
            float*dy1=new float[N]();for(int i=0;i<N;i++)for(int j=0;j<DIM;j++)dy1[j]+=dy2[i]*l2.W[(i%DIM)*DIM+j];
            l1.zg();for(int i=0;i<N;i++){int id=i%DIM;l1.db[id]+=dy1[i];for(int j=0;j<DIM;j++)l1.dW[id*DIM+j]+=dy1[i]*inp[j];}
            l3.upd(0.01f);l2.upd(0.01f);l1.upd(0.01f);
            delete[]dy2;delete[]dy1;
        }
        loss/=(samples*N);if(loss<best)best=loss;
        if(ep%50==0)printf("  Epoch %3d: loss=%.6f\n",ep,loss);
    }
    printf("✅ Trained in %.3fs | Best loss: %.6f\n\n",(float)(clock()-st)/CLOCKS_PER_SEC,best);
    
    // Q&A
    const char* questions[]={
        "What is Brahman?","How does yoga work?","What is karma?",
        "What is meditation?","What is artificial intelligence?",
        "What is organic farming?","Tell me about India",
        "What is the meaning of life?","How to find peace?",
        "What is consciousness?","भारत क्या है?","জ্ঞান কি?",
    };
    
    printf("🕉 VEDIC Q&A:\n============\n");
    for(int q=0;q<12;q++){
        float qv[N];t2v(questions[q],qv);
        int best_i=0;float best_s=-1;
        for(int s=0;s<samples;s++){
            float sim=cosine_sim(qv,&kb_emb[s*N]);
            if(sim>best_s){best_s=sim;best_i=s;}
        }
        printf("Q: %s\nA: %s\n\n",questions[q],kb[best_i]);
    }
    
    delete[]x;delete[]t;delete[]kb_emb;
    return 0;
}
