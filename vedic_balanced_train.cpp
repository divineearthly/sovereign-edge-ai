#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define PHI 1.618033988749895f
#define DIM 256
#define N (DIM * 2)
#define MAX_SAMPLES 1000
#define EPOCHS 200
#define LR 0.01f

inline float va(float x){float i=(x>=0)?x*1.05f:x*0.3f,p=(x>=0)?x*0.3f:x*1.05f;return 0.5f*x+0.25f*i+0.25f*p+0.1f*x;}
struct Layer{float W[DIM*DIM],b[DIM],dW[DIM*DIM],db[DIM];
    Layer(){for(int i=0;i<DIM*DIM;i++){W[i]=sinf(i*PHI*0.1f)*0.02f;dW[i]=0;}for(int i=0;i<DIM;i++)b[i]=db[i]=0;}
    void fwd(float*x,float*y,int n){for(int i=0;i<n;i++){float s=b[i%DIM];for(int j=0;j<DIM;j++)s+=x[(i/DIM)*DIM+j]*W[(i%DIM)*DIM+j];y[i]=s;}}
    void zg(){memset(dW,0,DIM*DIM*sizeof(float));memset(db,0,DIM*sizeof(float));}
    void upd(float lr){for(int i=0;i<DIM*DIM;i++){W[i]-=lr*dW[i];dW[i]*=0.9f;}for(int i=0;i<DIM;i++){b[i]-=lr*db[i];db[i]*=0.9f;}}};

void t2v(const char* t, float* v){
    for(int d=0;d<N;d++) v[d]=0;
    for(int i=0;t[i];i++) for(int d=0;d<N;d++) v[d]+=sinf(t[i]*d*PHI/N+i*0.1f)*cosf((t[i]+d)*3.14159f/N)*0.02f;
    float n=0;for(int d=0;d<N;d++)n+=v[d]*v[d];if(n>1e-8f){n=1.0f/sqrtf(n);for(int d=0;d<N;d++)v[d]*=n;}
}

float sim(float* a, float* b){float d=0,na=0,nb=0;for(int i=0;i<N;i++){d+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}return d/(sqrtf(na)*sqrtf(nb)+1e-8f);}

int main(){
    printf("🕉 VEDIC AI — BALANCED DATASET\n=============================\n");
    
    FILE* f = fopen("dataset_master.json", "r");
    if(!f){printf("Error: dataset not found\n");return 1;}
    fseek(f,0,SEEK_END);long sz=ftell(f);fseek(f,0,SEEK_SET);
    char* js=new char[sz+1];fread(js,1,sz,f);js[sz]=0;fclose(f);
    
    char** texts=new char*[MAX_SAMPLES];
    int count=0;
    char*p=js;
    while(*p&&count<MAX_SAMPLES){
        if(*p=='"'){p++;char*s=p;while(*p&&*p!='"'){if(*p=='\\')p++;p++;}
            int l=p-s;if(l>20&&l<600){texts[count]=new char[l+1];strncpy(texts[count],s,l);texts[count][l]=0;count++;}}
        p++;
    }
    delete[]js;
    
    // BALANCE: deduplicate similar texts, keep diverse samples
    printf("Loaded %d raw texts\n",count);
    
    // Encode & deduplicate
    float *emb=new float[count*N]();
    int* keep=new int[count];
    int kept=0;
    
    for(int i=0;i<count;i++){
        t2v(texts[i],&emb[i*N]);
        bool dup=false;
        for(int j=0;j<i;j++){
            if(sim(&emb[i*N],&emb[j*N])>0.95f){dup=true;break;}
        }
        if(!dup){keep[kept++]=i;}
    }
    
    printf("After dedup: %d unique texts\n\n",kept);
    
    // Train on unique texts
    float *x=new float[kept*N](),*t=new float[kept*N]();
    for(int s=0;s<kept;s++){t2v(texts[keep[s]],&x[s*N]);t2v(texts[keep[s]],&t[s*N]);}
    
    Layer l1,l2,l3;
    float *h1=new float[N],*h2=new float[N],*h3=new float[N];
    float *a1=new float[N],*a2=new float[N],*a3=new float[N],*g=new float[N];
    
    float best=1e9f;clock_t st=clock();
    for(int ep=0;ep<EPOCHS;ep++){
        float loss=0;
        for(int s=0;s<kept;s++){
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
            l3.upd(LR);l2.upd(LR);l1.upd(LR);
            delete[]dy2;delete[]dy1;
        }
        loss/=(kept*N);if(loss<best)best=loss;
        if(ep%40==0)printf("  Ep %3d: loss=%.6f\n",ep,loss);
    }
    printf("✅ Trained in %.2fs | Loss: %.6f\n\n",(float)(clock()-st)/CLOCKS_PER_SEC,best);
    
    // Q&A with diversity bonus
    const char* questions[]={
        "What is artificial intelligence?","What is machine learning?",
        "What is deep learning?","What is quantum computing?",
        "What is India?","What is philosophy?",
        "What is consciousness?","What is Brahman?",
        "What is yoga?","What is meditation?",
        "What is organic farming?","What is climate change?",
        "What is blockchain?","What is democracy?",
        "What is renewable energy?","What is the Internet?",
        "भारत क्या है?","জীবন কাকে বলে?",
    };
    
    printf("🕉 VEDIC Q&A:\n============\n");
    float* used=new float[kept]();
    for(int q=0;q<18;q++){
        float qv[N];t2v(questions[q],qv);
        int best_i=0;float best_s=-1;
        for(int s=0;s<kept;s++){
            float si=sim(qv,&emb[keep[s]*N])*(1.0f-used[s]*2.0f); // Penalize reused answers
            if(si>best_s){best_s=si;best_i=s;}
        }
        used[best_i]+=0.5f;
        printf("Q: %s\nA: %s\n\n",questions[q],texts[keep[best_i]]);
    }
    
    delete[]used;delete[]x;delete[]t;delete[]emb;delete[]keep;
    for(int i=0;i<count;i++)delete[]texts[i];delete[]texts;
    return 0;
}
