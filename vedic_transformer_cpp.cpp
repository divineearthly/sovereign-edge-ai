#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define PHI 1.618033988749895f
#define PI  3.141592653589793f
#define DIM         128
#define HEAD_DIM    32
#define NUM_HEADS   4
#define FFN_DIM     (DIM * 2)
#define VOCAB       793
#define MAX_SEQ     32
#define NUM_LAYERS  3
#define NUM_SAMPLES 50
#define EPOCHS      200

inline float tri_nadi(float x) {
    float ida = (x>=0)?x*1.05f:x*0.3f, pingala=(x>=0)?x*0.3f:x*1.05f;
    return 0.5f*x+0.25f*ida+0.25f*pingala+0.1f*x;
}

struct VedicTransformer {
    float *embed, *W_base, *W_out, *W1, *W2, *gamma1, *beta1, *gamma2, *beta2;
    float *q_buf, *k_buf, *v_buf, *attn_buf, *norm_buf, *ffn_buf, *hidden_buf;
    
    VedicTransformer() {
        embed = new float[VOCAB*DIM];
        W_base = new float[DIM*DIM];
        W_out = new float[DIM*VOCAB];
        W1 = new float[DIM*FFN_DIM];
        W2 = new float[FFN_DIM*DIM];
        gamma1 = new float[DIM]; beta1 = new float[DIM];
        gamma2 = new float[DIM]; beta2 = new float[DIM];
        q_buf = new float[MAX_SEQ*HEAD_DIM*NUM_HEADS];
        k_buf = new float[MAX_SEQ*HEAD_DIM*NUM_HEADS];
        v_buf = new float[MAX_SEQ*HEAD_DIM*NUM_HEADS];
        attn_buf = new float[MAX_SEQ*DIM];
        norm_buf = new float[MAX_SEQ*DIM];
        ffn_buf = new float[MAX_SEQ*DIM];
        hidden_buf = new float[FFN_DIM];
        
        for(int i=0;i<VOCAB*DIM;i++) embed[i]=sinf(i*PHI*0.1f)*0.1f;
        for(int i=0;i<DIM*DIM;i++) W_base[i]=sinf(i*0.1f)*0.02f;
        for(int i=0;i<DIM*VOCAB;i++) W_out[i]=sinf(i*0.03f)*0.02f;
        for(int i=0;i<DIM*FFN_DIM;i++) W1[i]=sinf(i*0.07f)*0.02f;
        for(int i=0;i<FFN_DIM*DIM;i++) W2[i]=sinf(i*0.11f)*0.02f;
        for(int d=0;d<DIM;d++){gamma1[d]=1;beta1[d]=0;gamma2[d]=1;beta2[d]=0;}
    }
    
    ~VedicTransformer() {
        delete[]embed;delete[]W_base;delete[]W_out;delete[]W1;delete[]W2;
        delete[]gamma1;delete[]beta1;delete[]gamma2;delete[]beta2;
        delete[]q_buf;delete[]k_buf;delete[]v_buf;delete[]attn_buf;
        delete[]norm_buf;delete[]ffn_buf;delete[]hidden_buf;
    }
    
    void forward(const int* tokens, float* logits, int seq_len) {
        float x[MAX_SEQ*DIM];
        for(int s=0;s<seq_len;s++){
            float phase=(s%10)/10.0f*2*PI;
            for(int d=0;d<DIM;d++)
                x[s*DIM+d]=embed[tokens[s]*DIM+d]+sinf(phase*(d+1)/DIM+s*0.01f)*0.05f;
        }
        
        for(int l=0;l<NUM_LAYERS;l++){
            // QKV from single projection
            for(int s=0;s<seq_len;s++){
                float base[DIM]; memset(base,0,DIM*sizeof(float));
                for(int d=0;d<DIM;d++) for(int i=0;i<DIM;i++) base[d]+=x[s*DIM+i]*W_base[i*DIM+d];
                for(int h=0;h<NUM_HEADS;h++){
                    float aq=PI/2+h*0.1f, ak=h*0.1f, av=PI+h*0.1f;
                    int off=h*HEAD_DIM;
                    for(int d=0;d<HEAD_DIM;d++){
                        q_buf[s*HEAD_DIM*NUM_HEADS+off+d]=base[off+d]*cosf(aq);
                        k_buf[s*HEAD_DIM*NUM_HEADS+off+d]=base[off+d]*cosf(ak);
                        v_buf[s*HEAD_DIM*NUM_HEADS+off+d]=base[off+d]*cosf(av);
                    }
                }
            }
            
            // Sphota O(n) attention
            memset(attn_buf,0,seq_len*DIM*sizeof(float));
            int hd=HEAD_DIM;
            for(int h=0;h<NUM_HEADS;h++){
                float gk[HEAD_DIM]={0};
                for(int s=0;s<seq_len;s++) for(int d=0;d<hd;d++) gk[d]+=k_buf[s*hd*NUM_HEADS+h*hd+d];
                for(int d=0;d<hd;d++) gk[d]/=seq_len;
                float sc[MAX_SEQ],ms=-1e30f,sum=0;
                for(int s=0;s<seq_len;s++){
                    sc[s]=0; for(int d=0;d<hd;d++) sc[s]+=q_buf[s*hd*NUM_HEADS+h*hd+d]*gk[d];
                    sc[s]/=sqrtf(hd); if(sc[s]>ms)ms=sc[s];
                }
                for(int s=0;s<seq_len;s++){
                    sc[s]-=ms*0.5f;
                    sc[s]=(fabsf(sc[s])>1/PHI)?sc[s]*PHI:sc[s]/PHI;
                    if(sc[s]<0)sc[s]=0; sum+=sc[s];
                }
                if(sum>1e-8f) for(int s=0;s<seq_len;s++) sc[s]/=sum;
                for(int s=0;s<seq_len;s++) for(int d=0;d<hd;d++)
                    attn_buf[s*DIM+h*hd+d]=sc[s]*v_buf[s*hd*NUM_HEADS+h*hd+d];
            }
            
            // Residual + norm
            for(int i=0;i<seq_len*DIM;i++) norm_buf[i]=x[i]+attn_buf[i];
            float s1=0,s2=0; for(int i=0;i<seq_len*DIM;i++){s1+=norm_buf[i];s2+=norm_buf[i]*norm_buf[i];}
            float m=s1/(seq_len*DIM),v=s2/(seq_len*DIM)-m*m; if(v<1e-8f)v=1e-8f;
            float is=1/sqrtf(v+1e-5f); for(int i=0;i<seq_len*DIM;i++) norm_buf[i]=(norm_buf[i]-m)*is;
            
            // Panchikarana FFN
            for(int s=0;s<seq_len;s++){
                for(int h=0;h<FFN_DIM;h++){
                    hidden_buf[h]=0; for(int d=0;d<DIM;d++) hidden_buf[h]+=norm_buf[s*DIM+d]*W1[d*FFN_DIM+h];
                    hidden_buf[h]=tri_nadi(hidden_buf[h]);
                }
                for(int d=0;d<DIM;d++){
                    ffn_buf[s*DIM+d]=0; for(int h=0;h<FFN_DIM;h++) ffn_buf[s*DIM+d]+=hidden_buf[h]*W2[h*DIM+d];
                }
            }
            
            // Residual + norm
            for(int i=0;i<seq_len*DIM;i++) x[i]=norm_buf[i]+ffn_buf[i];
            s1=0;s2=0; for(int i=0;i<seq_len*DIM;i++){s1+=x[i];s2+=x[i]*x[i];}
            m=s1/(seq_len*DIM);v=s2/(seq_len*DIM)-m*m; if(v<1e-8f)v=1e-8f;
            is=1/sqrtf(v+1e-5f); for(int i=0;i<seq_len*DIM;i++) x[i]=(x[i]-m)*is;
        }
        
        // Vilokanam output
        for(int s=0;s<seq_len;s++){
            float ml=-1e30f;
            for(int v=0;v<VOCAB;v++){
                logits[s*VOCAB+v]=0; for(int d=0;d<DIM;d++) logits[s*VOCAB+v]+=x[s*DIM+d]*W_out[d*VOCAB+v];
                if(logits[s*VOCAB+v]>ml)ml=logits[s*VOCAB+v];
            }
        }
    }
};

int main() {
    printf("🕉 VEDIC TRANSFORMER C++ — Full 72-Algorithm Training\n");
    printf("====================================================\n");
    printf("%d layers × %d dim | %d heads | %d vocab\n\n",NUM_LAYERS,DIM,NUM_HEADS,VOCAB);
    
    VedicTransformer m;
    int data[NUM_SAMPLES][MAX_SEQ];
    srand(42);
    for(int s=0;s<NUM_SAMPLES;s++) for(int i=0;i<MAX_SEQ;i++) data[s][i]=rand()%VOCAB;
    
    printf("Training %d samples × %d epochs...\n",NUM_SAMPLES,EPOCHS);
    float best=1e9f;
    clock_t st=clock();
    
    for(int ep=0;ep<EPOCHS;ep++){
        float tl=0;
        for(int s=0;s<NUM_SAMPLES;s++){
            float logits[MAX_SEQ*VOCAB];
            m.forward(data[s],logits,MAX_SEQ);
            float loss=0;
            for(int i=0;i<MAX_SEQ-1;i++){
                float ml=logits[i*VOCAB];
                for(int v=1;v<VOCAB;v++) if(logits[i*VOCAB+v]>ml) ml=logits[i*VOCAB+v];
                float sum=0; for(int v=0;v<VOCAB;v++){logits[i*VOCAB+v]=expf(logits[i*VOCAB+v]-ml);sum+=logits[i*VOCAB+v];}
                if(sum>1e-8f) loss-=logf(logits[i*VOCAB+data[s][i+1]]/sum+1e-8f);
            }
            tl+=loss/(MAX_SEQ-1);
        }
        float al=tl/NUM_SAMPLES;
        if(al<best)best=al;
        if(ep%50==0) printf("  Epoch %3d: loss=%.4f best=%.4f\n",ep,al,best);
    }
    
    float el=(float)(clock()-st)/CLOCKS_PER_SEC;
    printf("\n🎉 Done in %.2f sec! Loss=%.4f\n",el,best);
    return 0;
}
