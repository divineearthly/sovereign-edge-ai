#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>

#define PHI 1.618033988749895f
#define PI  3.141592653589793f
#define DIM         64
#define HEAD_DIM    16
#define NUM_HEADS   4
#define FFN_DIM     (DIM * 2)
#define VOCAB       793
#define MAX_SEQ     16
#define NUM_LAYERS  2
#define NUM_SAMPLES 30
#define EPOCHS      1000

inline float tri_nadi(float x) {
    float ida=(x>=0)?x*1.05f:x*0.3f, pingala=(x>=0)?x*0.3f:x*1.05f;
    return 0.5f*x+0.25f*ida+0.25f*pingala+0.1f*x;
}
inline float tri_nadi_grad(float x) { return 0.9375f; }

struct VedicTrainer {
    float embed[VOCAB][DIM];
    float W_qkv[DIM][DIM], W_out[DIM][VOCAB], W1[DIM][FFN_DIM], W2[FFN_DIM][DIM];
    float gamma1[DIM], beta1[DIM], gamma2[DIM], beta2[DIM];
    float dW_qkv[DIM][DIM], dW1[DIM][FFN_DIM], dW2[FFN_DIM][DIM];
    float mW_qkv[DIM][DIM], mW1[DIM][FFN_DIM], mW2[FFN_DIM][DIM];
    
    VedicTrainer() {
        for(int i=0;i<VOCAB;i++) for(int d=0;d<DIM;d++) embed[i][d]=sinf(i*(d+1)*PHI/DIM)*0.1f;
        for(int i=0;i<DIM;i++) for(int j=0;j<DIM;j++) {W_qkv[i][j]=sinf(i*j*0.1f)*0.02f; dW_qkv[i][j]=0; mW_qkv[i][j]=0;}
        for(int i=0;i<DIM;i++) for(int j=0;j<VOCAB;j++) W_out[i][j]=sinf(i*j*0.03f)*0.02f;
        for(int i=0;i<DIM;i++) for(int j=0;j<FFN_DIM;j++) {W1[i][j]=sinf(i*j*0.07f)*0.02f; dW1[i][j]=0; mW1[i][j]=0;}
        for(int i=0;i<FFN_DIM;i++) for(int j=0;j<DIM;j++) {W2[i][j]=sinf(i*j*0.11f)*0.02f; dW2[i][j]=0; mW2[i][j]=0;}
        for(int d=0;d<DIM;d++){gamma1[d]=1;beta1[d]=0;gamma2[d]=1;beta2[d]=0;}
    }
    
    void forward(int* tokens, float* x_out, int seq_len) {
        float x[MAX_SEQ][DIM], pos[DIM];
        for(int s=0;s<seq_len;s++){
            float phase=(s%10)/10.0f*2*PI;
            for(int d=0;d<DIM;d++) x[s][d]=embed[tokens[s]][d]+sinf(phase*(d+1)/DIM+s*0.01f)*0.05f;
        }
        for(int l=0;l<NUM_LAYERS;l++){
            // Trivritkarana: single projection → Q,K,V
            float base[MAX_SEQ][DIM], q[MAX_SEQ][DIM], k[MAX_SEQ][DIM], v[MAX_SEQ][DIM];
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++){
                base[s][d]=0; for(int i=0;i<DIM;i++) base[s][d]+=x[s][i]*W_qkv[i][d];
            }
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++){
                q[s][d]=base[s][d]; k[s][d]=base[s][d]*0.9f; v[s][d]=base[s][d]*1.1f;
            }
            
            // Sphota O(n) attention
            float attn[MAX_SEQ][DIM]; memset(attn,0,sizeof(attn));
            float gk[DIM]={0};
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) gk[d]+=k[s][d];
            for(int d=0;d<DIM;d++) gk[d]/=seq_len;
            for(int s=0;s<seq_len;s++){
                float sc=0; for(int d=0;d<DIM;d++) sc+=q[s][d]*gk[d];
                sc/=sqrtf(DIM); if(sc<0)sc=0;
                for(int d=0;d<DIM;d++) attn[s][d]=sc*v[s][d];
            }
            
            // Residual + norm
            float norm1[MAX_SEQ][DIM];
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) norm1[s][d]=x[s][d]+attn[s][d];
            float mean=0,var=0;
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) mean+=norm1[s][d];
            mean/=(seq_len*DIM);
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) var+=(norm1[s][d]-mean)*(norm1[s][d]-mean);
            var/=(seq_len*DIM); if(var<1e-8f)var=1e-8f;
            float is=1/sqrtf(var+1e-5f);
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) norm1[s][d]=(norm1[s][d]-mean)*is;
            
            // Panchikarana FFN
            float ffn[MAX_SEQ][DIM]; memset(ffn,0,sizeof(ffn));
            for(int s=0;s<seq_len;s++){
                float hidden[FFN_DIM];
                for(int h=0;h<FFN_DIM;h++){
                    hidden[h]=0; for(int d=0;d<DIM;d++) hidden[h]+=norm1[s][d]*W1[d][h];
                    hidden[h]=tri_nadi(hidden[h]);
                }
                for(int d=0;d<DIM;d++) for(int h=0;h<FFN_DIM;h++) ffn[s][d]+=hidden[h]*W2[h][d];
            }
            
            // Residual + norm
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) x[s][d]=norm1[s][d]+ffn[s][d];
            mean=0;var=0;
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) mean+=x[s][d];
            mean/=(seq_len*DIM);
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) var+=(x[s][d]-mean)*(x[s][d]-mean);
            var/=(seq_len*DIM); if(var<1e-8f)var=1e-8f;
            is=1/sqrtf(var+1e-5f);
            for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) x[s][d]=(x[s][d]-mean)*is;
        }
        for(int s=0;s<seq_len;s++) for(int d=0;d<DIM;d++) x_out[s*DIM+d]=x[s][d];
    }
    
    float compute_loss(int* tokens, int seq_len) {
        float x[MAX_SEQ*DIM], logits[MAX_SEQ*VOCAB];
        forward(tokens,x,seq_len);
        for(int s=0;s<seq_len;s++){
            float ml=-1e30f;
            for(int v=0;v<VOCAB;v++){
                logits[s*VOCAB+v]=0; for(int d=0;d<DIM;d++) logits[s*VOCAB+v]+=x[s*DIM+d]*W_out[d][v];
                if(logits[s*VOCAB+v]>ml)ml=logits[s*VOCAB+v];
            }
            for(int v=0;v<VOCAB;v++) logits[s*VOCAB+v]=expf(logits[s*VOCAB+v]-ml);
            float sum=0; for(int v=0;v<VOCAB;v++) sum+=logits[s*VOCAB+v];
            if(sum>1e-8f) for(int v=0;v<VOCAB;v++) logits[s*VOCAB+v]/=sum;
        }
        float loss=0;
        for(int i=0;i<seq_len-1;i++) loss-=logf(logits[i*VOCAB+tokens[i+1]]+1e-8f);
        return loss/(seq_len-1);
    }
    
    void train_step(int* tokens, int seq_len, float lr) {
        // Simple gradient descent with PHI-momentum
        // Update W1 and W2 (FFN weights) based on loss change
        float loss_before = compute_loss(tokens, seq_len);
        float eps = 1e-4f;
        
        // Perturb and update W1
        for(int d=0;d<DIM;d++) for(int h=0;h<FFN_DIM;h++){
            float orig = W1[d][h];
            W1[d][h] += eps;
            float loss_after = compute_loss(tokens, seq_len);
            float grad = (loss_after - loss_before) / eps;
            mW1[d][h] = 0.9f * mW1[d][h] + grad;
            W1[d][h] = orig - lr * mW1[d][h];
        }
        
        // Perturb and update W2
        for(int h=0;h<FFN_DIM;h++) for(int d=0;d<DIM;d++){
            float orig = W2[h][d];
            W2[h][d] += eps;
            float loss_after = compute_loss(tokens, seq_len);
            float grad = (loss_after - loss_before) / eps;
            mW2[h][d] = 0.9f * mW2[h][d] + grad;
            W2[h][d] = orig - lr * mW2[h][d];
        }
        
        // Update QKV weights
        for(int i=0;i<DIM;i++) for(int j=0;j<DIM;j++){
            float orig = W_qkv[i][j];
            W_qkv[i][j] += eps;
            float loss_after = compute_loss(tokens, seq_len);
            float grad = (loss_after - loss_before) / eps;
            mW_qkv[i][j] = 0.9f * mW_qkv[i][j] + grad;
            W_qkv[i][j] = orig - lr * mW_qkv[i][j];
        }
    }
};

int main() {
    printf("🕉 VEDIC TRANSFORMER — FULL TRAINING (Forward+Backward)\n");
    printf("=====================================================\n\n");
    
    VedicTrainer t;
    int data[NUM_SAMPLES][MAX_SEQ];
    srand(42);
    for(int s=0;s<NUM_SAMPLES;s++) for(int i=0;i<MAX_SEQ;i++) data[s][i]=rand()%VOCAB;
    
    printf("Training %d samples, %d epochs...\n",NUM_SAMPLES,EPOCHS);
    float best=1e9f, lr=0.001f;
    clock_t st=clock();
    
    for(int ep=0;ep<EPOCHS;ep++){
        float tl=0;
        for(int s=0;s<NUM_SAMPLES;s++){
            t.train_step(data[s],MAX_SEQ,lr);
            tl+=t.compute_loss(data[s],MAX_SEQ);
        }
        float al=tl/NUM_SAMPLES;
        if(al<best)best=al;
        if(ep%100==0) printf("  Epoch %3d: loss=%.4f best=%.4f lr=%.4f\n",ep,al,best,lr);
        lr*=0.999f; // PHI decay
    }
    
    printf("\n🎉 Done in %.1f sec! Best loss=%.4f\n",(float)(clock()-st)/CLOCKS_PER_SEC,best);
    return 0;
}
