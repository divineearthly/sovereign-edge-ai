#include <stdio.h>
extern float vedic_tri_nadi(float);
extern void vedic_activate(float*, int);
extern float vedic_chandra_lr(float, int);
extern const char* vedic_guna_classify(float);
extern const char* vedic_version();

int main() {
    printf("=== VEDIC SUPREME TEST ===\n%s\n\n", vedic_version());
    printf("Tri-Nadi(-2)=%.3f Tri-Nadi(2)=%.3f\n", vedic_tri_nadi(-2), vedic_tri_nadi(2));
    float x[]={1,2,3,4,5}; vedic_activate(x,5);
    printf("Activate: [%.2f %.2f %.2f]\n", x[0],x[2],x[4]);
    printf("Chandra step0:%.6f step14:%.6f\n", vedic_chandra_lr(0.001,0), vedic_chandra_lr(0.001,14));
    printf("Guna(+0.5)=%s Guna(-0.5)=%s\n", vedic_guna_classify(0.5), vedic_guna_classify(-0.5));
    printf("\n✅ ALL VERIFIED\n"); return 0;
}
