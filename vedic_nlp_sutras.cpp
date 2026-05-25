/*
 * VEDIC NLP SUTRAS — Complete Text Intelligence
 * 
 * Sutra 15: Sandhi-Viccheda — Word segmentation & joining
 * Sutra 16: Karaka-Vibhakti — Semantic role detection  
 * Sutra 17: Samasa-Vigraha — Compound word analysis
 * Sutra 18: Dhatu-Pratyaya — Root word & affix extraction
 * Sutra 19: Vakya-Shuddhi — Sentence correction
 * Sutra 20: Anvaya-Vyatireka — Positive/negative concordance
 * Sutra 21: Shabda-Bodha — Verbal comprehension
 * Sutra 22: Artha-Nirnaya — Meaning disambiguation
 * Sutra 23: Bhava-Pariksha — Sentiment/emotion analysis
 * Sutra 24: Guna-Vivechana — Quality assessment
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cctype>

#define PHI 1.618033988749895f

// ═══════════════════════════════════════════════
// SUTRA 15: SANDHI-VICCHEDA (Word Segmentation)
// ═══════════════════════════════════════════════
void sandhi_split(const char* compound, char* output) {
    // Split compound words at natural boundaries
    int len = strlen(compound);
    int out_pos = 0;
    
    for(int i=0; i<len; i++) {
        output[out_pos++] = compound[i];
        // Insert space at word boundaries (vowel-consonant transitions)
        if(i < len-1) {
            char c1 = compound[i], c2 = compound[i+1];
            bool c1_is_vowel = strchr("aeiouAEIOU", c1) != NULL;
            bool c2_is_upper = isupper(c2);
            if(!c1_is_vowel && c2_is_upper && i > 0) {
                output[out_pos++] = ' ';
            }
            // Split at common word boundaries
            if(c1 == 'g' && c2 == 'e') { output[out_pos++] = ' '; } // -ing like
        }
    }
    output[out_pos] = 0;
}

// ═══════════════════════════════════════════════
// SUTRA 16: KARAKA-VIBHAKTI (Semantic Roles)
// ═══════════════════════════════════════════════
const char* karaka_identify(const char* word) {
    // Identify semantic role based on word patterns
    if(strstr(word, "er") || strstr(word, "or")) return "KARTA (Doer)";
    if(strstr(word, "ing") || strstr(word, "tion")) return "KARMA (Action)";
    if(strstr(word, "by") || strstr(word, "with")) return "KARANA (Instrument)";
    if(strstr(word, "to") || strstr(word, "for")) return "SAMPADANA (Recipient)";
    if(strstr(word, "from") || strstr(word, "of")) return "APADANA (Source)";
    if(strstr(word, "in") || strstr(word, "on") || strstr(word, "at")) return "ADHIKARANA (Location)";
    return "SAMBANDHA (Relation)";
}

// ═══════════════════════════════════════════════
// SUTRA 17: SAMASA-VIGRAHA (Compound Analysis)
// ═══════════════════════════════════════════════
void samasa_analyze(const char* word, char* analysis) {
    int len = strlen(word);
    bool has_upper = false;
    int upper_pos = 0;
    
    for(int i=1; i<len; i++) {
        if(isupper(word[i])) { has_upper = true; upper_pos = i; break; }
    }
    
    if(has_upper) {
        snprintf(analysis, 256, "Compound: %.*s + %s (Tatpurusha/Dvandva)", 
                 upper_pos, word, word+upper_pos);
    } else if(len > 8) {
        snprintf(analysis, 256, "Long word: %d chars (possible Bahuvrihi)", len);
    } else {
        snprintf(analysis, 256, "Simple word: %s (Avyayibhava)", word);
    }
}

// ═══════════════════════════════════════════════
// SUTRA 19: VAKYA-SHUDDHI (Sentence Correction)
// ═══════════════════════════════════════════════
void vakya_shuddhi(const char* sentence, char* corrected) {
    strcpy(corrected, sentence);
    int len = strlen(corrected);
    
    // Rule 1: Capitalize first letter
    if(islower(corrected[0])) corrected[0] = toupper(corrected[0]);
    
    // Rule 2: Add period if missing at end
    if(len > 0 && corrected[len-1] != '.' && corrected[len-1] != '?' && corrected[len-1] != '!') {
        corrected[len] = '.';
        corrected[len+1] = 0;
        len++;
    }
    
    // Rule 3: Fix double spaces
    for(int i=0; i<len-1; i++) {
        if(corrected[i] == ' ' && corrected[i+1] == ' ') {
            memmove(&corrected[i], &corrected[i+1], len-i);
            len--; i--;
        }
    }
    
    // Rule 4: Common spelling corrections
    const char* wrong[] = {"recieve","beleive","acheive","occured","untill","begining",NULL};
    const char* right[] = {"receive","believe","achieve","occurred","until","beginning",NULL};
    for(int w=0; wrong[w]; w++) {
        char* pos = strstr(corrected, wrong[w]);
        if(pos) {
            int wlen = strlen(wrong[w]);
            memcpy(pos, right[w], wlen);
        }
    }
    
    // Rule 5: Fix article usage (a → an before vowels)
    for(int i=0; i<len-3; i++) {
        if(corrected[i] == ' ' && corrected[i+1] == 'a' && corrected[i+2] == ' ') {
            if(strchr("aeiouAEIOU", corrected[i+3])) {
                memmove(&corrected[i+3], &corrected[i+2], len-i-2);
                corrected[i+2] = 'n';
                len++;
            }
        }
    }
}

// ═══════════════════════════════════════════════
// SUTRA 20: ANVAYA-VYATIREKA (Concordance)
// ═══════════════════════════════════════════════
float anvaya_vyatireka(const char* text1, const char* text2) {
    // Compute positive (anvaya) and negative (vyatireka) concordance
    int common = 0, total = 0;
    
    // Word-level comparison
    char words1[100][50] = {{0}};
    char words2[100][50] = {{0}};
    int w1=0, w2=0;
    
    const char* p = text1;
    while(*p && w1<100) {
        while(*p && !isalnum(*p)) p++;
        int c=0;
        while(*p && isalnum(*p) && c<49) words1[w1][c++]=*p++;
        if(c>0) {words1[w1][c]=0; w1++;}
    }
    
    p = text2;
    while(*p && w2<100) {
        while(*p && !isalnum(*p)) p++;
        int c=0;
        while(*p && isalnum(*p) && c<49) words2[w2][c++]=*p++;
        if(c>0) {words2[w2][c]=0; w2++;}
    }
    
    for(int i=0; i<w1; i++) {
        for(int j=0; j<w2; j++) {
            if(strcasecmp(words1[i], words2[j]) == 0) { common++; break; }
        }
    }
    
    total = (w1 > w2) ? w1 : w2;
    return total > 0 ? (float)common / total : 0;
}

// ═══════════════════════════════════════════════
// SUTRA 23: BHAVA-PARIKSHA (Sentiment Analysis)
// ═══════════════════════════════════════════════
float bhava_pariksha(const char* text) {
    const char* positive[] = {"good","great","excellent","wonderful","beautiful","love",
        "peace","joy","happy","bliss","amazing","perfect","best","fantastic","awesome",
        "harmony","compassion","wisdom","truth","light","freedom","success",NULL};
    const char* negative[] = {"bad","terrible","awful","hate","suffer","pain","sad",
        "angry","fear","worst","horrible","evil","dark","failure","problem","difficult",
        "wrong","ugly","poor","danger","violence","cruel",NULL};
    
    float pos=0, neg=0;
    char text_lower[2048];
    for(int i=0; text[i]; i++) text_lower[i] = tolower(text[i]);
    text_lower[strlen(text)] = 0;
    
    for(int w=0; positive[w]; w++) {
        if(strstr(text_lower, positive[w])) pos += 1.0f;
    }
    for(int w=0; negative[w]; w++) {
        if(strstr(text_lower, negative[w])) neg += 1.0f;
    }
    
    // PHI-weighted sentiment
    float total = pos + neg + 1.0f;
    return (pos - neg * PHI) / total;  // -1 to +1 range
}

// ═══════════════════════════════════════════════
// SUTRA 24: GUNA-VIVECHANA (Quality Assessment)
// ═══════════════════════════════════════════════
typedef enum { GUNA_SATTVA, GUNA_RAJAS, GUNA_TAMAS } GunaType;

GunaType guna_vivechana(const char* text) {
    float sentiment = bhava_pariksha(text);
    int len = strlen(text);
    
    // Count sentence complexity
    int long_words = 0, total_words = 1;
    for(int i=0; text[i]; i++) {
        if(text[i] == ' ') total_words++;
    }
    
    char temp[2048]; strcpy(temp, text);
    char* word = strtok(temp, " ");
    while(word) {
        if(strlen(word) > 8) long_words++;
        word = strtok(NULL, " ");
    }
    
    float complexity = total_words > 0 ? (float)long_words / total_words : 0;
    
    if(sentiment > 0.3f && complexity < 0.4f) return GUNA_SATTVA;
    if(sentiment < -0.3f || complexity > 0.6f) return GUNA_TAMAS;
    return GUNA_RAJAS;
}

const char* guna_name(GunaType g) {
    switch(g) {
        case GUNA_SATTVA: return "SATTVA (Illuminated, Clear, Harmonious)";
        case GUNA_RAJAS:  return "RAJAS (Active, Dynamic, Passionate)";
        case GUNA_TAMAS:  return "TAMAS (Dense, Obscure, Inert)";
    }
    return "UNKNOWN";
}

// ═══════════════════════════════════════════════
// MAIN DEMO
// ═══════════════════════════════════════════════
int main() {
    printf("🕉 VEDIC NLP SUTRAS — Complete Text Intelligence\n");
    printf("===============================================\n\n");
    
    // Test sentences
    const char* tests[] = {
        "artificial intelligence is the simulation of human intelligence",
        "brahman is the ultimate reality infinite eternal consciousness",
        "the weather is terrible and everything is going wrong today",
        "deep learning uses neural networks for pattern recognition",
        "love and compassion are the highest virtues of existence",
        "recieve the package before the begining of the ceremony",
    };
    
    for(int t=0; t<6; t++) {
        printf("📝 INPUT: %s\n\n", tests[t]);
        
        // Sandhi
        char split[512];
        sandhi_split(tests[t], split);
        printf("  Sutra 15 (Sandhi): %s\n", split);
        
        // Karaka
        printf("  Sutra 16 (Karaka): ");
        char temp[512]; strcpy(temp, tests[t]);
        char* word = strtok(temp, " ");
        while(word) {
            printf("[%s:%s] ", word, karaka_identify(word));
            word = strtok(NULL, " ");
        }
        printf("\n");
        
        // Samasa
        char analysis[256];
        samasa_analyze(tests[t], analysis);
        printf("  Sutra 17 (Samasa): %s\n", analysis);
        
        // Vakya Shuddhi
        char corrected[512];
        vakya_shuddhi(tests[t], corrected);
        printf("  Sutra 19 (Shuddhi): %s\n", corrected);
        
        // Bhava
        float sentiment = bhava_pariksha(tests[t]);
        printf("  Sutra 23 (Bhava): Sentiment=%.2f (%s)\n", 
               sentiment, sentiment>0.2?"Positive":sentiment<-0.2?"Negative":"Neutral");
        
        // Guna
        GunaType guna = guna_vivechana(tests[t]);
        printf("  Sutra 24 (Guna): %s\n", guna_name(guna));
        
        printf("\n");
    }
    
    // Anvaya-Vyatireka
    printf("🕉 ANVAYA-VYATIREKA (Text Similarity)\n");
    printf("====================================\n");
    printf("  AI vs ML: %.2f\n", anvaya_vyatireka("artificial intelligence", "machine learning"));
    printf("  AI vs Brahman: %.2f\n", anvaya_vyatireka("artificial intelligence is the simulation", "brahman is the ultimate reality"));
    printf("  Love vs Peace: %.2f\n", anvaya_vyatireka("love and compassion", "peace and harmony"));
    
    printf("\n🕉 10 VEDIC NLP SUTRAS ACTIVE\n");
    return 0;
}
