#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int L_x = 2048000; // 进一步增大数组大小
    float* h_f_in = (float*)malloc(L_x * sizeof(float));
    float* h_f_target = (float*)malloc(L_x * sizeof(float));
    
    for (int i = 0; i < L_x; ++i) {
        h_f_in[i] = (float)(i);
        h_f_target[i] = (float)(i * 2);
    }
    
    clock_t start = clock();
    
    // 大幅增加外层循环次数
    for (int iter = 0; iter < 1000; iter++) {
        for (int k_x = 0; k_x < L_x; k_x++) {
            float tempval = 0.0f;
            tempval = h_f_in[k_x];
            h_f_in[k_x] = h_f_target[k_x];
            h_f_target[k_x] = tempval;
        }
    }
    
    clock_t end = clock();
    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        FILE *fp = fopen("c_68_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "h_f_in[%d]: %f, h_f_target[%d]: %f\n", i, h_f_in[i], i, h_f_target[i]);
            }
            fclose(fp);
        }
    }
    
    free(h_f_in);
    free(h_f_target);
  
    printf("Time: %.2fms\n", time_taken * 1000);  
    return 0;
}