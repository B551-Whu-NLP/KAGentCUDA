#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int data_size = 100000000;
    const int stride = 2;
    int* output_host = (int*)malloc(data_size * sizeof(int));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    clock_t start_time = clock();
    
    for (int i = 0; i < data_size; i++) {
        output_host[i] = i * stride;
    }
    
    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    
    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_184_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", output_host[i]);
            }
            fprintf(fp, "\n");
            fprintf(fp, "Time: %.2f(ms)\n", time_taken * 1000);
            fclose(fp);
        }
    }
    
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    
    free(output_host);
    return 0;
}