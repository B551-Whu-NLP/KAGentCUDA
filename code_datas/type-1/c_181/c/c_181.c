#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    const int data_size = 100000000;
    int* means_host = (int*)malloc(data_size * sizeof(int));
    
    for (int i = 0; i < data_size; ++i) {
        means_host[i] = i + 1;
    }
    
    clock_t start = clock();
    
    for (int id = 0; id < data_size; id++) {
        means_host[id] = 0;
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time: %.2fms\n", time_taken * 1000);
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    if (verbose) {
        FILE *fp = fopen("c_181_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", means_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    free(means_host);
    return 0;
}