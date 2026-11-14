#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void resetIndices(long* vec_out, const long N) {
    for (long idx = 0; idx < N; idx++) {
        vec_out[idx] = idx;
    }
}

int main(int argc, char *argv[]) {
    const long data_size = 100000000;
    long* vec_out_host = (long*)malloc(data_size * sizeof(long));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();
    
    resetIndices(vec_out_host, data_size);
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    if (verbose) {
        char output_file[256];
        snprintf(output_file, sizeof(output_file), "c_183_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (long i = 0; i < 1000; i++) {
                fprintf(fp, "%ld ", vec_out_host[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }
    
    printf("Time: %.2f(ms)\n", time_taken * 1000.0);
    
    free(vec_out_host);
    return 0;
}