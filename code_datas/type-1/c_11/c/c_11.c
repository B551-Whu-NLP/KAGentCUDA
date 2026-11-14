#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void set_sorting_offset(const int nrows, const int ncols, long long* offsets) {
    for (int tid = 0; tid <= ncols; tid++) {
        offsets[tid] = (long long)tid * nrows;
    }
}

int main(int argc, char *argv[]) {
    int nrows = 1000000;
    int ncols = 100000;
    long long* h_offsets = (long long*)malloc((ncols + 1) * sizeof(long long));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();
    set_sorting_offset(nrows, ncols, h_offsets);
    clock_t end = clock();
    
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2f(ms)\n", time_taken * 1000);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_11_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i <= 1000; i++) {
                fprintf(fp, "%lld ", h_offsets[i]);
            }
            fclose(fp);
        }
    }

    free(h_offsets);
    return 0;
}