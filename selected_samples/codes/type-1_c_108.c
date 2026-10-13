#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void subsample_ind_and_labels(int* d_ind_sub, const int* d_ind, unsigned int* d_label_sub, const unsigned int* d_label, int n_out, float inv_sub_factor) {
    for (int ind_out = 0; ind_out < n_out; ind_out++) {
        int ind_in = (int)floorf((float)(ind_out)*inv_sub_factor);
        d_ind_sub[ind_out] = d_ind[ind_in];
        d_label_sub[ind_out] = d_label[ind_in];
    }
}

int main(int argc, char** argv) {
    int n_in = 100000000;
    int n_out = 50000000;
    float inv_sub_factor = (float)n_in / (float)n_out;
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int* h_ind = (int*)malloc(n_in * sizeof(int));
    unsigned int* h_label = (unsigned int*)malloc(n_in * sizeof(unsigned int));
    int* h_ind_sub = (int*)malloc(n_out * sizeof(int));
    unsigned int* h_label_sub = (unsigned int*)malloc(n_out * sizeof(unsigned int));

    for (int i = 0; i < n_in; i++) {
        h_ind[i] = i;
        h_label[i] = i % 10;
    }

    clock_t start_time = clock();

    subsample_ind_and_labels(h_ind_sub, h_ind, h_label_sub, h_label, n_out, inv_sub_factor);

    clock_t end_time = clock();

    double elapsed_time = ((double)(end_time - start_time) * 1000.0) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_108_output.txt");
        FILE* f = fopen(output_file, "w");
        if (f != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(f, "Index: %d, Label: %u\n", h_ind_sub[i], h_label_sub[i]);
            }
            fclose(f);
        }
    }

    free(h_ind);
    free(h_label);
   
    printf("Time: %.2f(ms)\n", elapsed_time); free(h_ind_sub);
    free(h_label_sub);
    return 0;
}