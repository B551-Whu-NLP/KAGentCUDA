#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void create_p_vect(float* node_info1, float* node_info2, float* p, int n_nodes_1, int n_nodes_2) {
    int tx, ty;
    float cutoff = 0.5;

    clock_t start = clock();

    for (int iter = 0; iter < 1000; iter++) {
        for (tx = 0; tx < n_nodes_1; tx++) {
            for (ty = 0; ty < n_nodes_2; ty++) {
                int ind = tx * n_nodes_2 + ty;
                if ((node_info1[tx] < cutoff) && (node_info2[ty] < cutoff))
                    p[ind] = 0;
                else
                    p[ind] = node_info1[tx] * node_info2[ty];
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_spent);
}

int main(int argc, char* argv[]) {
    int n_nodes_1 = 1000;
    int n_nodes_2 = 1500;

    float* node_info1 = (float*)malloc(n_nodes_1 * sizeof(float));
    float* node_info2 = (float*)malloc(n_nodes_2 * sizeof(float));
    float* p = (float*)malloc(n_nodes_1 * n_nodes_2 * sizeof(float));

    for (int i = 0; i < n_nodes_1; i++) {
        node_info1[i] = 0.1 * (i + 1);
    }
    for (int i = 0; i < n_nodes_2; i++) {
        node_info2[i] = 0.2 * (i + 1);
    }

    create_p_vect(node_info1, node_info2, p, n_nodes_1, n_nodes_2);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_321_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < n_nodes_1; i++) {
                for (int j = 0; j < n_nodes_2; j++) {
                    fprintf(file, "%.2f ", p[i * n_nodes_2 + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(node_info1);
    free(node_info2);
    free(p);
    return 0;
}