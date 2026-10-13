#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void clearLabel(float* prA, float* prB, unsigned int num_nodes, float alpha) {
    for (unsigned int i = 0; i < num_nodes; i++) {
        prA[i] = alpha * prA[i] + (1.0f - alpha) * prB[i];
        prB[i] = alpha * prB[i] + (1.0f - alpha) * prA[i];
    }
}

int main(int argc, char *argv[]) {
    const unsigned int num_nodes = 5000000;
    float* h_prA = (float*)malloc(num_nodes * sizeof(float));
    float* h_prB = (float*)malloc(num_nodes * sizeof(float));

    for (unsigned int i = 0; i < num_nodes; i++) {
        h_prA[i] = (float)(i);
        h_prB[i] = (float)(2 * i);
    }

    clock_t start = clock();
    for(int iter = 0; iter < 1000; iter++) {
        clearLabel(h_prA, h_prB, num_nodes, 1.0);
    }
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time: %.2fms\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_50_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (unsigned int i = 0; i < 1000; ++i) {
                fprintf(fp, "prA[%u]: %f, prB[%u]: %f\n", i, h_prA[i], i, h_prB[i]);
            }
            fclose(fp);
        }
    }


    free(h_prA);
    free(h_prB);
    return 0;
}