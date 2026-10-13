#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int n = 100000000;
    int* h_head = (int*)malloc(n * sizeof(int));
    int* h_first_pts = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; ++i) {
        h_head[i] = (i % 2 == 0) ? 1 : 0;
    }

    clock_t start = clock();

    for (int i = 0; i < n; i++) {
        if (h_head[i] == 1) {
            h_first_pts[i] = i;
        }
        else {
            h_first_pts[i] = 0;
        }
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
        strcpy(output_file, "c_238_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%d ", h_first_pts[i]);
            }
            fclose(fp);
        }
    }

    free(h_head);
    free(h_first_pts);

    return 0;
}