#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void initialArray0(int tasks, int* f3) {
    for (int i = 0; i < tasks; i++) {
        f3[i] = 0;
    }
}

int main(int argc, char *argv[]) {
    int numTasks = 10000000;
    int* h_f3 = (int*)malloc(numTasks * sizeof(int));
    int verbose = 0;
    
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    clock_t start = clock();

    initialArray0(numTasks, h_f3);

    clock_t end = clock();
    double timeElapsed = (double)(end - start) / CLOCKS_PER_SEC;

    if (verbose) {
        char outputFileName[256];
        strcpy(outputFileName, "c_8_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d ", h_f3[i]);
            }
            fclose(fp);
        }
    }

    free(h_f3);
    printf("Time: %.2fms\n", timeElapsed * 1000);
    return 0;
}