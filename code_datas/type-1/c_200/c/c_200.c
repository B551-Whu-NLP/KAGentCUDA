#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void Copy_List(const int element_numbers, const float* origin_list, float* list) {
    for (int i = 0; i < element_numbers; i++) {
        list[i] = origin_list[i];
    }
}

int main(int argc, char *argv[]) {
    const int element_numbers = 2000000; // 增大数组大小
    float* origin_list = (float*)malloc(element_numbers * sizeof(float));
    float* list = (float*)malloc(element_numbers * sizeof(float));
    
    for (int i = 0; i < element_numbers; ++i) {
        origin_list[i] = i + 1.0; 
    }

    clock_t start_time = clock();

    // 增加复制次数以延长执行时间
    for (int iter = 0; iter < 500; iter++) {
        Copy_List(element_numbers, origin_list, list);
    }

    clock_t end_time = clock();

    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Time: %.2f(ms)\n", time_taken * 1000);

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_200_output.txt");
        FILE *fp = fopen(output_filename, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%.2f ", list[i]);
            }
            fclose(fp);
        }
    }


    free(origin_list);
    free(list);

    return 0;
}