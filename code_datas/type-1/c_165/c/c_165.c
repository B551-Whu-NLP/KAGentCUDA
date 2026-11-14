#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void get_positive_data(const float* all_box, const float* all_scores, const float* all_conf, const int* conf_inds, float* positive_box, float* positive_scores, float* positive_conf, int dims, int clsNum) {
    clock_t start = clock();

    for (int tid = 0; tid < dims; tid++) {
        if (conf_inds[tid] != (-1)) {
            positive_box[tid * 4 + 0] = all_box[tid * 4 + 0];
            positive_box[tid * 4 + 1] = all_box[tid * 4 + 1];
            positive_box[tid * 4 + 2] = all_box[tid * 4 + 2];
            positive_box[tid * 4 + 3] = all_box[tid * 4 + 3];
            for (int i = 0; i < clsNum; i++) {
                positive_scores[tid * clsNum + i] = all_scores[tid * clsNum + i];
            }
            positive_conf[tid] = all_conf[tid];
        }
        else {
            positive_box[tid * 4 + 0] = 0;
            positive_box[tid * 4 + 1] = 0;
            positive_box[tid * 4 + 2] = 0;
            positive_box[tid * 4 + 3] = 0;
            for (int i = 0; i < clsNum; i++) {
                positive_scores[tid * clsNum + i] = (-1);
            }
            positive_conf[tid] = (-1);
        }
    }

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char** argv) {
    int dims = 1000000;
    int clsNum = 500;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* all_box_host = (float*)malloc(dims * 4 * sizeof(float));
    float* all_scores_host = (float*)malloc(dims * clsNum * sizeof(float));
    float* all_conf_host = (float*)malloc(dims * sizeof(float));
    int* conf_inds_host = (int*)malloc(dims * sizeof(int));

    float* positive_box_host = (float*)malloc(dims * 4 * sizeof(float));
    float* positive_scores_host = (float*)malloc(dims * clsNum * sizeof(float));
    float* positive_conf_host = (float*)malloc(dims * sizeof(float));

    srand(42);
    for (int i = 0; i < dims; i++) {
        all_conf_host[i] = (float)(rand() % 100) / 100.0f;
        conf_inds_host[i] = (all_conf_host[i] > 0.5f) ? i : -1;
        for (int j = 0; j < 4; j++) {
            all_box_host[i * 4 + j] = (float)(rand() % 100);
        }
        for (int j = 0; j < clsNum; j++) {
            all_scores_host[i * clsNum + j] = (float)(rand() % 100) / 100.0f;
        }
    }

    get_positive_data(all_box_host, all_scores_host, all_conf_host, conf_inds_host, positive_box_host, positive_scores_host, positive_conf_host, dims, clsNum);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_165_output.txt");
        FILE* file = fopen(output_file, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(file, "Box %d: [%.6f, %.6f, %.6f, %.6f], Conf: %.6f\n", i,
                    positive_box_host[i * 4 + 0], positive_box_host[i * 4 + 1],
                    positive_box_host[i * 4 + 2], positive_box_host[i * 4 + 3],
                    positive_conf_host[i]);
            }
            fclose(file);
        }
    }

    free(all_box_host);
    free(all_scores_host);
    free(all_conf_host);
    free(conf_inds_host);

    free(positive_box_host);
    free(positive_scores_host);
    free(positive_conf_host);
    return 0;
}