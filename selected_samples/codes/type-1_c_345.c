#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void returnResult(const float* box, const float* score, const int* label,
    float* box_out, float* score_out, int* label_out,
    float score_thr, const int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (score[tid] < score_thr) {
            score_out[tid] = 0;
            box_out[tid * 4 + 0] = -1;
            box_out[tid * 4 + 1] = -1;
            box_out[tid * 4 + 2] = -1;
            box_out[tid * 4 + 3] = -1;
            label_out[tid] = -1;
        }
        else {
            score_out[tid] = score[tid];
            box_out[tid * 4 + 0] = box[tid * 4 + 0];
            box_out[tid * 4 + 1] = box[tid * 4 + 1];
            box_out[tid * 4 + 2] = box[tid * 4 + 2];
            box_out[tid * 4 + 3] = box[tid * 4 + 3];
            label_out[tid] = label[tid];
        }
    }
}

int main(int argc, char *argv[]) {
    const int dims = 10000000;  // 增加维度到1000万
    const float score_thr = 0.5;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* box, * score, * score_out, * box_out;
    int* label, * label_out;
    size_t size = dims * sizeof(float);

    box = (float*)malloc(size * 4);
    score = (float*)malloc(size);
    score_out = (float*)malloc(size);
    box_out = (float*)malloc(size * 4);
    label = (int*)malloc(size);
    label_out = (int*)malloc(size);

    srand(42);
    for (int i = 0; i < dims * 4; ++i) {
        box[i] = (float)(rand() % 1000) / 100.0f;
    }

    for (int i = 0; i < dims; ++i) {
        score[i] = (float)(rand() % 1000) / 1000.0f;
        label[i] = rand() % 100;
    }

    clock_t start = clock();
    returnResult(box, score, label, box_out, score_out, label_out, score_thr, dims);
    clock_t end = clock();

    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_345_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "Box[%d]: [%f, %f, %f, %f], Score[%d]: %f, Label[%d]: %d\n",
                    i, box_out[i * 4], box_out[i * 4 + 1], box_out[i * 4 + 2], box_out[i * 4 + 3],
                    i, score_out[i], i, label_out[i]);
            }
            fclose(fp);
        }
    }

    free(box);
    free(score);
    free(score_out);
    free(box_out);
    free(label);
    free(label_out);
    printf("Time: %.2f(ms)\n", time_taken * 1000);
    return 0;
}