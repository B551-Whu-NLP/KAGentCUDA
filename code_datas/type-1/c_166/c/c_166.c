#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void get_before_nms_data(const float* boxes, const float* scores, const int* labels, const int* index, float* boxes_out, float* scores_out, int* labels_out, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (index[tid] == 0) {
            boxes_out[tid * 4 + 0] = -1;
            boxes_out[tid * 4 + 1] = -1;
            boxes_out[tid * 4 + 2] = -1;
            boxes_out[tid * 4 + 3] = -1;
            scores_out[tid] = -1;
            labels_out[tid] = -1;
        }
        else {
            boxes_out[tid * 4 + 0] = boxes[tid * 4 + 0];
            boxes_out[tid * 4 + 1] = boxes[tid * 4 + 1];
            boxes_out[tid * 4 + 2] = boxes[tid * 4 + 2];
            boxes_out[tid * 4 + 3] = boxes[tid * 4 + 3];
            scores_out[tid] = scores[tid];
            labels_out[tid] = labels[tid];
        }
    }
}

int main(int argc, char* argv[]) {
    int dims = 1000000;  // 增加到100万
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float* boxes = (float*)malloc(dims * 4 * sizeof(float));
    float* scores = (float*)malloc(dims * sizeof(float));
    int* labels = (int*)malloc(dims * sizeof(int));
    int* index = (int*)malloc(dims * sizeof(int));
    float* boxes_out = (float*)malloc(dims * 4 * sizeof(float));
    float* scores_out = (float*)malloc(dims * sizeof(float));
    int* labels_out = (int*)malloc(dims * sizeof(int));

    srand(42);
    for (int i = 0; i < dims * 4; i++) {
        boxes[i] = (float)(rand() % 1000) / 10.0f;
    }

    for (int i = 0; i < dims; i++) {
        scores[i] = (float)(rand() % 1000) / 10.0f;
        labels[i] = rand() % 100;
        index[i] = rand() % 2;
    }

    for (int iter = 0; iter < 1000; iter++) {
        get_before_nms_data(boxes, scores, labels, index, boxes_out, scores_out, labels_out, dims);
    }

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_166_output.txt");
        FILE* fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "Box %d: (%.2f, %.2f, %.2f, %.2f) - Score: %.2f - Label: %d\n", i,
                        boxes_out[i * 4], boxes_out[i * 4 + 1], boxes_out[i * 4 + 2], boxes_out[i * 4 + 3],
                        scores_out[i], labels_out[i]);
            }
            fclose(fp);
        }
    }

    free(boxes);
    free(scores);
    free(labels);
    free(index);
    free(boxes_out);
    free(scores_out);
    free(labels_out);
    return 0;
}