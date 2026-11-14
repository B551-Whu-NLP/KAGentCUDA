#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void getOffsetBox_cpu(const int* clsIndex, const float* max_coordinate, float* offset, int dims, int batchSize, const float* before_nms_boxes) {
    for (int tid = 0; tid < dims; tid++) {
        int numPerbatch = dims;
        for (int i = 0; i < batchSize; i++) {
            if (before_nms_boxes[i * dims * 4 + tid * 4] == -1) {
                offset[i * numPerbatch + tid] = 0;
            }
            else {
                offset[i * numPerbatch + tid] = clsIndex[i * numPerbatch + tid] * (max_coordinate[i * dims * 4] + 1);
            }
        }
    }
}

int main(int argc, char** argv) {
    int dims = 1024;
    int batchSize = 1024;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    int* h_clsIndex = (int*)malloc(batchSize * dims * sizeof(int));
    float* h_max_coordinate = (float*)malloc(batchSize * dims * 4 * sizeof(float));
    float* h_offset = (float*)malloc(batchSize * dims * sizeof(float));
    float* h_before_nms_boxes = (float*)malloc(batchSize * dims * 4 * sizeof(float));

    srand(0);

    for (int i = 0; i < batchSize * dims; ++i) {
        h_clsIndex[i] = rand() % 10;
    }

    for (int i = 0; i < batchSize * dims * 4; ++i) {
        h_max_coordinate[i] = (float)(rand() % 10);
    }
    for (int i = 0; i < batchSize * dims * 4; ++i) {
        h_before_nms_boxes[i] = (float)(rand() % 10);
    }

    getOffsetBox_cpu(h_clsIndex, h_max_coordinate, h_offset, dims, batchSize, h_before_nms_boxes);

    if (verbose) {
        char output_filename[256];
        snprintf(output_filename, sizeof(output_filename), "c_142_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < batchSize; ++i) {
                for (int j = 0; j < dims; ++j) {
                    fprintf(file, "%.6f ", h_offset[i * dims + j]);
                }
                fprintf(file, "\n");
            }
            fclose(file);
        }
    }

    free(h_clsIndex);
    free(h_max_coordinate);
    free(h_offset);
    free(h_before_nms_boxes);
    return 0;
}