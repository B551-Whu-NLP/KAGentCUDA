#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void get_boxes_for_nms(const float* boxes_before_nms, const float* offset, float* boxes_for_nms, int dims) {
    for (int tid = 0; tid < dims; tid++) {
        if (boxes_before_nms[tid * 4] == (-1) && boxes_before_nms[tid * 4 + 1] == (-1) &&
            boxes_before_nms[tid * 4 + 2] == (-1) && boxes_before_nms[tid * 4 + 3] == (-1)) {
            boxes_for_nms[tid * 4] = -1;
            boxes_for_nms[tid * 4 + 1] = -1;
            boxes_for_nms[tid * 4 + 2] = -1;
            boxes_for_nms[tid * 4 + 3] = -1;
        }
        else {
            boxes_for_nms[tid * 4] = boxes_before_nms[tid * 4] + offset[tid];
            boxes_for_nms[tid * 4 + 1] = boxes_before_nms[tid * 4 + 1] + offset[tid];
            boxes_for_nms[tid * 4 + 2] = boxes_before_nms[tid * 4 + 2] + offset[tid];
            boxes_for_nms[tid * 4 + 3] = boxes_before_nms[tid * 4 + 3] + offset[tid];
        }
    }
}

int main(int argc, char *argv[]) {
    const int dims = 10000000;
    float *boxes_before_nms = (float*)malloc(dims * 4 * sizeof(float));
    float *offset = (float*)malloc(dims * sizeof(float));
    float *boxes_for_nms = (float*)malloc(dims * 4 * sizeof(float));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < dims * 4; ++i) {
        boxes_before_nms[i] = i + 1;
    }
    for (int i = 0; i < dims; ++i) {
        offset[i] = 0.5;
    }

    get_boxes_for_nms(boxes_before_nms, offset, boxes_for_nms, dims);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_174_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", boxes_for_nms[i]);
            }
            fprintf(fp, "\n");
            fclose(fp);
        }
    }

    free(boxes_before_nms);
    free(offset);
    free(boxes_for_nms);

    return 0;
}