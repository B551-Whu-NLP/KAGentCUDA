#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>

void decode(const float* anchor, const float* locData, float* predictBox, int dims, float scaleClamp, int batchSize) {
    for (int tid = 0; tid < dims; tid++) {
        for (int i = 0; i < batchSize; i++) {
            float anchorW = anchor[i * dims * 4 + tid * 4 + 2] - anchor[i * dims * 4 + tid * 4];
            float anchorH = anchor[i * dims * 4 + tid * 4 + 3] - anchor[i * dims * 4 + tid * 4 + 1];
            float anchorCx = anchor[i * dims * 4 + tid * 4] + 0.5 * anchorW;
            float anchorCy = anchor[i * dims * 4 + tid * 4 + 1] + 0.5 * anchorH;

            float dx = locData[i * dims * 4 + tid * 4];
            float dy = locData[i * dims * 4 + tid * 4 + 1];
            float dw = locData[i * dims * 4 + tid * 4 + 2];
            float dh = locData[i * dims * 4 + tid * 4 + 3];

            if (dw > scaleClamp) {
                dw = scaleClamp;
            }
            if (dh > scaleClamp) {
                dh = scaleClamp;
            }

            float preCx = dx * anchorW + anchorCx;
            float preCy = dy * anchorH + anchorCy;
            float preW = anchorW * expf(dw);
            float preH = anchorH * expf(dh);

            predictBox[i * dims * 4 + tid * 4] = preCx - 0.5 * preW;
            predictBox[i * dims * 4 + tid * 4 + 1] = preCy - 0.5 * preH;
            predictBox[i * dims * 4 + tid * 4 + 2] = preCx + 0.5 * preW;
            predictBox[i * dims * 4 + tid * 4 + 3] = preCy + 0.5 * preH;
        }
    }
}

int main(int argc, char *argv[]) {
    const int dims = 100;
    const int batchSize = 100;
    const float scaleClamp = 5.0;
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    float *anchor = (float*)malloc(batchSize * dims * 4 * sizeof(float));
    float *locData = (float*)malloc(batchSize * dims * 4 * sizeof(float));
    float *predictBox = (float*)malloc(batchSize * dims * 4 * sizeof(float));

    for (int i = 0; i < batchSize * dims * 4; ++i) {
        anchor[i] = (float)(i % 100 + 1);
        locData[i] = (float)((i % 100 + 1) * 0.1);
    }

    decode(anchor, locData, predictBox, dims, scaleClamp, batchSize);

    if (verbose) {
        char outputFileName[256];
        snprintf(outputFileName, sizeof(outputFileName), "c_176_output.txt");
        FILE *fp = fopen(outputFileName, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(fp, "%.2f ", predictBox[i]);
            }
            fclose(fp);
        }
    }

    free(anchor);
    free(locData);
    free(predictBox);

    return 0;
}