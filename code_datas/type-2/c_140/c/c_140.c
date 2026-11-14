#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

void GraphSum_forward(float* in, float* out, int* indptr, int* indices, int dim, int size) {
    clock_t start = clock();
    for (int src = 0; src < size; src++) {
        int src_degree = indptr[src + 1] - indptr[src];
        for (int i = indptr[src]; i < indptr[src + 1]; i++) {
            int dst = indices[i];
            int dst_degree = indptr[dst + 1] - indptr[dst];
            float coef = 1.0f / sqrtf((float)(src_degree * dst_degree));
            for (int j = 0; j < dim; j++) {
                out[src * dim + j] += coef * in[dst * dim + j];
            }
        }
    }
    clock_t end = clock();
    printf("Time: %.2fms\n", ((double)(end - start)) / CLOCKS_PER_SEC * 1000);
}

int main(int argc, char *argv[]) {
    int dim = 512;
    int numNodes = 10000;
    int edgesPerNode = 20;
    int totalEdges = numNodes * edgesPerNode;  // 32

    float* h_d_in_data  = (float*)malloc(numNodes * dim * sizeof(float));
    float* h_d_out_data = (float*)malloc(numNodes * dim * sizeof(float));
    int*   h_d_indptr   = (int*)malloc((numNodes + 1) * sizeof(int));
    int*   h_d_indices  = (int*)malloc(totalEdges * sizeof(int));  // ← 修复：用总边数

    srand(42);
    for (int i = 0; i < numNodes * dim; i++) {
        h_d_in_data[i]  = (float)(rand() % 100) / 10.0f;
        h_d_out_data[i] = 0.0f;
    }

    // indptr: 每个节点2条边
    for (int i = 0; i < numNodes + 1; i++) {
        h_d_indptr[i] = i * edgesPerNode;
    }

    // indices: 填满 totalEdges 个条目
    for (int i = 0; i < totalEdges; i++) {          // ← 修复：遍历到 totalEdges
        h_d_indices[i] = (i + 1) % numNodes;
    }

    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    GraphSum_forward(h_d_in_data, h_d_out_data, h_d_indptr, h_d_indices, dim, numNodes);

    if (verbose) {
        FILE *fp = fopen("c_140_output.txt", "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "%d: ", i);
                for (int j = 0; j < dim; j++)
                    fprintf(fp, "%f ", h_d_out_data[i * dim + j]);
                fprintf(fp, "\n");
            }
            fclose(fp);
        }
    }

    free(h_d_in_data);
    free(h_d_out_data);
    free(h_d_indptr);
    free(h_d_indices);
    return 0;
}