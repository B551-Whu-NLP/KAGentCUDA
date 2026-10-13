#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void DecodeBitstream(unsigned short* encoded, unsigned short* decoded, int size) {
    clock_t start = clock();
    
    for (int iter = 0; iter < 100000; iter++) {
        for (int i = 0; i < size; i++) {
            int bit_index = (i * 2);
            if (bit_index + 1 < size) {
                unsigned short curr_bit = encoded[bit_index];
                decoded[bit_index] = !encoded[bit_index] ^ encoded[bit_index + 1];
                decoded[bit_index + 1] = curr_bit ^ encoded[bit_index + 1];
            }
        }
    }
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC * 1000;
    printf("Time: %.2f(ms)\n", time_taken);
}

int main(int argc, char* argv[]) {
    const int size = 1000000;
    unsigned short* h_encoded = (unsigned short*)malloc(size * sizeof(unsigned short));
    unsigned short* h_decoded = (unsigned short*)calloc(size, sizeof(unsigned short));
    
    for (int i = 0; i < size; ++i) {
        h_encoded[i] = i % 65536;
    }
    
    int verbose = 0;
    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }
    
    DecodeBitstream(h_encoded, h_decoded, size);
    
    if (verbose) {
        char output_filename[256];
        strcpy(output_filename, "c_287_output.txt");
        FILE* file = fopen(output_filename, "w");
        if (file != NULL) {
            for (int i = 0; i < 1000; ++i) {
                fprintf(file, "%hu ", h_decoded[i]);
            }
            fclose(file);
        }
    }
    
    free(h_encoded);
    free(h_decoded);
    
    return 0;
}