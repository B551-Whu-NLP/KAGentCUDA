#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

void ConvertToBits(int* bit_decisions, unsigned short* bit_stream, int dec_size) {
    for (int dec_index = 0; dec_index < dec_size; dec_index++) {
        int bit_index = dec_index * 2;
        int curr_decision = bit_decisions[dec_index];
        bit_stream[bit_index] = ((curr_decision & 2) >> 1);
        bit_stream[bit_index + 1] = (curr_decision & 1);
    }
}

int main(int argc, char *argv[]) {
    int dec_size = 100000000; // Increased by 10x
    int* h_bit_decisions = (int*)malloc(dec_size * sizeof(int));
    unsigned short* h_bit_stream = (unsigned short*)malloc(dec_size * 2 * sizeof(unsigned short));
    int verbose = 0;

    if (argc > 1 && strcmp(argv[1], "--verbose") == 0) {
        verbose = 1;
    }

    for (int i = 0; i < dec_size; i++) {
        h_bit_decisions[i] = i + 1;
    }

    clock_t start_time = clock();

    ConvertToBits(h_bit_decisions, h_bit_stream, dec_size);

    clock_t end_time = clock();
    double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Time: %.2f\n", time_taken);

    if (verbose) {
        char output_file[256];
        strcpy(output_file, "c_89_output.txt");
        FILE *fp = fopen(output_file, "w");
        if (fp != NULL) {
            for (int i = 0; i < 1000; i++) {
                fprintf(fp, "Decision: %d -> Bits: %u %u\n", h_bit_decisions[i], h_bit_stream[i * 2], h_bit_stream[i * 2 + 1]);
            }
            fclose(fp);
        }
    }

    free(h_bit_decisions);
    free(h_bit_stream);

    return 0;
}