#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define MAX_ARGS 10
#define REC_LENGTH 49
#define REC_WINDOW 10
#define LATITUDE_POS 28
#define OPEN 10000

struct neighbor {
    char entry[REC_LENGTH];
    double dist;
};

int main(int argc, char* argv[]) {
    FILE   *flist, *fp;
    int    i=0, j=0, k=0, rec_count=0, done=0;
    char   sandbox[REC_LENGTH * REC_WINDOW], *rec_iter, dbname[64];
    struct neighbor *neighbors = NULL;
    float  target_lat, target_long;

    int verbose = 0;
    for (i = 1; i < argc; i++)
        if (strcmp(argv[i], "--verbose") == 0) verbose = 1;

    if (argc < 5) {
        fprintf(stderr, "Usage: %s <filelist> <num> <target_lat> <target_long> [--verbose]\n", argv[0]);
        exit(1);
    }

    flist = fopen(argv[1], "r");
    if (!flist) {
        fprintf(stderr, "ERROR: error opening filelist '%s'\n", argv[1]);
        exit(1);
    }

    k          = atoi(argv[2]);
    target_lat  = atof(argv[3]);
    target_long = atof(argv[4]);

    neighbors = malloc(k * sizeof(struct neighbor));
    if (neighbors == NULL) {
        fprintf(stderr, "ERROR: no room for neighbors\n");
        exit(1);
    }

    for (j = 0; j < k; j++)
        neighbors[j].dist = OPEN;

    if (fscanf(flist, "%s\n", dbname) != 1) {
        fprintf(stderr, "ERROR: error reading filelist\n");
        exit(1);
    }

    fp = fopen(dbname, "r");
    if (!fp) {
        fprintf(stderr, "ERROR: error opening db '%s'\n", dbname);
        exit(1);
    }

    float *z = (float *)malloc(REC_WINDOW * sizeof(float));

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (!done) {
        rec_count = fread(sandbox, REC_LENGTH, REC_WINDOW, fp);
        if (rec_count != REC_WINDOW) {
            if (!ferror(flist)) {
                fclose(fp);
                if (feof(flist)) {
                    done = 1;
                } else {
                    if (fscanf(flist, "%s\n", dbname) != 1) {
                        fprintf(stderr, "ERROR: error reading filelist\n");
                        exit(1);
                    }
                    fp = fopen(dbname, "r");
                    if (!fp) {
                        fprintf(stderr, "ERROR: error opening db '%s'\n", dbname);
                        exit(1);
                    }
                }
            } else {
                perror("ERROR");
                exit(1);
            }
        }

        #pragma omp parallel for shared(z, target_lat, target_long) private(i, rec_iter)
        for (i = 0; i < rec_count; i++) {
            rec_iter = sandbox + (i * REC_LENGTH + LATITUDE_POS - 1);
            float tmp_lat  = atof(rec_iter);
            float tmp_long = atof(rec_iter + 5);
            z[i] = sqrt(((tmp_lat  - target_lat)  * (tmp_lat  - target_lat)) +
                        ((tmp_long - target_long) * (tmp_long - target_long)));
        }

        for (i = 0; i < rec_count; i++) {
            float max_dist = -1;
            int   max_idx  = 0;
            for (j = 0; j < k; j++) {
                if (neighbors[j].dist > max_dist) {
                    max_dist = neighbors[j].dist;
                    max_idx  = j;
                }
            }
            if (z[i] < neighbors[max_idx].dist) {
                sandbox[(i + 1) * REC_LENGTH - 1] = '\0';
                strcpy(neighbors[max_idx].entry, sandbox + i * REC_LENGTH);
                neighbors[max_idx].dist = z[i];
            }
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double elapsed = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    printf("Time: %.6f s\n", elapsed);

    fclose(flist);

    if (verbose) {
        char result_path[512];
        const char *out_dir = getenv("RESULT_DIR");
        if (out_dir)
            snprintf(result_path, sizeof(result_path), "%s/nn_output.txt", out_dir);
        else
            snprintf(result_path, sizeof(result_path), "nn_output.txt");
        FILE *fout = fopen(result_path, "w");
        if (fout == NULL) {
            fprintf(stderr, "ERROR: could not open nn_output.txt\n");
        } else {
            // fprintf(fout, "The %d nearest neighbors are:\n", k);
            for (j = 0; j < k; j++) {
                if (neighbors[j].dist != OPEN)
                    fprintf(fout, "%s %f\n", neighbors[j].entry, neighbors[j].dist);
            }
            fclose(fout);
        }
    }

    free(neighbors);
    free(z);
    return 0;
}