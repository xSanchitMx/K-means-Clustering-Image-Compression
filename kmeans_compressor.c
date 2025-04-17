#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "stb_image.h"
#include "stb_image_write.h"

#define INPUT_FILE "input.png"
#define OUTPUT_TEMPLATE "output_%d.png"

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <k>\n", argv[0]);
        return 1;
    }

    int k = atoi(argv[1]);
    int max_threads = omp_get_num_procs();

    int width, height, channels;
    unsigned char *image = stbi_load(INPUT_FILE, &width, &height, &channels, 3);
    if (!image)
    {
        fprintf(stderr, "Failed to load %s\n", INPUT_FILE);
        return 1;
    }

    int img_size = width * height;
    FILE *log = fopen("performance_log.csv", "a");
    if (log)
        fprintf(log, "k,width,height,num_threads,time_seconds\n");

    for (int num_threads = 1; num_threads <= max_threads; num_threads++)
    {
        omp_set_num_threads(num_threads);

        unsigned char *output = (unsigned char *)malloc(img_size * 3);
        float *centroids = (float *)malloc(k * 3 * sizeof(float));
        int *labels = (int *)malloc(img_size * sizeof(int));

        // Init centroids randomly
        for (int i = 0; i < k; i++)
        {
            int idx = rand() % img_size;
            centroids[i * 3 + 0] = image[idx * 3 + 0];
            centroids[i * 3 + 1] = image[idx * 3 + 1];
            centroids[i * 3 + 2] = image[idx * 3 + 2];
        }

        double start_time = omp_get_wtime();

        for (int iter = 0; iter < 10; iter++)
        {
#pragma omp parallel for
            for (int i = 0; i < img_size; i++)
            {
                float min_dist = 1e9;
                int best = 0;
                for (int j = 0; j < k; j++)
                {
                    float dr = image[i * 3 + 0] - centroids[j * 3 + 0];
                    float dg = image[i * 3 + 1] - centroids[j * 3 + 1];
                    float db = image[i * 3 + 2] - centroids[j * 3 + 2];
                    float dist = dr * dr + dg * dg + db * db;
                    if (dist < min_dist)
                    {
                        min_dist = dist;
                        best = j;
                    }
                }
                labels[i] = best;
            }

            float *new_centroids = (float *)calloc(k * 3, sizeof(float));
            int *counts = (int *)calloc(k, sizeof(int));

#pragma omp parallel for
            for (int i = 0; i < img_size; i++)
            {
                int j = labels[i];
#pragma omp atomic
                new_centroids[j * 3 + 0] += image[i * 3 + 0];
#pragma omp atomic
                new_centroids[j * 3 + 1] += image[i * 3 + 1];
#pragma omp atomic
                new_centroids[j * 3 + 2] += image[i * 3 + 2];
#pragma omp atomic
                counts[j]++;
            }

            for (int j = 0; j < k; j++)
            {
                if (counts[j] > 0)
                {
                    centroids[j * 3 + 0] = new_centroids[j * 3 + 0] / counts[j];
                    centroids[j * 3 + 1] = new_centroids[j * 3 + 1] / counts[j];
                    centroids[j * 3 + 2] = new_centroids[j * 3 + 2] / counts[j];
                }
            }

            free(new_centroids);
            free(counts);
        }

        double end_time = omp_get_wtime();
        double elapsed = end_time - start_time;

// Generate image
#pragma omp parallel for
        for (int i = 0; i < img_size; i++)
        {
            int j = labels[i];
            output[i * 3 + 0] = (unsigned char)centroids[j * 3 + 0];
            output[i * 3 + 1] = (unsigned char)centroids[j * 3 + 1];
            output[i * 3 + 2] = (unsigned char)centroids[j * 3 + 2];
        }

        char output_filename[64];
        snprintf(output_filename, sizeof(output_filename), OUTPUT_TEMPLATE, num_threads);
        stbi_write_png(output_filename, width, height, 3, output, width * 3);

        printf("[Threads: %2d] Time = %.4f sec - Saved: %s\n", num_threads, elapsed, output_filename);

        if (log)
        {
            fprintf(log, "%d,%d,%d,%d,%.6f\n", k, width, height, num_threads, elapsed);
        }

        free(output);
        free(centroids);
        free(labels);
    }

    if (log)
        fclose(log);
    stbi_image_free(image);
    return 0;
}

// gcc -fopenmp kmeans_compressor.c -o kcompress.exe -lm
// ./kcompress <k-cluster amount>