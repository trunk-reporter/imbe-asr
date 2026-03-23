/*
 * bench_c.c -- Minimal C benchmark for IMBE-ASR ONNX inference.
 *
 * Tests pure inference speed without libimbe dependency.
 * Uses random features (same as Python bench scripts).
 *
 * Build (on Pi):
 *   gcc -O2 -o bench_c bench_c.c \
 *       -I$HOME/onnxruntime/include \
 *       -L$HOME/onnxruntime/lib -lonnxruntime \
 *       -lm
 *
 * Run:
 *   LD_LIBRARY_PATH=$HOME/onnxruntime/lib ./bench_c model.onnx
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "onnxruntime_c_api.h"

#define RAW_PARAM_DIM 170
#define VOCAB_SIZE 39

static const OrtApi *g_ort = NULL;

#define ORT_CHECK(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(_s)); \
        g_ort->ReleaseStatus(_s); \
        exit(1); \
    } \
} while(0)

/* CTC vocabulary: blank(0) + A-Z(1-26) + 0-9(27-36) + space(37) + apostrophe(38) */
static const char VOCAB[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '";

static void ctc_greedy_decode(const float *log_probs, int T, char *output, int max_len) {
    int pos = 0;
    int prev = -1;
    for (int t = 0; t < T && pos < max_len - 1; t++) {
        /* Find argmax */
        int best = 0;
        float best_val = log_probs[t * VOCAB_SIZE];
        for (int v = 1; v < VOCAB_SIZE; v++) {
            float val = log_probs[t * VOCAB_SIZE + v];
            if (val > best_val) { best_val = val; best = v; }
        }
        if (best > 0 && best != prev) {
            output[pos++] = VOCAB[best - 1];
        }
        prev = best;
    }
    output[pos] = '\0';
}

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static double bench_inference(OrtSession *session, int n_frames, int n_runs, int warmup) {
    float *features = (float *)malloc(n_frames * RAW_PARAM_DIM * sizeof(float));
    /* Fill with random-ish data */
    for (int i = 0; i < n_frames * RAW_PARAM_DIM; i++)
        features[i] = (float)(i % 1000) / 1000.0f - 0.5f;

    OrtMemoryInfo *mem_info;
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    int64_t feat_shape[] = {1, n_frames, RAW_PARAM_DIM};
    int64_t len_val = n_frames;
    int64_t len_shape[] = {1};

    const char *input_names[] = {"features", "lengths"};
    const char *output_names[] = {"log_probs", "out_lengths"};

    /* Warmup */
    for (int r = 0; r < warmup; r++) {
        OrtValue *feat_tensor, *len_tensor;
        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, features, n_frames * RAW_PARAM_DIM * sizeof(float),
            feat_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &feat_tensor));
        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, &len_val, sizeof(int64_t),
            len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor));

        OrtValue *inputs[] = {feat_tensor, len_tensor};
        OrtValue *outputs[] = {NULL, NULL};
        ORT_CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)inputs,
                              2, output_names, 2, outputs));

        g_ort->ReleaseValue(outputs[0]);
        g_ort->ReleaseValue(outputs[1]);
        g_ort->ReleaseValue(feat_tensor);
        g_ort->ReleaseValue(len_tensor);
    }

    /* Timed runs */
    double total_ms = 0;
    double min_ms = 1e9, max_ms = 0;
    char decode_buf[4096] = {0};

    for (int r = 0; r < n_runs; r++) {
        OrtValue *feat_tensor, *len_tensor;
        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, features, n_frames * RAW_PARAM_DIM * sizeof(float),
            feat_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &feat_tensor));
        ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
            mem_info, &len_val, sizeof(int64_t),
            len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor));

        OrtValue *inputs[] = {feat_tensor, len_tensor};
        OrtValue *outputs[] = {NULL, NULL};

        double t0 = get_time_ms();
        ORT_CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue *const *)inputs,
                              2, output_names, 2, outputs));
        double dt = get_time_ms() - t0;

        total_ms += dt;
        if (dt < min_ms) min_ms = dt;
        if (dt > max_ms) max_ms = dt;

        /* Decode last run */
        if (r == n_runs - 1) {
            float *log_probs;
            int64_t *out_lengths;
            ORT_CHECK(g_ort->GetTensorMutableData(outputs[0], (void **)&log_probs));
            ORT_CHECK(g_ort->GetTensorMutableData(outputs[1], (void **)&out_lengths));
            ctc_greedy_decode(log_probs, (int)out_lengths[0], decode_buf, sizeof(decode_buf));
        }

        g_ort->ReleaseValue(outputs[0]);
        g_ort->ReleaseValue(outputs[1]);
        g_ort->ReleaseValue(feat_tensor);
        g_ort->ReleaseValue(len_tensor);
    }

    g_ort->ReleaseMemoryInfo(mem_info);
    free(features);

    double mean_ms = total_ms / n_runs;
    double duration_s = n_frames / 50.0;
    printf("  %2ds (%4d fr)  %7.1fms  %7.1fms  %7.1fms  %7.4fx\n",
           (int)duration_s, n_frames, mean_ms, min_ms, max_ms,
           mean_ms / 1000.0 / duration_s);

    return mean_ms;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.onnx> [runs]\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    int n_runs = argc >= 3 ? atoi(argv[2]) : 5;

    /* Init ORT */
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env;
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "bench", &env));

    OrtSessionOptions *opts;
    ORT_CHECK(g_ort->CreateSessionOptions(&opts));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(opts, 4));
    ORT_CHECK(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    double t0 = get_time_ms();
    OrtSession *session;
    ORT_CHECK(g_ort->CreateSession(env, model_path, opts, &session));
    double load_ms = get_time_ms() - t0;

    printf("============================================================\n");
    printf("IMBE-ASR C Inference Benchmark\n");
    printf("============================================================\n");
    printf("Model: %s\n", model_path);
    printf("Load time: %.0fms\n", load_ms);
    printf("Runs per duration: %d\n\n", n_runs);

    printf("  Duration        Mean      Min       Max       RTF\n");
    printf("  --------------------------------------------------\n");

    int durations[] = {1, 2, 5, 10, 30};
    for (int i = 0; i < 5; i++) {
        bench_inference(session, durations[i] * 50, n_runs, 2);
    }

    printf("\nRTF < 1.0 = faster than real-time\n");

    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);

    return 0;
}
