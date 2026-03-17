/*
 * imbe_asr.c -- Standalone IMBE-to-text ASR inference engine.
 *
 * Reads P25 .tap files, decodes IMBE frames through libimbe, runs the
 * Conformer-CTC model via ONNX Runtime, and outputs transcribed text.
 *
 * No Python. No PyTorch. Single binary + model file + libimbe.so.
 *
 * Build:
 *   gcc -O2 -o imbe_asr imbe_asr.c \
 *       -I$HOME/onnxruntime/include \
 *       -L$HOME/onnxruntime/lib -lonnxruntime \
 *       -lm -ldl
 *
 * Usage:
 *   ./imbe_asr model.onnx stats.npz call.tap
 *   ./imbe_asr model.onnx stats.npz --watch /path/to/tap/dir
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <dlfcn.h>
#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

#include "onnxruntime_c_api.h"

/* ---- Constants ---- */
#define TAP_MAGIC       0x494D4245
#define TAP_HEADER_SIZE 8
#define TAP_FRAME_SIZE  32
#define IMBE_CODEWORDS  8
#define FRAME_SAMPLES   160
#define RAW_PARAM_DIM   170
#define SPEC_AMP_START  2
#define SPEC_AMP_END    58
#define MAX_FRAMES      4000
#define VOCAB_SIZE      39

/* CTC vocabulary: blank(0) + A-Z(1-26) + 0-9(27-36) + space(37) + apostrophe(38) */
static const char VOCAB[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 '";

/* ---- libimbe function pointers ---- */
typedef void* (*imbe_create_fn)(void);
typedef void  (*imbe_destroy_fn)(void*);
/* imbe_decode_params(dec, fv, snd_out, &f0, &num_harms, &num_bands, v_uv, sa) */
typedef int   (*imbe_decode_params_fn)(void*, short*, short*, float*, short*, short*, short*, short*);

static imbe_create_fn       lib_imbe_create;
static imbe_destroy_fn      lib_imbe_destroy;
static imbe_decode_params_fn lib_imbe_decode_params;

/* ---- ONNX Runtime globals ---- */
static const OrtApi *g_ort = NULL;

#define ORT_CHECK(expr) do { \
    OrtStatus *_s = (expr); \
    if (_s) { \
        fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(_s)); \
        g_ort->ReleaseStatus(_s); \
        exit(1); \
    } \
} while(0)

/* ---- Load libimbe ---- */
static int load_libimbe(const char *path) {
    void *lib = dlopen(path, RTLD_NOW);
    if (!lib) {
        fprintf(stderr, "Cannot load %s: %s\n", path, dlerror());
        return -1;
    }
    lib_imbe_create = (imbe_create_fn)dlsym(lib, "imbe_create");
    lib_imbe_destroy = (imbe_destroy_fn)dlsym(lib, "imbe_destroy");
    lib_imbe_decode_params = (imbe_decode_params_fn)dlsym(lib, "imbe_decode_params");

    if (!lib_imbe_create || !lib_imbe_destroy || !lib_imbe_decode_params) {
        fprintf(stderr, "Missing symbols in libimbe\n");
        return -1;
    }
    return 0;
}

/* ---- Read .tap file ---- */
typedef struct {
    short codewords[MAX_FRAMES][IMBE_CODEWORDS];
    int n_frames;
    int tgid;
} TapFile;

static int read_tap(const char *path, TapFile *tap) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;

    unsigned int header[2];
    if (fread(header, 4, 2, f) != 2 || header[0] != TAP_MAGIC) {
        fclose(f);
        return -1;
    }

    tap->n_frames = 0;
    tap->tgid = 0;

    unsigned char frame_buf[TAP_FRAME_SIZE];
    while (fread(frame_buf, 1, TAP_FRAME_SIZE, f) == TAP_FRAME_SIZE) {
        if (tap->n_frames >= MAX_FRAMES) break;

        /* Parse: uint32 seq, uint32 tgid, uint32 src_id, uint32 flags, uint16 u[8] */
        unsigned int tgid;
        memcpy(&tgid, frame_buf + 4, 4);
        if (tap->n_frames == 0) tap->tgid = tgid;

        short *u = tap->codewords[tap->n_frames];
        for (int j = 0; j < IMBE_CODEWORDS; j++) {
            unsigned short val;
            memcpy(&val, frame_buf + 16 + j * 2, 2);
            u[j] = (short)val;
        }
        tap->n_frames++;
    }

    fclose(f);
    return 0;
}

/* ---- Decode IMBE frames to raw_params + strip silence ---- */
#define MAX_HARMONICS 56

static int decode_frames(TapFile *tap, float *raw_params) {
    void *dec = lib_imbe_create();
    int out_frames = 0;
    short snd_out[FRAME_SAMPLES];

    for (int i = 0; i < tap->n_frames; i++) {
        float f0;
        short num_harms, num_bands;
        short v_uv[MAX_HARMONICS];
        short sa[MAX_HARMONICS];

        memset(v_uv, 0, sizeof(v_uv));
        memset(sa, 0, sizeof(sa));

        lib_imbe_decode_params(dec, tap->codewords[i], snd_out,
                               &f0, &num_harms, &num_bands, v_uv, sa);

        int L = num_harms < MAX_HARMONICS ? num_harms : MAX_HARMONICS;

        /* Build 170-dim raw_params vector */
        float params[RAW_PARAM_DIM];
        memset(params, 0, sizeof(params));
        params[0] = f0;
        params[1] = (float)L;
        for (int j = 0; j < L; j++) {
            params[2 + j] = sa[j] / 4.0f;      /* Q14.2 -> float */
            params[58 + j] = (float)v_uv[j];    /* voicing flags */
            params[114 + j] = 1.0f;             /* harmonic mask */
        }

        /* Check spectral energy for silence stripping */
        float energy = 0;
        for (int j = SPEC_AMP_START; j < SPEC_AMP_END; j++)
            energy += fabsf(params[j]);

        if (energy > 0) {
            memcpy(raw_params + out_frames * RAW_PARAM_DIM, params,
                   RAW_PARAM_DIM * sizeof(float));
            out_frames++;
        }
    }

    lib_imbe_destroy(dec);
    return out_frames;
}

/* ---- Load normalization stats (simple binary: 170 floats mean, 170 floats std) ---- */
typedef struct {
    float mean[RAW_PARAM_DIM];
    float std[RAW_PARAM_DIM];
} NormStats;

static int load_stats_bin(const char *path, NormStats *stats) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fread(stats->mean, sizeof(float), RAW_PARAM_DIM, f) != RAW_PARAM_DIM) { fclose(f); return -1; }
    if (fread(stats->std, sizeof(float), RAW_PARAM_DIM, f) != RAW_PARAM_DIM) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

/* ---- Normalize features ---- */
static void normalize(float *feats, int n_frames, const NormStats *stats) {
    for (int i = 0; i < n_frames; i++) {
        for (int j = 0; j < RAW_PARAM_DIM; j++) {
            feats[i * RAW_PARAM_DIM + j] =
                (feats[i * RAW_PARAM_DIM + j] - stats->mean[j]) / stats->std[j];
        }
    }
}

/* ---- CTC greedy decode ---- */
static void ctc_greedy_decode(const float *log_probs, int T, char *output, int max_len) {
    int prev = -1;
    int pos = 0;

    for (int t = 0; t < T && pos < max_len - 1; t++) {
        /* Find argmax */
        int best = 0;
        float best_val = log_probs[t * VOCAB_SIZE];
        for (int v = 1; v < VOCAB_SIZE; v++) {
            if (log_probs[t * VOCAB_SIZE + v] > best_val) {
                best_val = log_probs[t * VOCAB_SIZE + v];
                best = v;
            }
        }

        /* Skip blank (0) and repeats */
        if (best != 0 && best != prev) {
            output[pos++] = VOCAB[best - 1];
        }
        prev = best;
    }
    output[pos] = '\0';
}

/* ---- Batched inference ---- */
#define MAX_BATCH 32

typedef struct {
    char path[2048];
    float *feats;       /* normalized features */
    int n_frames;
    int tgid;
    char output[4096];
} BatchItem;

static int transcribe_batch(BatchItem *items, int batch_size, OrtSession *session) {
    if (batch_size <= 0) return 0;

    /* Find max sequence length for padding */
    int max_T = 0;
    for (int b = 0; b < batch_size; b++) {
        if (items[b].n_frames > max_T) max_T = items[b].n_frames;
    }

    /* Allocate padded feature tensor: (batch, max_T, 170) */
    float *padded = calloc(batch_size * max_T * RAW_PARAM_DIM, sizeof(float));
    int64_t *lengths = malloc(batch_size * sizeof(int64_t));

    for (int b = 0; b < batch_size; b++) {
        memcpy(padded + b * max_T * RAW_PARAM_DIM,
               items[b].feats,
               items[b].n_frames * RAW_PARAM_DIM * sizeof(float));
        lengths[b] = items[b].n_frames;
    }

    /* Create tensors */
    OrtMemoryInfo *mem_info;
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    int64_t feat_shape[] = {batch_size, max_T, RAW_PARAM_DIM};
    OrtValue *feat_tensor;
    ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, padded, batch_size * max_T * RAW_PARAM_DIM * sizeof(float),
        feat_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &feat_tensor));

    int64_t len_shape[] = {batch_size};
    OrtValue *len_tensor;
    ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, lengths, batch_size * sizeof(int64_t),
        len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor));

    /* Run batched inference */
    const char *input_names[] = {"features", "lengths"};
    const char *output_names[] = {"log_probs", "out_lengths"};
    OrtValue *inputs[] = {feat_tensor, len_tensor};
    OrtValue *outputs[] = {NULL, NULL};

    ORT_CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue*const*)inputs,
                          2, output_names, 2, outputs));

    /* Get outputs */
    float *log_probs;
    ORT_CHECK(g_ort->GetTensorMutableData(outputs[0], (void**)&log_probs));

    int64_t *out_lengths;
    ORT_CHECK(g_ort->GetTensorMutableData(outputs[1], (void**)&out_lengths));

    /* Get output shape to compute stride */
    OrtTensorTypeAndShapeInfo *shape_info;
    ORT_CHECK(g_ort->GetTensorTypeAndShape(outputs[0], &shape_info));
    int64_t out_shape[3];
    size_t dim_count = 3;
    ORT_CHECK(g_ort->GetDimensions(shape_info, out_shape, dim_count));
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    int out_T = (int)out_shape[1];  /* padded output time dim */

    /* CTC decode each item in the batch */
    for (int b = 0; b < batch_size; b++) {
        int T = (int)out_lengths[b];
        ctc_greedy_decode(log_probs + b * out_T * VOCAB_SIZE, T,
                          items[b].output, sizeof(items[b].output));
    }

    /* Cleanup */
    g_ort->ReleaseValue(outputs[0]);
    g_ort->ReleaseValue(outputs[1]);
    g_ort->ReleaseValue(feat_tensor);
    g_ort->ReleaseValue(len_tensor);
    g_ort->ReleaseMemoryInfo(mem_info);
    free(padded);
    free(lengths);

    return batch_size;
}

/* ---- Prepare a single tap file for batching ---- */
static int prepare_item(const char *tap_path, const NormStats *stats, BatchItem *item) {
    strncpy(item->path, tap_path, sizeof(item->path) - 1);
    item->output[0] = '\0';

    TapFile tap;
    if (read_tap(tap_path, &tap) < 0 || tap.n_frames < 15)
        return -1;

    item->tgid = tap.tgid;

    float *raw_params = malloc(tap.n_frames * RAW_PARAM_DIM * sizeof(float));
    item->n_frames = decode_frames(&tap, raw_params);

    if (item->n_frames < 10) {
        free(raw_params);
        return -1;
    }

    normalize(raw_params, item->n_frames, stats);
    item->feats = raw_params;
    return 0;
}

static void free_item(BatchItem *item) {
    if (item->feats) { free(item->feats); item->feats = NULL; }
}

/* ---- Single-file inference (convenience wrapper) ---- */
static int transcribe_tap(const char *tap_path, OrtSession *session,
                          const NormStats *stats, char *output, int max_len) {
    TapFile tap;
    if (read_tap(tap_path, &tap) < 0 || tap.n_frames < 15) {
        output[0] = '\0';
        return -1;
    }

    /* Decode IMBE frames to raw_params with silence stripping */
    float *raw_params = malloc(tap.n_frames * RAW_PARAM_DIM * sizeof(float));
    int n_frames = decode_frames(&tap, raw_params);

    if (n_frames < 10) {
        free(raw_params);
        output[0] = '\0';
        return -1;
    }

    /* Normalize */
    normalize(raw_params, n_frames, stats);

    /* Create ONNX Runtime tensors */
    OrtMemoryInfo *mem_info;
    ORT_CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info));

    int64_t feat_shape[] = {1, n_frames, RAW_PARAM_DIM};
    OrtValue *feat_tensor;
    ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, raw_params, n_frames * RAW_PARAM_DIM * sizeof(float),
        feat_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &feat_tensor));

    int64_t len_val = n_frames;
    int64_t len_shape[] = {1};
    OrtValue *len_tensor;
    ORT_CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem_info, &len_val, sizeof(int64_t),
        len_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &len_tensor));

    /* Run inference */
    const char *input_names[] = {"features", "lengths"};
    const char *output_names[] = {"log_probs", "out_lengths"};
    OrtValue *inputs[] = {feat_tensor, len_tensor};
    OrtValue *outputs[] = {NULL, NULL};

    ORT_CHECK(g_ort->Run(session, NULL, input_names, (const OrtValue*const*)inputs,
                          2, output_names, 2, outputs));

    /* Get output log_probs */
    float *log_probs;
    ORT_CHECK(g_ort->GetTensorMutableData(outputs[0], (void**)&log_probs));

    int64_t *out_lengths;
    ORT_CHECK(g_ort->GetTensorMutableData(outputs[1], (void**)&out_lengths));

    int T = (int)out_lengths[0];

    /* CTC greedy decode */
    ctc_greedy_decode(log_probs, T, output, max_len);

    /* Cleanup */
    g_ort->ReleaseValue(outputs[0]);
    g_ort->ReleaseValue(outputs[1]);
    g_ort->ReleaseValue(feat_tensor);
    g_ort->ReleaseValue(len_tensor);
    g_ort->ReleaseMemoryInfo(mem_info);
    free(raw_params);

    return tap.tgid;
}

/* ---- Main ---- */
int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model.onnx> <stats.bin> <file.tap | --watch dir>\n", argv[0]);
        return 1;
    }

    const char *model_path = argv[1];
    const char *stats_path = argv[2];

    /* Find and load libimbe */
    const char *imbe_paths[] = {
        "./vocoder/libimbe.so",
        "../vocoder/libimbe.so",
        "/mnt/disk/p25_train/vocoder/libimbe.so",
        NULL
    };
    int found = 0;
    for (int i = 0; imbe_paths[i]; i++) {
        if (load_libimbe(imbe_paths[i]) == 0) { found = 1; break; }
    }
    if (!found) {
        /* Try from LIBIMBE_PATH env var */
        const char *env = getenv("LIBIMBE_PATH");
        if (!env || load_libimbe(env) < 0) {
            fprintf(stderr, "Cannot find libimbe.so. Set LIBIMBE_PATH.\n");
            return 1;
        }
    }

    /* Load normalization stats */
    NormStats stats;
    if (load_stats_bin(stats_path, &stats) < 0) {
        fprintf(stderr, "Cannot load stats: %s\n", stats_path);
        return 1;
    }

    /* Initialize ONNX Runtime */
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

    OrtEnv *env;
    ORT_CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "imbe_asr", &env));

    OrtSessionOptions *opts;
    ORT_CHECK(g_ort->CreateSessionOptions(&opts));
    ORT_CHECK(g_ort->SetIntraOpNumThreads(opts, 4));
    ORT_CHECK(g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL));

    OrtSession *session;
    ORT_CHECK(g_ort->CreateSession(env, model_path, opts, &session));

    fprintf(stderr, "Model loaded: %s\n", model_path);

    if (strcmp(argv[3], "--watch") == 0) {
        if (argc < 5) {
            fprintf(stderr, "Usage: %s model stats --watch <dir> [--batch N]\n", argv[0]);
            return 1;
        }
        const char *watch_dir = argv[4];
        int max_batch = 8;
        if (argc >= 7 && strcmp(argv[5], "--batch") == 0)
            max_batch = atoi(argv[6]);
        if (max_batch > MAX_BATCH) max_batch = MAX_BATCH;

        fprintf(stderr, "Watching %s (batch=%d)...\n", watch_dir, max_batch);

        /* Batched polling watcher: collect files, process in batches */
        while (1) {
            DIR *dir = opendir(watch_dir);
            if (!dir) { sleep(1); continue; }

            /* Collect up to max_batch files */
            BatchItem batch[MAX_BATCH];
            int batch_count = 0;

            struct dirent *entry;
            while ((entry = readdir(dir)) && batch_count < max_batch) {
                int len = strlen(entry->d_name);
                if (len < 5 || strcmp(entry->d_name + len - 4, ".tap") != 0)
                    continue;

                char path[2048];
                snprintf(path, sizeof(path), "%s/%s", watch_dir, entry->d_name);

                if (prepare_item(path, &stats, &batch[batch_count]) == 0) {
                    batch_count++;
                }
            }
            closedir(dir);

            if (batch_count > 0) {
                struct timespec ts;
                clock_gettime(CLOCK_MONOTONIC, &ts);
                double t0 = ts.tv_sec + ts.tv_nsec / 1e9;

                transcribe_batch(batch, batch_count, session);

                clock_gettime(CLOCK_MONOTONIC, &ts);
                double dt = (ts.tv_sec + ts.tv_nsec / 1e9) - t0;

                for (int b = 0; b < batch_count; b++) {
                    if (batch[b].output[0]) {
                        const char *fname = strrchr(batch[b].path, '/');
                        fname = fname ? fname + 1 : batch[b].path;
                        printf("[TG=%d] %s\n  >> %s\n",
                               batch[b].tgid, fname, batch[b].output);
                    }
                    free_item(&batch[b]);
                }
                printf("--- batch=%d, %.0fms (%.0fms/file) ---\n\n",
                       batch_count, dt * 1000, dt * 1000 / batch_count);
                fflush(stdout);
            }

            usleep(200000);  /* 200ms poll */
        }
    } else {
        /* Single file mode */
        char output[4096];
        struct timespec ts;

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double t0 = ts.tv_sec + ts.tv_nsec / 1e9;

        int tgid = transcribe_tap(argv[3], session, &stats, output, sizeof(output));

        clock_gettime(CLOCK_MONOTONIC, &ts);
        double dt = (ts.tv_sec + ts.tv_nsec / 1e9) - t0;

        if (tgid >= 0) {
            printf("[TG=%d] (%.0fms)\n%s\n", tgid, dt * 1000, output);
        } else {
            fprintf(stderr, "Failed to transcribe: %s\n", argv[3]);
            return 1;
        }
    }

    g_ort->ReleaseSession(session);
    g_ort->ReleaseSessionOptions(opts);
    g_ort->ReleaseEnv(env);

    return 0;
}
