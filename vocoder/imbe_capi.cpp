/*
 * C API wrapper around imbe_vocoder for use via Python ctypes.
 *
 * Exposes:
 *   imbe_create()           -> opaque handle
 *   imbe_destroy(handle)
 *   imbe_decode(handle, frame_vector[8], snd[160])
 *   imbe_clear(handle)      -> reset vocoder state
 *   imbe_encode(handle, frame_vector[8], snd[160])
 *   imbe_decode_params(handle, frame_vector[8], snd[160],
 *                      &fund_freq, &num_harms, v_uv[56], sa[56])
 *                           -> decode + extract parameters
 */

#include "imbe_vocoder.h"
#include <cstdint>
#include <cmath>

extern "C" {

void *imbe_create(void) {
    return new imbe_vocoder();
}

void imbe_destroy(void *handle) {
    delete static_cast<imbe_vocoder *>(handle);
}

void imbe_decode(void *handle, int16_t *frame_vector, int16_t *snd) {
    static_cast<imbe_vocoder *>(handle)->imbe_decode(frame_vector, snd);
}

void imbe_encode(void *handle, int16_t *frame_vector, int16_t *snd) {
    static_cast<imbe_vocoder *>(handle)->imbe_encode(frame_vector, snd);
}

void imbe_clear(void *handle) {
    static_cast<imbe_vocoder *>(handle)->clear();
}

/*
 * Decode a frame and extract the intermediate IMBE parameters.
 *
 * Runs the full decode+synthesis chain (so vocoder state stays consistent),
 * then copies out the decoded parameters from IMBE_PARAM.
 *
 * Outputs:
 *   fund_freq_hz  - fundamental frequency in Hz (at 8kHz sample rate)
 *   num_harms     - number of harmonics L (9-56)
 *   num_bands     - number of voicing bands (3-12)
 *   v_uv[56]      - voiced(1)/unvoiced(0) per harmonic, only [0..L-1] valid
 *   sa[56]        - spectral amplitudes in Q14.2 fixed-point, [0..L-1] valid
 *
 * Returns: b_vec[0] (pitch index, 0-207). Returns -1 if frame was invalid
 *          (b_vec[0] out of range → frame repeat occurred).
 */
int imbe_decode_params(void *handle, int16_t *frame_vector, int16_t *snd,
                       float *fund_freq_hz, int16_t *num_harms, int16_t *num_bands,
                       int16_t *v_uv, int16_t *sa) {
    imbe_vocoder *voc = static_cast<imbe_vocoder *>(handle);

    // Run full decode + synthesis
    voc->imbe_decode(frame_vector, snd);

    // Read decoded parameters
    const IMBE_PARAM *p = voc->param();

    // Convert fund_freq from Q1.31 to Hz
    // fund_freq (Q1.31) = 2*f0/fs, so f0 = fund_freq * fs/2 = fund_freq * 4000
    double ff = (double)p->fund_freq / (double)(1u << 31);
    *fund_freq_hz = (float)(ff * 4000.0);

    *num_harms = p->num_harms;
    *num_bands = p->num_bands;

    for (int i = 0; i < p->num_harms && i < 56; i++) {
        v_uv[i] = p->v_uv_dsn[i];
        sa[i] = p->sa[i];
    }
    // Zero the rest
    for (int i = p->num_harms; i < 56; i++) {
        v_uv[i] = 0;
        sa[i] = 0;
    }

    // Return b_vec[0] or -1 if invalid
    int b0 = p->b_vec[0];
    if (b0 < 0 || b0 > 207)
        return -1;
    return b0;
}

} // extern "C"
