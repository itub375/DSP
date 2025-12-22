// dsp_deinterleave.c
// Port des Python-DSP-Codes (Interleaved Audio -> Segmente -> Deinterleave) nach C

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdbool.h>

//---------------------------------------
// Konstanten
//---------------------------------------
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Analyse-Parameter (wie im Python-Code)
#define FRAME_DURATION_S          0.002   // 2 ms
#define PERCENTILE_FEATURE        80.0
#define MIN_GAP_S                 0.01
#define TRUE_SEGMENT_DURATION_S   0.05    // nur für Referenzlinien im Plot
#define ENERGY_PERCENTILE         PERCENTILE_FEATURE
#define ENERGY_MIN_GAP_S          MIN_GAP_S
#define SHAPE_PERCENTILE          PERCENTILE_FEATURE
#define SHAPE_MIN_GAP_S           MIN_GAP_S
#define JUMP_PERCENTILE           98.0
#define JUMP_MIN_GAP_S            0.005
#define JOINT_MAX_DIFF_S          0.005
#define CLUSTER_THRESHOLD_HZ      500.0
#define MIN_SEGMENTS_PER_CLUSTER  3
#define USE_PRIO1_IN_BOUNDARIES   1       // 1 = Prio 1–3, 0 = nur 2–3

// Grenzen für Arrays (anpassen falls nötig)
#define MAX_FRAMES    50000
#define MAX_CHANGES   20000
#define MAX_SEGMENTS  10000
#define MAX_LABELS    26        // A–Z
#define FEAT_DIM      10        // energy, centroid, bw, rolloff, flatness, zcr, 4 Mel-Bänder

//---------------------------------------
// Strukturen
//---------------------------------------
typedef struct {
    int   segment_index;
    int   start_frame;
    int   end_frame;
    float start_time;
    float end_time;
    float mean_centroid_hz;
    int   cluster_id;
    char  label;               // 'A', 'B', 'C', ...
} Segment;

typedef struct {
    double t;
    int    method_id;  // 0: Centroid, 1: Energy, 2: Jump, 3: Shape
} TimeMethod;

//---------------------------------------
// Hilfsfunktionen: Vergleiche & Percentile
//---------------------------------------
int float_compare(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

int double_compare(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int int_compare(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return (ia > ib) - (ia < ib);
}

float percentile(const float *x, int n, double p) {
    if (n <= 0) return 0.0f;
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return 0.0f;
    memcpy(tmp, x, n * sizeof(float));
    qsort(tmp, n, sizeof(float), float_compare);
    double idx = (p / 100.0) * (n - 1);
    int i0 = (int)floor(idx);
    int i1 = (int)ceil(idx);
    float v0 = tmp[i0];
    float v1 = tmp[i1];
    float result = (i0 == i1) ? v0 : (v0 + (float)(idx - i0) * (v1 - v0));
    free(tmp);
    return result;
}

//---------------------------------------
// Peak-Picking aus Diffs (entspricht _peak_pick_from_diffs)
//---------------------------------------
int peak_pick_from_diffs(const float *diffs, int n_diffs,
                         double frame_duration,
                         double percentile_val,
                         double min_gap_s,
                         int *picked_indices, int max_peaks,
                         float *threshold_out)
{
    if (n_diffs <= 0) {
        *threshold_out = 0.0f;
        return 0;
    }

    float thr = percentile(diffs, n_diffs, percentile_val);
    *threshold_out = thr;

    int *cands = (int *)malloc(n_diffs * sizeof(int));
    int n_cands = 0;

    // Kandidaten über Threshold und lokale Maxima
    for (int i = 0; i < n_diffs; ++i) {
        if (diffs[i] >= thr) {
            float left  = (i > 0) ? diffs[i-1] : -FLT_MAX;
            float right = (i < n_diffs-1) ? diffs[i+1] : -FLT_MAX;
            if (diffs[i] >= left && diffs[i] >= right) {
                cands[n_cands++] = i;
            }
        }
    }

    int min_gap_frames = (int)ceil(min_gap_s / frame_duration);
    int n_peaks = 0;
    int last = -1000000000;

    for (int k = 0; k < n_cands; ++k) {
        int p = cands[k];
        if (p - last >= min_gap_frames) {
            if (n_peaks < max_peaks) {
                picked_indices[n_peaks++] = p;
                last = p;
            }
        }
    }

    free(cands);
    return n_peaks;
}

//---------------------------------------
// FFT-Stub: hier musst du FFTW / KissFFT o.Ä. einbauen
//---------------------------------------
void compute_rfft_mag_and_freqs(const float *frame, int L, float sr,
                                float *mag_out, float *freqs_out)
{
    int n_bins = L / 2 + 1;
    // TODO: Echte FFT implementieren!
    // Platzhalter: alle Magnituden 0, Frequenzachse linear
    for (int k = 0; k < n_bins; ++k) {
        mag_out[k] = 0.0f;
        freqs_out[k] = (float)k * (sr / (float)L);
    }
}

//---------------------------------------
// Spektral-Schwerpunkte pro Frame (compute_features)
//---------------------------------------
int compute_centroids(const float *y, int n_samples, int sr,
                      double frame_duration,
                      float *centroids_hz, int max_frames)
{
    int L = (int)(frame_duration * sr);
    if (L <= 0 || n_samples < L) return 0;
    int H = L;

    int frame_count = 0;
    int n_bins = L / 2 + 1;

    float *win   = (float *)malloc(L * sizeof(float));
    float *frame = (float *)malloc(L * sizeof(float));
    float *mag   = (float *)malloc(n_bins * sizeof(float));
    float *freqs = (float *)malloc(n_bins * sizeof(float));

    if (!win || !frame || !mag || !freqs) {
        free(win); free(frame); free(mag); free(freqs);
        return 0;
    }

    for (int i = 0; i < L; ++i) {
        win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(L - 1));
    }

    for (int start = 0; start + L <= n_samples; start += H) {
        if (frame_count >= max_frames) break;

        for (int i = 0; i < L; ++i) {
            frame[i] = y[start + i] * win[i];
        }

        compute_rfft_mag_and_freqs(frame, L, (float)sr, mag, freqs);

        double num = 0.0;
        double den = 0.0;
        for (int k = 0; k < n_bins; ++k) {
            num += freqs[k] * mag[k];
            den += mag[k];
        }
        float centroid = 0.0f;
        if (den > 1e-12) {
            centroid = (float)(num / den);
        }
        centroids_hz[frame_count++] = centroid;
    }

    free(win);
    free(frame);
    free(mag);
    free(freqs);
    return frame_count;
}

//---------------------------------------
// Mehrmerkmals-Features pro Frame (compute_frame_features)
// energy, centroid, bw, rolloff, flatness, zcr, 4 Mel-Proxies
//---------------------------------------
int compute_frame_features(const float *y, int n_samples, int sr,
                           double frame_duration,
                           float *frame_feats, int max_frames, int feat_dim)
{
    if (feat_dim < FEAT_DIM) {
        fprintf(stderr, "feat_dim (%d) < FEAT_DIM (%d)\n", feat_dim, FEAT_DIM);
        return 0;
    }

    int L = (int)(frame_duration * sr);
    if (L <= 0 || n_samples < L) return 0;
    int H = L;
    int n_bins = L / 2 + 1;

    float *win   = (float *)malloc(L * sizeof(float));
    float *frame = (float *)malloc(L * sizeof(float));
    float *mag   = (float *)malloc(n_bins * sizeof(float));
    float *freqs = (float *)malloc(n_bins * sizeof(float));

    if (!win || !frame || !mag || !freqs) {
        free(win); free(frame); free(mag); free(freqs);
        return 0;
    }

    for (int i = 0; i < L; ++i) {
        win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(L - 1));
    }

    int frame_idx = 0;
    while (frame_idx < max_frames && (frame_idx * H + L) <= n_samples) {
        int start = frame_idx * H;

        for (int i = 0; i < L; ++i) {
            frame[i] = y[start + i] * win[i];
        }

        // Grundgrößen im Zeitbereich
        double energy = 0.0;
        for (int i = 0; i < L; ++i) {
            double v = frame[i];
            energy += v * v;
        }
        energy /= (double)L;

        // Spektrum
        compute_rfft_mag_and_freqs(frame, L, (float)sr, mag, freqs);

        // Zahlen stabilisieren
        for (int k = 0; k < n_bins; ++k) {
            if (mag[k] < 1e-12f) mag[k] = 1e-12f;
        }

        // Centroid
        double num = 0.0, den = 0.0;
        for (int k = 0; k < n_bins; ++k) {
            num += freqs[k] * mag[k];
            den += mag[k];
        }
        double centroid = (den > 1e-12) ? (num / den) : 0.0;

        // Bandbreite
        double bw_num = 0.0;
        for (int k = 0; k < n_bins; ++k) {
            double diff = freqs[k] - centroid;
            bw_num += diff * diff * mag[k];
        }
        double bandwidth = (den > 1e-12) ? sqrt(bw_num / den) : 0.0;

        // Rolloff 85%
        double cumsum = 0.0;
        double target = 0.85 * den;
        float rolloff = freqs[n_bins-1];
        for (int k = 0; k < n_bins; ++k) {
            cumsum += mag[k];
            if (cumsum >= target) {
                rolloff = freqs[k];
                break;
            }
        }

        // Flatness
        double log_sum = 0.0;
        for (int k = 0; k < n_bins; ++k) {
            log_sum += log(mag[k]);
        }
        double geo_mean = exp(log_sum / (double)n_bins);
        double arith_mean = den / (double)n_bins;
        double flatness = (arith_mean > 1e-12) ? (geo_mean / arith_mean) : 0.0;

        // Zero-Crossing-Rate
        int zero_crossings = 0;
        int prev_sign = (frame[0] >= 0.0f) ? 1 : -1;
        for (int i = 1; i < L; ++i) {
            int sign = (frame[i] >= 0.0f) ? 1 : -1;
            if (sign != prev_sign) zero_crossings++;
            prev_sign = sign;
        }
        double zcr = (double)zero_crossings / (double)(L - 1);

        // 4 grobe Mel-Band-Proxies (hier einfach 4 Frequenzbänder der Magnitude)
        float melbands[4] = {0,0,0,0};
        int n_bands = 4;
        for (int b = 0; b < n_bands; ++b) {
            int start_bin = (int)((b    ) * (n_bins - 1) / (double)n_bands);
            int end_bin   = (int)((b + 1) * (n_bins - 1) / (double)n_bands);
            if (end_bin < start_bin) end_bin = start_bin;
            double s = 0.0;
            int cnt = 0;
            for (int k = start_bin; k <= end_bin && k < n_bins; ++k) {
                s += mag[k];
                cnt++;
            }
            melbands[b] = (cnt > 0) ? (float)(s / (double)cnt) : 0.0f;
        }

        float *feat = &frame_feats[frame_idx * feat_dim];
        feat[0] = (float)energy;
        feat[1] = (float)centroid;
        feat[2] = (float)bandwidth;
        feat[3] = (float)rolloff;
        feat[4] = (float)flatness;
        feat[5] = (float)zcr;
        feat[6] = melbands[0];
        feat[7] = melbands[1];
        feat[8] = melbands[2];
        feat[9] = melbands[3];

        frame_idx++;
    }

    int n_frames = frame_idx;
    if (n_frames <= 0) {
        free(win); free(frame); free(mag); free(freqs);
        return 0;
    }

    // Spaltenweise z-Normierung
    for (int d = 0; d < FEAT_DIM; ++d) {
        double mean = 0.0, var = 0.0;
        for (int i = 0; i < n_frames; ++i) {
            mean += frame_feats[i * feat_dim + d];
        }
        mean /= (double)n_frames;
        for (int i = 0; i < n_frames; ++i) {
            double val = frame_feats[i * feat_dim + d] - mean;
            var += val * val;
        }
        var /= (double)n_frames;
        double std = sqrt(var) + 1e-9;
        for (int i = 0; i < n_frames; ++i) {
            frame_feats[i * feat_dim + d] =
                (float)((frame_feats[i * feat_dim + d] - mean) / std);
        }
    }

    free(win);
    free(frame);
    free(mag);
    free(freqs);
    return n_frames;
}

//---------------------------------------
// Centroid-Detektor (detect_change_points_centroid_only)
//---------------------------------------
int detect_change_points_centroid_only(
    const float *centroids_hz, int n_frames,
    double frame_duration,
    double percentile_val,
    double min_gap_s,
    int *change_frame_indices,   // OUT
    int max_changes,
    float *centroid_diffs_out,   // optional OUT, Länge >= n_frames-1
    float *threshold_out)
{
    if (n_frames <= 1) {
        *threshold_out = 0.0f;
        return 0;
    }

    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n_frames; ++i) mean += centroids_hz[i];
    mean /= (double)n_frames;
    for (int i = 0; i < n_frames; ++i) {
        double d = centroids_hz[i] - mean;
        var += d * d;
    }
    var /= (double)n_frames;
    double std = sqrt(var) + 1e-9;

    int n_diffs = n_frames - 1;
    float *diffs = (float *)malloc(n_diffs * sizeof(float));
    if (!diffs) {
        *threshold_out = 0.0f;
        return 0;
    }

    for (int i = 0; i < n_diffs; ++i) {
        float c0 = (float)((centroids_hz[i]   - mean) / std);
        float c1 = (float)((centroids_hz[i+1] - mean) / std);
        diffs[i] = fabsf(c1 - c0);
        if (centroid_diffs_out) centroid_diffs_out[i] = diffs[i];
    }

    int *peaks = (int *)malloc(n_diffs * sizeof(int));
    if (!peaks) {
        free(diffs);
        *threshold_out = 0.0f;
        return 0;
    }

    float thr;
    int n_peaks = peak_pick_from_diffs(
        diffs, n_diffs,
        frame_duration,
        percentile_val,
        min_gap_s,
        peaks, n_diffs,
        &thr
    );

    int n_changes = (n_peaks < max_changes) ? n_peaks : max_changes;
    for (int i = 0; i < n_changes; ++i) {
        change_frame_indices[i] = peaks[i] + 1;   // Übergang i -> i+1
    }

    *threshold_out = thr;
    free(diffs);
    free(peaks);
    return n_changes;
}

//---------------------------------------
// Energie-Detektor (detect_change_points_energy_only)
//---------------------------------------
int detect_change_points_energy_only(
    const float *y, int n_samples, int sr,
    double frame_duration,
    double percentile_val,
    double min_gap_s,
    int *change_frame_indices, int max_changes,
    float *energies_out,      // OUT: pro Frame (optional, Länge >= MAX_FRAMES)
    int *n_frames_out,        // OUT: Anzahl Frames
    float *energy_diffs_out,  // optional: Diffs
    float *threshold_out)
{
    int L = (int)(frame_duration * sr);
    if (L <= 0 || n_samples < L) {
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }

    float *win = (float *)malloc(L * sizeof(float));
    if (!win) {
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }
    for (int i = 0; i < L; ++i) {
        win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(L - 1));
    }

    int n_frames = 0;
    int max_frames_local = MAX_FRAMES;
    float *energies = energies_out ? energies_out
                                   : (float *)malloc(max_frames_local * sizeof(float));
    if (!energies) {
        free(win);
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }

    for (int start = 0; start + L <= n_samples; start += L) {
        if (n_frames >= max_frames_local) break;
        double e = 0.0;
        for (int i = 0; i < L; ++i) {
            float v = y[start + i] * win[i];
            e += (double)v * (double)v;
        }
        e /= (double)L;
        energies[n_frames++] = (float)e;
    }

    if (n_frames_out) *n_frames_out = n_frames;

    if (n_frames <= 1) {
        if (!energies_out) free(energies);
        free(win);
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n_frames; ++i) mean += energies[i];
    mean /= (double)n_frames;
    for (int i = 0; i < n_frames; ++i) {
        double d = energies[i] - mean;
        var += d * d;
    }
    var /= (double)n_frames;
    double std = sqrt(var) + 1e-9;

    float *energies_norm = (float *)malloc(n_frames * sizeof(float));
    float *diffs = energy_diffs_out ? energy_diffs_out
                                    : (float *)malloc((n_frames - 1) * sizeof(float));
    if (!energies_norm || !diffs) {
        if (energies_norm) free(energies_norm);
        if (!energy_diffs_out && diffs) free(diffs);
        if (!energies_out) free(energies);
        free(win);
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    for (int i = 0; i < n_frames; ++i) {
        energies_norm[i] = (float)((energies[i] - mean) / std);
    }

    int n_diffs = n_frames - 1;
    for (int i = 0; i < n_diffs; ++i) {
        diffs[i] = fabsf(energies_norm[i+1] - energies_norm[i]);
    }

    int *peaks = (int *)malloc(n_diffs * sizeof(int));
    if (!peaks) {
        free(energies_norm);
        if (!energy_diffs_out) free(diffs);
        if (!energies_out) free(energies);
        free(win);
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    float thr;
    int n_peaks = peak_pick_from_diffs(
        diffs, n_diffs,
        frame_duration,
        percentile_val,
        min_gap_s,
        peaks, n_diffs,
        &thr
    );

    int n_changes = (n_peaks < max_changes) ? n_peaks : max_changes;
    for (int i = 0; i < n_changes; ++i) {
        change_frame_indices[i] = peaks[i] + 1;
    }

    if (threshold_out) *threshold_out = thr;

    free(energies_norm);
    free(peaks);
    if (!energy_diffs_out) free(diffs);
    if (!energies_out) free(energies);
    free(win);

    return n_changes;
}

//---------------------------------------
// Amplituden-Sprünge (detect_change_points_amplitude_jump)
//---------------------------------------
int detect_change_points_amplitude_jump(
    const float *y, int n_samples, int sr,
    double percentile_val,
    double min_gap_s,
    double *change_times,   // OUT: Sekunden
    int max_changes,
    float *diffs_out,       // optional
    float *threshold_out)
{
    if (n_samples <= 1) {
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    int n_diffs = n_samples - 1;
    float *diffs = diffs_out ? diffs_out
                             : (float *)malloc(n_diffs * sizeof(float));
    if (!diffs) {
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    for (int i = 0; i < n_diffs; ++i) {
        diffs[i] = fabsf(y[i+1] - y[i]);
    }

    int *peaks = (int *)malloc(n_diffs * sizeof(int));
    if (!peaks) {
        if (!diffs_out) free(diffs);
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    float thr;
    int n_peaks = peak_pick_from_diffs(
        diffs, n_diffs,
        1.0 / (double)sr,
        percentile_val,
        min_gap_s,
        peaks, n_diffs,
        &thr
    );

    int n_changes = (n_peaks < max_changes) ? n_peaks : max_changes;
    for (int i = 0; i < n_changes; ++i) {
        int sample_idx = peaks[i] + 1;
        change_times[i] = (double)sample_idx / (double)sr;
    }

    if (threshold_out) *threshold_out = thr;

    free(peaks);
    if (!diffs_out) free(diffs);
    return n_changes;
}

//---------------------------------------
// Shape-Change-Detektor (detect_change_points_shape_change)
//---------------------------------------
int detect_change_points_shape_change(
    const float *y, int n_samples, int sr,
    double frame_duration,
    double percentile_val,
    double min_gap_s,
    int *change_frame_indices,  // OUT: Frame-Indizes
    int max_changes,
    float *shape_diffs_out,     // optional
    float *threshold_out,       // OUT
    int *n_frames_out)          // OUT
{
    int L = (int)(frame_duration * sr);
    if (L <= 0 || n_samples < L) {
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }

    float *win  = (float *)malloc(L * sizeof(float));
    float *prev = (float *)malloc(L * sizeof(float));
    float *curr = (float *)malloc(L * sizeof(float));
    if (!win || !prev || !curr) {
        free(win); free(prev); free(curr);
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }

    for (int i = 0; i < L; ++i) {
        win[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / (float)(L - 1));
    }

    int max_frames_local = n_samples / L;
    if (max_frames_local < 2) {
        free(win); free(prev); free(curr);
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }

    float *diffs = (float *)malloc((max_frames_local - 1) * sizeof(float));
    if (!diffs) {
        free(win); free(prev); free(curr);
        if (threshold_out) *threshold_out = 0.0f;
        if (n_frames_out)  *n_frames_out = 0;
        return 0;
    }

    int n_frames = 0;
    int n_diffs  = 0;

    for (int start = 0; start + L <= n_samples; start += L) {
        for (int i = 0; i < L; ++i) {
            curr[i] = y[start + i] * win[i];
        }

        if (n_frames > 0) {
            double num = 0.0, den1 = 0.0, den2 = 0.0;
            for (int i = 0; i < L; ++i) {
                double a = prev[i];
                double b = curr[i];
                num  += a * b;
                den1 += a * a;
                den2 += b * b;
            }
            double den = sqrt(den1 * den2) + 1e-9;
            float corr = (float)(num / den);
            float diff = 1.0f - corr;
            if (n_diffs < max_frames_local - 1) {
                diffs[n_diffs++] = diff;
            }
        }

        memcpy(prev, curr, L * sizeof(float));
        n_frames++;
        if (n_frames >= max_frames_local) break;
    }

    if (n_frames_out) *n_frames_out = n_frames;

    if (n_diffs <= 0) {
        free(win); free(prev); free(curr); free(diffs);
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    if (shape_diffs_out) {
        memcpy(shape_diffs_out, diffs, n_diffs * sizeof(float));
    }

    int *peaks = (int *)malloc(n_diffs * sizeof(int));
    if (!peaks) {
        free(win); free(prev); free(curr); free(diffs);
        if (threshold_out) *threshold_out = 0.0f;
        return 0;
    }

    float thr;
    int n_peaks = peak_pick_from_diffs(
        diffs, n_diffs,
        frame_duration,
        percentile_val,
        min_gap_s,
        peaks, n_diffs,
        &thr
    );

    int n_changes = (n_peaks < max_changes) ? n_peaks : max_changes;
    for (int i = 0; i < n_changes; ++i) {
        change_frame_indices[i] = peaks[i] + 1;
    }

    if (threshold_out) *threshold_out = thr;

    free(peaks);
    free(diffs);
    free(win);
    free(prev);
    free(curr);
    return n_changes;
}

//---------------------------------------
// Joint-Prioritäten (find_joint_change_points)
//---------------------------------------
int timemethod_compare(const void *a, const void *b) {
    double ta = ((const TimeMethod*)a)->t;
    double tb = ((const TimeMethod*)b)->t;
    return (ta > tb) - (ta < tb);
}

void find_joint_change_points(
    const double *change_cent,   int n_cent,
    const double *change_energy, int n_energy,
    const double *change_jump,   int n_jump,
    const double *change_shape,  int n_shape,
    double max_diff_s,
    double *prio0, int *n_prio0,
    double *prio1, int *n_prio1,
    double *prio2, int *n_prio2,
    double *prio3, int *n_prio3)
{
    TimeMethod *points = (TimeMethod*)malloc(
        (n_cent + n_energy + n_jump + n_shape) * sizeof(TimeMethod)
    );
    int n_points = 0;

    for (int i = 0; i < n_cent;   ++i) points[n_points++] = (TimeMethod){ change_cent[i],   0 };
    for (int i = 0; i < n_energy; ++i) points[n_points++] = (TimeMethod){ change_energy[i], 1 };
    for (int i = 0; i < n_jump;   ++i) points[n_points++] = (TimeMethod){ change_jump[i],   2 };
    for (int i = 0; i < n_shape;  ++i) points[n_points++] = (TimeMethod){ change_shape[i],  3 };

    *n_prio0 = *n_prio1 = *n_prio2 = *n_prio3 = 0;
    if (n_points == 0) {
        free(points);
        return;
    }

    qsort(points, n_points, sizeof(TimeMethod), timemethod_compare);

    int start = 0;
    while (start < n_points) {
        double t0 = points[start].t;
        int end = start + 1;
        while (end < n_points && points[end].t - t0 <= max_diff_s) {
            ++end;
        }

        int methods[4] = {0,0,0,0};
        double mean_per_method[4] = {0,0,0,0};
        int count_per_method[4] = {0,0,0,0};

        for (int i = start; i < end; ++i) {
            int mid = points[i].method_id;
            methods[mid] = 1;
            mean_per_method[mid] += points[i].t;
            count_per_method[mid]++;
        }

        int unique_methods = 0;
        for (int m = 0; m < 4; ++m) if (methods[m]) unique_methods++;

        // Energy+Jump-only ignorieren
        if (unique_methods == 2 && methods[1] && methods[2] && !methods[0] && !methods[3]) {
            // skip
        } else if (unique_methods > 0) {
            double rep_sum = 0.0;
            int rep_count = 0;
            for (int m = 0; m < 4; ++m) {
                if (methods[m]) {
                    rep_sum += mean_per_method[m] / (double)count_per_method[m];
                    rep_count++;
                }
            }
            double t_mean = rep_sum / (double)rep_count;

            int prio = unique_methods - 1;
            if (prio < 0) prio = 0;
            if (prio > 3) prio = 3;

            switch (prio) {
                case 0: prio0[(*n_prio0)++] = t_mean; break;
                case 1: prio1[(*n_prio1)++] = t_mean; break;
                case 2: prio2[(*n_prio2)++] = t_mean; break;
                case 3: prio3[(*n_prio3)++] = t_mean; break;
            }
        }

        start = end;
    }

    free(points);
}

//---------------------------------------
// Segment-Clustering mit optionalen Features
// entspricht assign_segments_to_sources
//---------------------------------------
int assign_segments_to_sources(
    const float *centroids_hz, int n_frames,
    double frame_duration,
    const int *change_frame_indices, int n_change_frames,
    double cluster_threshold_hz,
    int min_segments_per_cluster,
    const float *frame_feats, int feat_dim,
    Segment *segments, int *n_segments,
    float *cluster_centroid_means, int *n_clusters)
{
    if (n_frames <= 0) {
        *n_segments = 0;
        *n_clusters = 0;
        return 0;
    }

    // Grenzen [0, frame_indices..., n_frames]
    int boundaries[MAX_SEGMENTS + 2];
    int nb = 0;
    boundaries[nb++] = 0;

    // sort + unique der change_frame_indices
    int *fb = (int *)malloc(n_change_frames * sizeof(int));
    if (!fb && n_change_frames > 0) return -1;
    for (int i = 0; i < n_change_frames; ++i) fb[i] = change_frame_indices[i];
    if (n_change_frames > 1) {
        qsort(fb, n_change_frames, sizeof(int), int_compare);
        int w = 1;
        for (int i = 1; i < n_change_frames; ++i) {
            if (fb[i] != fb[w-1]) fb[w++] = fb[i];
        }
        n_change_frames = w;
    }
    for (int i = 0; i < n_change_frames; ++i) {
        int idx = fb[i];
        if (idx > 0 && idx < n_frames) {
            boundaries[nb++] = idx;
        }
    }
    boundaries[nb++] = n_frames;
    free(fb);

    // Segmente aufbauen
    float seg_centroids[MAX_SEGMENTS];
    float seg_feats[MAX_SEGMENTS][FEAT_DIM];

    int seg_count = 0;
    int used_feat_dim = frame_feats ? feat_dim : 1;

    for (int i = 0; i < nb - 1; ++i) {
        int s = boundaries[i];
        int e = boundaries[i+1];
        if (e <= s) continue;
        if (seg_count >= MAX_SEGMENTS) break;

        double sum_c = 0.0;
        for (int f = s; f < e; ++f) sum_c += centroids_hz[f];
        float mean_c = (float)(sum_c / (double)(e - s));
        seg_centroids[seg_count] = mean_c;

        // Feature-Vektor
        if (frame_feats) {
            for (int d = 0; d < used_feat_dim; ++d) {
                double sum = 0.0;
                for (int f = s; f < e; ++f) {
                    sum += frame_feats[f * feat_dim + d];
                }
                seg_feats[seg_count][d] = (float)(sum / (double)(e - s));
            }
        } else {
            seg_feats[seg_count][0] = mean_c;
        }

        Segment seg;
        seg.segment_index    = seg_count;
        seg.start_frame      = s;
        seg.end_frame        = e;
        seg.start_time       = (float)(s * frame_duration);
        seg.end_time         = (float)(e * frame_duration);
        seg.mean_centroid_hz = mean_c;
        seg.cluster_id       = -1;
        seg.label            = '?';

        segments[seg_count++] = seg;
    }

    *n_segments = seg_count;
    if (seg_count == 0) {
        *n_clusters = 0;
        return 0;
    }

    // Hierarchisches Mergen der Cluster im Feature-Raum
    int cluster_id[MAX_SEGMENTS];
    float cluster_vec[MAX_SEGMENTS][FEAT_DIM];
    int cluster_size[MAX_SEGMENTS];
    int cluster_active[MAX_SEGMENTS];

    for (int i = 0; i < seg_count; ++i) {
        cluster_id[i] = i;
        cluster_size[i] = 1;
        cluster_active[i] = 1;
        for (int d = 0; d < used_feat_dim; ++d) {
            cluster_vec[i][d] = seg_feats[i][d];
        }
    }

    double MERGE_THRESH = frame_feats ? 1.8 : cluster_threshold_hz;

    while (1) {
        double min_d = 1e30;
        int best_a = -1, best_b = -1;

        for (int a = 0; a < seg_count; ++a) {
            if (!cluster_active[a]) continue;
            for (int b = a + 1; b < seg_count; ++b) {
                if (!cluster_active[b]) continue;
                double d = 0.0;
                if (frame_feats) {
                    for (int k = 0; k < used_feat_dim; ++k) {
                        double diff = cluster_vec[a][k] - cluster_vec[b][k];
                        d += diff * diff;
                    }
                    d = sqrt(d);
                } else {
                    double diff = (double)cluster_vec[a][0] - (double)cluster_vec[b][0];
                    d = fabs(diff);
                }
                if (d < min_d) {
                    min_d = d;
                    best_a = a;
                    best_b = b;
                }
            }
        }

        if (best_a < 0 || best_b < 0 || min_d > MERGE_THRESH) {
            break;
        }

        // B in A mergen
        double size_a = (double)cluster_size[best_a];
        double size_b = (double)cluster_size[best_b];
        double size_sum = size_a + size_b;

        for (int d = 0; d < used_feat_dim; ++d) {
            double va = cluster_vec[best_a][d];
            double vb = cluster_vec[best_b][d];
            cluster_vec[best_a][d] = (float)((va * size_a + vb * size_b) / size_sum);
        }
        cluster_size[best_a] += cluster_size[best_b];
        cluster_active[best_b] = 0;

        // Segment-IDs updaten
        for (int i = 0; i < seg_count; ++i) {
            if (cluster_id[i] == best_b) {
                cluster_id[i] = best_a;
            }
        }
    }

    // Clustergrößen neu zählen
    int size_per_cluster[MAX_SEGMENTS];
    for (int i = 0; i < seg_count; ++i) size_per_cluster[i] = 0;
    for (int i = 0; i < seg_count; ++i) {
        int cid = cluster_id[i];
        if (cid >= 0 && cid < seg_count) {
            size_per_cluster[cid]++;
        }
    }

    // Big-Cluster bestimmen
    int big_mask[MAX_SEGMENTS];
    for (int i = 0; i < seg_count; ++i) big_mask[i] = 0;
    int any_big = 0;
    for (int i = 0; i < seg_count; ++i) {
        if (!cluster_active[i]) continue;
        if (size_per_cluster[i] >= min_segments_per_cluster) {
            big_mask[i] = 1;
            any_big = 1;
        }
    }
    if (!any_big) {
        for (int i = 0; i < seg_count; ++i) {
            if (cluster_active[i]) big_mask[i] = 1;
        }
    }

    // Big-Cluster-Liste
    int big_ids[MAX_SEGMENTS];
    int n_big = 0;
    for (int i = 0; i < seg_count; ++i) {
        if (cluster_active[i] && big_mask[i]) big_ids[n_big++] = i;
    }

    // Kleine Segmente: zu nächstem Big-Cluster
    for (int i = 0; i < seg_count; ++i) {
        int cid = cluster_id[i];
        if (!cluster_active[cid] || big_mask[cid]) continue; // big oder tot -> fertig
        // dist seg_feats[i] zu jedem big
        double best_d = 1e30;
        int best_big = big_ids[0];
        for (int j = 0; j < n_big; ++j) {
            int bid = big_ids[j];
            double d = 0.0;
            if (frame_feats) {
                for (int k = 0; k < used_feat_dim; ++k) {
                    double diff = seg_feats[i][k] - cluster_vec[bid][k];
                    d += diff * diff;
                }
                d = sqrt(d);
            } else {
                double diff = (double)seg_feats[i][0] - (double)cluster_vec[bid][0];
                d = fabs(diff);
            }
            if (d < best_d) {
                best_d = d;
                best_big = bid;
            }
        }
        cluster_id[i] = best_big;
    }

    // Alte Cluster-IDs sammeln
    int unique_ids[MAX_SEGMENTS];
    int n_unique = 0;
    for (int i = 0; i < seg_count; ++i) {
        int cid = cluster_id[i];
        int known = 0;
        for (int j = 0; j < n_unique; ++j) {
            if (unique_ids[j] == cid) {
                known = 1; break;
            }
        }
        if (!known) unique_ids[n_unique++] = cid;
    }

    // Mittlere Centroids pro Cluster
    double centroid_sums[MAX_SEGMENTS];
    int centroid_counts[MAX_SEGMENTS];
    for (int j = 0; j < n_unique; ++j) {
        centroid_sums[j] = 0.0;
        centroid_counts[j] = 0;
    }
    for (int i = 0; i < seg_count; ++i) {
        int cid = cluster_id[i];
        // index in unique_ids suchen
        int idx = -1;
        for (int j = 0; j < n_unique; ++j) {
            if (unique_ids[j] == cid) { idx = j; break; }
        }
        if (idx >= 0) {
            centroid_sums[idx]   += seg_centroids[i];
            centroid_counts[idx] += 1;
        }
    }
    double centroid_means_local[MAX_SEGMENTS];
    for (int j = 0; j < n_unique; ++j) {
        centroid_means_local[j] =
            (centroid_counts[j] > 0) ?
            (centroid_sums[j] / (double)centroid_counts[j]) : 0.0;
    }

    // Cluster nach mittlerem Centroid sortieren
    int order[MAX_SEGMENTS];
    for (int j = 0; j < n_unique; ++j) order[j] = j;
    // sort via centroid_means_local
    int cmp_centroid(const void *a, const void *b) {
        int ia = *(const int*)a;
        int ib = *(const int*)b;
        double ca = centroid_means_local[ia];
        double cb = centroid_means_local[ib];
        return (ca > cb) - (ca < cb);
    }
    qsort(order, n_unique, sizeof(int), cmp_centroid);

    // Mapping old_cluster_id -> new_cluster_id
    int map_old_to_new[MAX_SEGMENTS];
    for (int i = 0; i < MAX_SEGMENTS; ++i) map_old_to_new[i] = -1;

    for (int rank = 0; rank < n_unique; ++rank) {
        int j = order[rank];
        int old_cid = unique_ids[j];
        map_old_to_new[old_cid] = rank;
        cluster_centroid_means[rank] = (float)centroid_means_local[j];
    }

    // Segmente updaten: cluster_id & label
    for (int i = 0; i < seg_count; ++i) {
        int old_cid = cluster_id[i];
        int new_cid = (old_cid >= 0 && old_cid < MAX_SEGMENTS)
                      ? map_old_to_new[old_cid]
                      : -1;
        segments[i].cluster_id = new_cid;
        if (new_cid >= 0 && new_cid < MAX_LABELS) {
            segments[i].label = (char)('A' + new_cid);
        } else {
            segments[i].label = '?';
        }
    }

    *n_clusters = n_unique;
    return 0;
}

//---------------------------------------
// segmente_zuordnen – baut Grenzen aus Prio 1–3 (oder 2–3)
// und ruft assign_segments_to_sources
//---------------------------------------
int segmente_zuordnen(
    const float *centroids_hz, int n_frames,
    const double *prio1_times, int n_prio1,
    const double *prio2_times, int n_prio2,
    const double *prio3_times, int n_prio3,
    const int *change_frames_cent, int n_change_cent,
    const float *frame_feats, int feat_dim,
    Segment *segments, int *n_segments,
    float *cluster_means, int *n_clusters)
{
    // Prio-Zeiten sammeln
    double prio_times_all[MAX_CHANGES];
    int n_prio_all = 0;

#if USE_PRIO1_IN_BOUNDARIES
    for (int i = 0; i < n_prio1; ++i) prio_times_all[n_prio_all++] = prio1_times[i];
#endif
    for (int i = 0; i < n_prio2; ++i) prio_times_all[n_prio_all++] = prio2_times[i];
    for (int i = 0; i < n_prio3; ++i) prio_times_all[n_prio_all++] = prio3_times[i];

    int frame_boundaries[MAX_FRAMES];
    int n_boundaries = 0;

    if (n_prio_all > 0) {
        for (int i = 0; i < n_prio_all; ++i) {
            int f_idx = (int)round(prio_times_all[i] / FRAME_DURATION_S);
            if (f_idx > 0 && f_idx < n_frames) {
                frame_boundaries[n_boundaries++] = f_idx;
            }
        }
        if (n_boundaries > 1) {
            qsort(frame_boundaries, n_boundaries, sizeof(int), int_compare);
            int w = 1;
            for (int i = 1; i < n_boundaries; ++i) {
                if (frame_boundaries[i] != frame_boundaries[w-1]) {
                    frame_boundaries[w++] = frame_boundaries[i];
                }
            }
            n_boundaries = w;
        }
    } else {
        // Fallback: Centroid-Wechsel
        for (int i = 0; i < n_change_cent; ++i) {
            frame_boundaries[n_boundaries++] = change_frames_cent[i];
        }
    }

    // assign_segments_to_sources aufrufen
    int err = assign_segments_to_sources(
        centroids_hz, n_frames,
        FRAME_DURATION_S,
        frame_boundaries, n_boundaries,
        CLUSTER_THRESHOLD_HZ,
        MIN_SEGMENTS_PER_CLUSTER,
        frame_feats, feat_dim,
        segments, n_segments,
        cluster_means, n_clusters
    );

    if (err != 0) return err;

    // Konsolenausgabe wie im Python-Code
    printf("\nSegment-Zuordnung (basierend auf priorisierten Wechselstellen):\n");
    for (int i = 0; i < *n_segments; ++i) {
        Segment *seg = &segments[i];
        printf("Segment %2d: %.3fs - %.3fs | Centroid ≈ %.0f Hz -> Signal %c\n",
               seg->segment_index,
               seg->start_time,
               seg->end_time,
               seg->mean_centroid_hz,
               seg->label);
    }

    printf("\nMittlere Schwerpunkte der gefundenen Quellen:\n");
    for (int cid = 0; cid < *n_clusters; ++cid) {
        char label = '?';
        for (int i = 0; i < *n_segments; ++i) {
            if (segments[i].cluster_id == cid) {
                label = segments[i].label;
                break;
            }
        }
        printf("Quelle %c: ca. %.0f Hz\n", label, cluster_means[cid]);
    }

    printf("\nAnzahl erkannter unterschiedlicher Signale (Cluster): %d\n", *n_clusters);

    return 0;
}

//---------------------------------------
// Rekonstruktion eines Signals aus Segmenten
//---------------------------------------
int reconstruct_source(const float *y, int n_samples, int sr,
                       const Segment *segments, int n_segments,
                       char label,
                       float *out, int max_out_samples)
{
    int out_pos = 0;
    for (int i = 0; i < n_segments; ++i) {
        if (segments[i].label != label) continue;

        int start_sample = (int)round(segments[i].start_time * sr);
        int end_sample   = (int)round(segments[i].end_time   * sr);

        if (start_sample < 0) start_sample = 0;
        if (end_sample > n_samples) end_sample = n_samples;

        for (int n = start_sample; n < end_sample; ++n) {
            if (out_pos >= max_out_samples) return out_pos;
            out[out_pos++] = y[n];
        }
    }
    return out_pos;
}

//---------------------------------------
// WAV-Export (16-bit PCM Mono) – als Ersatz für MP3-Export
//---------------------------------------
int write_wav_mono16(const char *filename, const float *x, int n, int sr)
{
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;

    int16_t *buf = (int16_t *)malloc(n * sizeof(int16_t));
    if (!buf) {
        fclose(f);
        return -1;
    }

    // Normalisieren
    float maxv = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = fabsf(x[i]);
        if (v > maxv) maxv = v;
    }
    if (maxv < 1e-9f) maxv = 1.0f;
    for (int i = 0; i < n; ++i) {
        float v = x[i] / maxv;
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        buf[i] = (int16_t)(v * 32767.0f);
    }

    uint32_t data_size = (uint32_t)(n * sizeof(int16_t));
    uint32_t chunk_size = 36 + data_size;

    // RIFF-Header
    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt-Chunk
    uint32_t subchunk1_size = 16;
    uint16_t audio_format = 1;      // PCM
    uint16_t num_channels = 1;
    uint32_t sample_rate = (uint32_t)sr;
    uint16_t bits_per_sample = 16;
    uint32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    uint16_t block_align = num_channels * bits_per_sample / 8;

    fwrite("fmt ", 1, 4, f);
    fwrite(&subchunk1_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    // data-Chunk
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    fwrite(buf, sizeof(int16_t), n, f);

    free(buf);
    fclose(f);
    return 0;
}

//---------------------------------------
// Plot-Stubs – hier könntest du z.B. gnuplot ansteuern
//---------------------------------------
void plot_signal_and_changes(/* alle nötigen Parameter */) {
    // TODO: optional implementieren
}

void plot_priority_changes(/* wie im Python-Code */) {
    // TODO: optional implementieren
}

void plot_close_priority_changes(/* wie im Python-Code */) {
    // TODO: optional implementieren
}

//---------------------------------------
// Audio laden – Stub
// Hier solltest du z.B. WAV mit libsndfile laden.
//---------------------------------------
int load_audio_mono(const char *filename, float **y_out, int *n_samples_out, int *sr_out)
{
    // TODO: echte Implementierung
    // Platzhalter: kein Audio
    *y_out = NULL;
    *n_samples_out = 0;
    *sr_out = 0;
    fprintf(stderr, "load_audio_mono: TODO – WAV/MP3-Loader implementieren!\n");
    return -1;
}

//---------------------------------------
// main – entspricht deinem Python-main (ohne I/O & Plots)
//---------------------------------------
int main(void)
{
    const char *INPUT_FILE = "input.wav";  // statt MP3: WAV-Datei

    float *y = NULL;
    int n_samples = 0;
    int sr = 0;

    if (load_audio_mono(INPUT_FILE, &y, &n_samples, &sr) != 0) {
        fprintf(stderr, "Konnte Audio nicht laden: %s\n", INPUT_FILE);
        return 1;
    }

    //---------------------------------------
    // 1) Features: Centroids + Frame-Features
    //---------------------------------------
    float centroids_hz[MAX_FRAMES];
    int n_frames_cent = compute_centroids(
        y, n_samples, sr,
        FRAME_DURATION_S,
        centroids_hz, MAX_FRAMES
    );

    float *frame_feats = (float *)malloc(MAX_FRAMES * FEAT_DIM * sizeof(float));
    if (!frame_feats) {
        fprintf(stderr, "Speicher für frame_feats fehlgeschlagen\n");
        free(y);
        return 1;
    }

    int n_frames_feats = compute_frame_features(
        y, n_samples, sr,
        FRAME_DURATION_S,
        frame_feats, MAX_FRAMES, FEAT_DIM
    );

    int n_frames = n_frames_cent;
    if (n_frames_feats < n_frames) n_frames = n_frames_feats;

    //---------------------------------------
    // 2) Wechselstellen der 4 Detektoren
    //---------------------------------------
    // Centroid
    int change_frames_cent[MAX_CHANGES];
    float diffs_cent[MAX_FRAMES];
    float thr_cent = 0.0f;
    int n_changes_cent = detect_change_points_centroid_only(
        centroids_hz, n_frames,
        FRAME_DURATION_S,
        PERCENTILE_FEATURE,
        MIN_GAP_S,
        change_frames_cent, MAX_CHANGES,
        diffs_cent, &thr_cent
    );
    double change_times_cent[MAX_CHANGES];
    for (int i = 0; i < n_changes_cent; ++i) {
        change_times_cent[i] = change_frames_cent[i] * FRAME_DURATION_S;
    }

    // Energie
    int change_frames_energy[MAX_CHANGES];
    float energies[MAX_FRAMES];
    float diffs_energy[MAX_FRAMES];
    float thr_energy = 0.0f;
    int n_frames_energy = 0;
    int n_changes_energy = detect_change_points_energy_only(
        y, n_samples, sr,
        FRAME_DURATION_S,
        ENERGY_PERCENTILE,
        ENERGY_MIN_GAP_S,
        change_frames_energy, MAX_CHANGES,
        energies, &n_frames_energy,
        diffs_energy, &thr_energy
    );
    double change_times_energy[MAX_CHANGES];
    for (int i = 0; i < n_changes_energy; ++i) {
        change_times_energy[i] = change_frames_energy[i] * FRAME_DURATION_S;
    }

    // Amplituden-Sprünge
    double change_times_jump[MAX_CHANGES];
    float diffs_jump[MAX_FRAMES];
    float thr_jump = 0.0f;
    int n_changes_jump = detect_change_points_amplitude_jump(
        y, n_samples, sr,
        JUMP_PERCENTILE,
        JUMP_MIN_GAP_S,
        change_times_jump, MAX_CHANGES,
        diffs_jump, &thr_jump
    );

    // Shape-Change
    int change_frames_shape[MAX_CHANGES];
    float diffs_shape[MAX_FRAMES];
    float thr_shape = 0.0f;
    int n_frames_shape = 0;
    int n_changes_shape = detect_change_points_shape_change(
        y, n_samples, sr,
        FRAME_DURATION_S,
        SHAPE_PERCENTILE,
        SHAPE_MIN_GAP_S,
        change_frames_shape, MAX_CHANGES,
        diffs_shape, &thr_shape,
        &n_frames_shape
    );
    double change_times_shape[MAX_CHANGES];
    for (int i = 0; i < n_changes_shape; ++i) {
        change_times_shape[i] = change_frames_shape[i] * FRAME_DURATION_S;
    }

    //---------------------------------------
    // 3) Priorisierte Wechselstellen
    //---------------------------------------
    double prio0_times[MAX_CHANGES], prio1_times[MAX_CHANGES],
           prio2_times[MAX_CHANGES], prio3_times[MAX_CHANGES];
    int n_prio0, n_prio1, n_prio2, n_prio3;

    find_joint_change_points(
        change_times_cent,   n_changes_cent,
        change_times_energy, n_changes_energy,
        change_times_jump,   n_changes_jump,
        change_times_shape,  n_changes_shape,
        JOINT_MAX_DIFF_S,
        prio0_times, &n_prio0,
        prio1_times, &n_prio1,
        prio2_times, &n_prio2,
        prio3_times, &n_prio3
    );

    printf("\nWechselstellen nach Priorität:\n");
    printf("\nPrio 1 (2 Methoden): %d Stellen\n", n_prio1);
    printf("Prio 2 (3 Methoden): %d Stellen\n", n_prio2);
    printf("Prio 3 (4 Methoden): %d Stellen\n", n_prio3);

    //---------------------------------------
    // 4) Segmente zuordnen (A/B/C/...)
    //---------------------------------------
    Segment segments[MAX_SEGMENTS];
    int n_segments = 0;
    float cluster_means[MAX_LABELS];
    int n_clusters = 0;

    segmente_zuordnen(
        centroids_hz, n_frames,
        prio1_times, n_prio1,
        prio2_times, n_prio2,
        prio3_times, n_prio3,
        change_frames_cent, n_changes_cent,
        frame_feats, FEAT_DIM,
        segments, &n_segments,
        cluster_means, &n_clusters
    );

    //---------------------------------------
    // 5) Rekonstruktion aller gefundenen Labels
    //---------------------------------------
    float *recon = (float *)malloc(n_samples * sizeof(float));
    if (!recon) {
        fprintf(stderr, "Speicher für recon fehlgeschlagen\n");
        free(frame_feats);
        free(y);
        return 1;
    }

    int label_used[MAX_LABELS] = {0};
    for (int i = 0; i < n_segments; ++i) {
        int cid = segments[i].cluster_id;
        if (cid >= 0 && cid < MAX_LABELS) label_used[cid] = 1;
    }

    for (int cid = 0; cid < n_clusters; ++cid) {
        if (!label_used[cid]) continue;
        char label = (char)('A' + cid);
        int n_out = reconstruct_source(
            y, n_samples, sr,
            segments, n_segments,
            label,
            recon, n_samples
        );
        printf("\nRekonstruierte Länge Signal %c: %d Samples (%.3f s)\n",
               label, n_out, (double)n_out / (double)sr);

        char fname[256];
        snprintf(fname, sizeof(fname), "deinterleaved_%c.wav", label);
        if (write_wav_mono16(fname, recon, n_out, sr) == 0) {
            printf("Deinterleavtes Signal %c als '%s' gespeichert.\n", label, fname);
        } else {
            printf("Fehler beim Speichern von %s\n", fname);
        }
    }

    free(recon);
    free(frame_feats);
    free(y);

    return 0;
}
