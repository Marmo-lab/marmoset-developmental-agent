package il.ac.huji.omerlab.marmolabel;

import android.graphics.Bitmap;

import java.util.Arrays;

/**
 * A clip's spectrogram, computed on the phone and cut into bitmap tiles narrow enough for
 * any GPU texture limit.
 *
 * <p>STFT: 1024-point Hann window (21 ms at 48 kHz), hop 240 (5 ms), so an edge can be
 * placed to ~5 ms when zoomed in [ours]. Pairs of FFT bins are summed into 256 rows of
 * 94 Hz over 0-24 kHz. Each row is shown in dB above its own median over the clip, the
 * same normalisation the learned detector sees (learned_detector.normalise), so stationary
 * colony noise sits near black and calls stand out at any loudness. Colour: magma over
 * 0-35 dB above the median.
 */
final class Spectrogram {
    static final int NFFT = 1024, HOP = 240, ROWS = 256, TILE = 1024, OVERVIEW = 1200;
    private static final float LO_DB = 0f, HI_DB = 35f;

    final Bitmap[] tiles;
    final Bitmap overview;
    final double hopS, nyquist;
    final int frames;

    private Spectrogram(Bitmap[] tiles, Bitmap overview, double hopS, double nyquist, int frames) {
        this.tiles = tiles;
        this.overview = overview;
        this.hopS = hopS;
        this.nyquist = nyquist;
        this.frames = frames;
    }

    static Spectrogram compute(short[] x, int rate) {
        int frames = x.length / HOP + 1;
        float[] db = new float[frames * ROWS];  // [frame][row], row 0 = lowest frequency
        double[] win = new double[NFFT], re = new double[NFFT], im = new double[NFFT];
        for (int i = 0; i < NFFT; i++) win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NFFT);
        Fft fft = new Fft(NFFT);
        for (int f = 0; f < frames; f++) {
            int start = f * HOP - NFFT / 2;  // centred frames, zero-padded at the clip's ends
            for (int i = 0; i < NFFT; i++) {
                int k = start + i;
                re[i] = k >= 0 && k < x.length ? x[k] * win[i] : 0;
                im[i] = 0;
            }
            fft.transform(re, im);
            for (int r = 0; r < ROWS; r++) {
                double p = 0;
                for (int k = 2 * r; k < 2 * r + 2; k++) p += re[k] * re[k] + im[k] * im[k];
                db[f * ROWS + r] = (float) (10 * Math.log10(p + 1e-3));
            }
        }
        float[] col = new float[frames];
        for (int r = 0; r < ROWS; r++) {
            for (int f = 0; f < frames; f++) col[f] = db[f * ROWS + r];
            Arrays.sort(col);
            float median = col[frames / 2];
            for (int f = 0; f < frames; f++) db[f * ROWS + r] -= median;
        }
        int[] lut = magma();
        Bitmap[] tiles = new Bitmap[(frames + TILE - 1) / TILE];
        for (int k = 0; k < tiles.length; k++) {
            int f0 = k * TILE, w = Math.min(TILE, frames - f0);
            int[] px = new int[w * ROWS];
            for (int c = 0; c < w; c++)
                for (int r = 0; r < ROWS; r++) px[(ROWS - 1 - r) * w + c] = lut[level(db[(f0 + c) * ROWS + r])];
            tiles[k] = Bitmap.createBitmap(px, w, ROWS, Bitmap.Config.ARGB_8888);
        }
        int ow = Math.min(OVERVIEW, frames);
        int[] px = new int[ow * ROWS];
        for (int c = 0; c < ow; c++) {
            int a = (int) ((long) c * frames / ow), b = Math.max(a + 1, (int) ((long) (c + 1) * frames / ow));
            for (int r = 0; r < ROWS; r++) {
                float m = -1e9f;
                for (int f = a; f < b; f++) m = Math.max(m, db[f * ROWS + r]);
                px[(ROWS - 1 - r) * ow + c] = lut[level(m)];
            }
        }
        Bitmap overview = Bitmap.createBitmap(px, ow, ROWS, Bitmap.Config.ARGB_8888);
        return new Spectrogram(tiles, overview, HOP / (double) rate, rate / 2.0, frames);
    }

    private static int level(float v) {
        int i = (int) ((v - LO_DB) / (HI_DB - LO_DB) * 255);
        return i < 0 ? 0 : Math.min(i, 255);
    }

    /** matplotlib's magma, from nine anchor colours. */
    private static int[] magma() {
        int[][] a = {{0, 0, 4}, {28, 16, 68}, {79, 18, 123}, {129, 37, 129}, {181, 54, 122},
                {229, 80, 100}, {251, 135, 97}, {254, 194, 135}, {252, 253, 191}};
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) {
            double p = i / 255.0 * (a.length - 1);
            int k = Math.min((int) p, a.length - 2);
            double u = p - k;
            int[] c = new int[3];
            for (int j = 0; j < 3; j++) c[j] = (int) Math.round(a[k][j] * (1 - u) + a[k + 1][j] * u);
            lut[i] = 0xff000000 | (c[0] << 16) | (c[1] << 8) | c[2];
        }
        return lut;
    }

    /** In-place radix-2 complex FFT. */
    private static final class Fft {
        private final int n;
        private final double[] cos, sin;
        private final int[] rev;

        Fft(int n) {
            this.n = n;
            cos = new double[n / 2];
            sin = new double[n / 2];
            for (int i = 0; i < n / 2; i++) {
                cos[i] = Math.cos(-2 * Math.PI * i / n);
                sin[i] = Math.sin(-2 * Math.PI * i / n);
            }
            rev = new int[n];
            int bits = Integer.numberOfTrailingZeros(n);
            for (int i = 0; i < n; i++) rev[i] = Integer.reverse(i) >>> (32 - bits);
        }

        void transform(double[] re, double[] im) {
            for (int i = 0; i < n; i++) {
                int j = rev[i];
                if (j > i) {
                    double t = re[i]; re[i] = re[j]; re[j] = t;
                    t = im[i]; im[i] = im[j]; im[j] = t;
                }
            }
            for (int size = 2; size <= n; size <<= 1) {
                int half = size / 2, step = n / size;
                for (int i = 0; i < n; i += size) {
                    for (int j = 0, k = 0; j < half; j++, k += step) {
                        int a = i + j, b = a + half;
                        double tr = re[b] * cos[k] - im[b] * sin[k], ti = re[b] * sin[k] + im[b] * cos[k];
                        re[b] = re[a] - tr;
                        im[b] = im[a] - ti;
                        re[a] += tr;
                        im[a] += ti;
                    }
                }
            }
        }
    }
}
