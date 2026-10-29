package il.ac.huji.omerlab.marmolabel;

import android.media.AudioAttributes;
import android.media.AudioFormat;
import android.media.AudioTrack;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/** A clip's samples (16-bit PCM WAV, first channel, as tools/labels/make_clips.py writes them) and their playback. */
final class Audio {
    final short[] x;
    final int rate;

    private Audio(short[] x, int rate) {
        this.x = x;
        this.rate = rate;
    }

    double duration() {
        return x.length / (double) rate;
    }

    static Audio read(File f) throws IOException {
        byte[] b = Files.readAllBytes(f.toPath());
        ByteBuffer bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN);
        if (b.length < 12 || !new String(b, 0, 4, StandardCharsets.US_ASCII).equals("RIFF")
                || !new String(b, 8, 4, StandardCharsets.US_ASCII).equals("WAVE")) throw new IOException("not a WAV file");
        int pos = 12, format = 0, channels = 1, rate = 0, bits = 0;
        while (pos + 8 <= b.length) {
            String id = new String(b, pos, 4, StandardCharsets.US_ASCII);
            int size = bb.getInt(pos + 4), body = pos + 8;
            if (id.equals("fmt ")) {
                format = bb.getShort(body) & 0xffff;
                channels = bb.getShort(body + 2);
                rate = bb.getInt(body + 4);
                bits = bb.getShort(body + 14);
            } else if (id.equals("data")) {
                if ((format != 1 && format != 0xfffe) || bits != 16) throw new IOException("expected 16-bit PCM");
                int n = Math.min(size, b.length - body) / (2 * channels);
                short[] x = new short[n];
                for (int i = 0; i < n; i++) x[i] = bb.getShort(body + 2 * channels * i);
                return new Audio(x, rate);
            }
            pos = body + size + (size & 1);
        }
        throw new IOException("no audio data");
    }

    /** Plays one stretch of a clip, peak-normalised so quiet calls are audible. */
    static final class Player {
        private AudioTrack track;
        private volatile boolean stopped = true;
        private double startT;
        private int n, rate;

        void play(Audio a, double t0, double t1) {
            stop();
            int i0 = Math.max(0, (int) (t0 * a.rate)), i1 = Math.min(a.x.length, (int) (t1 * a.rate));
            if (i1 <= i0) return;
            int peak = 1;
            for (int i = i0; i < i1; i++) peak = Math.max(peak, Math.abs((int) a.x[i]));
            double gain = Math.min(20.0, 0.9 * 32767 / peak);
            final short[] buf = new short[i1 - i0];
            for (int i = 0; i < buf.length; i++) buf[i] = (short) Math.round(a.x[i0 + i] * gain);
            int min = AudioTrack.getMinBufferSize(a.rate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT);
            final AudioTrack t = new AudioTrack.Builder()
                    .setAudioAttributes(new AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_MEDIA)
                            .setContentType(AudioAttributes.CONTENT_TYPE_MUSIC).build())
                    .setAudioFormat(new AudioFormat.Builder().setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                            .setSampleRate(a.rate).setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
                    .setTransferMode(AudioTrack.MODE_STREAM).setBufferSizeInBytes(Math.max(min, 8192) * 4).build();
            track = t;
            startT = i0 / (double) a.rate;
            n = buf.length;
            rate = a.rate;
            stopped = false;
            t.play();
            new Thread(() -> {
                for (int off = 0; off < buf.length && !stopped; ) {
                    int w = t.write(buf, off, Math.min(4096, buf.length - off));
                    if (w <= 0) break;
                    off += w;
                }
            }).start();
        }

        /** Seconds into the clip being played, or -1 when nothing plays (it stops itself at the end). */
        double position() {
            if (track == null) return -1;
            int h = track.getPlaybackHeadPosition();
            if (h >= n) {
                stop();
                return -1;
            }
            return startT + h / (double) rate;
        }

        boolean playing() {
            return track != null;
        }

        void stop() {
            stopped = true;
            if (track == null) return;
            try {
                track.pause();
                track.flush();
            } catch (IllegalStateException ignored) {
                // already stopped
            }
            track.release();
            track = null;
        }
    }
}
