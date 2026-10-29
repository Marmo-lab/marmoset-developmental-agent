package il.ac.huji.omerlab.marmolabel;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.ScaleGestureDetector;
import android.view.View;
import android.view.ViewConfiguration;

import java.util.Locale;

/**
 * The zoomable spectrogram with the clip's labels drawn over it.
 *
 * <p>Gestures: two fingers pinch to zoom in time and slide to scroll, in either mode. One
 * finger, in mark mode, drags out a new selection (white); in scroll mode it scrolls. A
 * tap selects the label under it (or clears the selection). The edges of the selection or
 * of the selected label have handles and can be dragged in either mode.
 */
final class SpectrogramView extends View {
    interface Listener {
        /** A label is about to change: the activity saves an undo step. */
        void beforeEdit();

        /** Labels changed: save. */
        void changed();

        /** The window or the selection changed: refresh the readouts. */
        void viewport();
    }

    static final double MIN_WINDOW_S = 0.2;
    static final int CALL = 0xff33d1c4, UNSURE = 0xffffb74d, OTHER = 0xffb0b0b0;

    Spectrogram spec;
    Labels labels;
    Listener listener;
    double duration = 60, t0 = 0, t1 = 6;
    Labels.Label selected;
    double pendS = -1, pendE = -1;
    boolean markMode = true;
    double playhead = -1;

    private final float dp, slop;
    private final Paint bmp = new Paint(Paint.FILTER_BITMAP_FLAG), fill = new Paint(),
            line = new Paint(Paint.ANTI_ALIAS_FLAG), text = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF dst = new RectF();
    private final ScaleGestureDetector scaler;

    private static final int NONE = 0, PAN = 1, CREATE = 2, EDGE = 3, EDGE_PENDING = 4, SCALE = 5;
    private int mode = NONE;
    private float downX, lastFocus;
    private long downTime;
    private double downT0, anchorT;
    private boolean moved, edgeStart, undoSaved, createdPending;

    SpectrogramView(Context c) {
        super(c);
        dp = c.getResources().getDisplayMetrics().density;
        slop = ViewConfiguration.get(c).getScaledTouchSlop();
        text.setTextSize(11 * dp);
        scaler = new ScaleGestureDetector(c, new ScaleGestureDetector.SimpleOnScaleGestureListener() {
            @Override
            public boolean onScaleBegin(ScaleGestureDetector d) {
                lastFocus = d.getFocusX();
                return true;
            }

            @Override
            public boolean onScale(ScaleGestureDetector d) {
                float prev = d.getPreviousSpanX(), cur = d.getCurrentSpanX();
                double factor = prev > 20 * dp && cur > 20 * dp ? prev / cur : 1;
                double focusT = timeAt(lastFocus);  // the time under the fingers stays under them
                double w = Math.max(MIN_WINDOW_S, Math.min(duration, (t1 - t0) * factor));
                double a = focusT - d.getFocusX() / getWidth() * w;
                setWindow(a, a + w);
                lastFocus = d.getFocusX();
                return true;
            }
        });
        scaler.setQuickScaleEnabled(false);
    }

    void setWindow(double a, double b) {
        double w = Math.min(b - a, duration);
        a = Math.max(0, Math.min(a, duration - w));
        t0 = a;
        t1 = a + w;
        invalidate();
        if (listener != null) listener.viewport();
    }

    double timeAt(float x) {
        return t0 + x / Math.max(1, getWidth()) * (t1 - t0);
    }

    float xAt(double t) {
        return (float) ((t - t0) / (t1 - t0) * getWidth());
    }

    void clearSelection() {
        selected = null;
        pendS = pendE = -1;
    }

    // ---- touch ------------------------------------------------------------------------------

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        if (spec == null) return true;
        scaler.onTouchEvent(e);
        switch (e.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                down(e.getX(), e.getEventTime());
                break;
            case MotionEvent.ACTION_POINTER_DOWN:
                if (mode == CREATE && createdPending) pendS = pendE = -1;
                if (mode == EDGE && undoSaved) finishEdge();
                mode = SCALE;
                break;
            case MotionEvent.ACTION_MOVE:
                if (mode != SCALE && e.getPointerCount() == 1) move(e.getX());
                break;
            case MotionEvent.ACTION_UP:
                if (mode != SCALE && !moved && e.getEventTime() - downTime < 500) tap(timeAt(downX));
                else if (mode == EDGE && undoSaved) finishEdge();
                else if (mode == EDGE_PENDING) orderPending();
                mode = NONE;
                if (listener != null) listener.viewport();
                break;
            case MotionEvent.ACTION_CANCEL:
                mode = NONE;
                break;
            default:
                break;
        }
        invalidate();
        return true;
    }

    private void down(float x, long time) {
        downX = x;
        downTime = time;
        downT0 = t0;
        moved = undoSaved = createdPending = false;
        float grab = 24 * dp;
        if (pendS >= 0 && Math.min(Math.abs(x - xAt(pendS)), Math.abs(x - xAt(pendE))) < grab) {
            mode = EDGE_PENDING;
            edgeStart = Math.abs(x - xAt(pendS)) < Math.abs(x - xAt(pendE));
        } else if (selected != null && Math.min(Math.abs(x - xAt(selected.s)), Math.abs(x - xAt(selected.e))) < grab) {
            mode = EDGE;
            edgeStart = Math.abs(x - xAt(selected.s)) < Math.abs(x - xAt(selected.e));
        } else if (markMode) {
            mode = CREATE;
            anchorT = clamp(timeAt(x));
        } else {
            mode = PAN;
        }
    }

    private void move(float x) {
        if (!moved && Math.abs(x - downX) < slop) return;
        moved = true;
        double t = clamp(timeAt(x));
        switch (mode) {
            case PAN: {
                double w = t1 - t0, a = downT0 - (x - downX) / getWidth() * w;
                setWindow(a, a + w);
                break;
            }
            case CREATE:
                createdPending = true;
                selected = null;
                pendS = Math.min(anchorT, t);
                pendE = Math.max(anchorT, t);
                break;
            case EDGE:
                if (!undoSaved && listener != null) listener.beforeEdit();
                undoSaved = true;
                if (edgeStart) selected.s = t;
                else selected.e = t;
                break;
            case EDGE_PENDING:
                if (edgeStart) pendS = t;
                else pendE = t;
                break;
            default:
                break;
        }
    }

    private void tap(double t) {
        Labels.Label best = null;
        for (Labels.Label l : labels.list)
            if (l.s <= t && t <= l.e && (best == null || l.e - l.s < best.e - best.s)) best = l;
        clearSelection();
        selected = best;
    }

    private void finishEdge() {
        if (selected.s > selected.e) {
            double s = selected.s;
            selected.s = selected.e;
            selected.e = s;
        }
        if (listener != null) listener.changed();
    }

    private void orderPending() {
        if (pendS > pendE) {
            double s = pendS;
            pendS = pendE;
            pendE = s;
        }
    }

    private double clamp(double t) {
        return Math.max(0, Math.min(duration, t));
    }

    // ---- drawing ----------------------------------------------------------------------------

    @Override
    protected void onDraw(Canvas c) {
        int w = getWidth(), h = getHeight();
        c.drawColor(0xff000000);
        text.setColor(0xffdddddd);
        if (spec == null) {
            text.setTextAlign(Paint.Align.CENTER);
            c.drawText("מחשב ספקטרוגרמה…", w / 2f, h / 2f, text);
            return;
        }
        float axis = 20 * dp, plot = h - axis;
        for (int k = 0; k < spec.tiles.length; k++) {
            int f0 = k * Spectrogram.TILE;
            double a = (f0 - 0.5) * spec.hopS, b = (f0 + spec.tiles[k].getWidth() - 0.5) * spec.hopS;
            if (b < t0 || a > t1) continue;
            dst.set(xAt(a), 0, xAt(b), plot);
            c.drawBitmap(spec.tiles[k], null, dst, bmp);
        }
        text.setTextAlign(Paint.Align.LEFT);
        line.setStrokeWidth(1);
        line.setColor(0x40ffffff);
        for (int khz = 5; khz < spec.nyquist / 1000; khz += 5) {
            float y = (float) (plot * (1 - khz * 1000 / spec.nyquist));
            c.drawLine(0, y, w, y, line);
            c.drawText(khz + "k", 3 * dp, y - 3 * dp, text);
        }
        for (Labels.Label l : labels.list) region(c, l.s, l.e, color(l.t), l == selected, l.t, plot);
        if (pendS >= 0) region(c, Math.min(pendS, pendE), Math.max(pendS, pendE), 0xffffffff, true, "", plot);
        if (labels.none && labels.list.isEmpty()) {
            text.setTextAlign(Paint.Align.CENTER);
            text.setTextSize(15 * dp);
            c.drawText("סומן: אין קריאות בקטע", w / 2f, 26 * dp, text);
            text.setTextSize(11 * dp);
        }
        if (playhead >= t0 && playhead <= t1) {
            line.setColor(0xffffeb3b);
            line.setStrokeWidth(2 * dp);
            c.drawLine(xAt(playhead), 0, xAt(playhead), plot, line);
        }
        timeAxis(c, w, plot, axis);
    }

    static int color(String t) {
        return t.equals("c") ? CALL : t.equals("?") ? UNSURE : OTHER;
    }

    private void region(Canvas c, double s, double e, int color, boolean sel, String tag, float plot) {
        if (e < t0 || s > t1) return;
        float x0 = xAt(s), x1 = xAt(e);
        fill.setColor((color & 0x00ffffff) | (sel ? 0x50000000 : 0x38000000));
        c.drawRect(x0, 0, x1, plot, fill);
        line.setColor(color);
        line.setStrokeWidth((sel ? 3 : 1.5f) * dp);
        c.drawLine(x0, 0, x0, plot, line);
        c.drawLine(x1, 0, x1, plot, line);
        if (sel) {
            fill.setColor(color);
            for (float x : new float[]{x0, x1}) c.drawCircle(x, plot / 2, 8 * dp, fill);
        }
        if (!tag.isEmpty()) {
            text.setColor(color);
            text.setTextAlign(Paint.Align.LEFT);
            text.setTextSize(14 * dp);
            c.drawText(tag, Math.max(x0, 0) + 4 * dp, 16 * dp, text);
            text.setTextSize(11 * dp);
            text.setColor(0xffdddddd);
        }
    }

    private void timeAxis(Canvas c, int w, float plot, float axis) {
        fill.setColor(0xff1b1b1b);
        c.drawRect(0, plot, w, plot + axis, fill);
        double[] steps = {0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20};
        double maxTicks = w / (70 * dp), step = 20;
        for (double s : steps) {
            if ((t1 - t0) / s <= maxTicks) {
                step = s;
                break;
            }
        }
        String fmt = step >= 1 ? "%.0f" : step >= 0.1 ? "%.1f" : "%.2f";
        line.setColor(0xff888888);
        line.setStrokeWidth(1);
        text.setTextAlign(Paint.Align.CENTER);
        for (long k = (long) Math.ceil(t0 / step); k * step <= t1; k++) {
            float x = xAt(k * step);
            c.drawLine(x, plot, x, plot + 5 * dp, line);
            c.drawText(String.format(Locale.US, fmt, k * step), x, plot + axis - 4 * dp, text);
        }
    }
}
