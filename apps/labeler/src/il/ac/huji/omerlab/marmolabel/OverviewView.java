package il.ac.huji.omerlab.marmolabel;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.MotionEvent;
import android.view.View;

/** The whole clip in one strip: labels as coloured bars, the zoomed window as a frame. Touch to jump there. */
final class OverviewView extends View {
    private final SpectrogramView main;
    private final float dp;
    private final Paint bmp = new Paint(Paint.FILTER_BITMAP_FLAG), fill = new Paint(), stroke = new Paint();
    private final RectF dst = new RectF();

    OverviewView(Context c, SpectrogramView main) {
        super(c);
        this.main = main;
        dp = c.getResources().getDisplayMetrics().density;
        stroke.setStyle(Paint.Style.STROKE);
        stroke.setStrokeWidth(2 * dp);
        stroke.setColor(0xffffffff);
    }

    private float x(double t) {
        return (float) (t / main.duration * getWidth());
    }

    @Override
    protected void onDraw(Canvas c) {
        int w = getWidth(), h = getHeight();
        c.drawColor(0xff000000);
        if (main.spec == null) return;
        float bar = 6 * dp;
        dst.set(0, 0, w, h - bar);
        c.drawBitmap(main.spec.overview, null, dst, bmp);
        for (Labels.Label l : main.labels.list) {
            fill.setColor(SpectrogramView.color(l.t));
            c.drawRect(x(l.s), h - bar, Math.max(x(l.e), x(l.s) + 2 * dp), h, fill);
        }
        float a = x(main.t0), b = x(main.t1);
        fill.setColor(0x80000000);
        c.drawRect(0, 0, a, h, fill);
        c.drawRect(b, 0, w, h, fill);
        c.drawRect(a, dp, Math.max(b, a + 2 * dp), h - dp, stroke);
    }

    @Override
    public boolean onTouchEvent(MotionEvent e) {
        int action = e.getActionMasked();
        if (main.spec != null && (action == MotionEvent.ACTION_DOWN || action == MotionEvent.ACTION_MOVE)) {
            double t = e.getX() / getWidth() * main.duration, w = main.t1 - main.t0;
            main.setWindow(t - w / 2, t + w / 2);
            invalidate();
        }
        return true;
    }
}
