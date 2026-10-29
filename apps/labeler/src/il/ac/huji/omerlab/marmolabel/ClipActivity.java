package il.ac.huji.omerlab.marmolabel;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Locale;

/**
 * Label one clip: overview strip, zoomable spectrogram, playback, and the label buttons.
 * Every change is saved at once (Store.save); "done" releases the clip to the Mac.
 */
public final class ClipActivity extends Activity implements SpectrogramView.Listener {
    private Store store;
    private String set, clip;
    private Labels labels;
    private Audio audio;
    private SpectrogramView view;
    private OverviewView overview;
    private TextView info;
    private CheckBox done;
    private Button modeBtn, playBtn, speedBtn;
    private final Audio.Player player = new Audio.Player();
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final ArrayDeque<Labels> undo = new ArrayDeque<>();
    private boolean half;
    private float dp;

    private final Runnable tick = new Runnable() {
        @Override
        public void run() {
            double p = player.position();
            view.playhead = p;
            view.invalidate();
            if (p >= 0) ui.postDelayed(this, 30);
            else playBtn.setText("▶ נגן");
        }
    };

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        dp = getResources().getDisplayMetrics().density;
        store = new Store(this);
        set = getIntent().getStringExtra("set");
        clip = getIntent().getStringExtra("clip");
        labels = store.load(set, clip);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setLayoutDirection(View.LAYOUT_DIRECTION_LTR);
        root.setBackgroundColor(0xff121212);
        root.setFitsSystemWindows(true);

        LinearLayout top = row();
        TextView title = new TextView(this);
        title.setText(String.format("%s · %s", clip, getIntent().getStringExtra("wall_time")));
        title.setTextColor(Color.WHITE);
        title.setTextSize(16);
        title.setPadding(px(10), 0, 0, 0);
        top.addView(title, new LinearLayout.LayoutParams(0, -2, 1));
        done = new CheckBox(this);
        done.setText("הקטע גמור");
        done.setTextColor(Color.WHITE);
        done.setChecked(store.isDone(set, clip));
        done.setOnCheckedChangeListener((b, on) -> setDone(on));
        top.addView(done);
        root.addView(top);

        view = new SpectrogramView(this);
        view.labels = labels;
        view.listener = this;
        overview = new OverviewView(this, view);
        root.addView(overview, new LinearLayout.LayoutParams(-1, px(54)));
        root.addView(view, new LinearLayout.LayoutParams(-1, 0, 1));

        info = new TextView(this);
        info.setTextColor(0xffbbbbbb);
        info.setTextSize(12);
        info.setPadding(px(10), px(4), px(10), px(4));
        root.addView(info);

        LinearLayout r1 = row();
        modeBtn = button(r1, "✎ סימון", 0, v -> {
            view.markMode = !view.markMode;
            modeBtn.setText(view.markMode ? "✎ סימון" : "✋ גלילה");
        });
        playBtn = button(r1, "▶ נגן", 0, v -> play());
        speedBtn = button(r1, "מהירות ×1", 0, v -> {
            half = !half;
            speedBtn.setText(half ? "מהירות ×½" : "מהירות ×1");
        });
        button(r1, "↶ בטל", 0, v -> undo());
        root.addView(r1);

        LinearLayout r2 = row();
        button(r2, "c  קריאה", SpectrogramView.CALL, v -> assign("c"));
        button(r2, "?  לא בטוח", SpectrogramView.UNSURE, v -> assign("?"));
        button(r2, "🗑 מחק", 0, v -> delete());
        button(r2, "אין קריאות", 0, v -> toggleNone());
        root.addView(r2);
        setContentView(root);
        updateInfo();

        new Thread(() -> {
            try {
                Audio a = Audio.read(store.audio(set, clip));
                Spectrogram s = Spectrogram.compute(a.x, a.rate);
                ui.post(() -> {
                    audio = a;
                    view.duration = a.duration();
                    view.spec = s;
                    view.setWindow(0, Math.min(6, a.duration()));
                    overview.invalidate();
                });
            } catch (IOException | OutOfMemoryError e) {
                ui.post(() -> {
                    Toast.makeText(this, "לא הצלחתי לפתוח את ההקלטה: " + e.getMessage(), Toast.LENGTH_LONG).show();
                    finish();
                });
            }
        }).start();
    }

    private LinearLayout row() {
        LinearLayout r = new LinearLayout(this);
        r.setOrientation(LinearLayout.HORIZONTAL);
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(px(4), px(2), px(4), px(2));
        return r;
    }

    private Button button(LinearLayout parent, String label, int color, View.OnClickListener l) {
        Button b = new Button(this);
        b.setText(label);
        b.setAllCaps(false);
        b.setTextSize(14);
        if (color != 0) b.setTextColor(color);
        b.setOnClickListener(l);
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(0, px(48), 1);
        p.setMargins(px(2), 0, px(2), 0);
        parent.addView(b, p);
        return b;
    }

    private int px(float v) {
        return Math.round(v * dp);
    }

    // ---- actions ----------------------------------------------------------------------------

    private void play() {
        if (player.playing()) {
            player.stop();
            return;
        }
        if (audio == null) return;
        double a = view.t0, b = view.t1;
        if (view.pendS >= 0) {
            a = view.pendS - 0.05;
            b = view.pendE + 0.05;
        } else if (view.selected != null) {
            a = view.selected.s - 0.05;
            b = view.selected.e + 0.05;
        }
        player.play(audio, a, b, half);
        playBtn.setText("■ עצור");
        ui.post(tick);
    }

    private void assign(String t) {
        if (view.pendS >= 0) {
            if (view.pendE - view.pendS < 0.01) {
                toast("הבחירה קצרה מדי");
                return;
            }
            beforeEdit();
            Labels.Label l = new Labels.Label(view.pendS, view.pendE, t);
            labels.list.add(l);
            labels.none = false;
            view.clearSelection();
            view.selected = l;
            changed();
        } else if (view.selected != null) {
            if (view.selected.t.equals(t)) return;
            beforeEdit();
            view.selected.t = t;
            changed();
        } else {
            toast("גרור על הספקטרוגרמה כדי לבחור אזור, או הקש על סימון קיים");
        }
    }

    private void delete() {
        if (view.pendS >= 0) {
            view.clearSelection();
        } else if (view.selected != null) {
            beforeEdit();
            labels.list.remove(view.selected);
            view.clearSelection();
            changed();
        }
        view.invalidate();
        updateInfo();
    }

    private void toggleNone() {
        if (!labels.list.isEmpty()) {
            toast("יש סימונים בקטע: מחק אותם קודם");
            return;
        }
        beforeEdit();
        labels.none = !labels.none;
        changed();
    }

    private void undo() {
        if (undo.isEmpty()) {
            toast("אין מה לבטל");
            return;
        }
        labels = undo.pop();
        view.labels = labels;
        view.clearSelection();
        changed();
    }

    private void setDone(boolean on) {
        if (on && !labels.finished()) {
            toast("סמן קריאות, או 'אין קריאות', לפני שמסמנים גמור");
            done.setChecked(false);
            return;
        }
        if (on && !store.hasOwn(set, clip)) save();  // a clip labelled on the computer, accepted as is
        store.setDone(set, clip, on);
    }

    // ---- SpectrogramView.Listener -----------------------------------------------------------

    @Override
    public void beforeEdit() {
        undo.push(labels.copy());
        while (undo.size() > 100) undo.removeLast();
    }

    @Override
    public void changed() {
        if (done.isChecked() && !labels.finished()) done.setChecked(false);
        save();
        view.invalidate();
        overview.invalidate();
        updateInfo();
    }

    @Override
    public void viewport() {
        overview.invalidate();
        updateInfo();
    }

    private void save() {
        try {
            store.save(set, clip, labels);
        } catch (IOException e) {
            toast("השמירה נכשלה: " + e.getMessage());
        }
    }

    private void updateInfo() {
        StringBuilder b = new StringBuilder(String.format(Locale.US, "%.2f–%.2f s", view.t0, view.t1));
        double s = view.pendS >= 0 ? view.pendS : view.selected != null ? view.selected.s : -1;
        double e = view.pendS >= 0 ? view.pendE : view.selected != null ? view.selected.e : -1;
        if (s >= 0) b.append(String.format(Locale.US, "   ·   selected %.3f–%.3f (%.0f ms)", s, e, (e - s) * 1000));
        b.append(String.format(Locale.US, "   ·   %d labels", labels.list.size()));
        info.setText(b.toString());
    }

    private void toast(String s) {
        Toast.makeText(this, s, Toast.LENGTH_SHORT).show();
    }

    @Override
    protected void onPause() {
        super.onPause();
        player.stop();
    }
}
