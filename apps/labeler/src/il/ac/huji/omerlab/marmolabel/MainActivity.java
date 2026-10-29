package il.ac.huji.omerlab.marmolabel;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * The clip list, with download (from the Mac, on Wi-Fi) and "send to computer".
 *
 * <p>On every return to this screen, if the phone is on Wi-Fi and a Mac address is set, the
 * list and any missing audio are fetched quietly; when the Mac is not reachable the app
 * simply works offline on what it has.
 */
public final class MainActivity extends Activity {
    private Store store;
    private EditText address, code;
    private TextView status;
    private LinearLayout list;
    private final Handler ui = new Handler(Looper.getMainLooper());
    private volatile boolean busy;
    private float dp;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        dp = getResources().getDisplayMetrics().density;
        store = new Store(this);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setLayoutDirection(View.LAYOUT_DIRECTION_RTL);
        root.setBackgroundColor(0xff121212);
        root.setPadding(px(12), px(8), px(12), 0);
        root.setFitsSystemWindows(true);

        TextView title = text("MarmoLabel · סימון קריאות", 20, Color.WHITE);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        root.addView(title);

        LinearLayout server = new LinearLayout(this);
        address = new EditText(this);
        address.setHint("כתובת המחשב, למשל 192.168.1.20:8766");
        address.setText(store.server());
        address.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_URI);
        address.setTextDirection(View.TEXT_DIRECTION_LTR);
        address.setTextColor(Color.WHITE);
        address.setHintTextColor(0xff777777);
        server.addView(address, new LinearLayout.LayoutParams(0, -2, 3));
        code = new EditText(this);
        code.setHint("קוד");
        code.setText(store.code());
        code.setInputType(InputType.TYPE_CLASS_NUMBER);
        code.setTextColor(Color.WHITE);
        code.setHintTextColor(0xff777777);
        server.addView(code, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(server);

        LinearLayout buttons = new LinearLayout(this);
        Button down = new Button(this);
        down.setText("⬇ הורד הקלטות");
        down.setAllCaps(false);
        down.setOnClickListener(v -> downloadClicked());
        buttons.addView(down, new LinearLayout.LayoutParams(0, -2, 1));
        Button up = new Button(this);
        up.setText("⬆ שלח למחשב");
        up.setAllCaps(false);
        up.setOnClickListener(v -> upload());
        buttons.addView(up, new LinearLayout.LayoutParams(0, -2, 1));
        root.addView(buttons);

        status = text("", 13, 0xffbbbbbb);
        status.setPadding(0, px(4), 0, px(6));
        root.addView(status);

        ScrollView scroll = new ScrollView(this);
        list = new LinearLayout(this);
        list.setOrientation(LinearLayout.VERTICAL);
        scroll.addView(list);
        root.addView(scroll, new LinearLayout.LayoutParams(-1, 0, 1));
        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        refresh();
        if (store.onWifi() && !store.server().isEmpty()) sync(true);
        else if (!store.onWifi()) status.setText("לא בוויפי: עובדים על מה שכבר הורד.");
    }

    private TextView text(String s, float size, int color) {
        TextView t = new TextView(this);
        t.setText(s);
        t.setTextSize(size);
        t.setTextColor(color);
        return t;
    }

    private int px(float v) {
        return Math.round(v * dp);
    }

    // ---- list -------------------------------------------------------------------------------

    private void refresh() {
        list.removeAllViews();
        JSONObject m = store.manifest();
        if (m == null) {
            list.addView(text("עוד אין רשימת הקלטות.\n\nעל המחשב הרץ:\npython tools/labels/phone_server.py\n"
                    + "הזן כאן את הכתובת והקוד שהוא מדפיס, ולחץ 'הורד הקלטות' (בוויפי, באותה רשת כמו המחשב).",
                    15, 0xffdddddd));
            return;
        }
        try {
            JSONArray sets = m.getJSONArray("sets");
            for (int i = 0; i < sets.length(); i++) {
                JSONObject set = sets.getJSONObject(i);
                String name = set.getString("name");
                JSONArray clips = set.getJSONArray("clips");
                int finished = 0;
                for (int j = 0; j < clips.length(); j++) if (labelled(name, clips.getJSONObject(j).getString("id"))) finished++;
                TextView h = text(String.format("%s   ·   %d/%d מסומנים", name, finished, clips.length()), 17, 0xff33d1c4);
                h.setTypeface(Typeface.DEFAULT_BOLD);
                h.setPadding(0, px(14), 0, px(4));
                list.addView(h);
                for (int j = 0; j < clips.length(); j++) list.addView(clipRow(name, clips.getJSONObject(j)));
            }
        } catch (JSONException e) {
            status.setText("רשימה פגומה: " + e.getMessage());
        }
    }

    private boolean labelled(String set, String clip) {
        return store.hasOwn(set, clip) ? store.isDone(set, clip) : store.hasComputer(set, clip);
    }

    private View clipRow(String set, JSONObject clip) throws JSONException {
        String id = clip.getString("id"), wall = clip.getString("wall_time").replace('T', ' ');
        boolean have = !store.needsAudio(set, clip);
        LinearLayout r = new LinearLayout(this);
        r.setGravity(Gravity.CENTER_VERTICAL);
        r.setPadding(px(8), px(10), px(8), px(10));
        r.setBackgroundColor(0xff1e1e1e);
        TextView name = text(id, 16, have ? Color.WHITE : 0xff777777);
        name.setTypeface(Typeface.MONOSPACE);
        r.addView(name, new LinearLayout.LayoutParams(px(80), -2));
        r.addView(text(wall.substring(0, Math.min(16, wall.length())), 13, 0xff999999), new LinearLayout.LayoutParams(0, -2, 1));
        String[] st = state(set, id, have);
        r.addView(text(st[0], 13, (int) Long.parseLong(st[1], 16)));
        r.setOnClickListener(v -> {
            if (!store.needsAudio(set, clip)) {
                startActivity(new Intent(this, ClipActivity.class).putExtra("set", set).putExtra("clip", id)
                        .putExtra("wall_time", wall));
            } else {
                Toast.makeText(this, "ההקלטה עוד לא הורדה", Toast.LENGTH_SHORT).show();
            }
        });
        LinearLayout.LayoutParams p = new LinearLayout.LayoutParams(-1, -2);
        p.setMargins(0, px(2), 0, px(2));
        r.setLayoutParams(p);
        return r;
    }

    /** Status text and colour (hex ARGB) of one clip. */
    private String[] state(String set, String id, boolean have) {
        if (!have) return new String[]{"לא הורד", "ff777777"};
        if (store.hasOwn(set, id)) {
            Labels l = store.load(set, id);
            String n = l.none && l.list.isEmpty() ? "אין קריאות" : l.list.size() + " סימונים";
            if (!store.isDone(set, id)) return new String[]{"✎ בעבודה · " + n, "ffffb74d"};
            return store.isSent(set, id) ? new String[]{"✓ נשלח · " + n, "ff66bb6a"} : new String[]{"✓ גמור, לא נשלח · " + n, "ff33d1c4"};
        }
        if (store.hasComputer(set, id)) {
            Labels l = store.load(set, id);
            return new String[]{"מהמחשב · " + (l.none && l.list.isEmpty() ? "אין קריאות" : l.list.size() + " סימונים"), "ff9e9e9e"};
        }
        return new String[]{"", "ff000000"};
    }

    // ---- network ----------------------------------------------------------------------------

    private void saveServer() {
        store.setServer(address.getText().toString(), code.getText().toString());
        address.setText(store.server());
    }

    private void downloadClicked() {
        saveServer();
        if (store.onWifi()) {
            sync(false);
            return;
        }
        new AlertDialog.Builder(this).setMessage("הטלפון לא מחובר לוויפי. ההורדה כולה היא כמה מאות MB. להוריד בכל זאת?")
                .setPositiveButton("הורד", (d, w) -> sync(false)).setNegativeButton("ביטול", null).show();
    }

    /** Fetch the list, then every clip whose audio is missing or incomplete. */
    private void sync(boolean quiet) {
        if (busy) return;
        busy = true;
        status.setText("מתחבר למחשב…");
        new Thread(() -> {
            try {
                JSONObject m = store.fetchManifest();
                List<String[]> todo = new ArrayList<>();
                JSONArray sets = m.getJSONArray("sets");
                for (int i = 0; i < sets.length(); i++) {
                    JSONObject set = sets.getJSONObject(i);
                    JSONArray clips = set.getJSONArray("clips");
                    for (int j = 0; j < clips.length(); j++)
                        if (store.needsAudio(set.getString("name"), clips.getJSONObject(j)))
                            todo.add(new String[]{set.getString("name"), clips.getJSONObject(j).getString("id")});
                }
                ui.post(this::refresh);
                for (int k = 0; k < todo.size(); k++) {
                    final String msg = String.format("מוריד %d/%d · %s", k + 1, todo.size(), todo.get(k)[1]);
                    ui.post(() -> status.setText(msg));
                    store.downloadAudio(todo.get(k)[0], todo.get(k)[1]);
                    if (k % 5 == 4) ui.post(this::refresh);
                }
                ui.post(() -> {
                    status.setText("✓ כל ההקלטות על הטלפון");
                    refresh();
                });
            } catch (IOException | JSONException e) {
                ui.post(() -> {
                    status.setText((quiet ? "המחשב לא זמין, עובדים במצב לא מקוון. " : "לא הצלחתי להוריד: ") + e.getMessage());
                    refresh();
                });
            } finally {
                busy = false;
            }
        }).start();
    }

    /** Post every finished clip that changed since it was last sent. */
    private void upload() {
        saveServer();
        if (busy) return;
        JSONObject m = store.manifest();
        if (m == null) return;
        List<String[]> todo = new ArrayList<>();
        try {
            JSONArray sets = m.getJSONArray("sets");
            for (int i = 0; i < sets.length(); i++) {
                String name = sets.getJSONObject(i).getString("name");
                JSONArray clips = sets.getJSONObject(i).getJSONArray("clips");
                for (int j = 0; j < clips.length(); j++) {
                    String id = clips.getJSONObject(j).getString("id");
                    if (store.hasOwn(name, id) && store.isDone(name, id) && !store.isSent(name, id)) todo.add(new String[]{name, id});
                }
            }
        } catch (JSONException e) {
            return;
        }
        if (todo.isEmpty()) {
            status.setText("אין קטעים גמורים שלא נשלחו. (קטע נשלח רק אחרי שסימנת 'הקטע גמור'.)");
            return;
        }
        busy = true;
        status.setText("שולח " + todo.size() + " קטעים…");
        new Thread(() -> {
            int sent = 0;
            String error = null;
            for (String[] t : todo) {
                try {
                    store.upload(t[0], t[1]);
                    sent++;
                } catch (IOException e) {
                    error = t[1] + ": " + e.getMessage();
                    break;
                }
            }
            final String msg = error == null ? "✓ נשלחו " + sent + " קטעים למחשב" : "נשלחו " + sent + ", ואז נכשל: " + error;
            ui.post(() -> {
                status.setText(msg);
                refresh();
            });
            busy = false;
        }).start();
    }
}
