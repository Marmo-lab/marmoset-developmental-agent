package il.ac.huji.omerlab.marmolabel;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.content.SharedPreferences;
import android.net.ConnectivityManager;
import android.net.NetworkCapabilities;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;

/**
 * Everything the app keeps: the clip list and audio downloaded from the Mac
 * (tools/labels/phone_server.py), and the labels.
 *
 * <p>Labels exist in two versions per clip. "computer" is what the Mac had at the last
 * download (e.g. labelled earlier in Audacity), shown when the phone has nothing of its own;
 * "own" is the phone's version, written on every edit. A clip leaves the phone only once
 * marked done: it is then copied to Download/MarmoLabel/&lt;set&gt;/&lt;clip&gt;.txt on every save,
 * and posted to the Mac by "send to computer".
 */
final class Store {
    private final Context ctx;
    private final File root;
    private final SharedPreferences prefs;

    Store(Context c) {
        ctx = c.getApplicationContext();
        root = ctx.getFilesDir();
        prefs = ctx.getSharedPreferences("marmolabel", Context.MODE_PRIVATE);
    }

    // ---- settings and per-clip state ------------------------------------------------------

    String server() {
        return prefs.getString("server", "");
    }

    String code() {
        return prefs.getString("code", "");
    }

    void setServer(String server, String code) {
        prefs.edit().putString("server", normalise(server)).putString("code", code.trim()).apply();
    }

    /** "10.0.0.5" -> "http://10.0.0.5:8766" (the server's default port). */
    static String normalise(String s) {
        s = s.trim();
        if (s.isEmpty()) return s;
        if (!s.contains("://")) s = "http://" + s;
        while (s.endsWith("/")) s = s.substring(0, s.length() - 1);
        if (s.indexOf(':', s.indexOf("://") + 3) < 0) s += ":8766";
        return s;
    }

    boolean isDone(String set, String clip) {
        return prefs.getBoolean("done:" + set + "/" + clip, false);
    }

    void setDone(String set, String clip, boolean done) {
        prefs.edit().putBoolean("done:" + set + "/" + clip, done).putBoolean("sent:" + set + "/" + clip, false).apply();
        if (done) exportToDownloads(set, clip, load(set, clip).format());
    }

    boolean isSent(String set, String clip) {
        return prefs.getBoolean("sent:" + set + "/" + clip, false);
    }

    boolean onWifi() {
        ConnectivityManager cm = ctx.getSystemService(ConnectivityManager.class);
        NetworkCapabilities nc = cm.getNetworkCapabilities(cm.getActiveNetwork());
        return nc != null && nc.hasTransport(NetworkCapabilities.TRANSPORT_WIFI);
    }

    // ---- files ------------------------------------------------------------------------------

    JSONObject manifest() {
        File f = new File(root, "manifest.json");
        try {
            return f.exists() ? new JSONObject(read(f)) : null;
        } catch (IOException | JSONException e) {
            return null;
        }
    }

    File audio(String set, String clip) {
        return new File(root, "audio/" + set + "/" + clip + ".wav");
    }

    private File own(String set, String clip) {
        return new File(root, "labels/" + set + "/" + clip + ".txt");
    }

    private File computer(String set, String clip) {
        return new File(root, "computer/" + set + "/" + clip + ".txt");
    }

    boolean hasOwn(String set, String clip) {
        return own(set, clip).exists();
    }

    boolean hasComputer(String set, String clip) {
        return computer(set, clip).exists();
    }

    Labels load(String set, String clip) {
        File f = own(set, clip).exists() ? own(set, clip) : computer(set, clip);
        try {
            return f.exists() ? Labels.parse(read(f)) : new Labels();
        } catch (IOException e) {
            return new Labels();
        }
    }

    void save(String set, String clip, Labels labels) throws IOException {
        String text = labels.format();
        write(own(set, clip), text);
        prefs.edit().putBoolean("sent:" + set + "/" + clip, false).apply();
        if (isDone(set, clip)) exportToDownloads(set, clip, text);
    }

    /** Copy a finished clip's labels to the phone's Download folder (the manual route to the Mac). */
    private void exportToDownloads(String set, String clip, String text) {
        ContentResolver cr = ctx.getContentResolver();
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        String key = "uri:" + set + "/" + clip;
        String saved = prefs.getString(key, null);
        if (saved != null) {
            try (OutputStream o = cr.openOutputStream(Uri.parse(saved), "wt")) {
                if (o != null) {
                    o.write(bytes);
                    return;
                }
            } catch (Exception ignored) {
                // the user deleted or moved it: create it again below
            }
        }
        ContentValues v = new ContentValues();
        v.put(MediaStore.MediaColumns.DISPLAY_NAME, clip + ".txt");
        v.put(MediaStore.MediaColumns.MIME_TYPE, "text/plain");
        v.put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/MarmoLabel/" + set);
        Uri u = cr.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, v);
        if (u == null) return;
        try (OutputStream o = cr.openOutputStream(u)) {
            if (o != null) o.write(bytes);
            prefs.edit().putString(key, u.toString()).apply();
        } catch (IOException ignored) {
            // the label is safe in the app; only the Download copy failed
        }
    }

    // ---- network ----------------------------------------------------------------------------

    private HttpURLConnection open(String path) throws IOException {
        if (server().isEmpty()) throw new IOException("חסרה כתובת המחשב");
        URL url = new URL(server() + path + "?t=" + URLEncoder.encode(code(), "UTF-8"));
        HttpURLConnection c = (HttpURLConnection) url.openConnection();
        c.setConnectTimeout(5000);
        c.setReadTimeout(60000);
        return c;
    }

    private static void check(HttpURLConnection c) throws IOException {
        int status = c.getResponseCode();
        if (status == 403) throw new IOException("קוד שגוי");
        if (status != 200) {
            InputStream err = c.getErrorStream();
            throw new IOException("שגיאה " + status + (err == null ? "" : ": " + new String(readAll(err), StandardCharsets.UTF_8)));
        }
    }

    /** Fetch the clip list, and the Mac's current labels of every clip. */
    JSONObject fetchManifest() throws IOException, JSONException {
        HttpURLConnection c = open("/api/manifest");
        try {
            check(c);
            String text = new String(readAll(c.getInputStream()), StandardCharsets.UTF_8);
            JSONObject m = new JSONObject(text);
            JSONArray sets = m.getJSONArray("sets");
            java.util.Set<String> listed = new java.util.HashSet<>();
            for (int i = 0; i < sets.length(); i++) {
                JSONObject set = sets.getJSONObject(i);
                JSONArray clips = set.getJSONArray("clips");
                for (int j = 0; j < clips.length(); j++) {
                    JSONObject clip = clips.getJSONObject(j);
                    listed.add(audio(set.getString("name"), clip.getString("id")).getPath());
                    File f = computer(set.getString("name"), clip.getString("id"));
                    if (clip.isNull("labels")) {
                        //noinspection ResultOfMethodCallIgnored
                        f.delete();
                    } else {
                        write(f, clip.getString("labels"));
                    }
                }
            }
            write(new File(root, "manifest.json"), text);
            // Audio of clips the Mac no longer lists (labelled, or another set) is dropped to free
            // space; labels are never deleted.
            File[] dirs = new File(root, "audio").listFiles();
            for (File d : dirs == null ? new File[0] : dirs) {
                File[] files = d.listFiles();
                for (File f : files == null ? new File[0] : files)
                    //noinspection ResultOfMethodCallIgnored
                    if (!listed.contains(f.getPath())) f.delete();
            }
            return m;
        } finally {
            c.disconnect();
        }
    }

    boolean needsAudio(String set, JSONObject clip) {
        File f = audio(set, clip.optString("id"));
        return !f.exists() || f.length() != clip.optLong("bytes");
    }

    void downloadAudio(String set, String clip) throws IOException {
        HttpURLConnection c = open("/audio/" + set + "/" + clip);
        try {
            check(c);
            File f = audio(set, clip);
            File tmp = new File(f.getPath() + ".part");
            //noinspection ResultOfMethodCallIgnored
            f.getParentFile().mkdirs();
            try (InputStream in = c.getInputStream(); OutputStream out = new FileOutputStream(tmp)) {
                byte[] buf = new byte[1 << 16];
                for (int n; (n = in.read(buf)) > 0; ) out.write(buf, 0, n);
            }
            if (!tmp.renameTo(f)) throw new IOException("cannot write " + f);
        } finally {
            c.disconnect();
        }
    }

    void upload(String set, String clip) throws IOException {
        byte[] body = load(set, clip).format().getBytes(StandardCharsets.UTF_8);
        HttpURLConnection c = open("/api/labels/" + set + "/" + clip);
        try {
            c.setDoOutput(true);
            c.setRequestMethod("POST");
            c.setRequestProperty("Content-Type", "text/plain; charset=utf-8");
            c.setFixedLengthStreamingMode(body.length);
            try (OutputStream o = c.getOutputStream()) {
                o.write(body);
            }
            check(c);
            prefs.edit().putBoolean("sent:" + set + "/" + clip, true).apply();
        } finally {
            c.disconnect();
        }
    }

    // ---- helpers ----------------------------------------------------------------------------

    private static String read(File f) throws IOException {
        return new String(Files.readAllBytes(f.toPath()), StandardCharsets.UTF_8);
    }

    private static void write(File f, String text) throws IOException {
        //noinspection ResultOfMethodCallIgnored
        f.getParentFile().mkdirs();
        File tmp = new File(f.getPath() + ".tmp");
        try (OutputStream o = new FileOutputStream(tmp)) {
            o.write(text.getBytes(StandardCharsets.UTF_8));
        }
        if (!tmp.renameTo(f)) throw new IOException("cannot write " + f);
    }

    private static byte[] readAll(InputStream in) throws IOException {
        ByteArrayOutputStream b = new ByteArrayOutputStream();
        byte[] buf = new byte[1 << 14];
        for (int n; (n = in.read(buf)) > 0; ) b.write(buf, 0, n);
        return b.toByteArray();
    }
}
