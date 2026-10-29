package il.ac.huji.omerlab.marmolabel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

/**
 * One clip's labels, read and written in the Audacity export format of LABELING.md, so
 * tools/labels/import_audacity.py reads phone and Audacity labels alike: one line per label,
 * {@code start<TAB>end<TAB>text} in seconds, text {@code c} (call), {@code ?} (unsure) or
 * {@code n} (non-call event). A checked clip without calls is the single line {@code none}.
 */
final class Labels {
    /** One labelled interval, in seconds from the clip start. */
    static final class Label {
        double s, e;
        String t;

        Label(double s, double e, String t) {
            this.s = s;
            this.e = e;
            this.t = t;
        }
    }

    final List<Label> list = new ArrayList<>();
    /** The clip was checked and holds no calls. */
    boolean none;

    static Labels parse(String text) {
        Labels out = new Labels();
        for (String line : text.split("\n")) {
            if (line.trim().isEmpty() || line.startsWith("\\")) continue;  // Audacity's spectral-selection lines
            String[] p = line.split("\t");
            if (p.length < 2) continue;
            String t = p.length > 2 ? p[2].trim() : "";
            if (t.equals("none")) {
                out.none = true;
                continue;
            }
            try {
                out.list.add(new Label(Double.parseDouble(p[0]), Double.parseDouble(p[1]), t.isEmpty() ? "c" : t));
            } catch (NumberFormatException ignored) {
                // a malformed line is skipped, as import_audacity does
            }
        }
        if (!out.list.isEmpty()) out.none = false;
        return out;
    }

    String format() {
        if (list.isEmpty()) return none ? "0.000000\t0.000000\tnone\n" : "";
        sort();
        StringBuilder b = new StringBuilder();
        for (Label l : list) b.append(String.format(Locale.US, "%.6f\t%.6f\t%s\n", l.s, l.e, l.t));
        return b.toString();
    }

    void sort() {
        Collections.sort(list, (a, b) -> Double.compare(a.s, b.s));
    }

    /** Labelled or explicitly empty: ready to leave the phone. */
    boolean finished() {
        return none || !list.isEmpty();
    }

    Labels copy() {
        Labels c = new Labels();
        for (Label l : list) c.list.add(new Label(l.s, l.e, l.t));
        c.none = none;
        return c;
    }
}
