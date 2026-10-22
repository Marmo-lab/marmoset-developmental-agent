# Hand-labelling marmoset calls

The goal is a small, trustworthy set of call boundaries, so that every detector and
channel strategy can finally be scored with real precision and recall. The paper's
0.975 / 0.78 came from a labelled set that no longer exists.

You mark **where** calls are. You do **not** need to know call types.

## What you get

```
labeling/
  clips/clip_01.wav ... clip_30.wav   one minute each, 48 kHz mono, volume-normalised
  clips.csv                           where each clip came from (don't edit)
  guide.png                           what calls look like vs. noise
  labels/                             put your exported label files here
```

## Setup (once)

1. Install **Audacity** (free): https://www.audacityteam.org
2. Open `guide.png` and look at it for a minute. Calls are clean tones or sweeps,
   mostly between **5 and 12 kHz**, often with a fainter copy (harmonic) above.
   Clicks and cage noise are short **vertical** smears that cover many frequencies at
   once.

## Labelling one clip (about 5-8 minutes)

1. **File > Open** `clips/clip_01.wav`.
2. Click the track name (the small arrow next to "clip_01") > **Spectrogram**.
3. Same menu > **Spectrogram Settings**:
   - Scale: **Linear**, Min frequency **0**, Max frequency **24000**
   - Gain **20 dB**, Range **80 dB** (raise Gain if the image is too dark)
   - Window size **1024**
4. Zoom so roughly **10 seconds** fill the screen (**View > Zoom In**, or `Cmd+1`),
   and scroll from left to right.
5. For each call: **drag** across it from start to end, press **`Cmd+B`**, type
   **`c`**, press **Enter**. Press **Space** to listen whenever you're unsure.
6. When the whole minute is done: **File > Export > Export Labels...** and save as
   `labels/clip_01.txt` (same number as the clip).

You can stop at any time; each clip is independent. Doing them in several sittings is
fine and probably better.

## The rules (please keep them consistent)

| situation | what to do |
|---|---|
| A clear call | label `c` |
| You think it might be a call but aren't sure | label `?` |
| A sequence of syllables with gaps **shorter than ~0.15 s** (e.g. a twitter) | **one** label over the whole sequence |
| Syllables separated by **longer** gaps | separate labels |
| Two animals calling at once, overlapping | one label covering both, `c` |
| A call cut off by the start or end of the clip | label it anyway |
| A loud, obvious non-call (door, cage rattle, human voice) | optional: label `n` |
| The whole clip has no calls at all | one label `none` anywhere, then export |
| Something visible in the spectrogram that you can't hear as a call | don't label it: the labeller's ear decides (anything unlabelled counts as noise) |

- Put each edge where the call **visibly starts and ends** in the spectrogram; zoom in on the edge if needed. Since
  2026-10-22 the pipeline outputs individual calls with exact times, so edges matter. Within about 20-30 ms is
  plenty. (The first 45 clips were labelled to ~50 ms; that is fine.)
- The 0.15 s rule is measured on your first 45 clips: gaps inside your labels were at most 0.21 s (median 0.03 s),
  and gaps between labels at least 0.16 s for 95% of them (median 0.59 s).
- **Label every call in the clip.** A missed call counts against the detector as much
  as a false one, so an unlabelled call makes the scores wrong.
- Please label **without** looking at any detector output, so the labels measure the
  detectors instead of echoing them.
- Night clips are often completely empty. "No calls" is a valid, useful answer (it tells
  us how many false alarms the detectors raise in silence). Audacity won't export an
  empty label track, so: click anywhere in the clip, press `Cmd+B`, type **`none`**,
  and export as usual. That one label means "I checked this clip, nothing here".

## When you're done (or partly done)

Tell Claude. The importer reads whatever is in `labeling/labels/`:

```bash
python tools/labels/import_audacity.py labeling/clips.csv labeling/labels -o data/labels/handlabels.csv
```
