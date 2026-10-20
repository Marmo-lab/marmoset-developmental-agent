# How the 1kD (Hasson) pipeline gets speech out of home recordings

Sources:
- Raviv et al., "The First 1,000 Days (1kD) Project", bioRxiv 10.64898/2026.03.19.712982 (data paper, methods
  "Algorithms deployed in the pipeline").
- Raviv, Hasenfratz, Hasson et al., "The Richness of Experience" / "The First 1000 Days: An Agent-Based Model of
  Early Language Acquisition", bioRxiv 10.64898/2026.03.28.715023 (v1/v2; methods "Corpus construction").
- The public code (`references/hasson_dev`, commit f961bc4) starts after all of this: its input is
  `transcripts.csv` + 16 kHz WAVs. None of the extraction steps below are in it.

## Pipeline (per home, per minute)
1. **Speech detection** (high recall). Each 1-minute file is cut into non-overlapping 4 s windows and passed to a
   wav2vec 2.0 transcriber. A window is "speech" if any meaningful transcription comes out. Speech seconds =
   windows × 4 s, and a minute is "speech present" if > 0.
2. **Motion detection** (video, high recall): frame differencing per camera, aggregated per room.
3. **Activity gate:** (motion > 0 and speech > 0) or ≥ 8 s of speech in any area. Only active minutes go on.
4. **Infant presence:** the middle frame (t = 30 s) of every camera goes to GPT-4o with yes/no questions (adult
   female, adult male, baby, older child). Scores are weighted per room (1.0 inside, 0.5 viewing from outside) and
   smoothed over ±4 min with weights 0.75 / 0.5 / 0.25 / 0.1.
5. **Channel selection:** each candidate microphone in a room is denoised with Demucs (Défossez, Synnaeve & Adi
   2020), and the one with the highest enhanced-speech energy is selected. The enhancer is used only for this choice.
6. **Transcription:** WhisperX large-v2 on 30 s chunks (temperature 0, VAD onset 0.20 / offset 0.363). The text is
   then normalised: fillers removed, punctuation stripped, spelling normalised.
7. **Assignment:** a minute enters the child's corpus if some room's smoothed infant score passes a threshold. It
   gets the transcript of that room. A lower threshold means more recall and more noise.
8. **Quality filters:** transcripts with a high zlib compression ratio (hallucinated repetition, 7.9%) are
   dropped, as are transcripts of ≤ 5 words.

Reported end-to-end, from 100 random minutes per family, 8 families:
- speech detection: P 0.94 / R 0.97
- infant + speech: P 0.76 / R 0.90
- WhisperX vs human transcripts: WER 0.51

## What the agent actually hears: TTS, not the recordings
The agent paper's methods are explicit:
- The far-field audio "contained substantial background noise and reverberation".
- The transcripts are split into sentences (spaCy), concatenated into segments of about 34 words, and
  **re-synthesised with IndexTTS**. Voices: 8 adult female voice prompts per home, with 30% of segments in a voice
  matched to the mother.
- Each synthesised segment is checked by re-transcribing it with WhisperX, and kept only if the normalised
  Levenshtein distance is < 0.1, with up to 5 retries.
- The training audio is 16 kHz clean TTS speech, with no noise, reverberation, speaker variation or prosody from
  the home.

## What this means for us
- There is no marmoset equivalent of steps 6–8. We cannot transcribe and re-synthesise calls, so our agent trains
  on the real far-field recordings, noise included. Our front-end has to do the job that TTS does for them.
- Their unit-to-phoneme purity (V = 512 vs about 39 phones) was measured on clean TTS speech from 8 voices. A
  marmoset unit inventory learned from noisy colony audio is not expected to be as clean, so D24 targets should be
  set with that in mind.
- What does transfer:
  - High-recall detection first, then cheaper filters (our spectral detector, then the tonality rejection).
  - Minute-level gating before expensive steps.
  - Choosing the best channel by denoised in-band energy (cf. our `best_snr` / `max_energy` strategies).
  - Quality filters with a precision/recall knob (our chunk-P constraint).
- There are two options worth testing later as ablations, not adopting by default:
  - (a) a denoiser used only to select the channel, as they do;
  - (b) a denoised training-audio arm, the closest analogue of their clean input.
- They train the uLM on deduplicated units plus durations; we never deduplicate (guardrail 1).
- The agent paper weights next-unit prediction 10 against masked prediction 1. The public config elides the uLM
  weight (`ulm_nll: ...`). See OPEN_DECISIONS D25.
