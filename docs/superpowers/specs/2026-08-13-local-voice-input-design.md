# Local voice input — design

**Date:** 2026-08-13
**Status:** Approved, implementing

## Problem

The repo is publicly described as *"A robotic manipulator controlled by voice, wiring an NLP pipeline
to a physical arm for natural-language control."* Neither half is true: there is no speech-to-text
anywhere in the codebase, and the arm is a MuJoCo simulation. The README is honest about this
(`Speech-to-text for voice input` sits unticked in the Roadmap), but the repo description is the
first line a reader sees.

Two ways to close the gap: delete the claim, or make it true. This design makes it true.

## Goal

Speak an instruction, have the arm execute it — with no cloud speech service, no GPU, and no change
to the existing agent or manipulation code.

## Non-goals

- Wake words. Push-to-talk is deliberate and demo-legible.
- Streaming/partial transcription. Utterances here are two seconds long; chunked streaming adds
  machinery for no perceptible gain.
- Speech output. The agent already reports in text.
- Physical hardware. Still a simulation, and the wording must keep saying so.

## Architecture

Voice is a *text source*. It produces the same string the user would have typed, hands it to the
existing queue, and stops being involved. `agent.py`, `skills.py`, and `sim_env.py` are untouched.

```
Mic (hold button)
      │  16 kHz mono float32
      ▼
┌──────────────┐   ┌───────────────┐
│  Recorder    │──▶│  Transcriber  │  faster-whisper, CPU, int8
└──────────────┘   └───────┬───────┘
                           │ text  (or "" if rejected)
                           ▼
                  existing instructions queue ──▶ run_instruction()
```

### `voice.py`

Two units, neither aware of the robot.

**`Recorder`** — wraps `sounddevice.InputStream` at 16 kHz mono float32 (Whisper's native rate, so
no resampling). `start()` opens the stream and clears the buffer; `stop()` closes it and returns the
concatenated `np.ndarray`. The stream callback only appends, so it stays non-blocking.

**`Transcriber`** — wraps `faster_whisper.WhisperModel(name, device="cpu", compute_type="int8")`.
Model name from `$WHISPER_MODEL`, default `base.en`. Loading takes seconds, so it happens once on a
background thread at startup and never on the hot path. `transcribe(audio) -> str`.

### The silence guard

Whisper hallucinates on near-silence — it emits `"Thank you."`, `"you"`, `"."`, and similar
artifacts of its training data. Releasing the mic button in a quiet room would otherwise send a junk
command to a robot arm. Three defences, in order:

1. **Duration floor.** Audio under `MIN_AUDIO_SECONDS` (0.3 s) returns `""` without invoking the
   model at all.
2. **Per-segment confidence.** Drop any segment with `no_speech_prob > 0.6` or
   `avg_logprob < -1.0`.
3. **Hallucination blocklist.** Drop the result if, once stripped of punctuation and lowercased, it
   matches a known phantom phrase.

A rejected transcript returns `""`, and an empty transcript never reaches the queue.

### `chat_ui.py` wiring

A **Mic** button, press-and-hold (`<ButtonPress-1>` / `<ButtonRelease-1>`), plus a status label
showing `loading model… → hold to talk → listening 1.2s → transcribing… → ready`.

Transcription runs on its own short-lived thread, so neither the Tkinter main loop nor the sim
worker blocks — the camera feeds keep updating while Whisper runs.

An accepted transcript goes straight into the existing `instructions` queue and is echoed to the
history as `you (voice): put the green box on the shelf`, so a demo recording shows what was heard,
not just what happened. If the model is still loading when the button is released, the audio is
dropped with a status message rather than queued.

## Error handling

| Failure | Behaviour |
|---|---|
| No input device | Mic button disabled at startup, status says why. Typing still works. |
| Model download/load fails | Status shows the error; button stays disabled; typing unaffected. |
| Audio too short, or silence-guard rejects | Status `no speech detected`; nothing queued. |
| Transcription raises | Status shows the error; nothing queued; UI stays usable. |

Voice failing must never take the text pipeline down with it.

## Testing

Mic hardware and ASR accuracy are not unit-testable; everything around them is. `test_voice.py`
covers, with a fake stream and a stubbed model:

- Buffer assembly, dtype, and empty-`stop()` behaviour
- `start()` clearing a previous take
- Duration floor rejection
- Silence-guard thresholds at their boundaries (`no_speech_prob`, `avg_logprob`)
- Hallucination blocklist matching, including punctuation/case normalisation
- Multi-segment joining and whitespace handling
- Model configured from `$WHISPER_MODEL` with the right device/compute type

Real audio is covered by `python voice.py --smoke`: records three seconds, prints the transcript and
elapsed time. Manual, documented in the README.

## Claim integrity

Shipping this closes the honest version of the `Repo tagline changed off "voice-ready"` evidence
item. The wording must still say **simulation** — voice becoming real does not make the arm
physical. The repo description becomes something like *"Voice-controlled 6-DOF Franka Panda in
MuJoCo — local speech-to-text into a language-conditioned agent."* Still not a VLA. Still not
perception-driven.
