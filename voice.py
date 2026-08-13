"""Local speech-to-text front-end for the robot arm agent.

Records from the microphone and transcribes with faster-whisper running on the
CPU — no cloud speech service and no GPU. The output is just text: it goes into
the same instruction queue a typed command would, so agent.py and skills.py know
nothing about voice.

Whisper hallucinates on near-silence, emitting artifacts of its training data
("Thank you.", "you", "."). Sending those to a robot arm as commands is not
acceptable, so transcribe() drops anything short, low-confidence, or matching a
known phantom phrase, and returns "" instead.

Smoke test with real audio and a real model:
    python voice.py --smoke
"""

import os
import re
import string
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000  # Whisper's native rate — anything else would need resampling
DEFAULT_MODEL = "base.en"
MIN_AUDIO_SECONDS = 0.3

# Confidence gates, applied per segment. Both are inclusive bounds: a segment is
# dropped only when it is strictly worse than the threshold.
MAX_NO_SPEECH_PROB = 0.6
MIN_AVG_LOGPROB = -1.0

# Phrases Whisper emits when handed silence. Matched against the whole
# transcript, not substrings, so "thank you, now put the green box on the shelf"
# is left alone.
HALLUCINATIONS = frozenset(
    [
        "",
        "you",
        "thank you",
        "thank you very much",
        "thanks for watching",
        "thanks for watching!",
        "bye",
        "okay",
        "so",
        "uh",
        "um",
    ]
)


class Recorder:
    """Microphone capture into a single float32 mono buffer at 16 kHz."""

    def __init__(self, device=None):
        self.device = device
        self._stream = None
        self._chunks = []

    def start(self):
        """Open the input stream and begin accumulating, discarding any previous take."""
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            callback=self._on_audio,
        )
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status):
        # Runs on the audio thread — append only, never block.
        self._chunks.append(indata.copy())

    def stop(self):
        """Close the stream and return everything captured as a 1-D float32 array."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if not self._chunks:
            return np.array([], dtype=np.float32)
        audio = np.concatenate(self._chunks, axis=0).reshape(-1)
        self._chunks = []
        return audio.astype(np.float32, copy=False)


class Transcriber:
    """faster-whisper wrapper with a silence guard.

    Model loading takes seconds, so load() is called once on a background thread
    at startup rather than on the first utterance.
    """

    def __init__(self, model_name=None, device=None):
        self.model_name = model_name or os.environ.get("WHISPER_MODEL", DEFAULT_MODEL)
        self.device = device
        self._model = None

    @property
    def ready(self):
        return self._model is not None

    def load(self):
        """Build the Whisper model. Safe to call more than once."""
        if self._model is None:
            self._model = WhisperModel(
                self.model_name, device="cpu", compute_type="int8"
            )
        return self._model

    def transcribe(self, audio):
        """Return the spoken text, or "" if nothing convincing was heard."""
        if self._model is None:
            raise RuntimeError("Whisper model is not loaded; call load() first")
        if audio.size < SAMPLE_RATE * MIN_AUDIO_SECONDS:
            return ""

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,  # greedy: these are two-second command phrases
            vad_filter=True,
        )
        kept = [seg.text.strip() for seg in segments if _is_confident(seg)]
        text = re.sub(r"\s+", " ", " ".join(kept)).strip()
        if _is_hallucination(text):
            return ""
        return text


def _is_confident(segment):
    return (
        segment.no_speech_prob <= MAX_NO_SPEECH_PROB
        and segment.avg_logprob >= MIN_AVG_LOGPROB
    )


def _is_hallucination(text):
    stripped = text.lower().strip().strip(string.punctuation + " ")
    return stripped in HALLUCINATIONS


def _smoke(seconds=3.0):  # pragma: no cover — needs a real mic and a real model
    """Record from the default mic and print what Whisper heard."""
    transcriber = Transcriber()
    print(f"loading {transcriber.model_name}...", flush=True)
    started = time.time()
    transcriber.load()
    print(f"loaded in {time.time() - started:.1f}s", flush=True)

    recorder = Recorder()
    print(f"speak now — recording {seconds:.0f}s...", flush=True)
    recorder.start()
    sd.sleep(int(seconds * 1000))
    audio = recorder.stop()
    print(f"captured {audio.size / SAMPLE_RATE:.1f}s", flush=True)

    started = time.time()
    text = transcriber.transcribe(audio)
    elapsed = time.time() - started
    print(f"transcribed in {elapsed:.2f}s")
    print(f"heard: {text!r}" if text else "heard: (nothing — silence guard rejected it)")


if __name__ == "__main__":  # pragma: no cover
    if "--smoke" in sys.argv:
        _smoke()
    else:
        print(__doc__)
