"""Tests for the local speech-to-text front-end.

Microphone hardware and Whisper's actual accuracy are not unit-testable, so the
stream and the model are both faked here. What is tested is everything around
them: buffer assembly, and the silence guard that stops Whisper's near-silence
hallucinations ("Thank you.", "you") from being sent to a robot arm as commands.
"""

import numpy as np
import pytest

import voice


class FakeStream:
    """Stand-in for sounddevice.InputStream that replays canned chunks."""

    def __init__(self, chunks=(), **kwargs):
        self.kwargs = kwargs
        self._chunks = list(chunks)
        self.started = False
        self.closed = False

    def start(self):
        self.started = True
        callback = self.kwargs["callback"]
        for chunk in self._chunks:
            callback(chunk, len(chunk), None, None)

    def close(self):
        self.closed = True


class FakeSegment:
    def __init__(self, text, no_speech_prob=0.0, avg_logprob=-0.1):
        self.text = text
        self.no_speech_prob = no_speech_prob
        self.avg_logprob = avg_logprob


class FakeModel:
    """Stand-in for faster_whisper.WhisperModel."""

    def __init__(self, name, device=None, compute_type=None):
        self.name = name
        self.device = device
        self.compute_type = compute_type
        self.segments = []
        self.calls = 0

    def transcribe(self, audio, **kwargs):
        self.calls += 1
        self.audio = audio
        self.kwargs = kwargs
        return list(self.segments), object()


def make_recorder(monkeypatch, chunks=()):
    created = {}

    def fake_input_stream(**kwargs):
        stream = FakeStream(chunks, **kwargs)
        created["stream"] = stream
        return stream

    monkeypatch.setattr(voice.sd, "InputStream", fake_input_stream)
    return voice.Recorder(), created


def make_transcriber(segments, monkeypatch, **kwargs):
    transcriber = voice.Transcriber(**kwargs)
    model = FakeModel("fake")
    model.segments = segments
    transcriber._model = model
    return transcriber, model


# --- Recorder ------------------------------------------------------------


def test_recorder_concatenates_chunks_in_order(monkeypatch):
    chunks = [
        np.array([[0.1], [0.2]], dtype=np.float32),
        np.array([[0.3]], dtype=np.float32),
    ]
    recorder, _ = make_recorder(monkeypatch, chunks)

    recorder.start()
    audio = recorder.stop()

    assert audio.tolist() == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)


def test_recorder_returns_float32_mono(monkeypatch):
    chunks = [np.array([[0.5], [0.5]], dtype=np.float32)]
    recorder, _ = make_recorder(monkeypatch, chunks)

    recorder.start()
    audio = recorder.stop()

    assert audio.dtype == np.float32
    assert audio.ndim == 1


def test_recorder_opens_stream_at_whisper_sample_rate(monkeypatch):
    recorder, created = make_recorder(monkeypatch)

    recorder.start()
    recorder.stop()

    assert created["stream"].kwargs["samplerate"] == voice.SAMPLE_RATE
    assert created["stream"].kwargs["channels"] == 1


def test_recorder_stop_without_audio_returns_empty(monkeypatch):
    recorder, _ = make_recorder(monkeypatch)

    recorder.start()
    audio = recorder.stop()

    assert audio.size == 0


def test_recorder_stop_closes_the_stream(monkeypatch):
    recorder, created = make_recorder(monkeypatch)

    recorder.start()
    recorder.stop()

    assert created["stream"].closed


def test_recorder_start_discards_the_previous_take(monkeypatch):
    chunks = [np.array([[0.9]], dtype=np.float32)]
    recorder, _ = make_recorder(monkeypatch, chunks)

    recorder.start()
    recorder.stop()
    recorder.start()
    second = recorder.stop()

    assert second.tolist() == pytest.approx([0.9], abs=1e-6)


def test_recorder_stop_when_never_started_returns_empty(monkeypatch):
    recorder, _ = make_recorder(monkeypatch)

    assert recorder.stop().size == 0


# --- Transcriber: duration floor ----------------------------------------


def test_audio_below_duration_floor_is_rejected_without_calling_model(monkeypatch):
    transcriber, model = make_transcriber(
        [FakeSegment("pick up the red box")], monkeypatch
    )
    too_short = np.zeros(int(voice.SAMPLE_RATE * 0.1), dtype=np.float32)

    assert transcriber.transcribe(too_short) == ""
    assert model.calls == 0


def test_empty_audio_is_rejected(monkeypatch):
    transcriber, model = make_transcriber([FakeSegment("hello")], monkeypatch)

    assert transcriber.transcribe(np.array([], dtype=np.float32)) == ""
    assert model.calls == 0


def test_audio_above_duration_floor_reaches_the_model(monkeypatch):
    transcriber, model = make_transcriber(
        [FakeSegment("pick up the red box")], monkeypatch
    )

    assert transcriber.transcribe(long_audio()) == "pick up the red box"
    assert model.calls == 1


# --- Transcriber: silence guard -----------------------------------------


def long_audio(seconds=2.0):
    return np.zeros(int(voice.SAMPLE_RATE * seconds), dtype=np.float32)


def test_segment_above_no_speech_threshold_is_dropped(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("Thank you.", no_speech_prob=0.9)], monkeypatch
    )

    assert transcriber.transcribe(long_audio()) == ""


def test_segment_at_no_speech_threshold_is_kept(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("pick up the red box", no_speech_prob=voice.MAX_NO_SPEECH_PROB)],
        monkeypatch,
    )

    assert transcriber.transcribe(long_audio()) == "pick up the red box"


def test_segment_below_logprob_threshold_is_dropped(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("mumble", avg_logprob=-2.5)], monkeypatch
    )

    assert transcriber.transcribe(long_audio()) == ""


def test_segment_at_logprob_threshold_is_kept(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("place it on the shelf", avg_logprob=voice.MIN_AVG_LOGPROB)],
        monkeypatch,
    )

    assert transcriber.transcribe(long_audio()) == "place it on the shelf"


def test_low_confidence_segments_are_dropped_but_good_ones_survive(monkeypatch):
    transcriber, _ = make_transcriber(
        [
            FakeSegment("pick up the blue box"),
            FakeSegment("you", no_speech_prob=0.95),
        ],
        monkeypatch,
    )

    assert transcriber.transcribe(long_audio()) == "pick up the blue box"


# --- Transcriber: hallucination blocklist -------------------------------


@pytest.mark.parametrize(
    "phantom", ["Thank you.", "you", ".", "Thanks for watching!", "  YOU  "]
)
def test_known_hallucinations_are_rejected(phantom, monkeypatch):
    transcriber, _ = make_transcriber([FakeSegment(phantom)], monkeypatch)

    assert transcriber.transcribe(long_audio()) == ""


def test_real_command_containing_a_blocklisted_word_survives(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("thank you, now put the green box on the shelf")], monkeypatch
    )

    result = transcriber.transcribe(long_audio())

    assert result == "thank you, now put the green box on the shelf"


# --- Transcriber: assembly ----------------------------------------------


def test_multiple_segments_are_joined(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("pick up the red box"), FakeSegment("and put it on the shelf")],
        monkeypatch,
    )

    result = transcriber.transcribe(long_audio())

    assert result == "pick up the red box and put it on the shelf"


def test_segment_whitespace_is_normalised(monkeypatch):
    transcriber, _ = make_transcriber(
        [FakeSegment("  put the yellow box on the shelf  ")], monkeypatch
    )

    assert transcriber.transcribe(long_audio()) == "put the yellow box on the shelf"


def test_no_segments_returns_empty(monkeypatch):
    transcriber, _ = make_transcriber([], monkeypatch)

    assert transcriber.transcribe(long_audio()) == ""


# --- Transcriber: model configuration -----------------------------------


def test_model_name_defaults_to_base_en(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)

    assert voice.Transcriber().model_name == "base.en"


def test_model_name_read_from_environment(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small.en")

    assert voice.Transcriber().model_name == "small.en"


def test_explicit_model_name_beats_environment(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "small.en")

    assert voice.Transcriber(model_name="tiny.en").model_name == "tiny.en"


def test_load_builds_a_cpu_int8_model(monkeypatch):
    built = {}

    def fake_whisper_model(name, device=None, compute_type=None):
        built.update(name=name, device=device, compute_type=compute_type)
        return FakeModel(name, device, compute_type)

    monkeypatch.setattr(voice, "WhisperModel", fake_whisper_model)
    transcriber = voice.Transcriber(model_name="tiny.en")

    transcriber.load()

    assert built == {"name": "tiny.en", "device": "cpu", "compute_type": "int8"}
    assert transcriber.ready


def test_load_is_idempotent(monkeypatch):
    calls = []

    def fake_whisper_model(name, device=None, compute_type=None):
        calls.append(name)
        return FakeModel(name, device, compute_type)

    monkeypatch.setattr(voice, "WhisperModel", fake_whisper_model)
    transcriber = voice.Transcriber(model_name="tiny.en")

    transcriber.load()
    transcriber.load()

    assert len(calls) == 1


def test_transcribe_before_load_raises(monkeypatch):
    transcriber = voice.Transcriber(model_name="tiny.en")

    with pytest.raises(RuntimeError, match="not loaded"):
        transcriber.transcribe(long_audio())


def test_transcriber_is_not_ready_before_load():
    assert not voice.Transcriber(model_name="tiny.en").ready
