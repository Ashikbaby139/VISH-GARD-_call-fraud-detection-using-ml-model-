import io
import json
import os
import threading
import wave
import audioop
from datetime import datetime


LOG_FILE = "transcripts.log"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-en-us")

# Global transcription state
live_partial = ""
live_final = ""

_model = None
_recognizer = None
_audio = None
_audio_backend = None
_input_device = None
_input_device_name = None
_sample_rate = 16000
_listener_started = False
_listener_lock = threading.Lock()
_listener_error = None


def _ensure_listener_resources():
    global _model, _recognizer, _audio, _audio_backend, _input_device, _input_device_name, _sample_rate, _listener_error

    if _model is not None and _recognizer is not None and _audio is not None:
        return

    try:
        from vosk import KaldiRecognizer, Model
    except ModuleNotFoundError as exc:
        _listener_error = f"Missing dependency: {exc.name}"
        raise RuntimeError(_listener_error) from exc

    try:
        import pyaudio

        _audio = pyaudio.PyAudio()
        _audio_backend = "pyaudio"
        _sample_rate = 16000
    except ModuleNotFoundError:
        try:
            import sounddevice as sd

            _audio = sd
            _audio_backend = "sounddevice"
            _input_device, _input_device_name, _sample_rate = _select_sounddevice_input(sd)
        except ModuleNotFoundError as exc:
            _listener_error = "Missing dependency: pyaudio or sounddevice"
            raise RuntimeError(_listener_error) from exc

    _model = Model(MODEL_PATH)
    _recognizer = KaldiRecognizer(_model, _sample_rate)
    _recognizer.SetWords(True)
    _listener_error = None


def _hostapi_name(sd, device):
    try:
        return str(sd.query_hostapis(device["hostapi"])["name"])
    except Exception:
        return ""


def _device_priority(name, hostapi_name):
    name = name.lower()
    hostapi_name = hostapi_name.lower()
    score = 0

    if "microphone" in name or "mic" in name:
        score += 100
    if "stereo mix" in name:
        score -= 100
    if "mapper" in name or "capture driver" in name:
        score -= 20

    if "wasapi" in hostapi_name:
        score += 40
    elif "wdm-ks" in hostapi_name:
        score += 30
    elif "directsound" in hostapi_name:
        score += 20
    elif "mme" in hostapi_name:
        score += 10

    return score


def _select_sounddevice_input(sd):
    devices = sd.query_devices()
    candidates = []

    for index, device in enumerate(devices):
        if device["max_input_channels"] < 1:
            continue

        name = str(device["name"])
        hostapi_name = _hostapi_name(sd, device)
        priority = _device_priority(name, hostapi_name)
        candidates.append((priority, index, device, name))

    candidates.sort(reverse=True)

    for _priority, index, device, name in candidates:
        for sample_rate in (16000, int(device.get("default_samplerate") or 16000)):
            try:
                sd.check_input_settings(
                    device=index,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="int16",
                )
                return index, name, sample_rate
            except Exception:
                continue

    default_input = sd.default.device[0]
    if default_input is None or default_input < 0:
        raise RuntimeError("No input device available")

    default_input = int(default_input)
    default_device = devices[default_input]
    default_rate = int(default_device.get("default_samplerate") or 16000)
    return default_input, str(default_device["name"]), default_rate


def audio_loop():
    global live_partial, live_final, _listener_error, _listener_started

    try:
        _ensure_listener_resources()

        if _audio_backend == "pyaudio":
            import pyaudio

            stream = _audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=_sample_rate,
                input=True,
                frames_per_buffer=4000,
            )
            stream.start_stream()
        elif _audio_backend == "sounddevice":
            stream = _audio.RawInputStream(
                samplerate=_sample_rate,
                blocksize=4000,
                channels=1,
                dtype="int16",
                device=_input_device,
            )
            stream.start()
        else:
            raise RuntimeError("No supported audio backend available")

        while True:
            if _audio_backend == "pyaudio":
                data = stream.read(4000, exception_on_overflow=False)
            else:
                data, _overflowed = stream.read(4000)

            if _recognizer.AcceptWaveform(data):
                result = json.loads(_recognizer.Result())
                text = result.get("text", "").strip()
                if text:
                    live_final += " " + text
                    live_partial = ""
            else:
                partial = json.loads(_recognizer.PartialResult())
                live_partial = partial.get("partial", "")
    except Exception as exc:
        _listener_error = str(exc)
        _listener_started = False


def ensure_listener_started():
    global _listener_started

    if _listener_started:
        return

    with _listener_lock:
        if _listener_started:
            return

        _ensure_listener_resources()

        threading.Thread(target=audio_loop, daemon=True).start()
        _listener_started = True


def get_live_text():
    if _listener_error:
        raise RuntimeError(_listener_error)

    ensure_listener_started()
    return (live_final + " " + live_partial).strip()


def get_live_snapshot():
    if _listener_error:
        raise RuntimeError(_listener_error)

    ensure_listener_started()
    return {
        "final": live_final.strip(),
        "partial": live_partial.strip(),
        "text": (live_final + " " + live_partial).strip(),
    }


def get_listener_error():
    return _listener_error


def get_listener_status():
    return {
        "backend": _audio_backend,
        "device": _input_device_name,
        "sample_rate": _sample_rate,
        "started": _listener_started,
        "error": _listener_error,
    }


def reset_live_text():
    global live_final, live_partial
    live_final = ""
    live_partial = ""


def _transcribe_wav_stream(stream):
    try:
        stream.seek(0)
        with wave.open(stream, "rb") as wav_file:
            if wav_file.getcomptype() != "NONE":
                raise RuntimeError("Compressed WAV files are not supported")

            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()

            if channels < 1 or channels > 2:
                raise RuntimeError("Only mono or stereo WAV files are supported")

            recognizer = _recognizer.__class__(_model, 16000)
            recognizer.SetWords(True)
            parts = []
            rate_state = None

            while True:
                data = wav_file.readframes(4000)
                if not data:
                    break

                if channels == 2:
                    data = audioop.tomono(data, sample_width, 0.5, 0.5)

                if sample_width != 2:
                    data = audioop.lin2lin(data, sample_width, 2)

                if sample_rate != 16000:
                    data, rate_state = audioop.ratecv(data, 2, 1, sample_rate, 16000, rate_state)

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        parts.append(text)

            final_result = json.loads(recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                parts.append(final_text)

            return " ".join(part for part in parts if part).strip()
    except wave.Error as exc:
        raise RuntimeError("Upload a valid WAV audio file") from exc


def transcribe_uploaded_audio(file_storage):
    _ensure_listener_resources()

    filename = (file_storage.filename or "").lower()

    try:
        file_storage.stream.seek(0)
    except Exception:
        pass

    if filename.endswith(".wav"):
        return _transcribe_wav_stream(file_storage.stream)

    if filename.endswith(".mp3") or filename.endswith(".mpeg"):
        try:
            from pydub import AudioSegment
        except ModuleNotFoundError as exc:
            raise RuntimeError("MP3/MPEG support requires pydub and ffmpeg") from exc

        try:
            file_storage.stream.seek(0)
            audio = AudioSegment.from_file(file_storage.stream)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            wav_buffer = io.BytesIO()
            audio.export(wav_buffer, format="wav")
            return _transcribe_wav_stream(wav_buffer)
        except Exception as exc:
            raise RuntimeError("Could not decode MP3/MPEG file. Install ffmpeg and try again.") from exc

    raise RuntimeError("Supported upload formats: WAV, MP3, MPEG")


def save_transcript_to_file(text, decision, confidence, keywords):
    if decision != 0:
        return

    entry = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text.strip(),
        "confidence": confidence,
        "keywords": keywords,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
