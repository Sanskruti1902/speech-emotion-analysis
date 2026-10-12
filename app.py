from flask import Flask, render_template, request, jsonify
from scipy.io import wavfile
from scipy.signal import resample_poly, find_peaks
from fractions import Fraction
import numpy as np
import io

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  # 20 MB upload limit

TARGET_FS = 16000

def to_mono_float(audio):
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = max(abs(info.min), info.max)
        audio = audio.astype(np.float64) / scale
    else:
        audio = audio.astype(np.float64)
    audio = np.nan_to_num(audio)
    if audio.size:
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
    return audio

def prepare_audio(file_bytes):
    try:
        fs, audio = wavfile.read(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError("Could not read this file as WAV. Please upload a WAV audio file.") from exc
    if fs <= 0 or audio.size == 0:
        raise ValueError("The uploaded WAV file is empty or invalid.")
    x = to_mono_float(audio)
    if fs != TARGET_FS:
        ratio = Fraction(TARGET_FS, int(fs)).limit_denominator()
        x = resample_poly(x, ratio.numerator, ratio.denominator)
    x = x - np.mean(x)
    if len(x) < int(0.25 * TARGET_FS):
        raise ValueError("Please provide at least 0.25 seconds of audio.")
    if len(x) > TARGET_FS * 120:
        raise ValueError("Audio must be 120 seconds or shorter.")
    peak = np.max(np.abs(x)) if len(x) else 0
    if peak > 0:
        x = x / peak
    return x.astype(np.float64)

def frame_signal(x, frame_len=480, hop=160):
    if len(x) < frame_len:
        x = np.pad(x, (0, frame_len - len(x)))
    count = 1 + max(0, (len(x) - frame_len) // hop)
    frames = np.stack([x[i*hop:i*hop+frame_len] for i in range(count)])
    return frames, hop

def estimate_pitch(frame, fs=TARGET_FS):
    frame = frame - np.mean(frame)
    if np.sqrt(np.mean(frame**2)) < 0.015:
        return 0.0
    min_lag = int(fs / 350)
    max_lag = min(int(fs / 60), len(frame) - 2)
    if min_lag >= max_lag:
        return 0.0
    # Normalized autocorrelation over plausible human voice pitch lags
    correlations = []
    lags = np.arange(min_lag, max_lag + 1)
    for lag in lags:
        a, b = frame[:-lag], frame[lag:]
        denom = np.sqrt(np.sum(a*a) * np.sum(b*b)) + 1e-12
        correlations.append(np.sum(a*b) / denom)
    idx = int(np.argmax(correlations))
    if correlations[idx] < 0.30:
        return 0.0
    return float(fs / lags[idx])

def analyze(x):
    fs = TARGET_FS
    frame_len, hop = 480, 160  # 30 ms frame, 10 ms hop
    frames, _ = frame_signal(x, frame_len, hop)
    window = np.hamming(frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    zcr = np.mean(np.abs(np.diff((frames >= 0).astype(float), axis=1)), axis=1)
    pitches = np.array([estimate_pitch(fr) for fr in frames])
    threshold = max(0.02, 0.10 * float(np.max(rms)))
    voiced = (rms > threshold) & (pitches >= 60) & (pitches <= 350)
    mean_pitch = float(np.median(pitches[voiced])) if np.any(voiced) else None
    mean_energy = float(np.mean(rms))
    mean_zcr = float(np.mean(zcr))
    active_ratio = float(np.mean(rms > threshold))

    if mean_pitch is None:
        emotion = "Uncertain — insufficient voiced speech"
        explanation = "Try speaking clearly for several seconds, with less background noise."
    elif mean_pitch > 190 and mean_energy > 0.18:
        emotion = "Possible high vocal activation / excitement"
        explanation = "The simple rules found relatively high pitch and energy."
    elif mean_pitch < 120 and mean_energy < 0.10:
        emotion = "Possible low vocal activation / calmness"
        explanation = "The simple rules found relatively low pitch and energy."
    else:
        emotion = "Neutral or uncertain"
        explanation = "The measured features do not match the demonstration rules strongly."

    # Full-recording spectrum
    nfft = max(2048, 2 ** int(np.ceil(np.log2(max(2, len(x))))))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    spec_db = 20 * np.log10(spec / max(1, len(x)) + 1e-8)
    keep = freqs <= 4000
    spectrum = {
        "frequency": freqs[keep][::max(1, int(np.sum(keep) / 500))].tolist(),
        "magnitude": spec_db[keep][::max(1, int(np.sum(keep) / 500))].tolist()
    }

    # Spectrogram using short-time FFT; limit to <=4 kHz and compact response
    n_spec = 512
    spec_frames = []
    for fr in frames:
        mag = np.abs(np.fft.rfft(fr * window, n=n_spec))
        spec_frames.append(20 * np.log10(mag + 1e-8))
    spec_db_matrix = np.array(spec_frames).T
    max_bins = n_spec // 2 + 1
    # keep all bins through 4 kHz; reduce time columns if needed
    spectrogram = spec_db_matrix[:max_bins, :]
    if spectrogram.shape[1] > 240:
        cols = np.linspace(0, spectrogram.shape[1]-1, 240).astype(int)
        spectrogram = spectrogram[:, cols]
    # Waveform downsampled for display
    step = max(1, len(x)//1500)
    wave_x = (np.arange(0, len(x), step) / fs).tolist()
    wave_y = x[::step].tolist()
    frame_times = ((np.arange(len(rms)) * hop + frame_len/2) / fs)
    energy = {"time": frame_times[::max(1, len(frame_times)//500)].tolist(),
              "rms": rms[::max(1, len(frame_times)//500)].tolist()}

    return {
        "duration": round(len(x)/fs, 2),
        "sample_rate": fs,
        "mean_pitch": round(mean_pitch, 2) if mean_pitch is not None else None,
        "mean_energy": round(mean_energy, 4),
        "mean_zcr": round(mean_zcr, 4),
        "active_ratio": round(active_ratio, 3),
        "emotion": emotion,
        "explanation": explanation,
        "waveform": {"time": wave_x, "amplitude": wave_y},
        "spectrum": spectrum,
        "energy": energy,
        "spectrogram": spectrogram.tolist(),
        "spectrogram_min": float(np.min(spectrogram)),
        "spectrogram_max": float(np.max(spectrogram)),
        "spectrogram_times": round(spectrogram.shape[1], 0),
        "spectrogram_freq_bins": max_bins
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_route():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file received. Record speech or choose a WAV file."}), 400
    f = request.files["audio"]
    if not f.filename:
        return jsonify({"error": "Please select an audio file."}), 400
    try:
        x = prepare_audio(f.read())
        return jsonify(analyze(x))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Audio analysis failed")
        return jsonify({"error": "Analysis failed unexpectedly. Try a shorter, clear WAV recording."}), 500

@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "File is too large. Please use audio under 20 MB."}), 413

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
