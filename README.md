# Speech Emotion Lab — Online Deployable Website

A beginner-friendly Flask website for a Speech Signal Processing and Emotion Analysis project.

## Features
- Browser microphone recording (permission required)
- WAV audio upload
- Speech waveform
- Frequency spectrum
- Short-time RMS energy
- Spectrogram
- Pitch, energy, zero-crossing rate and activity metrics
- Simple rule-based emotion estimate

**Important:** This is an educational DSP demonstration, not a validated emotion-recognition system. Pitch and energy alone cannot reliably identify a person's emotional state.

## Files
- `app.py` — Python/Flask backend and DSP processing
- `templates/index.html` — website interface and browser recording
- `requirements.txt` — Python dependencies
- `render.yaml` — deployment configuration for Render

## Run locally on Windows

1. Install Python 3.10 or newer from https://www.python.org/downloads/. During installation, select **Add Python to PATH**.
2. Extract the ZIP file into a folder, for example `speech_emotion_website`.
3. Open that folder in File Explorer. Click the address bar, type `cmd`, and press Enter.
4. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

5. Activate it:

   ```bash
   .venv\Scripts\activate
   ```

6. Install dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

7. Start the website:

   ```bash
   python app.py
   ```

8. Open `http://127.0.0.1:5000` in Chrome or Edge.
9. Click **Start microphone recording**, allow microphone permission, speak, then click **Stop recording** and **Analyze recording**.

Microphone access is normally allowed on `localhost`. A deployed website must use HTTPS for browser microphone access.

## Uploading audio
Upload a WAV file (mono or stereo, common integer/float WAV formats). The server converts it to mono 16 kHz for analysis. Maximum duration is 120 seconds and maximum upload size is 20 MB.

If you have an MP3 or M4A file, convert it to WAV first, for example with an audio editor such as Audacity. This starter project intentionally accepts WAV only to avoid requiring external codecs.

## Deploy online with Render

1. Create a GitHub account if needed and create a new repository.
2. Upload the extracted project files (`app.py`, `requirements.txt`, `render.yaml`, and the `templates` folder) to the repository.
3. Sign in to https://render.com and choose **New → Blueprint** (or create a Web Service connected to the repository).
4. Select your repository and deploy. The included `render.yaml` configures the service.
5. When deployment finishes, open the HTTPS URL Render provides.
6. Allow microphone access when prompted.

Free hosting availability, sleep behavior, and plan limits can change. For a college demonstration, test the deployed link before presenting. Do not upload sensitive or private recordings.

## Troubleshooting
- **Microphone does not start:** use Chrome or Edge, allow microphone access in browser site settings, and use HTTPS or localhost.
- **No sound:** check Windows Settings → System → Sound → Input, select the correct mic, and test its input meter.
- **Unsupported audio:** upload a `.wav` file.
- **`python` is not recognized:** reinstall Python with PATH enabled, or try the Windows `py` command.
- **Port 5000 is in use:** stop the other application or change the port in `app.py`.

## Academic explanation
The server resamples audio to 16 kHz, converts stereo to mono, removes DC offset, and normalizes the waveform. It computes short-time RMS energy, zero-crossing rate, pitch estimation using autocorrelation, a full-recording FFT spectrum, and a short-time FFT spectrogram. A small set of heuristic thresholds creates the displayed possible result. For a research-grade emotion classifier, train and validate a model on a labeled speech-emotion dataset and evaluate its performance.
