# Spline Interpolation Based Audio and Video Enhancement

This project enhances degraded video and audio using **spline interpolation techniques**.  
It reconstructs smoother frames and higher-quality audio from degraded inputs.

---

## 📂 Project Structure

# Spline Interpolation Based Audio and Video Enhancement

This project enhances degraded video and audio by using **spline interpolation** techniques.  
The objective is to reconstruct smoother, high-quality frames and audio signals  
from low frame-rate video and low sample-rate audio.

---

## 📂 Project Structure

```plaintext
/ (Root Directory)
├── src/                  # Source code
│   ├── audio/             # Audio utilities and interpolation
│   ├── video/             # Video utilities and frame interpolation
│   ├── evaluation/        # Evaluation scripts (audio and video metrics)
│   ├── integration/       # Scripts to merge enhanced audio and video
├── data/
│   ├── input/             # Input degraded video and audio (you place them here)
│   ├── output/            # Output enhanced audio and video (created after running)
├── results/
│   ├── audio_charts/      # Saved audio evaluation plots and metrics
│   ├── video_charts/      # Saved video evaluation plots and metrics
├── Dockerfile             # Docker configuration
├── requirements.txt       # Python dependencies (pip version)
├── environment.yml        # Conda environment file (alternative)
├── main.py                # Main script to run full enhancement pipeline
└── README.md              # This file
```

---

## ⚙️ Setup Instructions

### Option 1: Using Conda (Recommended)

1. Run the setup script:

```bash
bash setup.sh
```

2. After setup, manually activate the environment:

```bash
conda activate spline-interpolation-enhancement
```

3. Install ffmpeg if needed:

```bash
brew install ffmpeg      # For Mac
sudo apt install ffmpeg  # For Linux
```

---

### Option 2: Using Docker

```bash
docker build -t spline-enhancement .
docker run --rm -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results spline-enhancement
```

✅ This automatically runs `main.py` inside the container.

---

## 🚀 How to Run the Project

After setting up the environment and activating it:

```bash
python main.py
```

✅ This will:
- Load degraded audio and video
- Apply spline interpolation
- Enhance and save outputs
- Evaluate quality (MSE, SNR, SSIM, PSNR)
- Merge enhanced audio and video

---

## 📋 Outputs Generated

| Folder | Description |
|--------|-------------|
| `data/output/` | Enhanced audio, enhanced video, final merged video |
| `results/audio_charts/` | Audio waveform, spectrogram, error plots, metrics |
| `results/video_charts/` | Frame comparisons, SSIM/PSNR graphs, video metrics |

---

## 📢 Important Notes

- Place your original degraded video as `data/input/original_video.mp4` before running.
- Outputs are auto-saved into `data/output/` and `results/`.
- Large files are excluded from GitHub by `.gitignore`.

---

## 📚 Technologies Used

- Python 3.12
- NumPy, SciPy, OpenCV, Matplotlib
- Scikit-Learn, Scikit-Image
- TQDM (progress bars)
- MoviePy (audio-video merging)
- Docker (for containerized runs)

---

## 👌 Credits

Developed as part of the **Spline Interpolation Based Audio-Video Enhancement** project.

