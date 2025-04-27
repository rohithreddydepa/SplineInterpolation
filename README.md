# Spline Interpolation Based Audio and Video Enhancement

This project enhances degraded video and audio using **spline interpolation techniques**.  
It reconstructs smoother frames and higher-quality audio from degraded inputs.

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

### Option 1: Using Docker (Recommended)

Docker is the recommended way to run this project to avoid any dependency or environment issues.
If Docker is not installed, please download and install it from [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/).

1. Build the Docker image:

```bash
docker build -t spline-enhancement .
```

2. Run the Docker container:

```bash
docker run --rm -it -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results spline-enhancement
```

✅ This automatically runs `main.py` inside the container.

---

### Option 2: Using Conda (Alternative)

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
- **Note:** In this project, the original video was not low-quality. Therefore, a `degrade.py` script was used to artificially downgrade a normal video (by reducing frame rate and audio quality). You can use `degrade.py` similarly on any video to create degraded input for testing purposes.

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

