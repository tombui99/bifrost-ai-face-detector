# AI Face Detector 🚀

A real-time facial recognition system built with Python, utilizing state-of-the-art deep learning models for accurate face detection and identity matching.

## 🌟 Features

- **Real-time Detection**: Uses `RetinaFace` for robust and accurate face detection.
- **High-Precision Recognition**: Leverages the `Facenet512` model for deep representational matching.
- **Dynamic Database**: Automatically indexes images placed in the `faces_db` directory.
- **Visual Feedback**: Real-time bounding boxes with identity labels and confidence scores.

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Deep Learning Framework**: [DeepFace](https://github.com/serengil/deepface)
- **Models**: RetinaFace (Detection), Facenet512 (Recognition)
- **Backend**: TensorFlow / Keras

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- A webcam for real-time detection

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/tombui99/bifrost-ai-face-detector.git
   cd bifrost-ai-face-detector
   ```

2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

Add photos of people you want to recognize into the `faces_db` folder:

- Create a subfolder for each person: `faces_db/John_Doe/`
- Add one or more `.jpg` or `.png` images of that person into their respective folder.

## 💻 Usage

Run the main script to start the recognition system:

```bash
python main.py
```

- Press **'q'** to quit the application.
- The system will automatically clear the DeepFace cache (`.pkl` files) on startup to ensure new images are indexed.

## 📝 License

This project is for educational and personal use.

---

Built with ❤️ by [tombui99](https://github.com/tombui99)
