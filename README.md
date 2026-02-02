# AI Face Detector 🚀

A real-time facial recognition system built with Python, utilizing state-of-the-art deep learning models for accurate face detection and identity matching.

## 🌟 Features

- **Real-time Tracking**: Optimized face detection counting for live feeds.
- **Snapshot Recognition**: Capture snapshots to identify faces with high precision.
- **Admin Dashboard**: Web interface to manage employees and view records.
- **Employee Management**: Register staff with metadata (ID, Department) and face photos.
- **Time Attendance**: Automatically log identified faces to a local database with timestamps.
- **Visual Feedback**: Real-time bounding boxes with identity labels and confidence scores.

## 🛠️ Tech Stack

- **Language**: Python 3.x
- **Backend API**: [Flask](https://flask.palletsprojects.com/)
- **Database**: SQLite (for Attendance Logs & Employees)
- **Computer Vision**: [OpenCV](https://opencv.org/)
- **Deep Learning**: [DeepFace](https://github.com/serengil/deepface)
- **Models**: RetinaFace (Detection), Facenet512 (Recognition)

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

### 1. Web API & Dashboard

Run the Flask server to access the web interface and admin dashboard:

```bash
python app.py
```

- **Scanner**: `http://localhost:8080/`
- **Admin Dash**: `http://localhost:8080/admin`

### 2. Standalone OpenCV Tool

For local desktop testing:

```bash
python main.py
```

- Press **'s'** or **Click SCAN** to take a snapshot and identify people.
- Press **'q'** to quit.

## 📝 License

This project is for educational and personal use.

---

Built with ❤️ by [tombui99](https://github.com/tombui99)
