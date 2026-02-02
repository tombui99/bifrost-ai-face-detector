import os
import cv2
import base64
import numpy as np
import glob
import time
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from deepface import DeepFace
import db_utils

app = Flask(__name__, template_folder="frontend", static_folder="frontend/static")
CORS(app)

# Configuration
DB_PATH = os.path.abspath("faces_db")
MODEL_NAME = "Facenet512"
DISTANCE_THRESHOLD = 0.6

def clear_deepface_cache(db_path):
    """Deletes .pkl files in the database directory to force DeepFace to re-index."""
    pkl_files = glob.glob(os.path.join(db_path, "*.pkl"))
    for f in pkl_files:
        try:
            os.remove(f)
            print(f"Cleared cache file: {f}")
        except Exception as e:
            print(f"Failed to clear cache {f}: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/upload_face", methods=["POST"])
def upload_face():
    try:
        data = request.json
        name = data["name"].strip()
        employee_id = data.get("employee_id", "").strip()
        department = data.get("department", "").strip()
        image_data = data["image"]
        
        if not name:
            return jsonify({"status": "error", "message": "Name is required"}), 400
            
        # 1. Save to Database
        db_utils.add_employee(name, employee_id, department)
        
        # 2. Save image to disk for DeepFace
        person_dir = os.path.join(DB_PATH, name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Save image
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Unique filename with timestamp
        filename = f"face_{int(time.time())}.jpg"
        file_path = os.path.join(person_dir, filename)
        
        with open(file_path, "wb") as f:
            f.write(image_bytes)
            
        # Force re-indexing by clearing pkl
        clear_deepface_cache(DB_PATH)
        
        return jsonify({"status": "success", "message": f"Face registered for {name}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/logs")
def api_logs():
    try:
        logs = db_utils.get_logs()
        return jsonify({"status": "success", "logs": logs})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/employees")
def api_employees():
    try:
        employees = db_utils.get_employees()
        return jsonify({"status": "success", "employees": employees})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/process_snapshot", methods=["POST"])
def process_snapshot():
    try:
        data = request.json
        image_data = data["image"]
        # Remove header from base64 string
        image_data = image_data.split(",")[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to numpy array for OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Identification
        results_find = []
        db_exists = os.path.exists(DB_PATH) and len([f for f in os.listdir(DB_PATH) if not f.startswith('.')]) > 0
        
        if db_exists:
            results_find = DeepFace.find(img_path = img, 
                                      db_path = DB_PATH,
                                      model_name = MODEL_NAME,
                                      detector_backend = 'retinaface',
                                      enforce_detection = False,
                                      silent = True)
        
        # 2. Extract faces for UI
        faces_snap = DeepFace.extract_faces(img_path = img,
                                          detector_backend = 'retinaface',
                                          enforce_detection = False)

        detections = []
        for f_obj in faces_snap:
            if f_obj.get('confidence', 0) > 0:
                fa = f_obj["facial_area"]
                fx, fy, fw, fh = fa['x'], fa['y'], fa['w'], fa['h']
                
                identity_name = "Unknown"
                distance = 1.0
                
                # Match detection to finding results
                if db_exists and results_find:
                    for df in results_find:
                        if not df.empty:
                            match = df.iloc[0]
                            mx, my, mw, mh = match['source_x'], match['source_y'], match['source_w'], match['source_h']
                            
                            # Simple overlap check
                            if abs(fx - mx) < 50 and abs(fy - my) < 50:
                                dist_cols = [c for c in match.index if 'distance' in c or 'cosine' in c]
                                dist = match[dist_cols[0]] if dist_cols else 1.0
                                
                                if dist < DISTANCE_THRESHOLD:
                                    id_path = match['identity']
                                    name = os.path.basename(os.path.dirname(id_path))
                                    if name in ['faces_db', '']:
                                        name = os.path.splitext(os.path.basename(id_path))[0]
                                    identity_name = name
                                    distance = float(dist)
                                break
                
                detections.append({
                    "x": int(fx),
                    "y": int(fy),
                    "w": int(fw),
                    "h": int(fh),
                    "name": identity_name,
                    "distance": distance
                })
                
                # Log attendance if recognized
                if identity_name != "Unknown":
                    db_utils.log_attendance(identity_name)

        return jsonify({"status": "success", "detections": detections})

    except Exception as e:
        print(f"Error processing snapshot: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "healthy"}), 200

if __name__ == "__main__":
    clear_deepface_cache(DB_PATH)
    # Production servers handle SSL. We listen on port 8080.
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
