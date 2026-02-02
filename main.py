import cv2
from deepface import DeepFace
import time
import os
import glob

def clear_deepface_cache(db_path):
    """Deletes .pkl files in the database directory to force DeepFace to re-index."""
    pkl_files = glob.glob(os.path.join(db_path, "*.pkl"))
    for f in pkl_files:
        try:
            os.remove(f)
            print(f"Cleared cache file: {f}")
        except Exception as e:
            print(f"Failed to clear cache {f}: {e}")

def main():
    print("--- AI Facial Recognition System v2.2 (Facenet512) ---")
    
    # Absolute Path to Database
    db_path = os.path.abspath("faces_db")
    print(f"Database Path: {db_path}")
    
    # Verify Database Content
    if not os.path.exists(db_path):
        print(f"Error: Database directory '{db_path}' not found.")
        return
        
    db_files = [f for f in os.listdir(db_path) if not f.startswith('.')]
    print(f"Items in Database: {db_files}")
    
    # Clear cache at startup to ensure new photos are indexed
    clear_deepface_cache(db_path)
    
    # Configuration
    MODEL_NAME = "Facenet512" # More accurate than VGG-Face
    DISTANCE_THRESHOLD = 0.6  # Standard for Facenet512 cosine is ~0.3, so 0.6 is very relaxed
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    # Frame skipping and smoothing
    frame_count = 0
    skip_frames = 2  # Process every 3rd frame
    last_detected_results = []
    
    db_exists = len(db_files) > 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame for a mirror effect
        frame = cv2.flip(frame, 1)

        if frame_count % (skip_frames + 1) == 0:
            try:
                processed_results = []
                
                # Check for matches in full frame
                results_find = []
                if db_exists:
                    try:
                        results_find = DeepFace.find(img_path = frame, 
                                                  db_path = db_path,
                                                  model_name = MODEL_NAME,
                                                  detector_backend = 'retinaface',
                                                  enforce_detection = False,
                                                  silent = True)
                    except Exception as e:
                        print(f"DeepFace.find error: {e}")

                # Detection for all faces (to show red boxes for unknowns)
                faces = DeepFace.extract_faces(img_path = frame,
                                             detector_backend = 'retinaface',
                                             enforce_detection = False)
                
                for face_obj in faces:
                    if face_obj.get('confidence', 0) > 0:
                        facial_area = face_obj["facial_area"]
                        x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                        
                        identity = "Unknown"
                        display_dist = ""
                        
                        if db_exists and results_find:
                            # Match detection to finding results
                            for df in results_find:
                                if not df.empty:
                                    match = df.iloc[0]
                                    mx, my, mw, mh = match['source_x'], match['source_y'], match['source_w'], match['source_h']
                                    
                                    # Overlap check
                                    if abs(x - mx) < 60 and abs(y - my) < 60:
                                        distance_cols = [col for col in match.index if MODEL_NAME in col and ('cosine' in col or 'euclidean' in col)]
                                        # If specific col not found, try generic distance
                                        if not distance_cols:
                                            distance_cols = [col for col in match.index if 'distance' in col or 'cosine' in col]
                                            
                                        if distance_cols:
                                            distance = match[distance_cols[0]]
                                            display_dist = f"({distance:.2f})"
                                            
                                            # Console logging
                                            print(f"Face at [{x},{y}] - Match: {os.path.basename(match['identity'])} Distance: {distance:.4f}")
                                            
                                            if distance < DISTANCE_THRESHOLD:
                                                identity_path = match['identity']
                                                # Look for directory name (person's name)
                                                name = os.path.basename(os.path.dirname(identity_path))
                                                if name == 'faces_db' or name == '':
                                                    name = os.path.splitext(os.path.basename(identity_path))[0]
                                                identity = name
                                        break
                                        
                        processed_results.append({
                            "facial_area": facial_area,
                            "name": identity,
                            "distance": display_dist
                        })
                
                last_detected_results = processed_results
            except Exception as e:
                # print(f"Processing error: {e}")
                pass
        
        face_count = 0
        if last_detected_results:
            face_count = len(last_detected_results)
            for res in last_detected_results:
                facial_area = res["facial_area"]
                name = res["name"]
                dist_str = res["distance"]
                x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} {dist_str}".strip()
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.putText(frame, f"Faces Detected: {face_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Model: {MODEL_NAME}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
