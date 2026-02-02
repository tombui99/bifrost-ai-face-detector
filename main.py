import cv2
from deepface import DeepFace
import time
import os
import glob
import db_utils

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
    print("--- AI Facial Recognition System ---")
    
    # Absolute Path to Database
    db_path = os.path.abspath("faces_db")
    db_exists = os.path.exists(db_path) and len([f for f in os.listdir(db_path) if not f.startswith('.')]) > 0
    
    # Clear cache at startup to ensure new photos are indexed
    clear_deepface_cache(db_path)
    
    # Configuration
    MODEL_NAME = "Facenet512" 
    DISTANCE_THRESHOLD = 0.6  
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return

    # Get frame dimensions
    ret, frame = cap.read()
    if not ret: return
    height, width, _ = frame.shape
    
    # UI Button Configuration
    button_w, button_h = 160, 50
    btn_x = (width - button_w) // 2
    btn_y = height - button_h - 20
    button_rect = (btn_x, btn_y, button_w, button_h)

    # State variables
    scan_requested = False
    processing = False
    current_faces = []        # Stores {x, y, w, h} from real-time DeepFace.extract_faces
    frame_count = 0
    skip_frames = 2

    def handle_mouse(event, x, y, flags, param):
        nonlocal scan_requested
        if event == cv2.EVENT_LBUTTONDOWN:
            bx, by, bw, bh = button_rect
            if bx <= x <= bx + bw and by <= y <= by + bh:
                scan_requested = True

    cv2.namedWindow('Face Recognition')
    cv2.setMouseCallback('Face Recognition', handle_mouse)

    print("Controls:")
    print("  - Click 'SCAN' button: Take snapshot and identify")
    print("  - Press 's': Take snapshot and identify")
    print("  - Press 'q': Quit application")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        # 1. Lean Real-time Tracking (Only boxes and count)
        if frame_count % (skip_frames + 1) == 0:
            try:
                # Optimized detection for tracking only
                faces = DeepFace.extract_faces(img_path = frame,
                                             detector_backend = 'retinaface',
                                             enforce_detection = False)
                current_faces = []
                for f_obj in faces:
                    if f_obj.get('confidence', 0) > 0:
                        current_faces.append(f_obj["facial_area"])
            except:
                pass

        # 2. Keyboard Control
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            scan_requested = True
        elif key == ord('q'):
            break

        # 3. Snapshot and Recognition Flow
        if scan_requested and not processing:
            processing = True
            scan_requested = False
            
            # Use current frame as snapshot
            snapshot = frame.copy()
            snapshot_display = snapshot.copy()
            
            # Feedback on main window
            cv2.putText(display_frame, "CAPTURING SNAPSHOT...", (width//2 - 150, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.imshow('Face Recognition', display_frame)
            cv2.waitKey(1)

            try:
                # Identification on Snapshot
                results_find = []
                if db_exists:
                    results_find = DeepFace.find(img_path = snapshot, 
                                              db_path = db_path,
                                              model_name = MODEL_NAME,
                                              detector_backend = 'retinaface',
                                              enforce_detection = False,
                                              silent = True)
                
                # Full detection for the snapshot display
                faces_snap = DeepFace.extract_faces(img_path = snapshot,
                                                  detector_backend = 'retinaface',
                                                  enforce_detection = False)

                for f_obj in faces_snap:
                    if f_obj.get('confidence', 0) > 0:
                        fa = f_obj["facial_area"]
                        fx, fy, fw, fh = fa['x'], fa['y'], fa['w'], fa['h']
                        
                        identity_name = "Unknown"
                        dist_str = ""
                        
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
                                            dist_str = f"({dist:.2f})"
                                        break
                        
                        color = (0, 255, 0) if identity_name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(snapshot_display, (fx, fy), (fx+fw, fy+fh), color, 2)
                        label = f"{identity_name} {dist_str}".strip()
                        cv2.putText(snapshot_display, label, (fx, fy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        # Log attendance
                        if identity_name != "Unknown":
                            db_utils.log_attendance(identity_name)

                # Show Snapshot Result in a new window
                cv2.imshow('Snapshot Result', snapshot_display)
                print("Snapshot captured. View 'Snapshot Result' window.")
            except Exception as e:
                print(f"Snapshot error: {e}")
            
            processing = False

        # 4. Draw Real-time Tracking Boxes (Generic Red)
        for face in current_faces:
            fx, fy, fw, fh = face['x'], face['y'], face['w'], face['h']
            cv2.rectangle(display_frame, (fx, fy), (fx+fw, fy+fh), (0, 0, 255), 2)

        # 5. UI Overlays
        bx, by, bw, bh = button_rect
        cv2.rectangle(display_frame, (bx-2, by-2), (bx+bw+2, by+bh+2), (50, 50, 50), -1)
        cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh), (0, 200, 0) if not processing else (0, 100, 0), -1)
        cv2.putText(display_frame, "SCAN", (bx+40, by+35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(display_frame, f"Faces Detected: {len(current_faces)}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, "Click SCAN to identify faces in a snapshot", (20, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow('Face Recognition', display_frame)
        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
