import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import av
import threading
from streamlit_webrtc import VideoProcessorBase

# Local imports
from config.config import EAR_THRESHOLD, CONSEC_FRAMES, MODEL_PATH, ALARM_FILE
from src.core.ear import calculate_ear
from src.utils.audio import play_alarm_sound

class DrowsinessDetector(VideoProcessorBase):
    def __init__(self):
        # Load Face Landmarker
        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.FaceLandmarkerOptions(base_options=base_options,
                                                   output_face_blendshapes=False,
                                                   output_facial_transformation_matrixes=False,
                                                   num_faces=1)
            self.detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error initializing MediaPipe FaceLandmarker: {e}")
            raise e

        self.frame_count = 0
        self.drowsy = False
        self.alarm_thread = None

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    h, w, _ = img.shape
                    # face_landmarks is already a list of landmarks in Tasks API
                    landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks]

                    # Left and right eye indices (MediaPipe Face Mesh)
                    left_eye = [362, 385, 387, 263, 373, 380]
                    right_eye = [33, 160, 158, 133, 153, 144]

                    left_ear = calculate_ear(landmarks, left_eye)
                    right_ear = calculate_ear(landmarks, right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    
                    # Visualize eye landmarks for debugging
                    for idx in left_eye:
                        cv2.circle(img, landmarks[idx], 2, (0, 255, 0), -1)
                    for idx in right_eye:
                        cv2.circle(img, landmarks[idx], 2, (0, 255, 0), -1)

                    if ear < EAR_THRESHOLD:
                        self.frame_count += 1
                    else:
                        self.frame_count = 0
                        self.drowsy = False

                    if self.frame_count >= CONSEC_FRAMES:
                        self.drowsy = True
                        cv2.putText(img, "DROWSY!", (30, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
                        # Play alarm in separate thread
                        if self.alarm_thread is None or not self.alarm_thread.is_alive():
                            self.alarm_thread = threading.Thread(target=play_alarm_sound, args=(ALARM_FILE,))
                            self.alarm_thread.start()
                    else:
                        cv2.putText(img, f"EAR: {ear:.2f}", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            print(f"Error in processing frame: {e}")
            return frame
