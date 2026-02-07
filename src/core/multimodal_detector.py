
import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

# --- CONSTANTS & HELPERS ---

def normalize_angle(angle):
    """Normalize angle to -180 to 180 range and filter impossible values"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    # Reject impossible head poses
    if abs(angle) > 90:
        return 0.0
    return angle

class DriverState(Enum):
    CALIBRATING = "Calibrating..."
    NORMAL = "Normal"
    LOW_RISK = "Low Risk"
    MODERATE_RISK = "Moderate Risk"
    HIGH_RISK = "High Risk"     # Corresponds to INTOXICATED/IMPAIRED
    DROWSY = "Drowsy"           # Specific high risk state
    ASLEEP = "Asleep"           # Critical state

@dataclass
class DriverMetrics:
    ear: float
    mar: float
    head_pose: Tuple[float, float, float]
    blink_rate: float
    perclos: float
    risk_score: float
    confidence: float
    signals: Dict[str, float]
    gaze_deviation: float
    timestamp: float

# --- ANALYSIS MODULES ---

class VisionAnalyzer:
    def __init__(self):
        self.baseline_established = False
        self.baseline_window_frames = 120 * 30  # 2 minutes @ 30fps typically
        
        # History Buffers
        self.ear_history = deque(maxlen=1800)      # 1 min @ 30fps
        self.gaze_history = deque(maxlen=150)      # 5 sec
        self.blink_durations = deque(maxlen=20)
        self.blink_timestamps = deque(maxlen=60)   # timestamps of blinks
        self.head_pose_history = deque(maxlen=180) # 6 sec
        self.microsleep_events = deque(maxlen=20)  # timestamps
        
        # State
        self.in_blink = False
        self.blink_start = 0.0
        self.closure_counter = 0
        
        # Calibration Data
        self.baseline_gaze_variance = 0.0
        self.baseline_blink_rate = 0.0
        
        # Indices (MediaPipe)
        # Left Eye (Subject Left, Screen Right): 362, 385, 387, 263, 373, 380
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        # Right Eye (Subject Right, Screen Left): 33, 160, 158, 133, 153, 144
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        # Iris (Centers approx)
        self.LEFT_IRIS = 473 # Subject Left
        self.RIGHT_IRIS = 468 # Subject Right
        
    def calculate_ear(self, landmarks, indices):
        """Calculate Eye Aspect Ratio"""
        try:
            # Vertical
            v1 = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
            v2 = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
            # Horizontal
            h = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]]))
            return (v1 + v2) / (2.0 * h) if h != 0 else 0.0
        except:
            return 0.0

    def calculate_mar(self, landmarks) -> float:
        """Calculate Mouth Aspect Ratio"""
        try:
            # Vertical distance (Inner lip height: 13-14)
            v = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[14]))
            # Horizontal distance (Mouth corners: 61-291)
            h = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))
            return v / h if h != 0 else 0.0
        except:
            return 0.0

    def estimate_head_pose(self, landmarks, img_shape) -> Tuple[float, float, float]:
        """Estimate head pose (pitch, yaw, roll)"""
        # Reusing the PnP logic from previous version
        try:
            model_points = np.array([
                (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
                (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
            ])
            image_points = np.array([
                landmarks[1], landmarks[152], landmarks[263],
                landmarks[33], landmarks[61], landmarks[291]
            ], dtype="double")
            
            focal_length = img_shape[1]
            center = (img_shape[1]/2, img_shape[0]/2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0,0,1]], dtype="double")
            dist_coeffs = np.zeros((4,1))
            
            success, rot_vec, trans_vec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                return (normalize_angle(angles[0]), normalize_angle(angles[1]), normalize_angle(angles[2]))
        except:
            pass
        return (0.0, 0.0, 0.0)

    def analyze_frame(self, landmarks_list, img_shape):
        """Main vision analysis loop"""
        current_time = time.time()
        signals = {}
        
        # 1. EAR & PERCLOS
        left_ear = self.calculate_ear(landmarks_list, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks_list, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        
        # PERCLOS (Percent > 80% closed, threshold ~0.2)
        ear_closure_threshold = 0.2
        if len(self.ear_history) > 0:
            closed_frames = sum(1 for e in self.ear_history if e < ear_closure_threshold)
            perclos = closed_frames / len(self.ear_history)
        else:
            perclos = 0.0
            
        # Score PERCLOS
        if perclos > 0.15: signals['eye_closure_rate'] = 3
        elif perclos > 0.08: signals['eye_closure_rate'] = 2
        elif perclos > 0.04: signals['eye_closure_rate'] = 1
        else: signals['eye_closure_rate'] = 0
        
        # 2. Blink Pattern
        if avg_ear < 0.2 and not self.in_blink:
            self.blink_start = current_time
            self.in_blink = True
        elif avg_ear > 0.25 and self.in_blink:
            duration = current_time - self.blink_start
            self.blink_durations.append(duration)
            self.blink_timestamps.append(current_time)
            self.in_blink = False
            
        # Analyze blinks
        blink_score = 0
        blink_rate = 0.0
        if len(self.blink_timestamps) > 1:
            # Calculate rate per minute based on recent blinks
            recent_blinks = [t for t in self.blink_timestamps if current_time - t < 60.0]
            blink_rate = len(recent_blinks) # Blinks in last minute
            
        if len(self.blink_durations) >= 5:
            avg_duration = np.mean(self.blink_durations)
            # Duration check
            if avg_duration > 0.5: blink_score += 2
            elif avg_duration > 0.35: blink_score += 1
            
        signals['blink_pattern'] = min(blink_score, 3)

        # 3. Gaze Stability
        # Need Iris Landmarks
        gaze_variance = 0.0
        if len(landmarks_list) > 473:
            left_iris = landmarks_list[self.LEFT_IRIS]
            right_iris = landmarks_list[self.RIGHT_IRIS]
            # Just tracking mean position variance
            self.gaze_history.append((left_iris, right_iris))
            
            if len(self.gaze_history) >= 30:
                # Convert to numpy for var
                recent = list(self.gaze_history)[-30:]
                # Check variance of Left Iris X and Y
                l_x = [p[0][0] for p in recent]
                l_y = [p[0][1] for p in recent]
                var = np.var(l_x) + np.var(l_y)
                gaze_variance = var
                
                if self.baseline_established and self.baseline_gaze_variance > 0:
                    ratio = var / (self.baseline_gaze_variance + 1e-6)
                    if ratio > 3.0: signals['gaze_stability'] = 3
                    elif ratio > 2.0: signals['gaze_stability'] = 2
                    else: signals['gaze_stability'] = 0
                else:
                    self.baseline_gaze_variance = var # auto update if not established
                    signals['gaze_stability'] = 0
        else:
            signals['gaze_stability'] = 0

        # 4. Head Drift
        pitch, yaw, roll = self.estimate_head_pose(landmarks_list, img_shape)
        self.head_pose_history.append((pitch, yaw, roll))
        
        drift_score = 0
        if len(self.head_pose_history) > 60:
            recent_pitch = [p[0] for p in self.head_pose_history]
            # Check for excessive leaning/nodding
            sustained_lean = sum(1 for p in recent_pitch if abs(p) > 25) / len(recent_pitch)
            if sustained_lean > 0.3:
                drift_score += 1
            
            # Check for variance ("bobbing")
            var_pitch = np.var(recent_pitch)
            if var_pitch > 500: # similar to old logic
                drift_score += 2
                
        signals['head_pose_drift'] = min(drift_score, 3)
        
        # 5. Micro-sleep
        if avg_ear < 0.18:
            self.closure_counter += 1
        else:
            if 60 < self.closure_counter < 120: # 2-4 seconds
                self.microsleep_events.append(current_time)
            self.closure_counter = 0
        
        recent_sleeps = [t for t in self.microsleep_events if current_time - t < 300]
        if len(recent_sleeps) >= 3: signals['micro_sleep'] = 3
        elif len(recent_sleeps) >= 1: signals['micro_sleep'] = 2
        else: signals['micro_sleep'] = 0
        
        # 6. MAR
        mar = self.calculate_mar(landmarks_list)
        
        return signals, avg_ear, mar, (pitch, yaw, roll), perclos, blink_rate, gaze_variance

class VehicleDynamicsAnalyzer:
    def __init__(self):
        self.baseline_established = False
        # Placeholder for vehicle signal history
        self.steering_history = deque(maxlen=300)
        self.lane_deviation_history = deque(maxlen=300)
        
    def analyze_driving_pattern(self, vehicle_data: Dict[str, Any]):
        """Analyze vehicle telemetry"""
        signals = {
            'steering_instability': 0,
            'lane_weaving': 0,
            'pedal_jerkiness': 0,
            'speed_variance': 0
        }
        
        if not vehicle_data:
            return signals
            
        # Example Implementation for Steering
        steering = vehicle_data.get('steering_angle', None)
        if steering is not None:
            self.steering_history.append(steering)
            if len(self.steering_history) > 30:
                # Calculate entropy or high frequency changes
                # Simplified: count direction reversals
                changes = np.diff(list(self.steering_history)[-30:])
                reversals = sum(1 for i in range(len(changes)-1) if changes[i]*changes[i+1] < 0)
                if reversals > 10: # >10 reversals in 1 sec (30 frames)
                    signals['steering_instability'] = 2
                    
        # Lane Deviation
        lane_offset = vehicle_data.get('lane_offset', None)
        if lane_offset is not None:
             self.lane_deviation_history.append(lane_offset)
             # Check weaving (variance)
             if len(self.lane_deviation_history) > 60:
                 var = np.var(list(self.lane_deviation_history)[-60:])
                 if var > 0.5: # Arbitrary unit threshold
                     signals['lane_weaving'] = 2
                     
        return signals

class ImpairmentDetectionEngine:
    def __init__(self):
        self.vision = VisionAnalyzer()
        self.vehicle = VehicleDynamicsAnalyzer()
        
        self.confidence_history = deque(maxlen=90)
        self.state_history = deque(maxlen=30)
        
        # Weights
        self.weights = {
            'eye_closure_rate': 0.15,
            'micro_sleep': 0.15,
            'gaze_stability': 0.10,
            'blink_pattern': 0.05,
            'head_pose_drift': 0.05,
            'lane_weaving': 0.20,
            'steering_instability': 0.15,
            'pedal_jerkiness': 0.10,
            'speed_variance': 0.05
        }
        
    def process_frame(self, landmarks_list, img_shape, vehicle_data: Dict = None) -> Tuple[DriverState, DriverMetrics]:
        current_time = time.time()
        
        # 1. Vision Signals
        vision_signals, ear, mar, head_pose, perclos, blink_rate, gaze_var = self.vision.analyze_frame(landmarks_list, img_shape)
        
        # 2. Vehicle Signals
        vehicle_signals = self.vehicle.analyze_driving_pattern(vehicle_data or {})
        
        # 3. Fusion
        all_signals = {**vision_signals, **vehicle_signals}
        
        # Calculate Weighted Score
        total_score = 0.0
        max_possible = 0.0
        
        vision_keys = ['eye_closure_rate', 'micro_sleep', 'gaze_stability', 'blink_pattern', 'head_pose_drift']
        vehicle_keys = ['lane_weaving', 'steering_instability', 'pedal_jerkiness', 'speed_variance']
        
        has_vehicle_data = bool(vehicle_data)
        
        for key, weight in self.weights.items():
            # Skip vehicle weights if no vehicle data
            if not has_vehicle_data and key in vehicle_keys:
                continue
                
            val = all_signals.get(key, 0)
            total_score += val * weight
            max_possible += 3.0 * weight # Max signal is 3
            
        confidence = total_score / max_possible if max_possible > 0 else 0.0
        self.confidence_history.append(confidence)
        
        # Smooth confidence
        avg_confidence = np.mean(self.confidence_history)
        
        # Determine State
        if avg_confidence > 0.7:
            # More sensitive sleep detection
            # 1. High PERCLOS (>0.4)
            # 2. Any recent micro-sleep event (eyes closed > 2s)
            if perclos > 0.4 or all_signals.get('micro_sleep', 0) > 0: 
                state = DriverState.ASLEEP
            else:
                state = DriverState.HIGH_RISK # Intoxicated/Impaired
        elif avg_confidence > 0.4:
             # Check for sleep even in moderate confidence if signals are strong
            if perclos > 0.5 or all_signals.get('micro_sleep', 0) > 0:
                state = DriverState.ASLEEP
            else:
                state = DriverState.MODERATE_RISK
        elif avg_confidence > 0.2:
            state = DriverState.LOW_RISK
        else:
             # Safety net: if eyes are closed for long enough, it IS sleep regardless of confidence
            if perclos > 0.6 or all_signals.get('micro_sleep', 0) > 1:
                state = DriverState.ASLEEP
            else:
                state = DriverState.NORMAL
            
        # Specific Drowsy Detection (Legacy Support)
        # If PERCLOS is high (>0.15) but not yet 'ASLEEP'
        if state != DriverState.ASLEEP and perclos > 0.15:
            state = DriverState.DROWSY
            
        metrics = DriverMetrics(
            ear=ear,
            mar=mar,
            head_pose=head_pose,
            blink_rate=blink_rate,
            perclos=perclos,
            risk_score=total_score,
            confidence=avg_confidence,
            signals=all_signals,
            gaze_deviation=gaze_var,
            timestamp=current_time
        )
        
        return state, metrics

# --- ALIAS FOR BACKWARD COMPATIBILITY ---
# The app expects `EnhancedDriverDetector`
class EnhancedDriverDetector(ImpairmentDetectionEngine):
    pass