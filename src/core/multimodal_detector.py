import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from scipy import signal as scipy_signal
from scipy.stats import entropy

# --- CONSTANTS & HELPERS ---

def normalize_angle(angle):
    """Normalize angle to -180 to 180 range with sanity checks"""
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    # Physiologically impossible head poses
    if abs(angle) > 90:
        return 0.0
    return angle

def calculate_entropy(data: List[float], bins: int = 10) -> float:
    """Calculate Shannon entropy for pattern irregularity detection"""
    if len(data) < 2:
        return 0.0
    try:
        hist, _ = np.histogram(data, bins=bins, density=True)
        hist = hist[hist > 0]  # Remove zeros
        return entropy(hist)
    except:
        return 0.0

class DriverState(Enum):
    CALIBRATING = "Calibrating..."
    NORMAL = "Normal"
    LOW_RISK = "Low Risk"
    MODERATE_RISK = "Moderate Risk"
    HIGH_RISK = "High Risk"
    DROWSY = "Drowsy"
    ASLEEP = "Asleep"
    DISTRACTED = "Distracted"

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
    attention_score: float = 0.0
    cognitive_load: float = 0.0

# --- ADVANCED ANALYSIS MODULES ---

class AdaptiveVisionAnalyzer:
    """Enhanced vision analysis with adaptive thresholds and ML-ready features"""
    
    def __init__(self):
        self.baseline_established = False
        self.calibration_frames = 0
        self.baseline_window_frames = 5 * 30  # 5 seconds @ 30fps
        
        # Extended History Buffers with temporal analysis
        self.ear_history = deque(maxlen=3600)       # 2 min @ 30fps
        self.gaze_history = deque(maxlen=300)       # 10 sec high-res
        self.blink_durations = deque(maxlen=50)     # More samples
        self.blink_timestamps = deque(maxlen=100)
        self.head_pose_history = deque(maxlen=300)  # 10 sec
        self.microsleep_events = deque(maxlen=30)
        self.yawn_history = deque(maxlen=60)        # Yawn detection
        
        # Advanced State Tracking
        self.in_blink = False
        self.blink_start = 0.0
        self.closure_counter = 0
        self.distraction_counter = 0
        self.phone_use_counter = 0
        
        # Adaptive Thresholds (learned during calibration)
        self.adaptive_ear_threshold = 0.2
        self.adaptive_mar_threshold = 0.6
        self.baseline_gaze_variance = 0.0
        self.baseline_blink_rate = 15.0  # Average blinks/min
        self.baseline_head_variance = 0.0
        
        # Frequency Analysis Buffers
        self.ear_fft_buffer = deque(maxlen=256)  # Power for FFT
        
        # MediaPipe Landmark Indices
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        self.LEFT_IRIS = 473
        self.RIGHT_IRIS = 468
        self.MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409]
        
        # Performance metrics
        self.last_process_time = 0.0
        
    def calculate_ear(self, landmarks, indices) -> float:
        """Enhanced EAR with outlier rejection"""
        try:
            v1 = np.linalg.norm(np.array(landmarks[indices[1]]) - np.array(landmarks[indices[5]]))
            v2 = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[4]]))
            h = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[3]]))
            
            if h == 0:
                return 0.0
                
            ear = (v1 + v2) / (2.0 * h)
            
            # Outlier rejection (physiologically impossible values)
            if ear < 0.05 or ear > 0.5:
                return 0.0
                
            return ear
        except:
            return 0.0

    def calculate_mar(self, landmarks) -> float:
        """Enhanced MAR for yawn detection"""
        try:
            # Multiple vertical measurements for robustness
            v1 = np.linalg.norm(np.array(landmarks[13]) - np.array(landmarks[14]))
            v2 = np.linalg.norm(np.array(landmarks[82]) - np.array(landmarks[87]))
            h = np.linalg.norm(np.array(landmarks[61]) - np.array(landmarks[291]))
            
            if h == 0:
                return 0.0
                
            mar = (v1 + v2) / (2.0 * h)
            
            # Physiological bounds
            if mar < 0.1 or mar > 2.0:
                return 0.0
                
            return mar
        except:
            return 0.0

    def estimate_head_pose(self, landmarks, img_shape) -> Tuple[float, float, float]:
        """Robust head pose estimation with error handling"""
        try:
            # 3D model points (canonical face)
            model_points = np.array([
                (0.0, 0.0, 0.0),           # Nose tip
                (0.0, -330.0, -65.0),      # Chin
                (-225.0, 170.0, -135.0),   # Left eye corner
                (225.0, 170.0, -135.0),    # Right eye corner
                (-150.0, -150.0, -125.0),  # Left mouth corner
                (150.0, -150.0, -125.0)    # Right mouth corner
            ], dtype=np.float64)
            
            # 2D image points
            image_points = np.array([
                landmarks[1],    # Nose tip
                landmarks[152],  # Chin
                landmarks[263],  # Left eye corner
                landmarks[33],   # Right eye corner
                landmarks[61],   # Left mouth corner
                landmarks[291]   # Right mouth corner
            ], dtype=np.float64)
            
            # Camera matrix
            focal_length = img_shape[1]
            center = (img_shape[1] / 2, img_shape[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            dist_coeffs = np.zeros((4, 1))
            
            success, rot_vec, trans_vec = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                rot_mat, _ = cv2.Rodrigues(rot_vec)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                
                pitch = normalize_angle(angles[0] * 360)
                yaw = normalize_angle(angles[1] * 360)
                roll = normalize_angle(angles[2] * 360)
                
                return (pitch, yaw, roll)
        except Exception as e:
            pass
            
        return (0.0, 0.0, 0.0)

    def detect_phone_use(self, landmarks, img_shape) -> bool:
        """Detect potential phone usage from head angle"""
        pitch, yaw, roll = self.estimate_head_pose(landmarks, img_shape)
        
        # Phone usage: looking down (pitch > 30) with head tilted
        if pitch > 30 and abs(yaw) < 20:
            return True
        return False

    def analyze_blink_pattern(self, current_time: float) -> Tuple[float, int]:
        """Advanced blink pattern analysis"""
        blink_score = 0
        blink_rate = 0.0
        
        if len(self.blink_timestamps) > 1:
            # Calculate instantaneous rate
            recent_blinks = [t for t in self.blink_timestamps if current_time - t < 60.0]
            blink_rate = len(recent_blinks)
            
            # Abnormal blink rate detection
            if self.baseline_established:
                if blink_rate < 5:  # Too few (drowsiness)
                    blink_score += 2
                elif blink_rate > 30:  # Too many (stress/irritation)
                    blink_score += 1
            
        # Blink duration analysis
        if len(self.blink_durations) >= 10:
            avg_duration = np.mean(self.blink_durations)
            std_duration = np.std(self.blink_durations)
            
            # Prolonged blinks (drowsiness indicator)
            if avg_duration > 0.5:
                blink_score += 3
            elif avg_duration > 0.35:
                blink_score += 2
            elif avg_duration > 0.25:
                blink_score += 1
                
            # Irregular blink pattern (high variance)
            if std_duration > 0.15:
                blink_score += 1
                
        return blink_rate, min(blink_score, 3)

    def analyze_gaze_stability(self, landmarks, current_time: float) -> Tuple[float, int]:
        """Multi-modal gaze stability analysis"""
        gaze_variance = 0.0
        gaze_score = 0
        
        if len(landmarks) > 473:
            left_iris = landmarks[self.LEFT_IRIS]
            right_iris = landmarks[self.RIGHT_IRIS]
            
            # Store iris positions
            self.gaze_history.append({
                'left': left_iris,
                'right': right_iris,
                'time': current_time
            })
            
            if len(self.gaze_history) >= 60:
                recent = list(self.gaze_history)[-60:]
                
                # Calculate 2D variance
                left_x = [p['left'][0] for p in recent]
                left_y = [p['left'][1] for p in recent]
                
                var_x = np.var(left_x)
                var_y = np.var(left_y)
                gaze_variance = var_x + var_y
                
                # Frequency analysis (saccade detection)
                gaze_entropy = calculate_entropy(left_x)
                
                if self.baseline_established and self.baseline_gaze_variance > 0:
                    ratio = gaze_variance / (self.baseline_gaze_variance + 1e-6)
                    
                    # Excessive wandering
                    if ratio > 4.0:
                        gaze_score = 3
                    elif ratio > 2.5:
                        gaze_score = 2
                    elif ratio > 1.5:
                        gaze_score = 1
                        
                    # Fixation (too little movement - microsleep)
                    if ratio < 0.3 and gaze_entropy < 1.0:
                        gaze_score = max(gaze_score, 2)
                else:
                    # Update baseline during calibration
                    if self.calibration_frames < self.baseline_window_frames:
                        self.baseline_gaze_variance = gaze_variance
                        
        return gaze_variance, gaze_score

    def analyze_head_dynamics(self) -> int:
        """Advanced head pose dynamics analysis"""
        if len(self.head_pose_history) < 90:
            return 0
            
        recent = list(self.head_pose_history)[-90:]
        pitches = [p[0] for p in recent]
        yaws = [p[1] for p in recent]
        rolls = [p[2] for p in recent]
        
        drift_score = 0
        
        # 1. Sustained abnormal position (nodding off)
        sustained_lean = sum(1 for p in pitches if abs(p) > 25) / len(pitches)
        if sustained_lean > 0.4:
            drift_score += 2
        elif sustained_lean > 0.25:
            drift_score += 1
            
        # 2. Head bobbing (variance analysis)
        var_pitch = np.var(pitches)
        var_yaw = np.var(yaws)
        
        if var_pitch > 600 or var_yaw > 600:
            drift_score += 2
        elif var_pitch > 350 or var_yaw > 350:
            drift_score += 1
            
        # 3. Frequency domain analysis (detect rhythmic nodding)
        if len(pitches) >= 128:
            try:
                # FFT for periodic nodding detection
                fft = np.fft.fft(pitches[-128:])
                freqs = np.fft.fftfreq(128, d=1/30.0)  # 30fps
                
                # Peak detection in 0.1-0.5 Hz range (slow nodding)
                low_freq_mask = (np.abs(freqs) > 0.1) & (np.abs(freqs) < 0.5)
                low_freq_power = np.sum(np.abs(fft[low_freq_mask]))
                
                if low_freq_power > 1000:  # Threshold for rhythmic nodding
                    drift_score += 2
            except:
                pass
                
        return min(drift_score, 3)

    def detect_microsleep(self, ear: float, current_time: float) -> int:
        """Enhanced microsleep detection with temporal patterns"""
        # Track eye closure duration
        if ear < 0.18:
            self.closure_counter += 1
        else:
            # Microsleep: 2-4 seconds of eye closure
            if 60 <= self.closure_counter <= 120:  # 2-4 sec @ 30fps
                self.microsleep_events.append({
                    'time': current_time,
                    'duration': self.closure_counter / 30.0
                })
            self.closure_counter = 0
            
        # Analyze microsleep frequency
        recent_events = [e for e in self.microsleep_events 
                        if current_time - e['time'] < 300]  # Last 5 min
        
        if len(recent_events) >= 5:
            return 3
        elif len(recent_events) >= 3:
            return 2
        elif len(recent_events) >= 1:
            return 1
            
        return 0

    def detect_yawning(self, mar: float, current_time: float) -> int:
        """Yawn detection and frequency analysis"""
        self.yawn_history.append(mar)
        
        # Yawn: sustained high MAR
        if mar > self.adaptive_mar_threshold and len(self.yawn_history) >= 15:
            recent_mar = list(self.yawn_history)[-15:]
            sustained_opening = sum(1 for m in recent_mar if m > self.adaptive_mar_threshold)
            
            if sustained_opening > 10:  # >66% of last 0.5 sec
                yawn_score = 2
                return yawn_score
                
        return 0

    def calibrate_baseline(self):
        """Update adaptive thresholds based on collected data"""
        if self.calibration_frames >= self.baseline_window_frames:
            if not self.baseline_established:
                # Calculate adaptive EAR threshold
                if len(self.ear_history) > 1000:
                    ear_array = np.array(list(self.ear_history))
                    self.adaptive_ear_threshold = np.percentile(ear_array, 20)
                    
                # Calculate adaptive MAR threshold
                if len(self.yawn_history) > 100:
                    mar_array = np.array(list(self.yawn_history))
                    self.adaptive_mar_threshold = np.percentile(mar_array, 80)
                    
                # Baseline blink rate
                if len(self.blink_timestamps) > 10:
                    total_time = max(self.blink_timestamps) - min(self.blink_timestamps)
                    if total_time > 0:
                        self.baseline_blink_rate = len(self.blink_timestamps) / (total_time / 60.0)
                        
                self.baseline_established = True
                
        self.calibration_frames += 1

    def analyze_frame(self, landmarks_list, img_shape) -> Tuple[Dict, float, float, Tuple, float, float, float, float]:
        """Main vision analysis pipeline"""
        start_time = time.time()
        current_time = time.time()
        signals = {}
        
        # 1. Eye Aspect Ratio & PERCLOS
        left_ear = self.calculate_ear(landmarks_list, self.LEFT_EYE)
        right_ear = self.calculate_ear(landmarks_list, self.RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        
        self.ear_history.append(avg_ear)
        self.ear_fft_buffer.append(avg_ear)
        
        # PERCLOS calculation
        ear_threshold = self.adaptive_ear_threshold
        if len(self.ear_history) > 30:
            closed_frames = sum(1 for e in list(self.ear_history)[-1800:] if e < ear_threshold)
            perclos = closed_frames / min(len(self.ear_history), 1800)
        else:
            perclos = 0.0
            
        # PERCLOS scoring (more granular)
        if perclos > 0.25:
            signals['eye_closure_rate'] = 3
        elif perclos > 0.15:
            signals['eye_closure_rate'] = 2
        elif perclos > 0.08:
            signals['eye_closure_rate'] = 1
        else:
            signals['eye_closure_rate'] = 0
            
        # 2. Blink Analysis
        if avg_ear < ear_threshold and not self.in_blink:
            self.blink_start = current_time
            self.in_blink = True
        elif avg_ear > (ear_threshold + 0.05) and self.in_blink:
            duration = current_time - self.blink_start
            if 0.05 < duration < 1.0:  # Valid blink
                self.blink_durations.append(duration)
                self.blink_timestamps.append(current_time)
            self.in_blink = False
            
        blink_rate, blink_score = self.analyze_blink_pattern(current_time)
        signals['blink_pattern'] = blink_score
        
        # 3. Gaze Stability
        gaze_variance, gaze_score = self.analyze_gaze_stability(landmarks_list, current_time)
        signals['gaze_stability'] = gaze_score
        
        # 4. Head Pose Analysis
        pitch, yaw, roll = self.estimate_head_pose(landmarks_list, img_shape)
        self.head_pose_history.append((pitch, yaw, roll))
        
        head_score = self.analyze_head_dynamics()
        signals['head_pose_drift'] = head_score
        
        # 5. Distraction Detection
        if self.detect_phone_use(landmarks_list, img_shape):
            self.phone_use_counter += 1
        else:
            self.phone_use_counter = max(0, self.phone_use_counter - 1)
            
        if abs(yaw) > 45 or abs(pitch) > 35:
            self.distraction_counter += 1
        else:
            self.distraction_counter = max(0, self.distraction_counter - 1)
            
        distraction_score = 0
        if self.distraction_counter > 90:  # 3 sec sustained
            distraction_score = 3
        elif self.distraction_counter > 30:
            distraction_score = 2
        elif self.phone_use_counter > 30:
            distraction_score = 2
            
        signals['distraction'] = distraction_score
        
        # 6. Microsleep Detection
        microsleep_score = self.detect_microsleep(avg_ear, current_time)
        signals['micro_sleep'] = microsleep_score
        
        # 7. Yawn Detection
        mar = self.calculate_mar(landmarks_list)
        yawn_score = self.detect_yawning(mar, current_time)
        signals['yawning'] = yawn_score
        
        # 8. Attention Score (0-1, higher is better)
        attention_score = 1.0 - (
            signals['eye_closure_rate'] * 0.2 +
            signals['gaze_stability'] * 0.15 +
            signals['distraction'] * 0.25 +
            signals['head_pose_drift'] * 0.1
        ) / 3.0
        attention_score = max(0.0, min(1.0, attention_score))
        
        # Update calibration
        self.calibrate_baseline()
        
        self.last_process_time = time.time() - start_time
        
        return (signals, avg_ear, mar, (pitch, yaw, roll), 
                perclos, blink_rate, gaze_variance, attention_score)


class VehicleDynamicsAnalyzer:
    """Enhanced vehicle telemetry analysis"""
    
    def __init__(self):
        self.baseline_established = False
        
        # Extended history buffers
        self.steering_history = deque(maxlen=600)      # 20 sec @ 30Hz
        self.lane_deviation_history = deque(maxlen=600)
        self.speed_history = deque(maxlen=600)
        self.acceleration_history = deque(maxlen=600)
        
        # Baseline metrics
        self.baseline_steering_entropy = 0.0
        self.baseline_lane_variance = 0.0
        
    def analyze_steering_pattern(self, steering_angle: float) -> int:
        """Analyze steering micro-corrections and weaving"""
        self.steering_history.append(steering_angle)
        
        if len(self.steering_history) < 90:
            return 0
            
        recent = list(self.steering_history)[-90:]
        
        # 1. Count direction reversals (micro-corrections)
        changes = np.diff(recent)
        reversals = sum(1 for i in range(len(changes) - 1) 
                       if changes[i] * changes[i+1] < 0)
        
        # 2. Calculate steering entropy
        steering_entropy = calculate_entropy(recent, bins=15)
        
        score = 0
        if reversals > 15:  # >15 reversals in 3 sec
            score += 2
        elif reversals > 10:
            score += 1
            
        if steering_entropy > 2.5:
            score += 1
            
        return min(score, 3)

    def analyze_lane_position(self, lane_offset: float) -> int:
        """Analyze lane keeping and weaving patterns"""
        self.lane_deviation_history.append(lane_offset)
        
        if len(self.lane_deviation_history) < 90:
            return 0
            
        recent = list(self.lane_deviation_history)[-180:]
        
        # 1. Variance analysis (weaving)
        variance = np.var(recent)
        
        # 2. Trend analysis (drifting)
        mean_deviation = np.mean(recent)
        
        # 3. Crossing frequency
        crossings = sum(1 for i in range(len(recent) - 1)
                       if recent[i] * recent[i+1] < 0)
        
        score = 0
        if variance > 0.8:
            score += 2
        elif variance > 0.5:
            score += 1
            
        if abs(mean_deviation) > 0.7:
            score += 1
            
        if crossings > 5:  # Crossing center multiple times
            score += 1
            
        return min(score, 3)

    def analyze_speed_pattern(self, speed: float) -> int:
        """Analyze speed variance and control"""
        self.speed_history.append(speed)
        
        if len(self.speed_history) < 90:
            return 0
            
        recent = list(self.speed_history)[-180:]
        
        # Calculate variance and coefficient of variation
        std = np.std(recent)
        mean = np.mean(recent)
        
        if mean > 0:
            cv = std / mean
        else:
            cv = 0
            
        score = 0
        if cv > 0.15:  # High variance
            score = 2
        elif cv > 0.10:
            score = 1
            
        return score

    def analyze_pedal_control(self, acceleration: float) -> int:
        """Analyze acceleration smoothness"""
        self.acceleration_history.append(acceleration)
        
        if len(self.acceleration_history) < 60:
            return 0
            
        recent = list(self.acceleration_history)[-60:]
        
        # Jerk analysis (rate of change of acceleration)
        jerk = np.diff(recent)
        jerk_variance = np.var(jerk)
        
        score = 0
        if jerk_variance > 5.0:
            score = 2
        elif jerk_variance > 2.5:
            score = 1
            
        return score

    def analyze_driving_pattern(self, vehicle_data: Dict[str, Any]) -> Dict[str, int]:
        """Comprehensive vehicle dynamics analysis"""
        signals = {
            'steering_instability': 0,
            'lane_weaving': 0,
            'pedal_jerkiness': 0,
            'speed_variance': 0
        }
        
        if not vehicle_data:
            return signals
            
        # Steering analysis
        if 'steering_angle' in vehicle_data:
            signals['steering_instability'] = self.analyze_steering_pattern(
                vehicle_data['steering_angle']
            )
            
        # Lane position analysis
        if 'lane_offset' in vehicle_data:
            signals['lane_weaving'] = self.analyze_lane_position(
                vehicle_data['lane_offset']
            )
            
        # Speed analysis
        if 'speed' in vehicle_data:
            signals['speed_variance'] = self.analyze_speed_pattern(
                vehicle_data['speed']
            )
            
        # Acceleration analysis
        if 'acceleration' in vehicle_data:
            signals['pedal_jerkiness'] = self.analyze_pedal_control(
                vehicle_data['acceleration']
            )
            
        return signals


class MultiModalFusionEngine:
    """Advanced sensor fusion with Bayesian reasoning"""
    
    def __init__(self):
        self.vision = AdaptiveVisionAnalyzer()
        self.vehicle = VehicleDynamicsAnalyzer()
        
        # State tracking
        self.confidence_history = deque(maxlen=30)  # 1 sec smoothing
        self.state_history = deque(maxlen=60)
        self.risk_trajectory = deque(maxlen=300)
        
        # Dynamic weight adaptation
        self.weights = {
            # Vision signals (60% weight)
            'eye_closure_rate': 0.15,
            'micro_sleep': 0.15,
            'gaze_stability': 0.10,
            'blink_pattern': 0.05,
            'head_pose_drift': 0.05,
            'yawning': 0.05,
            'distraction': 0.05,
            
            # Vehicle signals (40% weight)
            'lane_weaving': 0.15,
            'steering_instability': 0.15,
            'pedal_jerkiness': 0.05,
            'speed_variance': 0.05
        }
        
        # Bayesian priors
        self.state_transition_matrix = {
            DriverState.NORMAL: {DriverState.NORMAL: 0.85, DriverState.LOW_RISK: 0.10, DriverState.DISTRACTED: 0.05},
            DriverState.LOW_RISK: {DriverState.NORMAL: 0.30, DriverState.LOW_RISK: 0.50, DriverState.MODERATE_RISK: 0.15, DriverState.DROWSY: 0.05},
            DriverState.MODERATE_RISK: {DriverState.LOW_RISK: 0.20, DriverState.MODERATE_RISK: 0.50, DriverState.HIGH_RISK: 0.20, DriverState.DROWSY: 0.10},
            DriverState.DROWSY: {DriverState.MODERATE_RISK: 0.15, DriverState.DROWSY: 0.50, DriverState.ASLEEP: 0.30, DriverState.HIGH_RISK: 0.05},
            DriverState.HIGH_RISK: {DriverState.MODERATE_RISK: 0.20, DriverState.HIGH_RISK: 0.60, DriverState.ASLEEP: 0.20},
            DriverState.ASLEEP: {DriverState.ASLEEP: 0.70, DriverState.DROWSY: 0.25, DriverState.HIGH_RISK: 0.05},
            DriverState.DISTRACTED: {DriverState.NORMAL: 0.40, DriverState.DISTRACTED: 0.40, DriverState.LOW_RISK: 0.20}
        }
        
    def calculate_risk_score(self, signals: Dict[str, int], has_vehicle_data: bool) -> float:
        """Calculate weighted risk score with normalization"""
        total_score = 0.0
        max_possible = 0.0
        
        vision_keys = ['eye_closure_rate', 'micro_sleep', 'gaze_stability', 
                      'blink_pattern', 'head_pose_drift', 'yawning', 'distraction']
        vehicle_keys = ['lane_weaving', 'steering_instability', 
                       'pedal_jerkiness', 'speed_variance']
        
        for key, weight in self.weights.items():
            # Skip vehicle weights if no data
            if not has_vehicle_data and key in vehicle_keys:
                continue
                
            signal_value = signals.get(key, 0)
            total_score += signal_value * weight
            max_possible += 3.0 * weight
            
        # Normalize to 0-1
        if max_possible > 0:
            normalized_score = total_score / max_possible
        else:
            normalized_score = 0.0
            
        return normalized_score

    def determine_state(self, confidence: float, signals: Dict[str, int], 
                       perclos: float, attention: float) -> DriverState:
        """Multi-factor state determination with hysteresis"""
        
        # Critical sleep indicators
        has_microsleep = signals.get('micro_sleep', 0) > 0
        has_severe_closure = perclos > 0.4
        has_extended_closure = perclos > 0.6
        
        # Drowsiness indicators
        is_drowsy_signal = (perclos > 0.15 or signals.get('yawning', 0) > 1 or
                           signals.get('eye_closure_rate', 0) > 1)
        
        # Distraction indicators
        is_distracted = signals.get('distraction', 0) >= 2
        
        # State determination with clear hierarchy
        if has_extended_closure or (has_severe_closure and has_microsleep):
            return DriverState.ASLEEP
            
        elif confidence > 0.7:
            if has_microsleep or has_severe_closure:
                return DriverState.ASLEEP
            else:
                return DriverState.HIGH_RISK
                
        elif confidence > 0.5:
            if is_drowsy_signal:
                return DriverState.DROWSY
            else:
                return DriverState.MODERATE_RISK
                
        elif confidence > 0.3:
            if has_severe_closure:
                return DriverState.ASLEEP
            elif is_drowsy_signal:
                return DriverState.DROWSY
            elif is_distracted:
                return DriverState.DISTRACTED
            else:
                return DriverState.LOW_RISK
                
        else:
            # Safety net for sleep even at low confidence
            if has_severe_closure:
                return DriverState.ASLEEP
            elif is_drowsy_signal:
                return DriverState.DROWSY
            elif is_distracted:
                return DriverState.DISTRACTED
            else:
                return DriverState.NORMAL

    def apply_temporal_smoothing(self, current_state: DriverState) -> DriverState:
        """Apply Bayesian state transition smoothing"""
        if len(self.state_history) < 10:
            self.state_history.append(current_state)
            return current_state
            
        # Get most recent stable state
        recent_states = list(self.state_history)[-10:]
        
        # Count state occurrences
        state_counts = {}
        for state in recent_states:
            state_counts[state] = state_counts.get(state, 0) + 1
            
        # Require consistency for critical states
        if current_state in [DriverState.ASLEEP, DriverState.HIGH_RISK]:
            # Need 3+ confirmations in last 10 frames
            if state_counts.get(current_state, 0) >= 3:
                self.state_history.append(current_state)
                return current_state
            else:
                # Keep previous state
                return self.state_history[-1] if self.state_history else current_state
        else:
            # Accept state change for non-critical
            self.state_history.append(current_state)
            return current_state

    def process_frame(self, landmarks_list, img_shape, 
                     vehicle_data: Optional[Dict] = None) -> Tuple[DriverState, DriverMetrics]:
        """Main fusion pipeline"""
        current_time = time.time()
        
        # 1. Vision Analysis
        (vision_signals, ear, mar, head_pose, perclos, 
         blink_rate, gaze_var, attention) = self.vision.analyze_frame(landmarks_list, img_shape)
        
        # 2. Vehicle Analysis
        vehicle_signals = self.vehicle.analyze_driving_pattern(vehicle_data or {})
        
        # 3. Merge signals
        all_signals = {**vision_signals, **vehicle_signals}
        
        # 4. Calculate risk score
        has_vehicle = bool(vehicle_data)
        raw_confidence = self.calculate_risk_score(all_signals, has_vehicle)
        self.confidence_history.append(raw_confidence)
        
        # Temporal smoothing of confidence
        smoothed_confidence = np.mean(self.confidence_history)
        
        # 5. Determine state
        raw_state = self.determine_state(smoothed_confidence, all_signals, perclos, attention)
        final_state = self.apply_temporal_smoothing(raw_state)
        
        # 6. Track risk trajectory
        self.risk_trajectory.append(smoothed_confidence)
        
        # 7. Calculate cognitive load estimate
        cognitive_load = 0.0
        if has_vehicle:
            # High cognitive load = high variance in controls
            steering_var = all_signals.get('steering_instability', 0) / 3.0
            speed_var = all_signals.get('speed_variance', 0) / 3.0
            cognitive_load = (steering_var + speed_var) / 2.0
        
        # 8. Build metrics
        metrics = DriverMetrics(
            ear=ear,
            mar=mar,
            head_pose=head_pose,
            blink_rate=blink_rate,
            perclos=perclos,
            risk_score=raw_confidence,
            confidence=smoothed_confidence,
            signals=all_signals,
            gaze_deviation=gaze_var,
            timestamp=current_time,
            attention_score=attention,
            cognitive_load=cognitive_load
        )
        
        return final_state, metrics


# --- BACKWARD COMPATIBILITY ALIASES ---

class EnhancedDriverDetector(MultiModalFusionEngine):
    """Alias for backward compatibility"""
    pass

class ImpairmentDetectionEngine(MultiModalFusionEngine):
    """Alias for backward compatibility"""
    pass