#!/usr/bin/env python3
"""
GuardianDrive AI - Integrated Safety Platform
Complete multimodal detection with risk mapping, insurance bridge, and stakeholder alerts
"""

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import threading
import time
import base64
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.multimodal_detector import EnhancedDriverDetector, DriverState
from src.utils.alert_system import SecureAlertSystem
from src.utils.risk_mapping import RiskMappingSystem
from src.utils.insurance_bridge import InsuranceDataBridge, DrivingSession
from src.utils.stakeholder_alerts import MultiStakeholderAlertSystem

# Global instances - integrated into the system
risk_mapper = RiskMappingSystem()
insurance_bridge = InsuranceDataBridge("DRIVER_001")
stakeholder_alerts = MultiStakeholderAlertSystem()

# Enhanced thresholds based on research
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.6  # Yawning threshold
CONSEC_FRAMES = 20
PERCLOS_THRESHOLD = 0.8

# PWA Static Files Setup
def serve_pwa_files():
    """Add routes for PWA files"""
    
    # Get base64 encoded icons
    def get_icon_base64(icon_path):
        try:
            if os.path.exists(icon_path):
                with open(icon_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode()
        except:
            pass
        return ""
    
    # Get manifest base64
    def get_manifest_base64():
        manifest_path = 'manifest.json'
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, 'rb') as f:
                    return base64.b64encode(f.read()).decode()
            except:
                pass
        return base64.b64encode(b'{}').decode()
    
    icon_192 = get_icon_base64('icons/icon-192x192.png')
    manifest_b64 = get_manifest_base64()
    
    # Add custom CSS and PWA meta tags
    pwa_head = f"""
    <link rel="manifest" href="data:application/json;base64,{manifest_b64}">
    <meta name="theme-color" content="#ff6b6b">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="DrowsinessDetect">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    
    {f'<link rel="icon" type="image/png" sizes="192x192" href="data:image/png;base64,{icon_192}">' if icon_192 else ''}
    {f'<link rel="apple-touch-icon" href="data:image/png;base64,{icon_192}">' if icon_192 else ''}
    
    <style>
        .install-button {{
            position: fixed;
            top: 10px;
            right: 10px;
            background: #ff6b6b;
            color: white;
            border: none;
            padding: 8px 12px;
            border-radius: 6px;
            cursor: pointer;
            z-index: 1000;
            font-size: 12px;
            display: none;
        }}
        .install-button:hover {{
            background: #ff5252;
        }}
        
        @media (max-width: 768px) {{
            .main .block-container {{
                padding-top: 2rem;
                padding-bottom: 2rem;
            }}
        }}
        
        .viewerBadge_container__1QSob {{
            display: none;
        }}
        
        footer {{
            visibility: hidden;
        }}
    </style>
    """
    
    # PWA JavaScript for service worker and install prompt
    pwa_js = """
    <script>
        // Register service worker
        if ('serviceWorker' in navigator) {
            window.addEventListener('load', function() {
                // Create service worker content as data URL
                const swContent = `
                    const CACHE_NAME = 'drowsiness-detection-v1';
                    const urlsToCache = ['/'];
                    
                    self.addEventListener('install', (event) => {
                        event.waitUntil(
                            caches.open(CACHE_NAME)
                                .then((cache) => cache.addAll(urlsToCache))
                        );
                    });
                    
                    self.addEventListener('fetch', (event) => {
                        if (event.request.url.includes('webrtc') || 
                            event.request.url.includes('ws://') || 
                            event.request.url.includes('wss://')) {
                            return;
                        }
                        
                        event.respondWith(
                            caches.match(event.request)
                                .then((response) => response || fetch(event.request))
                        );
                    });
                `;
                
                const blob = new Blob([swContent], { type: 'application/javascript' });
                const swUrl = URL.createObjectURL(blob);
                
                navigator.serviceWorker.register(swUrl)
                    .then(function(registration) {
                        console.log('ServiceWorker registration successful');
                    })
                    .catch(function(err) {
                        console.log('ServiceWorker registration failed: ', err);
                    });
            });
        }
        
        // Install prompt
        let deferredPrompt;
        const installButton = document.createElement('button');
        installButton.textContent = '📱 Install App';
        installButton.className = 'install-button';
        
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            installButton.style.display = 'block';
            document.body.appendChild(installButton);
        });
        
        installButton.addEventListener('click', () => {
            if (deferredPrompt) {
                deferredPrompt.prompt();
                deferredPrompt.userChoice.then((choiceResult) => {
                    if (choiceResult.outcome === 'accepted') {
                        installButton.style.display = 'none';
                    }
                    deferredPrompt = null;
                });
            }
        });
        
        // Add to home screen for iOS
        if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {
            const iosInstall = document.createElement('div');
            iosInstall.innerHTML = `
                <div style="position: fixed; bottom: 20px; left: 20px; right: 20px; background: #ff6b6b; color: white; padding: 15px; border-radius: 8px; text-align: center; z-index: 1000;">
                    📱 Install this app: Tap <strong>Share</strong> then <strong>Add to Home Screen</strong>
                    <button onclick="this.parentElement.remove()" style="position: absolute; top: 5px; right: 10px; background: none; border: none; color: white; font-size: 16px;">×</button>
                </div>
            `;
            document.body.appendChild(iosInstall);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (iosInstall.parentElement) {
                    iosInstall.remove();
                }
            }, 10000);
        }
    </script>
    """
    
    # Inject PWA components
    st.markdown(pwa_head + pwa_js, unsafe_allow_html=True)

# Load alarm
def play_alarm():
    """Play audio alert using pygame (Cross-platform & Robust)."""
    import os
    import time
    
    try:
        # Determine the root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        
        # Potential paths for the alarm file
        potential_paths = [
            os.path.join(root_dir, 'alarm.mp3'),
            os.path.join(root_dir, 'alarm.wav'),
            os.path.join(current_dir, 'alarm.mp3'),
            os.path.abspath('alarm.mp3')
        ]
        
        target_alarm = None
        for p in potential_paths:
            if os.path.exists(p):
                target_alarm = p
                break
        
        if not target_alarm:
            print(f"❌ Alarm file not found.")
            return

        # Initialize pygame mixer
        import pygame
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(target_alarm)
            pygame.mixer.music.play()
            
            # Wait for audio to finish (non-blocking in main thread, but blocking here for the thread)
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except pygame.error as e:
            print(f"⚠️ Pygame audio error (No audio device on server?): {e}")
        except Exception as e:
            print(f"⚠️ Audio playback failed: {e}")
            
    except ImportError:
        print("❌ 'pygame' not installed.")
    except Exception as e:
        print(f"❌ Audio system error: {e}")



alarm_thread = None

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices):
    # Eye landmarks
    p1, p2 = landmarks[eye_indices[1]], landmarks[eye_indices[5]]
    p3, p4 = landmarks[eye_indices[2]], landmarks[eye_indices[4]]
    p5, p6 = landmarks[eye_indices[0]], landmarks[eye_indices[3]]

    # Calculate vertical eye distances
    # Point 1 (top-left) to Point 5 (bottom-left)
    v1 = euclidean(landmarks[eye_indices[1]], landmarks[eye_indices[5]])
    # Point 2 (top-right) to Point 4 (bottom-right)
    v2 = euclidean(landmarks[eye_indices[2]], landmarks[eye_indices[4]])
    
    # Calculate horizontal eye distance
    # Point 0 (leftmost) to Point 3 (rightmost)
    h = euclidean(landmarks[eye_indices[0]], landmarks[eye_indices[3]])

    # Prevent division by zero
    if h == 0:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear

class GuardianDriveDetector(VideoProcessorBase):
    def __init__(self):
        # Load Face Landmarker
        model_path = os.path.join('..', 'models', 'face_landmarker.task')
        if not os.path.exists(model_path):
            model_path = os.path.join('models', 'face_landmarker.task')
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at {model_path}")
            raise FileNotFoundError(f"face_landmarker.task not found")
        
        try:
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
                num_faces=1
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            self.enhanced_detector = EnhancedDriverDetector()
            self.alert_system = SecureAlertSystem()
        except Exception as e:
            st.error(f"❌ Failed to initialize detector: {e}")
            raise
        
        # Session tracking for insurance
        self.session_start = time.time()
        self.session_id = f"SESSION_{int(self.session_start)}"
        self.current_state = DriverState.SOBER_ALERT
        self.state_duration = 0
        self.last_alert_time = 0
        self.total_alerts = 0
        self.drowsy_events = 0
        self.metrics_history = []
        
    def recv(self, frame):
        global alarm_thread
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # --- LIGHTING NORMALIZATION (For Indian Skin Tones) ---
            # Apply CLAHE to L channel of LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img_enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
            # Use enhanced image for detection
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB))
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                for face_landmarks in detection_result.face_landmarks:
                    h, w, _ = img.shape
                    landmarks = [(int(l.x * w), int(l.y * h)) for l in face_landmarks]
                    
                    # Process with enhanced detector
                    state, metrics = self.enhanced_detector.process_frame(landmarks, (h, w))
                    
                    # Update state tracking
                    if state == self.current_state:
                        self.state_duration += 1/30  # Assuming 30 FPS
                    else:
                        self.current_state = state
                        self.state_duration = 0
                    
                    # Store metrics
                    self.metrics_history.append(metrics)
                    if len(self.metrics_history) > 100:
                        self.metrics_history.pop(0)
                    
                    # Get location for risk mapping
                    location = {"lat": 28.6139, "lng": 77.2090}  # Mock GPS
                    current_time = time.time()
                    
                    # Handle critical states with full GuardianDrive AI integration
                    if state in [DriverState.DROWSY, DriverState.ASLEEP, DriverState.INTOXICATED]:
                        if current_time - self.last_alert_time > 5.0:
                            self.total_alerts += 1
                            
                            if state == DriverState.DROWSY:
                                self.drowsy_events += 1
                            
                            # 1. Log to Risk Mapping System
                            severity = 5 if state == DriverState.ASLEEP else 4 if state == DriverState.INTOXICATED else 3
                            risk_mapper.log_risk_event(
                                location["lat"], location["lng"],
                                state.value.lower(), severity
                            )
                            
                            # 2. Trigger Multi-Stakeholder Alerts for severe cases
                            if state in [DriverState.ASLEEP, DriverState.INTOXICATED]:
                                stakeholder_alerts.trigger_coordinated_response(
                                    driver_state=state.value,
                                    location=location,
                                    vehicle_speed=60,
                                    duration=self.state_duration
                                )
                            
                            # 3. Regular Alert System
                            self.alert_system.trigger_alert(
                                driver_state=state.value,
                                metrics={
                                    'ear': metrics.ear,
                                    'mar': metrics.mar,
                                    'perclos': metrics.perclos,
                                    'confidence': 0.85
                                },
                                duration=self.state_duration,
                                confidence=0.85
                            )
                            
                            self.last_alert_time = current_time
                        
                        # Play alarm for critical states
                        if state in [DriverState.ASLEEP, DriverState.INTOXICATED]:
                            if alarm_thread is None or not alarm_thread.is_alive():
                                alarm_thread = threading.Thread(target=play_alarm)
                                alarm_thread.start()
                    
                    # Visualize landmarks and metrics
                    self.draw_enhanced_visualization(img, landmarks, metrics, state)
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)[:40]}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Detection error: {e}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def draw_enhanced_visualization(self, img, landmarks, metrics, state):
        """Draw enhanced visualization with all metrics"""
        # MediaPipe face landmarker has 478 landmarks
        # Draw ALL eye landmarks for visibility
        # Left eye: 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
        # Right eye: 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
        
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Draw left eye
        for idx in left_eye_indices:
            if idx < len(landmarks):
                cv2.circle(img, landmarks[idx], 2, (0, 255, 0), -1)
        
        # Draw right eye
        for idx in right_eye_indices:
            if idx < len(landmarks):
                cv2.circle(img, landmarks[idx], 2, (0, 255, 0), -1)
        
        # Draw mouth landmarks
        mouth_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]
        for idx in mouth_indices:
            if idx < len(landmarks):
                cv2.circle(img, landmarks[idx], 1, (255, 0, 0), -1)
        
        # State indicator with color coding
        state_colors = {
            DriverState.SOBER_ALERT: (0, 255, 0),
            DriverState.DROWSY: (0, 255, 255),
            DriverState.ASLEEP: (0, 0, 255),
            DriverState.INTOXICATED: (255, 0, 255)
        }
        
        color = state_colors.get(state, (255, 255, 255))
        
        # Main state display
        cv2.putText(img, f"State: {state.value}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Metrics display
        y_offset = 60
        metrics_text = [
            f"EAR: {metrics.ear:.3f}",
            f"MAR: {metrics.mar:.3f}",
            f"PERCLOS: {metrics.perclos:.2f}",
            f"Blink Rate: {metrics.blink_rate:.1f}/min",
            f"Head Pose: P{metrics.head_pose[0]:.1f} Y{metrics.head_pose[1]:.1f} R{metrics.head_pose[2]:.1f}",
            f"Gaze Dev: {metrics.gaze_deviation:.2f}",
            f"Duration: {self.state_duration:.1f}s"
        ]
        
        for text in metrics_text:
            cv2.putText(img, text, (30, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
        
        # Alert indicator
        if state != DriverState.SOBER_ALERT:
            cv2.rectangle(img, (10, 10), (img.shape[1]-10, 50), color, 3)
            if state == DriverState.ASLEEP:
                cv2.putText(img, "CRITICAL ALERT!", (img.shape[1]//2-100, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Main Streamlit app
def main():
    # Change to streamlit_app directory for proper file paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Set page config
    st.set_page_config(
        page_title="GuardianDrive AI", 
        layout="wide",
        page_icon="🛡️",
        initial_sidebar_state="expanded"
    )
    
    # Serve PWA files
    serve_pwa_files()
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🛡️ GuardianDrive AI")
        st.markdown("**Connected Safety Platform**")
        st.markdown("---")
        
        page = st.radio("📍 Navigation", [
            "🚗 Live Detection",
            "🗺️ Risk Mapping",
            "💼 Insurance Bridge",
            "🚨 Alert System",
            "📊 Statistics"
        ], index=0)
        
        st.markdown("---")
        st.markdown("**🔒 Privacy:** 100% Local")
        st.markdown("**⚡ Status:** 🟢 Active")
    
    # Page routing
    if page == "🚗 Live Detection":
        render_live_detection()
    elif page == "🗺️ Risk Mapping":
        render_risk_mapping()
    elif page == "💼 Insurance Bridge":
        render_insurance_bridge()
    elif page == "🚨 Alert System":
        render_alert_system()
    elif page == "📊 Statistics":
        render_statistics()

def render_live_detection():
    """Render live detection page"""
    st.title("🛡️ GuardianDrive AI - Live Detection")
    st.markdown("**Real-time multimodal driver state monitoring with integrated safety features**")
    
    # Main camera interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 🎥 Live Camera Feed")
        webrtc_streamer(
            key="guardiandrive-detection",
            video_processor_factory=GuardianDriveDetector,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {"urls": ["stun:stun1.l.google.com:19302"]},
                    {"urls": ["stun:stun2.l.google.com:19302"]},
                    {"urls": ["stun:stun3.l.google.com:19302"]},
                    {"urls": ["stun:stun4.l.google.com:19302"]},
                ]
            },
            async_processing=True,
        )
    
    with col2:
        st.markdown("### 📊 Live Metrics")
        st.metric("Detection States", "4")
        st.metric("AI Processing", "Local")
        st.metric("Privacy", "🔒 Secure")
        st.metric("Status", "🟢 Active")
        
        st.markdown("---")
        st.markdown("**🎯 Features Active:**")
        st.markdown("- ✅ Multimodal Detection")
        st.markdown("- ✅ Risk Mapping")
        st.markdown("- ✅ Stakeholder Alerts")
        st.markdown("- ✅ Insurance Tracking")
    
    # Enhanced metrics display
    st.markdown("---")
    st.markdown("### 📊 Advanced Detection Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("EAR Threshold", f"{EAR_THRESHOLD}")
        st.metric("PERCLOS Threshold", f"{PERCLOS_THRESHOLD}")
    with col2:
        st.metric("MAR Threshold", f"{MAR_THRESHOLD}")
        st.metric("Alert Frames", f"{CONSEC_FRAMES}")
    with col3:
        st.metric("Detection States", "4")
        st.metric("Multimodal", "✅ Active")
    with col4:
        st.metric("Alert System", "🔒 Secure")
        st.metric("Privacy", "🛡️ Local")

def render_risk_mapping():
    """Render risk mapping page"""
    st.title("🗺️ Predictive Risk Mapping")
    st.markdown("**Community-driven micro-risk zone heatmaps**")
    
    stats = risk_mapper.get_statistics()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Zones Monitored", stats["total_zones_monitored"])
    col2.metric("Total Incidents", stats["total_incidents"])
    col3.metric("High Risk Zones", stats["high_risk_zones"])
    col4.metric("Data Points", stats["data_points"])
    
    st.markdown("### 🎯 Risk Score Lookup")
    col1, col2 = st.columns(2)
    lat = col1.number_input("Latitude", value=28.6139, format="%.4f")
    lng = col2.number_input("Longitude", value=77.2090, format="%.4f")
    
    if st.button("Check Risk Score"):
        risk_info = risk_mapper.get_risk_score(lat, lng)
        risk_colors = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴", "unknown": "⚪"}
        st.markdown(f"### {risk_colors.get(risk_info['risk_level'], '⚪')} Risk Level: {risk_info['risk_level'].upper()}")
        
        col1, col2 = st.columns(2)
        col1.metric("Risk Score", f"{risk_info['score']:.2f}/5")
        col2.metric("Incidents", risk_info['incidents'])

def render_insurance_bridge():
    """Render insurance bridge page"""
    st.title("💼 Insurance Data Bridge")
    st.markdown("**Secure behavior-based insurance integration**")
    
    tab1, tab2 = st.tabs(["Driver Profile", "Premium Calculator"])
    
    with tab1:
        if st.button("Generate API Key"):
            new_key = insurance_bridge.generate_api_key("Demo Insurer")
            st.success(f"API Key: {new_key}")
        
        if st.button("Get Driver Profile"):
            demo_key = insurance_bridge.generate_api_key("Demo")
            profile = insurance_bridge.get_driver_profile(demo_key)
            if profile:
                st.json(profile)
            else:
                st.warning("No driving data available")
    
    with tab2:
        st.markdown("### 💰 Premium Recommendation")
        demo_key = insurance_bridge.generate_api_key("Demo")
        recommendation = insurance_bridge.get_premium_recommendation(demo_key)
        
        if recommendation:
            col1, col2, col3 = st.columns(3)
            col1.metric("Base Premium", f"₹{recommendation['base_premium']:,.0f}")
            col2.metric("Discount", f"{recommendation['discount_percentage']}%")
            col3.metric("Final Premium", f"₹{recommendation['adjusted_premium']:,.0f}")

def render_alert_system():
    """Render alert system page"""
    st.title("🚨 Multi-Stakeholder Alert System")
    st.markdown("**Coordinated emergency response ecosystem**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 👨👩👧👦 Family Contacts")
        st.info("2 contacts configured")
    with col2:
        st.markdown("#### 🚨 Emergency Services")
        st.warning("Demo Mode (APIs Disabled)")
    
    st.markdown("### 🧪 Test Alert")
    test_state = st.selectbox("Driver State", ["Drowsy", "Asleep", "Drunk"])
    if st.button("🚨 Trigger Test Alert"):
        location = {"lat": 28.6139, "lng": 77.2090}
        response = stakeholder_alerts.trigger_coordinated_response(
            driver_state=test_state, location=location, vehicle_speed=60, duration=5.0
        )
        st.json(response)

def render_statistics():
    """Render statistics page"""
    st.title("📊 System Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🗺️ Risk Mapping")
        stats = risk_mapper.get_statistics()
        st.metric("Zones", stats["total_zones_monitored"])
        st.metric("Incidents", stats["total_incidents"])
    
    with col2:
        st.markdown("### 💼 Insurance")
        demo_key = insurance_bridge.generate_api_key("Demo")
        profile = insurance_bridge.get_driver_profile(demo_key)
        if profile:
            st.metric("Safety Score", f"{profile['average_safety_score']:.1f}")
            st.metric("Distance", f"{profile['total_distance_km']:.1f} km")
    
    with col3:
        st.markdown("### 🚨 Alerts")
        st.metric("Active Incidents", len(stakeholder_alerts.active_incidents))
        st.metric("System Status", "🟢 Online")

if __name__ == "__main__":
    main()
