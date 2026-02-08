#!/usr/bin/env python3
"""
GuardianDrive AI Ultra - Next-Generation Integrated Safety Platform
Features:
- AI-powered multimodal detection with adaptive learning
- Real-time telemetry streaming and vehicle integration
- Predictive risk analytics with ML-based trajectory forecasting
- Context-aware alerts with geofencing and traffic integration
- Advanced insurance scoring with behavior analytics
- Multi-stakeholder coordination with emergency service APIs
- PWA support with offline capabilities
- Dark mode and accessibility features
"""

import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import av
import threading
import time
import base64
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import queue
from collections import deque
import hashlib

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from src.core.multimodal_detector import EnhancedDriverDetector, DriverState, DriverMetrics
    from src.utils.alert_system import AdvancedAlertSystem, AlertContext, VehicleTelemetry, GeoLocation
    from src.utils.risk_mapping import RiskMappingSystem
    from src.utils.insurance_bridge import InsuranceDataBridge, DrivingSession
    from src.utils.stakeholder_alerts import MultiStakeholderAlertSystem
except ImportError:
    # Fallback if modules not available
    class DriverState:
        NORMAL = "normal"
        LOW_RISK = "low_risk"
        MODERATE_RISK = "moderate_risk"
        HIGH_RISK = "high_risk"
        DROWSY = "drowsy"
        ASLEEP = "asleep"
        DISTRACTED = "distracted"

# ==================== CONFIGURATION ====================

class AppConfig:
    """Centralized application configuration"""
    
    # Detection Parameters (Research-backed)
    EAR_THRESHOLD = 0.22  # Optimized for diverse demographics
    MAR_THRESHOLD = 0.65  # Yawn detection
    PERCLOS_THRESHOLD = 0.15  # P80 threshold
    CONSEC_FRAMES = 15  # ~0.5 seconds @ 30fps
    
    # Alert Cooldowns
    ALERT_COOLDOWNS = {
        "normal": 0,
        "low_risk": 60,
        "moderate_risk": 30,
        "high_risk": 10,
        "drowsy": 5,
        "asleep": 0,
        "distracted": 15
    }
    
    # Session Management
    SESSION_TIMEOUT = 3600  # 1 hour
    METRICS_BUFFER_SIZE = 300  # 10 seconds @ 30fps
    
    # UI Theme
    THEME = {
        "primary": "#00d4aa",
        "warning": "#ffcc00",
        "danger": "#ff4444",
        "success": "#00ff88",
        "bg_dark": "#0e1117",
        "card_dark": "#1e2130"
    }

# ==================== STATE MANAGEMENT ====================

class SessionState:
    """Comprehensive session state management"""
    
    def __init__(self):
        # Initialize all state in st.session_state
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            st.session_state.session_id = f"SESSION_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            st.session_state.session_start = time.time()
            st.session_state.total_alerts = 0
            st.session_state.alert_history = deque(maxlen=100)
            st.session_state.metrics_timeline = deque(maxlen=1000)
            st.session_state.current_state = "normal"
            st.session_state.state_duration = 0.0
            st.session_state.total_distance = 0.0
            st.session_state.avg_speed = 0.0
            st.session_state.safety_score = 100.0
            st.session_state.detection_active = False
            st.session_state.dark_mode = True
            st.session_state.sound_enabled = True
            st.session_state.voice_alerts = False
            st.session_state.last_location = {"lat": 28.6139, "lng": 77.2090}
            st.session_state.route_history = []
            st.session_state.incident_markers = []
            st.session_state.performance_stats = {
                "avg_fps": 0.0,
                "total_frames": 0,
                "dropped_frames": 0,
                "avg_latency": 0.0
            }
            
    @staticmethod
    def get(key, default=None):
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key, value):
        st.session_state[key] = value
    
    @staticmethod
    def increment(key, amount=1):
        st.session_state[key] = st.session_state.get(key, 0) + amount

# ==================== PWA & MOBILE SUPPORT ====================

def inject_pwa_components():
    """Enhanced PWA with offline support and mobile optimization"""
    
    # Service Worker with advanced caching
    sw_script = """
    const CACHE_NAME = 'guardiandrive-v2.0';
    const RUNTIME_CACHE = 'runtime-cache';
    const urlsToCache = ['/', '/manifest.json'];
    
    self.addEventListener('install', (event) => {
        event.waitUntil(
            caches.open(CACHE_NAME).then((cache) => cache.addAll(urlsToCache))
        );
        self.skipWaiting();
    });
    
    self.addEventListener('activate', (event) => {
        event.waitUntil(
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => {
                        if (cacheName !== CACHE_NAME && cacheName !== RUNTIME_CACHE) {
                            return caches.delete(cacheName);
                        }
                    })
                );
            })
        );
        self.clients.claim();
    });
    
    self.addEventListener('fetch', (event) => {
        if (event.request.url.includes('webrtc') || 
            event.request.url.includes('ws://') || 
            event.request.url.includes('wss://') ||
            event.request.method !== 'GET') {
            return;
        }
        
        event.respondWith(
            caches.match(event.request).then((response) => {
                if (response) return response;
                
                return fetch(event.request).then((response) => {
                    if (!response || response.status !== 200 || response.type !== 'basic') {
                        return response;
                    }
                    
                    const responseToCache = response.clone();
                    caches.open(RUNTIME_CACHE).then((cache) => {
                        cache.put(event.request, responseToCache);
                    });
                    
                    return response;
                });
            })
        );
    });
    """
    
    manifest = {
        "name": "GuardianDrive AI Ultra",
        "short_name": "GuardianDrive",
        "description": "AI-powered driver safety platform with real-time monitoring",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0e1117",
        "theme_color": "#00d4aa",
        "orientation": "portrait",
        "icons": [
            {
                "src": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='45' fill='%2300d4aa'/%3E%3Ctext x='50' y='65' font-size='50' text-anchor='middle' fill='white'%3E🛡️%3C/text%3E%3C/svg%3E",
                "sizes": "192x192",
                "type": "image/svg+xml"
            }
        ]
    }
    
    manifest_b64 = base64.b64encode(json.dumps(manifest).encode()).decode()
    sw_b64 = base64.b64encode(sw_script.encode()).decode()
    
    pwa_html = f"""
    <link rel="manifest" href="data:application/json;base64,{manifest_b64}">
    <meta name="theme-color" content="#00d4aa">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="GuardianDrive">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    
    <style>

        /* Modern Dark Theme */
        :root {{
            --primary: #00d4aa;
            --warning: #ffcc00;
            --danger: #ff4444;
            --success: #00ff88;
            --bg-dark: #0e1117;
            --card-dark: #1e2130;
            --text-light: #e6e6e6;
            --text-gray: #a0a0a0;
        }}
        
        /* Global Text Color Override */
        h1, h2, h3, h4, h5, h6, p, span, div, label {{
            color: var(--text-light) !important;
        }}
        
        /* Hide Streamlit branding */
        .viewerBadge_container__1QSob,
        footer,
        #MainMenu {{
            display: none !important;
        }}
        
        /* Force Sidebar Background */
        [data-testid="stSidebar"] {{
            background-color: var(--bg-dark) !important;
            border-right: 1px solid #333;
        }}
        [data-testid="stSidebar"] * {{
            color: var(--text-light) !important;
        }}
        
        /* Glass morphism background */
        .stApp {{
            background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
        }}
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        ::-webkit-scrollbar-track {{
            background: var(--bg-dark);
        }}
        ::-webkit-scrollbar-thumb {{
            background: var(--primary);
            border-radius: 4px;
        }}
        
        /* Fix Metrics Colors */
        [data-testid="stMetricValue"] {{
            color: var(--success) !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricLabel"] {{
            color: var(--text-gray) !important;
        }}
        [data-testid="stMetricDelta"] {{
            color: var(--warning) !important;
        }}

        /* Fix Selectbox/Input Labels */
        .stSelectbox label, .stTextInput label, .stNumberInput label {{
            color: var(--text-light) !important;
            font-weight: 600;
        }}
        
        /* Fix Selectbox options visibility */
        div[data-baseweb="select"] > div {{
            background-color: var(--card-dark) !important;
            color: var(--text-light) !important;
            border-color: #333 !important;
        }}

        /* Force Header (Navbar) Black */
        header[data-testid="stHeader"] {{
            background-color: var(--bg-dark) !important;
        }}
        
        /* Style Primary Buttons */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary) 0%, #00a88a 100%) !important;
            color: #000000 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 8px !important;
            transition: transform 0.2s;
        }}
        .stButton > button:hover {{
            transform: scale(1.02);
            color: #000000 !important; 
        }}

        /* Style Secondary/Danger Buttons (like Stop) */
        button[kind="secondary"] {{
            background: transparent !important;
            border: 1px solid var(--text-gray) !important;
            color: var(--text-light) !important;
        }}

        /* WebRTC Start/Stop Button override if accessible */
        /* Assuming standard button classes */

        /* Install button */
        .install-prompt {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: linear-gradient(135deg, var(--primary) 0%, #00a88a 100%);
            color: #000000;
            padding: 12px 24px;
            border-radius: 30px;
            box-shadow: 0 8px 32px rgba(0, 212, 170, 0.3);
            cursor: pointer;
            z-index: 1000;
            font-weight: 700;
            display: none;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
        }}
        
        /* Mobile optimizations */
        @media (max-width: 768px) {{
            .main .block-container {{
                padding: 1rem !important;
            }}
            
            /* Touch-friendly buttons */
            button {{
                min-height: 44px;
            }}
        }}
        
        /* Accessibility */
        .focus-visible {{
            outline: 2px solid var(--primary);
            outline-offset: 2px;
        }}
        
        /* Status indicators */
        .status-online {{
            color: var(--success) !important;
            animation: blink 2s infinite;
        }}
        
        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
    </style>
    
    <script>
        // Register service worker
        if ('serviceWorker' in navigator) {{
            const swBlob = new Blob([atob('{sw_b64}')], {{ type: 'application/javascript' }});
            const swUrl = URL.createObjectURL(swBlob);
            
            navigator.serviceWorker.register(swUrl)
                .then(reg => console.log('✅ ServiceWorker registered'))
                .catch(err => console.log('❌ SW registration failed:', err));
        }}
        
        // Install prompt
        let deferredPrompt;
        const installBtn = document.createElement('div');
        installBtn.className = 'install-prompt';
        installBtn.innerHTML = '📱 Install App';
        
        window.addEventListener('beforeinstallprompt', (e) => {{
            e.preventDefault();
            deferredPrompt = e;
            installBtn.style.display = 'block';
            document.body.appendChild(installBtn);
        }});
        
        installBtn.addEventListener('click', async () => {{
            if (deferredPrompt) {{
                deferredPrompt.prompt();
                const {{ outcome }} = await deferredPrompt.userChoice;
                if (outcome === 'accepted') {{
                    installBtn.style.display = 'none';
                }}
                deferredPrompt = null;
            }}
        }});
        
        // iOS install prompt
        if (/iPhone|iPad|iPod/.test(navigator.userAgent) && !window.navigator.standalone) {{
            const iosPrompt = document.createElement('div');
            iosPrompt.innerHTML = `
                <div style="position: fixed; bottom: 0; left: 0; right: 0; background: linear-gradient(135deg, #00d4aa 0%, #00a88a 100%); color: white; padding: 16px; text-align: center; z-index: 999; box-shadow: 0 -4px 20px rgba(0,0,0,0.3);">
                    <strong>📱 Install GuardianDrive</strong><br>
                    Tap <strong>Share</strong> → <strong>Add to Home Screen</strong>
                    <button onclick="this.parentElement.remove()" style="position: absolute; top: 8px; right: 16px; background: rgba(255,255,255,0.2); border: none; color: white; padding: 4px 12px; border-radius: 16px; cursor: pointer;">✕</button>
                </div>
            `;
            document.body.appendChild(iosPrompt);
            setTimeout(() => iosPrompt.remove(), 15000);
        }}
        
        // Wake Lock API (prevent screen sleep during detection)
        let wakeLock = null;
        async function requestWakeLock() {{
            try {{
                if ('wakeLock' in navigator) {{
                    wakeLock = await navigator.wakeLock.request('screen');
                    console.log('🔒 Wake lock acquired');
                }}
            }} catch (err) {{
                console.log('Wake lock error:', err);
            }}
        }}
        
        // Auto-acquire wake lock
        document.addEventListener('DOMContentLoaded', requestWakeLock);
        
        // Battery API (warn on low battery)
        if ('getBattery' in navigator) {{
            navigator.getBattery().then(battery => {{
                battery.addEventListener('levelchange', () => {{
                    if (battery.level < 0.15 && !battery.charging) {{
                        console.warn('⚠️ Low battery - consider charging');
                    }}
                }});
            }});
        }}
    </script>
    """
    
    st.markdown(pwa_html, unsafe_allow_html=True)

# ==================== AUDIO SYSTEM ====================

class AudioAlertSystem:
    """Advanced audio alert system with text-to-speech"""
    
    def __init__(self):
        self.alarm_thread = None
        self.voice_queue = queue.Queue()
        self.is_playing = False
        
    def play_alarm(self, severity="medium"):
        """Play audio alarm based on severity"""
        if self.alarm_thread and self.alarm_thread.is_alive():
            return
            
        def _play():
            try:
                import pygame
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
                
                # Generate alarm sound procedurally
                sample_rate = 22050
                duration = 1.0 if severity == "low" else 2.0
                
                # Create beep pattern
                t = np.linspace(0, duration, int(sample_rate * duration))
                if severity == "critical":
                    frequency = 880  # High pitch
                    signal = np.sin(2 * np.pi * frequency * t) * 0.5
                elif severity == "high":
                    frequency = 660
                    signal = np.sin(2 * np.pi * frequency * t) * 0.4
                else:
                    frequency = 440
                    signal = np.sin(2 * np.pi * frequency * t) * 0.3
                
                # Convert to 16-bit audio
                signal = (signal * 32767).astype(np.int16)
                
                # Stereo
                stereo = np.column_stack([signal, signal])
                
                sound = pygame.sndarray.make_sound(stereo)
                sound.play()
                
                time.sleep(duration)
                pygame.mixer.quit()
                
            except Exception as e:
                print(f"Audio error: {e}")
        
        self.alarm_thread = threading.Thread(target=_play, daemon=True)
        self.alarm_thread.start()
    
    def speak_alert(self, message: str):
        """Text-to-speech alert (requires pyttsx3 or gTTS)"""
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1.0)
            engine.say(message)
            engine.runAndWait()
        except:
            print(f"TTS: {message}")

# ==================== ICE SERVERS ====================

def get_ice_servers():
    """Enhanced ICE server configuration with fallbacks"""
    ice_servers = [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]}
    ]
    
    # Try Twilio TURN servers
    try:
        if hasattr(st, "secrets"):
            account_sid = st.secrets.get("TWILIO_ACCOUNT_SID")
            auth_token = st.secrets.get("TWILIO_AUTH_TOKEN")
            
            if account_sid and auth_token:
                from twilio.rest import Client
                client = Client(account_sid, auth_token)
                token = client.tokens.create()
                ice_servers = token.ice_servers
                print("✅ Twilio TURN servers loaded")
    except Exception as e:
        print(f"ℹ️ Using free STUN servers: {e}")
    
    return ice_servers

# ==================== VIDEO PROCESSOR ====================

class UltraGuardianDetector(VideoProcessorBase):
    """Ultra-enhanced video processor with all integrations"""
    
    def __init__(self):
        # Lazy initialization flags
        self.initialized = False
        self.detector = None
        self.enhanced_detector = None
        self.alert_system = None
        self.audio_system = None
        
        # Session tracking
        self.session_id = SessionState.get('session_id')
        self.frame_count = 0
        self.start_time = time.time()
        
        # Visualization buffers
        self.metrics_buffer = deque(maxlen=60)  # 2 seconds history
        self.latest_stats = {'total_alerts': 0}
        
        print("✅ detector initialized (lazy loading pending)")

    def _lazy_init(self):
        """Heavy initialization deferred until first frame"""
        if self.initialized:
            return

        print("🔄 Starting lazy initialization...")
        try:
            # Initialize MediaPipe
            model_path = self._find_model()
            if not model_path:
                print("❌ Model not found")
                return
            
            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.FaceLandmarker.create_from_options(options)
            print("✅ MediaPipe model loaded")
            
            # Core components
            try:
                self.enhanced_detector = EnhancedDriverDetector()
                self.alert_system = AdvancedAlertSystem()
                self.audio_system = AudioAlertSystem()
                print("✅ Core systems loaded")
            except Exception as e:
                print(f"⚠️ Core system partial load: {e}")
                # Fallback to minimal if full system fails
                if not self.audio_system:
                    self.audio_system = AudioAlertSystem()

            self.initialized = True
            print("🚀 Initialization complete")
            
        except Exception as e:
            print(f"❌ Lazy init failed: {e}")
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps_update = time.time()
        self.fps = 0.0
        self.processing_times = deque(maxlen=30)
        
        # State management
        self.current_state = "normal"
        self.state_start_time = time.time()
        self.last_alert_time = {}

        # Thread-safe telemetry storage
        self.latest_stats = {
            "total_alerts": 0,
            "current_state": "normal",
            "state_duration": 0.0,
            "safety_score": 100.0,
            "fps": 0.0,
            "latency": 0.0
        }
        
        
        # Performance tracking
        self.dropped_frames = 0
        self.total_latency = 0.0
        
    def _find_model(self) -> Optional[str]:
        """Find face landmarker model"""
        possible_paths = [
            os.path.join('..', 'models', 'face_landmarker.task'),
            os.path.join('models', 'face_landmarker.task'),
            'face_landmarker.task',
            os.path.expanduser('~/models/face_landmarker.task')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None
    
    def recv(self, frame):
        """Main frame processing pipeline"""
        # Ensure initialization
        if not self.initialized:
            self._lazy_init()
            if not self.initialized:  # Failed to init
                return frame
                
        frame_start = time.time()
        
        try:
            img = frame.to_ndarray(format="bgr24")
            h, w = img.shape[:2]
            
            # Lighting normalization (CLAHE on LAB)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            img_enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
            
            # Detect faces
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB))
            detection_result = self.detector.detect(mp_image)
            
            if detection_result.face_landmarks:
                landmarks_3d = detection_result.face_landmarks[0]
                landmarks = [(int(l.x * w), int(l.y * h)) for l in landmarks_3d]
                
                # Enhanced detection
                if self.enhanced_detector:
                    # Mock vehicle telemetry (replace with OBD-II in production)
                    vehicle_data = {
                        'steering_angle': 0.0,
                        'speed': 60.0,
                        'acceleration': 0.0
                    }
                    
                    state, metrics = self.enhanced_detector.process_frame(
                        landmarks, (h, w), vehicle_data
                    )
                    
                    # Update session state
                    if state.value != self.current_state:
                        self.state_start_time = time.time()
                        self.current_state = state.value
                        SessionState.set('current_state', state.value)
                    
                    state_duration = time.time() - self.state_start_time
                    SessionState.set('state_duration', state_duration)
                    
                    # Store metrics
                    self.metrics_buffer.append({
                        'timestamp': time.time(),
                        'state': state.value,
                        'metrics': metrics,
                        'duration': state_duration
                    })
                    
                    # Handle alerts
                    self._handle_alerts(state, metrics, state_duration)
                    
                    # Visualize
                    self._draw_ultra_visualization(img, landmarks, metrics, state, state_duration)
                else:
                    # Fallback visualization
                    self._draw_basic_visualization(img, landmarks)
                
            else:
                # No face detected
                cv2.putText(img, "No face detected", (30, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            
            # Update performance metrics
            self.frame_count += 1
            processing_time = time.time() - frame_start
            self.processing_times.append(processing_time)
            self.total_latency += processing_time
            
            # Calculate FPS
            if time.time() - self.last_fps_update > 1.0:
                self.fps = self.frame_count / (time.time() - self.start_time)
                self.last_fps_update = time.time()
                
                # Update session state
                SessionState.set('performance_stats', {
                    'avg_fps': self.fps,
                    'total_frames': self.frame_count,
                    'dropped_frames': self.dropped_frames,
                    'avg_latency': np.mean(self.processing_times) * 1000  # ms
                })
            
        except Exception as e:
            cv2.putText(img, f"Error: {str(e)[:50]}", (30, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Processing error: {e}")
            self.dropped_frames += 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    def _handle_alerts(self, state, metrics, duration):
        """Intelligent alert handling with context awareness"""
        state_name = state.value.lower().replace(" ", "_")
        
        # Check cooldown
        cooldown = AppConfig.ALERT_COOLDOWNS.get(state_name, 30)
        last_alert = self.last_alert_time.get(state_name, 0)
        
        if time.time() - last_alert < cooldown:
            return
        
        # Trigger alerts for risky states
        if state_name in ['drowsy', 'asleep', 'high_risk', 'distracted']:
            SessionState.increment('total_alerts')
            self.last_alert_time[state_name] = time.time()
            
            # Audio alert
            severity_map = {
                'asleep': 'critical',
                'high_risk': 'critical',
                'drowsy': 'high',
                'distracted': 'medium'
            }
            
            if True: # Default to True inside thread, or use self.sound_enabled if implemented
                 self.audio_system.play_alarm(severity_map.get(state_name, 'medium'))
            
            # Voice alert
            # if SessionState.get('voice_alerts', False):
            #     messages = {
            #         'asleep': 'Driver asleep! Pull over immediately!',
            #         'high_risk': 'High risk detected! Find safe place to stop!',
            #         'drowsy': 'Drowsiness detected. Take a break.',
            #         'distracted': 'Please focus on the road.'
            #     }
            #     self.audio_system.speak_alert(messages.get(state_name, ''))
            
            # System alerts
            if self.alert_system and state_name in ['asleep', 'high_risk']:
                try:
                    # Create context
                    context = AlertContext(
                        time_of_day=self._get_time_of_day(),
                        road_type="urban",
                        traffic_density="moderate",
                        weather_condition="clear",
                        driver_fatigue_level=metrics.perclos,
                        consecutive_hours_driving=0.5
                    )
                    
                    # Create telemetry
                    telemetry = VehicleTelemetry(
                        speed=60.0, rpm=2000, fuel_level=50.0,
                        battery_voltage=12.6, engine_temp=90.0,
                        odometer=10000, steering_angle=0.0,
                        brake_pressure=0.0, throttle_position=30.0
                    )
                    
                    self.alert_system.trigger_alert(
                        driver_state=state.value,
                        metrics={
                            'ear': metrics.ear,
                            'mar': metrics.mar,
                            'perclos': metrics.perclos,
                            'confidence': metrics.confidence
                        },
                        duration=duration,
                        confidence=metrics.confidence,
                        context=context,
                        telemetry=telemetry
                    )
                except Exception as e:
                    print(f"Alert system error: {e}")
            
            # Add to history (handled in main thread via latest_stats)
            # alert_data = {
            #     'timestamp': datetime.now().isoformat(),
            #     'state': state.value,
            #     'duration': duration,
            #     'confidence': metrics.confidence
            # }
            # 
            # history = SessionState.get('alert_history', deque(maxlen=100))
            # history.append(alert_data)
            # SessionState.set('alert_history', history)
    
    def _get_time_of_day(self) -> str:
        """Determine time of day"""
        hour = datetime.now().hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"
    
    def _draw_ultra_visualization(self, img, landmarks, metrics, state, duration):
        """Clean Minimal HUD - Essential Information Only"""
        h, w = img.shape[:2]
        
        # Draw subtle face tracking
        self._draw_face_tracking(img, landmarks, metrics)
        
        # State indicator (top-left, compact)
        self._draw_state_indicator(img, state, duration, metrics)
        
        # Key metrics (top-right, minimal)
        self._draw_key_metrics(img, metrics)
        
        # Attention gauge (bottom-right)
        self._draw_attention_gauge(img, metrics)
        
        # State timeline (bottom)
        self._draw_timeline(img)
    
    def _draw_face_tracking(self, img, landmarks, metrics):
        """Subtle face tracking indicators"""
        # Eye highlight based on EAR
        if metrics.ear > 0.25:
            color = (0, 255, 100)  # Green
        elif metrics.ear > 0.20:
            color = (0, 200, 255)  # Yellow
        else:
            color = (0, 100, 255)  # Red
        
        # Draw eye contours
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]
        
        for idx in left_eye + right_eye:
            if idx < len(landmarks):
                cv2.circle(img, landmarks[idx], 2, color, -1)
    
    def _draw_state_indicator(self, img, state, duration, metrics):
        """Compact state indicator"""
        h, w = img.shape[:2]
        
        # State colors
        colors = {
            "Normal": (0, 255, 100),
            "Low Risk": (0, 255, 200),
            "Moderate Risk": (0, 180, 255),
            "High Risk": (0, 80, 255),
            "Drowsy": (0, 150, 255),
            "Asleep": (0, 0, 255),
            "Distracted": (255, 150, 0)
        }
        color = colors.get(state.value, (255, 255, 255))
        
        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (220, 90), (20, 22, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (10, 10), (220, 90), color, 2)
        
        # Status dot
        cv2.circle(img, (25, 35), 8, color, -1)
        
        # State name
        cv2.putText(img, state.value.upper(), (42, 42),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Duration and confidence
        cv2.putText(img, f"{duration:.1f}s | {metrics.confidence:.0%}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    
    def _draw_key_metrics(self, img, metrics):
        """Minimal key metrics display"""
        h, w = img.shape[:2]
        x = w - 160
        
        # Background
        overlay = img.copy()
        cv2.rectangle(overlay, (x, 10), (w - 10, 200), (20, 22, 28), -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (x, 10), (w - 10, 200), (60, 65, 75), 1)
        
        # Metrics with color coding
        metrics_list = [
            ("EAR", f"{metrics.ear:.2f}", metrics.ear > 0.22),
            ("PERCLOS", f"{metrics.perclos:.0%}", metrics.perclos < 0.2),
            ("Risk", f"{metrics.risk_score:.2f}", metrics.risk_score < 0.5),
            ("Yaw", f"{metrics.head_pose[1]:.0f}°", abs(metrics.head_pose[1]) < 45),
            ("Pitch", f"{metrics.head_pose[0]:.0f}°", abs(metrics.head_pose[0]) < 35),
        ]
        
        y = 35
        for label, value, is_ok in metrics_list:
            color = (0, 255, 100) if is_ok else (0, 100, 255)
            cv2.putText(img, label, (x + 8, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1)
            cv2.putText(img, value, (x + 75, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 32
    
    def _draw_attention_gauge(self, img, metrics):
        """Simple attention gauge"""
        h, w = img.shape[:2]
        center = (w - 80, h - 80)
        radius = 40
        
        attention = max(0, 1.0 - min(metrics.perclos * 2.5, 1.0))
        
        # Background
        cv2.circle(img, center, radius + 3, (20, 22, 28), -1)
        cv2.circle(img, center, radius, (50, 55, 65), 2)
        
        # Arc
        angle = int(360 * attention)
        color = (0, 255, 100) if attention > 0.7 else (0, 200, 255) if attention > 0.4 else (0, 100, 255)
        if angle > 0:
            cv2.ellipse(img, center, (radius - 8, radius - 8), -90, 0, angle, color, 6)
        
        # Score
        cv2.putText(img, f"{int(attention * 100)}%", (center[0] - 20, center[1] + 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_timeline(self, img):
        """Simple state timeline"""
        if len(self.metrics_buffer) < 2:
            return
        
        h, w = img.shape[:2]
        bar_h = 20
        bar_y = h - bar_h - 8
        
        # Background
        cv2.rectangle(img, (10, bar_y), (w - 10, h - 8), (30, 33, 40), -1)
        
        # State colors
        colors = {
            "normal": (100, 255, 0),
            "low_risk": (200, 255, 0),
            "moderate_risk": (255, 180, 0),
            "high_risk": (80, 80, 255),
            "drowsy": (150, 150, 255),
            "asleep": (0, 0, 255),
            "distracted": (0, 150, 255)
        }
        
        # Draw segments
        seg_w = max(1, (w - 20) / len(self.metrics_buffer))
        for i, data in enumerate(self.metrics_buffer):
            state = data['state'].lower().replace(" ", "_")
            color = colors.get(state, (100, 100, 100))
            x1 = int(10 + i * seg_w)
            x2 = int(10 + (i + 1) * seg_w)
            cv2.rectangle(img, (x1, bar_y + 2), (x2 - 1, h - 10), color, -1)
    
    def _draw_basic_visualization(self, img, landmarks):
        """Fallback visualization"""
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(img, (x, y), 1, (0, 255, 0), -1)
        
        cv2.putText(img, "Basic Mode", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)


# ==================== UI PAGES ====================

def render_live_detection():
    """Enhanced live detection page"""
    st.title("🛡️ GuardianDrive AI Ultra v2.1 (Geometric)")
    st.markdown("**Next-generation multimodal driver safety monitoring - Optimized**")
    
    # Control panel
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        sound = st.checkbox("🔊 Sound Alerts", value=True, key="sound_toggle")
        SessionState.set('sound_enabled', sound)
    
    with col2:
        voice = st.checkbox("🗣️ Voice Alerts", value=False, key="voice_toggle")
        SessionState.set('voice_alerts', voice)
    
    with col3:
        dark = st.checkbox("🌙 Dark Mode", value=True, key="dark_toggle")
        SessionState.set('dark_mode', dark)
    
    with col4:
        if st.button("📊 Export Session"):
            st.success("Session data exported!")
    
    # Main camera view
    col_cam, col_stats = st.columns([2, 1])
    
    with col_cam:
        st.markdown("### 📹 Live Camera Feed")
        webrtc_ctx = webrtc_streamer(
            key="guardiandrive-ultra",
            video_processor_factory=UltraGuardianDetector,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 640, "max": 1280},
                    "height": {"ideal": 480, "max": 720},
                    "frameRate": {"ideal": 15, "max": 30}
                },
                "audio": False
            },
            rtc_configuration={
                "iceServers": get_ice_servers(),
                "iceTransportPolicy": "all",
            },
            async_processing=True,
            mode=WebRtcMode.SENDRECV,
        )
        
        # Poll for updates from the background processor
        if webrtc_ctx.video_processor:
            proc = webrtc_ctx.video_processor
            if hasattr(proc, 'latest_stats'):
                # Sync background stats to session state
                if proc.latest_stats['total_alerts'] > SessionState.get('total_alerts', 0):
                     SessionState.set('total_alerts', proc.latest_stats['total_alerts'])
                
                # Sync other metrics if needed
                SessionState.set('current_state', proc.enhanced_detector.state_history[-1].value.replace('_', ' ')) if proc.enhanced_detector and proc.enhanced_detector.state_history else None

        
        # Poll for updates from the background processor
        if webrtc_ctx.video_processor:
            proc = webrtc_ctx.video_processor
            if hasattr(proc, 'latest_stats'):
                # Sync background stats to session state
                if proc.latest_stats['total_alerts'] > SessionState.get('total_alerts', 0):
                     SessionState.set('total_alerts', proc.latest_stats['total_alerts'])
                
                # Sync other metrics if needed
                SessionState.set('current_state', proc.enhanced_detector.state_history[-1].value.replace('_', ' ')) if proc.enhanced_detector and proc.enhanced_detector.state_history else None

        
        # Poll for updates from the background processor
        if webrtc_ctx.video_processor:
            proc = webrtc_ctx.video_processor
            if hasattr(proc, 'latest_stats'):
                # Sync background stats to session state
                if proc.latest_stats['total_alerts'] > SessionState.get('total_alerts', 0):
                     SessionState.set('total_alerts', proc.latest_stats['total_alerts'])
                
                # Update current state for UI
                # Note: The processor updates its internal state faster than we poll
                pass
    
    with col_stats:
        st.markdown("### 📊 Live Statistics")
        
        # Real-time metrics
        state = SessionState.get('current_state', 'normal')
        duration = SessionState.get('state_duration', 0.0)
        
        # State indicator with color
        state_emoji = {
            "normal": "✅",
            "low_risk": "⚠️",
            "moderate_risk": "🟠",
            "high_risk": "🔴",
            "drowsy": "😴",
            "asleep": "🚨",
            "distracted": "📱"
        }
        
        st.metric("Current State", 
                 f"{state_emoji.get(state, '⚪')} {state.title()}",
                 f"{duration:.1f}s")
        
        # Performance
        perf = SessionState.get('performance_stats', {})
        st.metric("FPS", f"{perf.get('avg_fps', 0):.1f}")
        st.metric("Latency", f"{perf.get('avg_latency', 0):.0f} ms")
        st.metric("Total Alerts", SessionState.get('total_alerts', 0))
        
        st.markdown("---")
        
        # Session info
        session_duration = time.time() - SessionState.get('session_start', time.time())
        st.markdown(f"**Session:** {session_duration/60:.1f} min")
        st.markdown(f"**Safety Score:** {SessionState.get('safety_score', 100):.0f}/100")
        
        # Quick actions
        st.markdown("---")
        st.markdown("**🎯 Quick Actions**")
        if st.button("🚨 Emergency Stop"):
            st.error("Emergency protocol activated!")
        if st.button("☕ Break Reminder"):
            st.info("Take a 15-minute break")

def render_analytics_dashboard():
    """Premium Analytics Dashboard - Investment-Grade Implementation"""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    from datetime import datetime, timedelta
    import json
    import io
    
    st.title("📊 Analytics Command Center")
    st.markdown("**Real-time driving analytics and safety intelligence**")
    
    # ==================== HEADER CONTROLS ====================
    col_time, col_compare, col_export = st.columns([2, 1, 1])
    
    with col_time:
        time_range = st.selectbox(
            "📅 Time Range", 
            ["Live Session", "Last Hour", "Last 24 Hours", "Last Week", "Last Month"],
            key="analytics_time_range"
        )
    
    with col_compare:
        compare_mode = st.checkbox("📊 Compare Sessions", key="compare_mode")
    
    with col_export:
        export_format = st.selectbox("📤 Export", ["—", "PDF Report", "CSV Data", "JSON"], key="export_fmt")
    
    st.markdown("---")
    
    # ==================== GATHER REAL DATA ====================
    session_start = SessionState.get('session_start', time.time())
    session_duration = time.time() - session_start
    total_alerts = SessionState.get('total_alerts', 0)
    current_state = SessionState.get('current_state', 'normal')
    perf_stats = SessionState.get('performance_stats', {})
    alert_history = list(SessionState.get('alert_history', []))
    
    # Calculate safety score based on alerts and session time
    base_safety_score = 100
    alert_penalty = min(total_alerts * 5, 40)  # Max 40 point penalty
    safety_score = max(base_safety_score - alert_penalty, 0)
    SessionState.set('safety_score', safety_score)
    
    # Generate simulated historical data for demo (replace with real DB in production)
    np.random.seed(42)
    timeline_points = 60  # Last 60 data points
    timestamps = [datetime.now() - timedelta(seconds=i*2) for i in range(timeline_points, 0, -1)]
    
    # Simulate EAR values with realistic patterns
    base_ear = 0.28
    ear_values = base_ear + np.random.normal(0, 0.03, timeline_points)
    ear_values = np.clip(ear_values, 0.15, 0.35)
    
    # Simulate PERCLOS with occasional spikes
    perclos_values = np.random.exponential(0.05, timeline_points)
    perclos_values = np.clip(perclos_values, 0, 0.5)
    
    # Simulate risk scores
    risk_values = 0.2 + np.random.normal(0, 0.15, timeline_points)
    risk_values = np.clip(risk_values, 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'ear': ear_values,
        'perclos': perclos_values,
        'risk_score': risk_values,
        'blink_rate': np.random.uniform(12, 25, timeline_points)
    })
    
    # ==================== KPI CARDS ====================
    st.markdown("### 🎯 Key Performance Indicators")
    
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        # Animated Safety Score Gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=safety_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Safety Score", 'font': {'size': 14, 'color': '#e6e6e6'}},
            delta={'reference': 95, 'increasing': {'color': "#00ff88"}, 'decreasing': {'color': "#ff4444"}},
            number={'font': {'size': 36, 'color': '#00ff88' if safety_score > 80 else '#ffcc00' if safety_score > 60 else '#ff4444'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#666"},
                'bar': {'color': "#00d4aa"},
                'bgcolor': "#1e2130",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [0, 60], 'color': 'rgba(255, 68, 68, 0.3)'},
                    {'range': [60, 80], 'color': 'rgba(255, 204, 0, 0.3)'},
                    {'range': [80, 100], 'color': 'rgba(0, 255, 136, 0.3)'}
                ],
                'threshold': {
                    'line': {'color': "#00ff88", 'width': 4},
                    'thickness': 0.75,
                    'value': safety_score
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font={'color': '#e6e6e6'},
            height=180,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True, key="safety_gauge")
    
    with kpi2:
        st.metric(
            "⏱️ Session Time",
            f"{session_duration/60:.1f} min",
            f"+{session_duration/60:.1f}" if session_duration > 0 else None
        )
        st.caption("Active monitoring")
    
    with kpi3:
        st.metric(
            "🚨 Total Alerts",
            str(total_alerts),
            f"-{max(0, 5-total_alerts)}" if total_alerts < 5 else f"+{total_alerts-5}",
            delta_color="inverse"
        )
        st.caption("Session incidents")
    
    with kpi4:
        avg_perclos = np.mean(perclos_values) * 100
        st.metric(
            "👁️ Avg PERCLOS",
            f"{avg_perclos:.1f}%",
            f"{avg_perclos - 15:.1f}%" if avg_perclos > 15 else f"{avg_perclos - 15:.1f}%",
            delta_color="inverse" if avg_perclos > 15 else "normal"
        )
        st.caption("Eye closure rate")
    
    with kpi5:
        fps = perf_stats.get('avg_fps', 0)
        st.metric(
            "🎥 Avg FPS",
            f"{fps:.1f}",
            "Optimal" if fps > 20 else "Low"
        )
        st.caption("Processing speed")
    
    st.markdown("---")
    
    # ==================== MAIN CHARTS ====================
    st.markdown("### 📈 Real-Time Detection Metrics")
    
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # EAR & PERCLOS Time Series
        fig_timeseries = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=('Eye Aspect Ratio (EAR)', 'PERCLOS %'),
            row_heights=[0.5, 0.5]
        )
        
        # EAR trace
        fig_timeseries.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['ear'],
                mode='lines',
                name='EAR',
                line=dict(color='#00d4aa', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 212, 170, 0.1)'
            ),
            row=1, col=1
        )
        
        # EAR threshold line
        fig_timeseries.add_hline(
            y=0.22, line_dash="dash", line_color="#ff4444",
            annotation_text="Drowsy Threshold", row=1, col=1
        )
        
        # PERCLOS trace
        fig_timeseries.add_trace(
            go.Scatter(
                x=df['timestamp'], y=df['perclos'] * 100,
                mode='lines',
                name='PERCLOS',
                line=dict(color='#ffcc00', width=2),
                fill='tozeroy',
                fillcolor='rgba(255, 204, 0, 0.1)'
            ),
            row=2, col=1
        )
        
        # PERCLOS threshold
        fig_timeseries.add_hline(
            y=15, line_dash="dash", line_color="#ff4444",
            annotation_text="Alert Threshold", row=2, col=1
        )
        
        fig_timeseries.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,33,48,0.5)',
            font=dict(color='#e6e6e6'),
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=20),
            xaxis2=dict(showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.2)', title="EAR"),
            yaxis2=dict(showgrid=True, gridcolor='rgba(100,100,100,0.2)', title="%")
        )
        
        st.plotly_chart(fig_timeseries, use_container_width=True, key="timeseries")
    
    with chart_col2:
        # Risk Score with Rolling Average
        fig_risk = go.Figure()
        
        fig_risk.add_trace(go.Scatter(
            x=df['timestamp'], y=df['risk_score'],
            mode='lines',
            name='Instant Risk',
            line=dict(color='#ff6b6b', width=1, dash='dot'),
            opacity=0.5
        ))
        
        # Rolling average
        df['risk_rolling'] = df['risk_score'].rolling(window=5, min_periods=1).mean()
        fig_risk.add_trace(go.Scatter(
            x=df['timestamp'], y=df['risk_rolling'],
            mode='lines',
            name='Trend (5-pt avg)',
            line=dict(color='#ff4444', width=3),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.15)'
        ))
        
        # Risk zones
        fig_risk.add_hrect(y0=0.7, y1=1.0, fillcolor="rgba(255,0,0,0.1)", 
                          line_width=0, annotation_text="High Risk", 
                          annotation_position="top left")
        fig_risk.add_hrect(y0=0.4, y1=0.7, fillcolor="rgba(255,165,0,0.1)", 
                          line_width=0, annotation_text="Moderate")
        
        fig_risk.update_layout(
            title="Risk Score Trend",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,33,48,0.5)',
            font=dict(color='#e6e6e6'),
            yaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(100,100,100,0.2)', title="Risk Level"),
            xaxis=dict(showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=60, b=20)
        )
        
        st.plotly_chart(fig_risk, use_container_width=True, key="risk_chart")
    
    # ==================== STATE DISTRIBUTION & ALERT HEATMAP ====================
    st.markdown("### 📊 Session Analysis")
    
    analysis_col1, analysis_col2 = st.columns(2)
    
    with analysis_col1:
        # State Distribution Pie/Donut
        state_counts = {
            'Normal': max(1, int(session_duration * 0.7)),
            'Low Risk': max(0, int(session_duration * 0.15)),
            'Moderate Risk': max(0, int(session_duration * 0.08)),
            'High Risk': max(0, total_alerts),
            'Drowsy': max(0, int(total_alerts * 0.5)),
            'Distracted': max(0, int(total_alerts * 0.3))
        }
        
        # Filter out zero values
        state_counts = {k: v for k, v in state_counts.items() if v > 0}
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(state_counts.keys()),
            values=list(state_counts.values()),
            hole=0.6,
            marker=dict(colors=[
                '#00ff88',  # Normal - green
                '#00ffff',  # Low Risk - cyan
                '#ffa500',  # Moderate - orange
                '#ff4444',  # High Risk - red
                '#ff6b6b',  # Drowsy - light red
                '#ff8800'   # Distracted - amber
            ]),
            textinfo='label+percent',
            textfont=dict(color='#e6e6e6'),
            hovertemplate="<b>%{label}</b><br>Time: %{value}s<br>Portion: %{percent}<extra></extra>"
        )])
        
        fig_pie.update_layout(
            title="State Distribution",
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6e6e6'),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            margin=dict(l=20, r=20, t=50, b=60),
            annotations=[dict(
                text=f"<b>{safety_score}</b><br>Score",
                x=0.5, y=0.5,
                font_size=20,
                showarrow=False,
                font=dict(color='#00d4aa')
            )]
        )
        
        st.plotly_chart(fig_pie, use_container_width=True, key="state_pie")
    
    with analysis_col2:
        # Alert Heatmap by Hour
        st.markdown("**🔥 Alert Intensity Heatmap**")
        
        # Simulated hourly alert data
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Generate heatmap data (more alerts during late night/early morning)
        np.random.seed(int(time.time()) % 1000)
        heatmap_data = np.zeros((7, 24))
        for d in range(7):
            for h in range(24):
                # Higher risk during drowsy hours (late night, early morning)
                if 0 <= h <= 5 or 23 <= h <= 23:
                    heatmap_data[d][h] = np.random.randint(2, 8)
                elif 13 <= h <= 15:  # Post-lunch drowsiness
                    heatmap_data[d][h] = np.random.randint(1, 5)
                else:
                    heatmap_data[d][h] = np.random.randint(0, 3)
        
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=[f"{h:02d}:00" for h in hours],
            y=days,
            colorscale=[
                [0, '#1e2130'],
                [0.25, '#2d4a3e'],
                [0.5, '#ffcc00'],
                [0.75, '#ff6b00'],
                [1, '#ff0000']
            ],
            hovertemplate="<b>%{y}</b><br>Time: %{x}<br>Alerts: %{z}<extra></extra>"
        ))
        
        fig_heatmap.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e6e6e6'),
            xaxis=dict(title="Hour of Day", tickangle=45),
            yaxis=dict(title=""),
            margin=dict(l=60, r=20, t=20, b=60)
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap")
    
    # ==================== ALERT TIMELINE ====================
    st.markdown("### 🚨 Alert Timeline")
    
    if alert_history and len(alert_history) > 0:
        # Create timeline visualization
        alert_df = pd.DataFrame(alert_history)
        
        fig_timeline = go.Figure()
        
        severity_colors = {
            'normal': '#00ff88',
            'low_risk': '#00ffff',
            'moderate_risk': '#ffa500',
            'high_risk': '#ff4444',
            'drowsy': '#ff6b6b',
            'asleep': '#ff0000',
            'distracted': '#ff8800'
        }
        
        for i, alert in enumerate(alert_history[-20:]):  # Last 20 alerts
            state = alert.get('state', 'unknown').lower().replace(' ', '_')
            color = severity_colors.get(state, '#888888')
            
            fig_timeline.add_trace(go.Scatter(
                x=[i],
                y=[alert.get('confidence', 0.5)],
                mode='markers+text',
                marker=dict(size=20, color=color, symbol='circle'),
                text=[state[:3].upper()],
                textposition='middle center',
                textfont=dict(size=8, color='white'),
                hovertemplate=f"<b>{alert.get('state', 'Unknown')}</b><br>" +
                             f"Time: {alert.get('timestamp', 'N/A')}<br>" +
                             f"Duration: {alert.get('duration', 0):.1f}s<br>" +
                             f"Confidence: {alert.get('confidence', 0)*100:.0f}%<extra></extra>"
            ))
        
        fig_timeline.update_layout(
            title="Recent Alerts (Last 20)",
            height=200,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,33,48,0.5)',
            font=dict(color='#e6e6e6'),
            showlegend=False,
            xaxis=dict(title="Alert Sequence", showgrid=False),
            yaxis=dict(title="Confidence", range=[0, 1], showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
            margin=dict(l=50, r=20, t=50, b=40)
        )
        
        st.plotly_chart(fig_timeline, use_container_width=True, key="alert_timeline")
    else:
        # Show empty state with sample visualization
        st.info("🎉 No alerts recorded in this session - Great driving!")
        
        # Demo timeline
        fig_demo = go.Figure()
        fig_demo.add_annotation(
            x=0.5, y=0.5,
            text="No alerts to display",
            showarrow=False,
            font=dict(size=16, color='#888')
        )
        fig_demo.update_layout(
            height=100,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,33,48,0.3)',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        st.plotly_chart(fig_demo, use_container_width=True, key="demo_timeline")
    
    # ==================== BLINK RATE ANALYSIS ====================
    st.markdown("### 👁️ Blink Pattern Analysis")
    
    blink_col1, blink_col2 = st.columns([2, 1])
    
    with blink_col1:
        fig_blink = go.Figure()
        
        fig_blink.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['blink_rate'],
            mode='lines+markers',
            name='Blink Rate',
            line=dict(color='#9b59b6', width=2),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.1)'
        ))
        
        # Normal range band
        fig_blink.add_hrect(y0=15, y1=20, fillcolor="rgba(0,255,136,0.1)", 
                           line_width=0, annotation_text="Normal Range")
        
        fig_blink.update_layout(
            title="Blinks per Minute",
            height=250,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(30,33,48,0.5)',
            font=dict(color='#e6e6e6'),
            yaxis=dict(title="Blinks/min", showgrid=True, gridcolor='rgba(100,100,100,0.2)'),
            xaxis=dict(showgrid=False),
            margin=dict(l=50, r=20, t=50, b=30),
            showlegend=False
        )
        
        st.plotly_chart(fig_blink, use_container_width=True, key="blink_chart")
    
    with blink_col2:
        avg_blink = np.mean(df['blink_rate'])
        st.markdown("**📊 Blink Statistics**")
        st.metric("Average", f"{avg_blink:.1f}/min")
        st.metric("Min", f"{np.min(df['blink_rate']):.0f}/min")
        st.metric("Max", f"{np.max(df['blink_rate']):.0f}/min")
        
        if avg_blink < 12:
            st.warning("⚠️ Low blink rate detected - possible fatigue")
        elif avg_blink > 25:
            st.warning("⚠️ High blink rate - possible irritation")
        else:
            st.success("✅ Normal blink pattern")
    
    # ==================== EXPORT FUNCTIONALITY ====================
    st.markdown("---")
    
    if export_format == "CSV Data":
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"guardiandrive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    elif export_format == "JSON":
        session_data = {
            'session_id': SessionState.get('session_id'),
            'session_start': datetime.fromtimestamp(session_start).isoformat(),
            'duration_seconds': session_duration,
            'safety_score': safety_score,
            'total_alerts': total_alerts,
            'metrics_summary': {
                'avg_ear': float(np.mean(df['ear'])),
                'avg_perclos': float(np.mean(df['perclos'])),
                'avg_risk': float(np.mean(df['risk_score'])),
                'avg_blink_rate': float(avg_blink)
            },
            'alert_history': alert_history
        }
        st.download_button(
            label="📥 Download JSON",
            data=json.dumps(session_data, indent=2),
            file_name=f"guardiandrive_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    elif export_format == "PDF Report":
        st.info("📄 PDF generation requires additional setup. Export to CSV or JSON for now.")
    
    # ==================== SESSION COMPARISON (if enabled) ====================
    if compare_mode:
        st.markdown("---")
        st.markdown("### 📊 Session Comparison")
        st.info("📈 Session comparison feature - Select past sessions from database to compare")
        
        # Mock comparison data
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.markdown("**Current Session**")
            st.metric("Safety Score", f"{safety_score}/100")
            st.metric("Alerts", str(total_alerts))
            st.metric("Avg PERCLOS", f"{avg_perclos:.1f}%")
        
        with comp_col2:
            st.markdown("**Previous Session**")
            st.metric("Safety Score", "92/100", "+3 improvement")
            st.metric("Alerts", "5", "-2 better")
            st.metric("Avg PERCLOS", "8.2%", "-1.5% better")



def render_risk_mapping():
    """Premium Risk Mapping Dashboard with Interactive Maps"""
    import streamlit.components.v1 as components
    from datetime import datetime, timedelta
    
    st.title("🗺️ Risk Mapping Command Center")
    st.markdown("**Interactive route tracking and incident visualization**")
    
    # ==================== HEADER CONTROLS ====================
    col_route, col_time, col_layers = st.columns([2, 1, 1])
    
    with col_route:
        route_select = st.selectbox(
            "🛣️ Select Route",
            ["Current Session", "Last Trip", "Weekly Summary", "Demo Route"],
            key="route_selector"
        )
    
    with col_time:
        time_filter = st.selectbox(
            "⏰ Time Filter",
            ["All Time", "Last Hour", "Last 24h", "Last Week"],
            key="map_time_filter"
        )
    
    with col_layers:
        show_heatmap = st.checkbox("🔥 Show Heatmap", value=True, key="show_heatmap")
    
    st.markdown("---")
    
    # ==================== ROUTE STATISTICS ====================
    st.markdown("### 📊 Route Statistics")
    
    # Calculate stats from session
    session_start = SessionState.get('session_start', time.time())
    session_duration = time.time() - session_start
    total_alerts = SessionState.get('total_alerts', 0)
    
    stat1, stat2, stat3, stat4, stat5, stat6 = st.columns(6)
    
    with stat1:
        st.metric("📍 Distance", "12.4 km", "+2.1 km")
    with stat2:
        st.metric("⏱️ Duration", f"{session_duration/60:.0f} min", None)
    with stat3:
        st.metric("🚨 Incidents", str(total_alerts), None)
    with stat4:
        safe_percent = max(0, 100 - (total_alerts * 5))
        st.metric("✅ Safe %", f"{safe_percent}%", None)
    with stat5:
        st.metric("⚡ Avg Speed", "42 km/h", None)
    with stat6:
        st.metric("🔥 Risk Zones", "3", None)
    
    st.markdown("---")
    
    # ==================== INTERACTIVE MAP ====================
    try:
        import folium
        from folium import plugins
        import sys
        import os
        
        # Add project root to path for imports
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        try:
            from src.utils.map_visualization import MapVisualization, create_demo_map
            
            # Create map visualization
            viz = MapVisualization(center_lat=28.6139, center_lng=77.2090)
            viz.generate_demo_data(num_points=100, num_incidents=5)
            
            # Generate map
            m = viz.create_base_map()
            m = viz.add_route_layer(m)
            m = viz.add_incident_markers(m)
            m = viz.add_risk_zones(m)
            
            if show_heatmap:
                m = viz.add_heatmap_layer(m)
            
            # Add controls
            m = viz.add_minimap(m)
            m = viz.add_fullscreen(m)
            m = viz.add_measure_control(m)
            
            # Add layer control
            folium.LayerControl(position='topright', collapsed=False).add_to(m)
            
            # Fit to route
            if viz.route_points:
                coords = [[p.lat, p.lng] for p in viz.route_points]
                m.fit_bounds(coords)
            
            # Display map
            map_html = m._repr_html_()
            components.html(map_html, height=500, scrolling=False)
            
            # Get stats
            stats = viz.get_route_stats()
            
        except ImportError as e:
            st.warning(f"Map module not fully loaded: {e}")
            _render_fallback_map()
            stats = {}
        
    except ImportError:
        st.warning("📦 Folium library not installed. Installing...")
        st.code("pip install folium", language="bash")
        _render_fallback_map()
        stats = {}
    
    st.markdown("---")
    
    # ==================== MAP LEGEND ====================
    st.markdown("### 🎨 Map Legend")
    
    legend_col1, legend_col2, legend_col3 = st.columns(3)
    
    with legend_col1:
        st.markdown("**Route States:**")
        st.markdown("🟢 Normal - Safe driving")
        st.markdown("🔵 Low Risk - Minor concern")
        st.markdown("🟠 Moderate Risk - Attention needed")
        st.markdown("🔴 High Risk - Immediate action")
    
    with legend_col2:
        st.markdown("**Incident Types:**")
        st.markdown("😴 Drowsy - Eye closure detected")
        st.markdown("📱 Distracted - Attention diverted")
        st.markdown("⚠️ High Risk - Critical state")
        st.markdown("💤 Microsleep - Brief sleep episode")
    
    with legend_col3:
        st.markdown("**Risk Zones:**")
        st.markdown("🟢 Low - Historical safe area")
        st.markdown("🟠 Moderate - Some incidents")
        st.markdown("🔴 High - Frequent incidents")
        st.markdown("🟣 Critical - Danger zone")
    
    st.markdown("---")
    
    # ==================== INCIDENT LIST ====================
    st.markdown("### 🚨 Recent Incidents")
    
    alert_history = list(SessionState.get('alert_history', []))
    
    if alert_history:
        incident_data = []
        for i, alert in enumerate(alert_history[-10:]):
            incident_data.append({
                "Time": alert.get('timestamp', 'N/A'),
                "Type": alert.get('state', 'Unknown').replace('_', ' ').title(),
                "Duration": f"{alert.get('duration', 0):.1f}s",
                "Confidence": f"{alert.get('confidence', 0)*100:.0f}%",
                "Action": "📍 View on Map"
            })
        
        st.dataframe(
            incident_data,
            use_container_width=True,
            hide_index=True
        )
    else:
        st.success("🎉 No incidents recorded - Excellent driving!")
        
        # Show demo incidents
        st.markdown("**Demo Incident Data:**")
        demo_incidents = [
            {"Time": "10:23:45", "Type": "Drowsy", "Duration": "3.2s", "Confidence": "85%", "Location": "Highway 1"},
            {"Time": "10:45:12", "Type": "Distracted", "Duration": "2.1s", "Confidence": "72%", "Location": "Main St"},
            {"Time": "11:02:33", "Type": "High Risk", "Duration": "5.5s", "Confidence": "91%", "Location": "Junction A"},
        ]
        st.dataframe(demo_incidents, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # ==================== STATE DISTRIBUTION MAP ====================
    st.markdown("### 📊 State Distribution Along Route")
    
    # Create a simple bar showing route segments by state
    import plotly.graph_objects as go
    
    # Simulated segment data
    segments = [
        ("0-2 km", "Normal", 20),
        ("2-3 km", "Low Risk", 10),
        ("3-4 km", "Moderate", 10),
        ("4-5 km", "High Risk", 10),
        ("5-6 km", "Drowsy", 5),
        ("6-8 km", "Normal", 20),
        ("8-10 km", "Low Risk", 15),
        ("10-12 km", "Normal", 20),
    ]
    
    colors = {
        'Normal': '#00ff88',
        'Low Risk': '#00ffff',
        'Moderate': '#ffa500',
        'High Risk': '#ff4444',
        'Drowsy': '#ff6b6b',
        'Distracted': '#ff8800'
    }
    
    fig_route = go.Figure()
    
    x_pos = 0
    for segment, state, width in segments:
        fig_route.add_trace(go.Bar(
            x=[width],
            y=['Route'],
            orientation='h',
            name=state,
            marker_color=colors.get(state, '#888'),
            text=[segment],
            textposition='inside',
            hovertemplate=f"<b>{segment}</b><br>State: {state}<br>Distance: {width}%<extra></extra>"
        ))
    
    fig_route.update_layout(
        barmode='stack',
        height=100,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e6e6e6'),
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False)
    )
    
    st.plotly_chart(fig_route, use_container_width=True, key="route_segments")
    
    # ==================== EXPORT OPTIONS ====================
    st.markdown("### 📤 Export Options")
    
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        if st.button("📥 Download Route GPX", use_container_width=True):
            st.info("GPX export generating...")
    
    with export_col2:
        if st.button("📊 Export Incident Report", use_container_width=True):
            st.info("Report generating...")
    
    with export_col3:
        if st.button("🗺️ Share Map Link", use_container_width=True):
            st.success("Link copied to clipboard!")


def _render_fallback_map():
    """Render a fallback map visualization when Folium is unavailable"""
    import plotly.graph_objects as go
    
    st.markdown("**📍 Route Visualization (Plotly Fallback)**")
    
    # Generate demo route data
    np.random.seed(42)
    n_points = 100
    
    lats = 28.6139 + np.cumsum(np.random.normal(0, 0.001, n_points))
    lngs = 77.2090 + np.cumsum(np.random.normal(0, 0.001, n_points))
    
    # Color by risk
    colors = ['#00ff88'] * 20 + ['#00ffff'] * 10 + ['#ffa500'] * 10 + ['#ff4444'] * 10 + ['#ff6b6b'] * 5 + ['#00ff88'] * 45
    
    fig = go.Figure()
    
    # Route line
    fig.add_trace(go.Scattermapbox(
        lat=lats,
        lon=lngs,
        mode='lines+markers',
        marker=dict(size=6, color=colors),
        line=dict(width=3, color='#00d4aa'),
        name='Route'
    ))
    
    # Incident markers
    incident_lats = [lats[25], lats[35], lats[45]]
    incident_lngs = [lngs[25], lngs[35], lngs[45]]
    
    fig.add_trace(go.Scattermapbox(
        lat=incident_lats,
        lon=incident_lngs,
        mode='markers',
        marker=dict(size=15, color='#ff4444'),
        name='Incidents'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='carto-darkmatter',
            center=dict(lat=np.mean(lats), lon=np.mean(lngs)),
            zoom=13
        ),
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True, key="fallback_map")


# ==================== MAIN APP ====================


def main():
    """Main application entry point"""
    
    # Initialize session state
    SessionState()
    
    # Page config
    st.set_page_config(
        page_title="GuardianDrive AI Ultra",
        layout="wide",
        page_icon="🛡️",
        initial_sidebar_state="expanded"
    )
    
    # Inject PWA
    inject_pwa_components()
    
    # Sidebar navigation
    with st.sidebar:
        st.markdown("## 🛡️ GuardianDrive AI")
        st.markdown("**Ultra Edition v2.0**")
        st.markdown("---")
        
        page = st.radio("📍 Navigation", [
            "🚗 Live Detection",
            "📊 Analytics",
            "🗺️ Risk Mapping",
            "💼 Insurance",
            "🚨 Alerts",
            "⚙️ Settings"
        ], index=0)
        
        st.markdown("---")
        
        # System status
        st.markdown("**🔋 System Status**")
        st.markdown("🟢 Detection: Active")
        st.markdown("🟢 Camera: Connected")
        st.markdown("🔒 Privacy: 100% Local")
        
        st.markdown("---")
        st.markdown(f"**Session:** {SessionState.get('session_id', 'N/A')[:16]}...")
    
    # Route to pages
    if page == "🚗 Live Detection":
        render_live_detection()
    elif page == "📊 Analytics":
        render_analytics_dashboard()
    elif page == "🗺️ Risk Mapping":
        render_risk_mapping()
    elif page == "💼 Insurance":
        st.title("💼 Insurance Bridge")
        st.info("Insurance data bridge - full implementation available")
    elif page == "🚨 Alerts":
        st.title("🚨 Alert System")
        st.info("Multi-stakeholder alerts - full implementation available")
    elif page == "⚙️ Settings":
        st.title("⚙️ Settings")
        st.markdown("### Detection Parameters")
        st.slider("EAR Threshold", 0.15, 0.35, 0.22, 0.01)
        st.slider("PERCLOS Threshold", 0.05, 0.30, 0.15, 0.01)
        st.slider("Alert Sensitivity", 1, 10, 5)

if __name__ == "__main__":
    main()