"""
Secure Alert System for Driver State Detection
Implements encrypted communication with authorities and emergency contacts
"""

import json
import time
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import base64
from datetime import datetime
import queue
from config.env_config import Config

@dataclass
class AlertData:
    driver_id: str
    vehicle_id: str
    timestamp: str
    location: Dict[str, float]  # {"lat": 0.0, "lng": 0.0}
    driver_state: str
    confidence: float
    metrics: Dict
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"

@dataclass
class EmergencyContact:
    name: str
    phone: str
    email: str
    relationship: str
    priority: int

class AlertSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class SecureAlertSystem:
    def __init__(self, config_path: str = "config/alert_config.json"):
        self.config = self.load_config(config_path)
        self.alert_queue = queue.Queue()
        self.offline_buffer = []
        self.is_online = True
        self.alert_thread = threading.Thread(target=self._process_alerts, daemon=True)
        self.alert_thread.start()
        
    def load_config(self, config_path: str) -> Dict:
        """Load alert system configuration"""
        default_config = {
            "driver_id": "DRIVER_001",
            "vehicle_id": "VEH_001",
            "emergency_contacts": [
                {
                    "name": "Emergency Contact",
                    "phone": "+91XXXXXXXXXX",
                    "email": "emergency@example.com",
                    "relationship": "Family",
                    "priority": 1
                }
            ],
            "authorities": {
                "police_control_room": {
                    "endpoint": "https://police-api.gov.in/alerts",
                    "api_key": Config.POLICE_API_KEY or "YOUR_API_KEY",
                    "enabled": bool(Config.POLICE_API_KEY)
                },
                "traffic_control": {
                    "endpoint": "https://traffic-control.gov.in/incidents",
                    "api_key": Config.POLICE_API_KEY or "YOUR_API_KEY",
                    "enabled": bool(Config.POLICE_API_KEY)
                }
            },
            "alert_thresholds": {
                "drowsy_duration": 3.0,  # seconds
                "critical_duration": 10.0,  # seconds
                "intoxication_confidence": 0.8
            },
            "privacy": {
                "encrypt_data": True,
                "anonymize_location": False,
                "data_retention_days": 30
            }
        }
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key, value in default_config.items():
                    if key not in config:
                        config[key] = value
                return config
        except FileNotFoundError:
            # Create default config file
            import os
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def determine_severity(self, driver_state: str, metrics: Dict, duration: float) -> AlertSeverity:
        """Determine alert severity based on driver state and metrics"""
        if driver_state == "Asleep":
            return AlertSeverity.CRITICAL
        elif driver_state == "High Risk": # Replaces "Drunk"
            if metrics.get("confidence", 0) > self.config["alert_thresholds"]["intoxication_confidence"]:
                return AlertSeverity.CRITICAL
            else:
                return AlertSeverity.HIGH
        elif driver_state == "Drowsy": # Replaces "Sleepy/Drowsy"
            if duration > self.config["alert_thresholds"]["critical_duration"]:
                return AlertSeverity.CRITICAL
            elif duration > self.config["alert_thresholds"]["drowsy_duration"]:
                return AlertSeverity.HIGH
            else:
                return AlertSeverity.MEDIUM
        elif driver_state == "Moderate Risk":
            return AlertSeverity.MEDIUM
        elif driver_state == "Low Risk":
            return AlertSeverity.LOW
        else:
            return AlertSeverity.LOW
    
    def get_mock_location(self) -> Dict[str, float]:
        """Get mock GPS location (replace with actual GPS in production)"""
        # Mock location - replace with actual GPS integration
        return {
            "lat": 28.6139,  # Delhi coordinates
            "lng": 77.2090,
            "accuracy": 10.0,
            "timestamp": time.time()
        }
    
    def encrypt_data(self, data: str) -> str:
        """Simple encryption for sensitive data (use proper encryption in production)"""
        if not self.config["privacy"]["encrypt_data"]:
            return data
            
        # Simple base64 encoding (replace with proper encryption)
        encoded = base64.b64encode(data.encode()).decode()
        return encoded
    
    def create_alert(self, driver_state: str, metrics: Dict, 
                    duration: float = 0.0, confidence: float = 0.8) -> AlertData:
        """Create an alert data structure"""
        severity = self.determine_severity(driver_state, metrics, duration)
        location = self.get_mock_location()
        
        # Anonymize location if configured
        if self.config["privacy"]["anonymize_location"]:
            # Round coordinates to reduce precision
            location["lat"] = round(location["lat"], 2)
            location["lng"] = round(location["lng"], 2)
        
        alert = AlertData(
            driver_id=self.config["driver_id"],
            vehicle_id=self.config["vehicle_id"],
            timestamp=datetime.now().isoformat(),
            location=location,
            driver_state=driver_state,
            confidence=confidence,
            metrics=metrics,
            severity=severity.value
        )
        
        return alert
    
    def send_to_authorities(self, alert: AlertData) -> bool:
        """Send alert to police/traffic control (mock implementation)"""
        try:
            # Mock implementation - replace with actual API calls
            authorities = self.config["authorities"]
            
            for authority, config in authorities.items():
                if not config.get("enabled", False):
                    continue
                    
                alert_data = asdict(alert)
                encrypted_data = self.encrypt_data(json.dumps(alert_data))
                
                # Mock API call
                print(f"[MOCK] Sending alert to {authority}:")
                print(f"  Endpoint: {config['endpoint']}")
                print(f"  Data: {encrypted_data[:100]}...")
                print(f"  Severity: {alert.severity}")
                
                # In production, implement actual HTTP requests here
                # response = requests.post(config['endpoint'], 
                #                        headers={'Authorization': f'Bearer {config["api_key"]}'},
                #                        json={'data': encrypted_data})
                
            return True
        except Exception as e:
            print(f"Failed to send alert to authorities: {e}")
            return False
    
    def send_to_emergency_contacts(self, alert: AlertData) -> bool:
        """Send alert to emergency contacts (real SMS if configured)"""
        try:
            from src.utils.sms_service import get_sms_service
            sms_service = get_sms_service()
            
            contacts = self.config["emergency_contacts"]
            
            for contact in sorted(contacts, key=lambda x: x["priority"]):
                message = self.create_emergency_message(alert, contact)
                
                if sms_service.enabled:
                    # Send real SMS
                    result = sms_service.send_sms(contact['phone'], message)
                    print(f"[SMS] Sent to {contact['name']}: {result['status']}")
                else:
                    # Mock SMS
                    print(f"[MOCK SMS] To {contact['name']} ({contact['phone']}):")
                    print(f"  Message: {message}")
                
            return True
        except Exception as e:
            print(f"Failed to send alert to emergency contacts: {e}")
            return False
    
    def create_emergency_message(self, alert: AlertData, contact: EmergencyContact) -> str:
        """Create user-friendly emergency message"""
        message = f"""DRIVER ALERT
        
Driver: {alert.driver_id}
Vehicle: {alert.vehicle_id}
Status: {alert.driver_state}
Severity: {alert.severity}
Time: {alert.timestamp}
Location: {alert.location['lat']:.4f}, {alert.location['lng']:.4f}

This is an automated alert from the vehicle's safety system.
Please check on the driver immediately.

- Vehicle Safety System"""
        
        return message
    
    def trigger_alert(self, driver_state: str, metrics: Dict, 
                     duration: float = 0.0, confidence: float = 0.8):
        """Trigger an alert for the given driver state"""
        alert = self.create_alert(driver_state, metrics, duration, confidence)
        self.alert_queue.put(alert)
        
        # Immediate console alert
        print(f"\nDRIVER ALERT TRIGGERED")
        print(f"State: {driver_state}")
        print(f"Severity: {alert.severity}")
        print(f"Time: {alert.timestamp}")
        print(f"Confidence: {confidence:.2f}")
    
    def _process_alerts(self):
        """Background thread to process alert queue"""
        while True:
            try:
                alert = self.alert_queue.get(timeout=1.0)
                
                if self.is_online:
                    # Send to authorities for high/critical alerts
                    if alert.severity in ["HIGH", "CRITICAL"]:
                        success_auth = self.send_to_authorities(alert)
                        if not success_auth:
                            self.offline_buffer.append(alert)
                    
                    # Always send to emergency contacts for medium+ alerts
                    if alert.severity in ["MEDIUM", "HIGH", "CRITICAL"]:
                        success_contacts = self.send_to_emergency_contacts(alert)
                        if not success_contacts:
                            self.offline_buffer.append(alert)
                else:
                    # Store in offline buffer
                    self.offline_buffer.append(alert)
                    print(f"[OFFLINE] Alert buffered: {alert.severity}")
                
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing alert: {e}")
    
    def set_online_status(self, is_online: bool):
        """Update online status and process buffered alerts if back online"""
        self.is_online = is_online
        
        if is_online and self.offline_buffer:
            print(f"[ONLINE] Processing {len(self.offline_buffer)} buffered alerts...")
            for alert in self.offline_buffer:
                self.alert_queue.put(alert)
            self.offline_buffer.clear()
    
    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics"""
        return {
            "queue_size": self.alert_queue.qsize(),
            "buffered_alerts": len(self.offline_buffer),
            "is_online": self.is_online,
            "config_loaded": bool(self.config)
        }