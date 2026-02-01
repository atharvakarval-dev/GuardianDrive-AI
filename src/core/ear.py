import numpy as np

def euclidean(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices):
    """
    Calculate Eye Aspect Ratio (EAR) for a given eye.
    
    Args:
        landmarks: List of (x, y) coordinates for facial landmarks.
        eye_indices: List of indices for the specific eye (6 points).
        
    Returns:
        float: The calculated EAR value.
    """
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
