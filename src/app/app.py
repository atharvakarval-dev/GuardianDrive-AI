import streamlit as st
from streamlit_webrtc import webrtc_streamer

# Local imports
from config.config import EAR_THRESHOLD, CONSEC_FRAMES
from src.core.detector import DrowsinessDetector
from src.app.pwa_utils import serve_pwa_files, render_install_guide

def main():
    # Set page config
    st.set_page_config(
        page_title="Driver Drowsiness Detection", 
        layout="centered",
        page_icon="🚗",
        initial_sidebar_state="collapsed"
    )
    
    # Serve PWA files
    serve_pwa_files()
    
    # App header
    st.title("🚗 Driver Drowsiness Detection")
    st.markdown("**Real-time drowsiness detection using computer vision**")
    
    # PWA status indicator
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("ℹ️ PWA Info"):
            st.info("This app supports offline use when installed!")
    
    # Installation guide
    render_install_guide()
    
    # Main camera interface
    st.markdown("---")
    st.markdown("### Live Detection")
    st.markdown("Allow camera access to start drowsiness detection.")
    
    # Run webcam with Streamlit
    webrtc_streamer(
        key="drowsiness-app",
        video_processor_factory=DrowsinessDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    
    # Additional info
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("EAR Threshold", f"{EAR_THRESHOLD}")
    with col2:
        st.metric("Alert Frames", f"{CONSEC_FRAMES}")
    with col3:
        st.metric("Status", "🟢 Ready")
    
    # Footer with PWA info
    st.markdown("---")
    st.markdown(
        "<small>💡 **Tip**: Install this app for offline access and better performance!</small>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
