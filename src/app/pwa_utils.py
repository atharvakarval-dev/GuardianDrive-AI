import streamlit as st
import base64
import os

def get_pwa_head():
    """Returns the HTML head content for PWA."""
    
    # Read manifest if exists
    manifest_content = "{}"
    if os.path.exists('manifest.json'):
        with open('manifest.json', 'rb') as f:
            manifest_content = f.read()
    
    # Read icon if exists
    icon_b64 = ""
    if os.path.exists('icons/icon-192x192.png'):
        with open('icons/icon-192x192.png', 'rb') as f:
            icon_b64 = base64.b64encode(f.read()).decode()

    pwa_head = f"""
    <head>
        <link rel="manifest" href="data:application/json;base64,{base64.b64encode(manifest_content).decode()}">
        <meta name="theme-color" content="#ff6b6b">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="default">
        <meta name="apple-mobile-web-app-title" content="DrowsinessDetect">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        
        <!-- PWA Icons -->
        <link rel="icon" type="image/png" sizes="32x32" href="data:image/png;base64,{icon_b64}">
        <link rel="apple-touch-icon" href="data:image/png;base64,{icon_b64}">
        
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
            
            /* PWA-friendly responsive design */
            @media (max-width: 768px) {{
                .main .block-container {{
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                }}
            }}
            
            /* Hide Streamlit branding for PWA */
            .viewerBadge_container__1QSob {{
                display: none;
            }}
            
            footer {{
                visibility: hidden;
            }}
        </style>
    </head>
    """
    return pwa_head

def get_pwa_js():
    """Returns the JavaScript content for PWA service worker and install prompt."""
    return """
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

def serve_pwa_files():
    """Injects PWA head and JS into the Streamlit app"""
    pwa_head = get_pwa_head()
    pwa_js = get_pwa_js()
    st.markdown(pwa_head + pwa_js, unsafe_allow_html=True)

def render_install_guide():
    """Renders the installation instructions expander."""
    with st.expander("📱 Install as Mobile App", expanded=False):
        st.markdown("""
        ### Install this Progressive Web App for the best experience:
        
        **📱 On Mobile:**
        - **Chrome/Edge**: Tap menu → "Add to Home Screen"
        - **Safari**: Tap share → "Add to Home Screen"
        
        **💻 On Desktop:**
        - **Chrome/Edge**: Click install icon in address bar
        - Or look for "Install App" button
        
        **✨ Benefits:**
        - 🚀 Faster loading
        - 📱 App-like experience
        - 🔒 Works offline
        - 💾 Cached for speed
        """)
