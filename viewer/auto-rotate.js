/**
 * Auto-Rotate Controller for PlayCanvas Model Viewer
 *
 * Features:
 * - Y-axis auto-rotation using OrbitCamera yaw manipulation
 * - User input detection (mouse/touch)
 * - Idle timer for auto-resume
 * - postMessage API for external control
 *
 * Based on: https://playcanvas.com/project/561263 (Auto-Rotate Orbit Camera)
 */

(function() {
    'use strict';

    // Configuration
    const CONFIG = {
        SEARCH_ATTEMPTS: 100,           // Try for 10 seconds
        SEARCH_INTERVAL: 100,           // ms between attempts
        DEFAULT_SPEED: 15,              // degrees per second
        IDLE_TIMEOUT: 2000,             // ms until auto-rotate resumes
    };

    // State
    let viewer = null;
    let orbitCamera = null;
    let app = null;
    let autoRotate = false;
    let rotationSpeed = CONFIG.DEFAULT_SPEED;
    let userInteracting = false;
    let idleTimer = null;
    let updateCallback = null;

    /**
     * Find OrbitCamera in PlayCanvas Model Viewer
     */
    function findOrbitCamera() {
        console.log('[AutoRotate] Searching for OrbitCamera...');

        // Check if window.viewer exists
        if (!window.viewer) {
            return false;
        }

        viewer = window.viewer;

        // Method 1: Check viewer.cameraControls (PlayCanvas Model Viewer v5.6.3+)
        if (viewer.cameraControls && viewer.cameraControls._pose) {
            orbitCamera = viewer.cameraControls;
            app = viewer.app;
            console.log('[AutoRotate] ✅ Found at: viewer.cameraControls');
            logOrbitInfo();
            return true;
        }

        // Method 2: Check viewer.camera.script.orbitCamera
        if (viewer.camera?.script?.orbitCamera) {
            orbitCamera = viewer.camera.script.orbitCamera;
            app = viewer.app;
            console.log('[AutoRotate] ✅ Found at: viewer.camera.script.orbitCamera');
            logOrbitInfo();
            return true;
        }

        // Method 3: Check viewer.orbitCamera directly
        if (viewer.orbitCamera) {
            orbitCamera = viewer.orbitCamera;
            app = viewer.app;
            console.log('[AutoRotate] ✅ Found at: viewer.orbitCamera');
            logOrbitInfo();
            return true;
        }

        // Method 4: Search for objects with yaw/pitch/distance properties
        for (const key in viewer) {
            const obj = viewer[key];
            if (obj && typeof obj === 'object') {
                if (('yaw' in obj || 'pitch' in obj) && 'distance' in obj) {
                    orbitCamera = obj;
                    app = viewer.app;
                    console.log(`[AutoRotate] ✅ Found at: viewer.${key}`);
                    logOrbitInfo();
                    return true;
                }
            }
        }

        return false;
    }

    /**
     * Log OrbitCamera info for debugging
     */
    function logOrbitInfo() {
        if (!orbitCamera) return;

        console.log('[AutoRotate] OrbitCamera properties:');

        // Check for cameraControls._pose structure
        if (orbitCamera._pose) {
            console.log('  yaw (angles.y):', orbitCamera._pose.angles.y);
            console.log('  pitch (angles.x):', orbitCamera._pose.angles.x);
            console.log('  distance:', orbitCamera._pose.distance);
        } else {
            // Legacy structure
            console.log('  yaw:', orbitCamera.yaw);
            console.log('  pitch:', orbitCamera.pitch);
            console.log('  distance:', orbitCamera.distance);
        }

        console.log('  App found:', !!app);
    }

    /**
     * Start auto-rotation
     */
    function startAutoRotate(speed) {
        if (!orbitCamera || !app) {
            console.error('[AutoRotate] Cannot start: OrbitCamera or app not found');
            return false;
        }

        if (autoRotate) {
            console.log('[AutoRotate] Already running');
            return false;
        }

        if (speed !== undefined) {
            rotationSpeed = speed;
        }

        autoRotate = true;

        // Create update callback
        updateCallback = function(dt) {
            if (autoRotate && !userInteracting && orbitCamera) {
                // Check for cameraControls._pose structure
                if (orbitCamera._pose) {
                    // Increase yaw (angles.y) by speed degrees per second
                    orbitCamera._pose.angles.y += rotationSpeed * dt;

                    // Wrap around 360 degrees
                    if (orbitCamera._pose.angles.y >= 360) {
                        orbitCamera._pose.angles.y -= 360;
                    } else if (orbitCamera._pose.angles.y < 0) {
                        orbitCamera._pose.angles.y += 360;
                    }

                    // Update camera controls to apply changes
                    if (typeof orbitCamera.update === 'function') {
                        orbitCamera.update(dt);
                    }
                } else {
                    // Legacy structure
                    orbitCamera.yaw += rotationSpeed * dt;

                    // Wrap around 360 degrees
                    if (orbitCamera.yaw >= 360) {
                        orbitCamera.yaw -= 360;
                    } else if (orbitCamera.yaw < 0) {
                        orbitCamera.yaw += 360;
                    }
                }
            }
        };

        // Register with PlayCanvas app update loop
        app.on('update', updateCallback);

        console.log(`[AutoRotate] ✅ Started! Speed: ${rotationSpeed} deg/sec`);
        return true;
    }

    /**
     * Stop auto-rotation
     */
    function stopAutoRotate() {
        if (!autoRotate) {
            console.log('[AutoRotate] Not running');
            return false;
        }

        autoRotate = false;

        // Unregister update callback
        if (app && updateCallback) {
            app.off('update', updateCallback);
            updateCallback = null;
        }

        console.log('[AutoRotate] ⏹️ Stopped');
        return true;
    }

    /**
     * Handle user input start (mouse down / touch start)
     */
    function handleInputStart() {
        if (!autoRotate) return;

        userInteracting = true;

        // Clear existing idle timer
        if (idleTimer) {
            clearTimeout(idleTimer);
            idleTimer = null;
        }

        console.log('[AutoRotate] 👆 User interaction - pausing');
    }

    /**
     * Handle user input end (mouse up / touch end)
     */
    function handleInputEnd() {
        if (!autoRotate || !userInteracting) return;

        // Set idle timer to resume after timeout
        idleTimer = setTimeout(() => {
            userInteracting = false;
            idleTimer = null;
            console.log('[AutoRotate] ⏱️ Idle timeout - resuming');
        }, CONFIG.IDLE_TIMEOUT);
    }

    /**
     * Setup input listeners for user interaction detection
     */
    function setupInputListeners() {
        if (!app) return;

        // Mouse events
        if (app.mouse) {
            app.mouse.on('mousedown', handleInputStart);
            app.mouse.on('mouseup', handleInputEnd);
            console.log('[AutoRotate] ✅ Mouse listeners registered');
        }

        // Touch events
        if (app.touch) {
            app.touch.on('touchstart', handleInputStart);
            app.touch.on('touchend', handleInputEnd);
            console.log('[AutoRotate] ✅ Touch listeners registered');
        }
    }

    /**
     * Setup postMessage listener for external control
     */
    function setupMessageListener() {
        window.addEventListener('message', function(event) {
            const message = event.data;

            if (!message || typeof message !== 'object') {
                return;
            }

            switch (message.action) {
                case 'startAutoRotate':
                    const speed = message.speed || CONFIG.DEFAULT_SPEED;
                    const success = startAutoRotate(speed);
                    console.log(`[AutoRotate] postMessage: startAutoRotate(${speed}) -> ${success}`);
                    break;

                case 'stopAutoRotate':
                    stopAutoRotate();
                    console.log('[AutoRotate] postMessage: stopAutoRotate()');
                    break;

                case 'setSpeed':
                    if (message.speed !== undefined) {
                        rotationSpeed = message.speed;
                        console.log(`[AutoRotate] postMessage: setSpeed(${message.speed})`);
                    }
                    break;

                default:
                    // Ignore unknown messages
                    break;
            }
        });

        console.log('[AutoRotate] ✅ postMessage listener registered');
    }

    /**
     * Initialize auto-rotate controller
     */
    function initialize() {
        let attempts = 0;

        const searchInterval = setInterval(() => {
            attempts++;

            if (findOrbitCamera()) {
                clearInterval(searchInterval);

                // Setup listeners
                setupInputListeners();
                setupMessageListener();

                // Expose global API for debugging
                window.autoRotate = {
                    start: (speed) => startAutoRotate(speed),
                    stop: () => stopAutoRotate(),
                    setSpeed: (speed) => { rotationSpeed = speed; },
                    getState: () => ({
                        active: autoRotate,
                        userInteracting,
                        speed: rotationSpeed,
                        yaw: orbitCamera?._pose ? orbitCamera._pose.angles.y : orbitCamera?.yaw,
                        pitch: orbitCamera?._pose ? orbitCamera._pose.angles.x : orbitCamera?.pitch,
                        distance: orbitCamera?._pose ? orbitCamera._pose.distance : orbitCamera?.distance
                    }),
                    orbitCamera: () => orbitCamera,
                    app: () => app
                };

                // Notify parent window
                window.parent.postMessage({
                    type: 'autoRotateReady',
                    success: true
                }, '*');

                console.log('[AutoRotate] ✅ Initialized successfully!');
                console.log('[AutoRotate] Available at: window.autoRotate');

            } else if (attempts >= CONFIG.SEARCH_ATTEMPTS) {
                clearInterval(searchInterval);

                console.error(`[AutoRotate] ❌ Failed to find OrbitCamera after ${attempts} attempts`);
                console.log('[AutoRotate] Available window.viewer keys:', Object.keys(window.viewer || {}));

                // Notify parent window of failure
                window.parent.postMessage({
                    type: 'autoRotateReady',
                    success: false
                }, '*');
            }
        }, CONFIG.SEARCH_INTERVAL);
    }

    // Start initialization when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initialize);
    } else {
        // DOM already loaded, wait a bit for viewer to initialize
        setTimeout(initialize, 1000);
    }

    // Also try on window load as fallback
    window.addEventListener('load', function() {
        if (!orbitCamera) {
            setTimeout(() => {
                if (!orbitCamera) {
                    initialize();
                }
            }, 500);
        }
    });

})();
