/**
 * Alert Tone Generator Setup
 * 
 * This directory should contain 3 alert tone MP3 files:
 * - alert_low.mp3 (220 Hz tone, 1 second, -6dB volume)
 * - alert_med.mp3 (440 Hz tone, 1 second, -3dB volume)
 * - alert_high.mp3 (880 Hz tone, 1 second, 0dB volume)
 * 
 * SETUP OPTIONS:
 * 
 * Option 1: Generate using FFmpeg (Recommended)
 * ============================================
 * Install FFmpeg if not already installed
 * 
 * # Generate alert_low.mp3 (220 Hz, 1 sec)
 * ffmpeg -f lavfi -i sine=f=220:d=1 -q:a 9 -acodec libmp3lame alert_low.mp3
 * 
 * # Generate alert_med.mp3 (440 Hz, 1 sec)
 * ffmpeg -f lavfi -i sine=f=440:d=1 -q:a 9 -acodec libmp3lame alert_med.mp3
 * 
 * # Generate alert_high.mp3 (880 Hz, 1 sec)
 * ffmpeg -f lavfi -i sine=f=880:d=1 -q:a 9 -acodec libmp3lame alert_high.mp3
 * 
 * 
 * Option 2: Generate using Python script
 * =======================================
 * Run: python3 generate_alerts.py (from this directory)
 * Requires: pip install scipy
 * 
 * 
 * Option 3: Download from CDN
 * ===========================
 * Use online tone generators to create the files and download
 * Place files in this directory with the correct names
 * 
 * 
 * Option 4: Use existing audio files
 * ==================================
 * If you have alert tone MP3 files from another source,
 * rename them appropriately and place them here:
 * - alert_low.mp3
 * - alert_med.mp3
 * - alert_high.mp3
 * 
 * 
 * VALIDATION:
 * ===========
 * After adding files, verify they work by opening DevTools console:
 * 
 * new Audio("/static/sfx/alert_low.mp3").play();
 * new Audio("/static/sfx/alert_med.mp3").play();
 * new Audio("/static/sfx/alert_high.mp3").play();
 * 
 * Each should produce a distinct tone.
 */
