#!/usr/bin/env python3
"""
Generate Alert Tone MP3 Files
 
Script to create 3 alert tone MP3 files for the INIDS system.
Frequencies:
- alert_low.mp3: 220 Hz (low alert)
- alert_med.mp3: 440 Hz (medium alert)
- alert_high.mp3: 880 Hz (high alert)

Requirements:
  pip install scipy
  ffmpeg (must be installed on system)

Run from the /static/sfx/ directory:
  python3 generate_alerts.py
"""

import os
import subprocess
import sys

def generate_tone(frequency, duration, output_file):
    """Generate a sine wave tone and save as MP3"""
    print(f"Generating {output_file} ({frequency} Hz, {duration}s)...")
    
    cmd = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', f'sine=f={frequency}:d={duration}',
        '-q:a', '9',
        '-acodec', 'libmp3lame',
        '-y',  # Overwrite output file
        output_file
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return False
        print(f"✓ {output_file} created successfully")
        return True
    except FileNotFoundError:
        print("ERROR: ffmpeg not found. Please install ffmpeg:")
        print("  Ubuntu/Debian: sudo apt-get install ffmpeg")
        print("  macOS: brew install ffmpeg")
        print("  Windows: choco install ffmpeg")
        return False

def main():
    """Generate all alert tones"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=== INIDS Alert Tone Generator ===\n")
    
    # Define tones to generate
    tones = [
        (220, 1, 'alert_low.mp3'),
        (440, 1, 'alert_med.mp3'),
        (880, 1, 'alert_high.mp3')
    ]
    
    success_count = 0
    for freq, duration, output in tones:
        if generate_tone(freq, duration, output):
            success_count += 1
    
    print(f"\n✓ Generated {success_count}/{len(tones)} alert tones")
    
    if success_count == len(tones):
        print("\n✓ All alert tones ready!")
        print("Testing playback in browser console:")
        print("  new Audio('/static/sfx/alert_low.mp3').play();")
        print("  new Audio('/static/sfx/alert_med.mp3').play();")
        print("  new Audio('/static/sfx/alert_high.mp3').play();")
        return 0
    else:
        print(f"\nERROR: Only {success_count}/{len(tones)} tones generated")
        return 1

if __name__ == '__main__':
    sys.exit(main())
