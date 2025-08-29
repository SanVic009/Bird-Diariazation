#!/usr/bin/env python3
"""
Simple Audio Input Test
======================
This script just tests audio input functionality without any ML model.
"""

import sounddevice as sd
import numpy as np
import time

# Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_DURATION = 0.5  # seconds
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

def audio_callback(indata, frames, time_info, status):
    """This is called for each audio block."""
    if status:
        print('Status:', status)
    
    # Calculate audio levels
    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    
    # Create a visual meter
    meter_chunks = 40
    meter = '|' + '=' * int(volume_norm * meter_chunks) + ' ' * (meter_chunks - int(volume_norm * meter_chunks)) + '|'
    
    print(f'Volume: {volume_norm:.4f} {meter}')

def main():
    print("\nAudio Input Test")
    print("===============")
    
    # List available devices
    print("\nAvailable Audio Devices:")
    print("----------------------")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"Device {i}: {device['name']}")
            print(f"  Channels: {device['max_input_channels']}")
            print(f"  Sample Rate: {device['default_samplerate']}")
    
    # Get device selection
    device_id = int(input("\nEnter device number to test: "))
    
    print(f"\nTesting device {device_id}")
    print("Press Ctrl+C to stop\n")
    
    try:
        with sd.InputStream(
            device=device_id,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            callback=audio_callback
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest finished.")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == '__main__':
    main()
