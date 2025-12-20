"""
test_microphone.py
Quick test to verify microphone is working and capturing audio
"""

import sounddevice as sd
import numpy as np
import whisper

print("=" * 60)
print("🎤 Microphone Test")
print("=" * 60)

# Test 1: Check available devices
print("\n1. Available Audio Devices:")
print("-" * 60)
devices = sd.query_devices()
for i, device in enumerate(devices):
    if device['max_input_channels'] > 0:
        print(f"   [{i}] {device['name']} (Input)")

# Test 2: Check default input
print("\n2. Default Input Device:")
print("-" * 60)
default_input = sd.query_devices(kind='input')
print(f"   Device: {default_input['name']}")
print(f"   Sample Rate: {default_input['default_samplerate']} Hz")
print(f"   Input Channels: {default_input['max_input_channels']}")

# Test 3: Record 5 seconds
print("\n3. Recording Test (5 seconds):")
print("-" * 60)
print("   🔴 Recording... SPEAK NOW!")

duration = 5
sample_rate = 16000

try:
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    audio = audio.flatten()
    
    # Check if audio was captured
    max_amplitude = np.max(np.abs(audio))
    mean_amplitude = np.mean(np.abs(audio))
    
    print(f"   ✅ Recording complete!")
    print(f"   Max amplitude: {max_amplitude:.4f}")
    print(f"   Mean amplitude: {mean_amplitude:.4f}")
    
    if max_amplitude < 0.001:
        print("   ⚠️  WARNING: Very low audio level!")
        print("   Check:")
        print("      - Microphone is plugged in")
        print("      - Microphone is not muted")
        print("      - Windows microphone permissions")
        print("      - Microphone volume in Windows settings")
    else:
        print("   ✅ Audio levels look good!")
    
    # Test 4: Transcribe with Whisper
    print("\n4. Whisper Transcription Test:")
    print("-" * 60)
    print("   Loading Whisper model...")
    
    model = whisper.load_model("base")
    
    print("   Transcribing...")
    result = model.transcribe(audio, language="mr", fp16=False)
    
    text = result["text"].strip()
    
    print(f"   📝 Transcribed text: '{text}'")
    
    if not text:
        print("   ⚠️  No text detected!")
        print("   Try:")
        print("      - Speaking louder")
        print("      - Getting closer to microphone")
        print("      - Checking microphone is selected as default")
    else:
        print("   ✅ Transcription successful!")
    
    # Show segments
    if result.get("segments"):
        print(f"   Segments: {len(result['segments'])}")
        for i, seg in enumerate(result["segments"][:3]):
            print(f"      [{i+1}] {seg['text']} (confidence: ~{1-seg.get('no_speech_prob', 0):.2f})")

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)

# Recommendations
print("\n💡 Troubleshooting Tips:")
print("   1. Check Windows Sound Settings:")
print("      - Right-click speaker icon → Sounds → Recording")
print("      - Select your microphone → Properties → Levels")
print("      - Set to 80-100%")
print("")
print("   2. Grant Microphone Permission:")
print("      - Windows Settings → Privacy → Microphone")
print("      - Allow apps to access microphone")
print("")
print("   3. Test in Windows:")
print("      - Settings → System → Sound → Input")
print("      - Test your microphone there first")
print("")
print("   4. If using headset/external mic:")
print("      - Make sure it's set as default device")
print("      - Check cable connection")