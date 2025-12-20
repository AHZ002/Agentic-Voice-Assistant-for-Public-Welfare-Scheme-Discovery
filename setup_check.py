"""
setup_check.py
Verifies that all dependencies and configuration are correct before running main.py
"""

import sys
import os

def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   Required: Python 3.8 or higher")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required_packages = [
        ("google.generativeai", "Google Gemini API"),
        ("whisper", "OpenAI Whisper"),
        ("sounddevice", "Audio capture"),
        ("gtts", "Google Text-to-Speech"),
        ("pygame", "Audio playback"),
        ("numpy", "Numerical operations"),
        ("scipy", "Scientific computing")
    ]
    
    all_installed = True
    
    for package, description in required_packages:
        try:
            __import__(package)
            print(f"✅ {package:25s} - {description}")
        except ImportError:
            print(f"❌ {package:25s} - {description} (NOT INSTALLED)")
            all_installed = False
    
    return all_installed

def check_api_key():
    """Check if Google Gemini API key is set"""
    print("\nChecking API key...")
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if not api_key:
        print("❌ GEMINI_API_KEY not found in environment variables")
        print("\n   To set the API key:")
        print("   PowerShell: $env:GEMINI_API_KEY=\"your-key-here\"")
        print("   CMD:        set GEMINI_API_KEY=your-key-here")
        print("   Bash:       export GEMINI_API_KEY=\"your-key-here\"")
        print("\n   Get a FREE API key at: https://makersuite.google.com/app/apikey")
        print("   ✨ FREE TIER: 60 requests/minute!")
        return False
    
    if len(api_key) < 20:
        print("⚠️  API key seems too short - please verify")
        return False
    
    print(f"✅ API key found (length: {len(api_key)})")
    return True

def check_schemes_file():
    """Check if schemes.json exists"""
    print("\nChecking schemes data file...")
    
    # Update this path to match your setup
    schemes_path = r"C:\Users\ABDUL_HADI\Desktop\Voice Agent\data\schemes.json"
    
    if not os.path.exists(schemes_path):
        print(f"❌ schemes.json not found at: {schemes_path}")
        print("   Please update the path in main.py and this script")
        return False
    
    print(f"✅ schemes.json found")
    
    # Check if it's valid JSON
    try:
        import json
        with open(schemes_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            print(f"   Contains {len(data)} schemes")
        elif isinstance(data, dict) and 'schemes' in data:
            print(f"   Contains {len(data['schemes'])} schemes")
        else:
            print("   ⚠️  Unexpected JSON structure")
    
    except Exception as e:
        print(f"   ⚠️  Error reading file: {e}")
    
    return True

def check_audio_devices():
    """Check if audio devices are available"""
    print("\nChecking audio devices...")
    
    try:
        import sounddevice as sd
        
        # Get default input device
        try:
            input_device = sd.query_devices(kind='input')
            print(f"✅ Microphone found: {input_device['name']}")
        except Exception as e:
            print(f"❌ No microphone detected: {e}")
            return False
        
        # Get default output device
        try:
            output_device = sd.query_devices(kind='output')
            print(f"✅ Speaker found: {output_device['name']}")
        except Exception as e:
            print(f"⚠️  No speaker detected: {e}")
        
        return True
    
    except ImportError:
        print("❌ sounddevice not installed - cannot check audio devices")
        return False

def check_whisper_model():
    """Check if Whisper model can be loaded"""
    print("\nChecking Whisper model...")
    
    try:
        import whisper
        print("✅ Whisper package available")
        print("   Note: Model will auto-download on first run (~140MB for 'base' model)")
        return True
    except ImportError:
        print("❌ Whisper not installed")
        return False

def check_pygame():
    """Check if pygame can initialize"""
    print("\nChecking pygame audio...")
    
    try:
        import pygame
        pygame.mixer.init()
        print("✅ Pygame audio initialized successfully")
        pygame.mixer.quit()
        return True
    except Exception as e:
        print(f"⚠️  Pygame initialization issue: {e}")
        print("   Audio playback might not work properly")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("🔍 Voice Agent Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("API Key", check_api_key),
        ("Schemes File", check_schemes_file),
        ("Audio Devices", check_audio_devices),
        ("Whisper Model", check_whisper_model),
        ("Pygame Audio", check_pygame)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ {check_name} check failed: {e}")
            results.append((check_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10s} {check_name}")
    
    print("=" * 60)
    print(f"Result: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run main.py")
        print("\nRun: python main.py")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
        print("\nFor help, see README.md or check the documentation.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()