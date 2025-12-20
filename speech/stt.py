"""
Speech-to-Text (STT) Module - Google Speech Recognition

This module captures audio from the system microphone and transcribes
speech using Google's Speech Recognition API (FREE, no API key needed).
Supports Indian languages including Hindi, Marathi, Telugu, Tamil, etc.

MUCH MORE RELIABLE than Whisper for live audio capture!

NO reasoning logic
NO user prompts
NO conversational text
Pure transcription only
"""
import os
import io
import wave
import tempfile
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not installed. Audio capture will not work.")

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False
    print("Warning: SpeechRecognition not installed. Install with: pip install SpeechRecognition")

try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class TranscriptionError(Exception):
    """Base exception for transcription errors"""
    pass


class AudioCaptureError(TranscriptionError):
    """Error during audio capture"""
    pass


class TranscriptionFailedError(TranscriptionError):
    """Error during transcription"""
    pass


class LowConfidenceError(TranscriptionError):
    """Transcription confidence too low"""
    pass


class ConfidenceLevel(Enum):
    """Confidence level enumeration"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


@dataclass
class TranscriptionResult:
    """Structured transcription result"""
    transcribed_text: str
    language: str
    confidence_score: float  # 0.0 to 1.0
    confidence_level: ConfidenceLevel
    is_reliable: bool
    audio_duration: float  # seconds
    model_used: str
    segments: Optional[list] = None  # Detailed segment information
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "transcribed_text": self.transcribed_text,
            "language": self.language,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "is_reliable": self.is_reliable,
            "audio_duration": self.audio_duration,
            "model_used": self.model_used,
            "segments": self.segments
        }


class SpeechToText:
    """
    Speech-to-Text transcription system using Google Speech Recognition
    FREE and RELIABLE for live audio!
    """
    
    # Supported Indian languages (Google Speech Recognition language codes)
    SUPPORTED_LANGUAGES = {
        "hindi": "hi-IN",
        "marathi": "mr-IN",
        "telugu": "te-IN",
        "tamil": "ta-IN",
        "bengali": "bn-IN",
        "gujarati": "gu-IN",
        "kannada": "kn-IN",
        "malayalam": "ml-IN",
        "punjabi": "pa-IN",
        "urdu": "ur-IN",
        "english": "en-IN"
    }
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.65
    LOW_CONFIDENCE_THRESHOLD = 0.40
    
    def __init__(
        self,
        model_size: str = "base",  # Ignored, for compatibility
        language: str = "hindi",
        sample_rate: int = 16000,
        channels: int = 1,
        confidence_threshold: float = 0.40,
        device: Optional[str] = None  # Ignored, for compatibility
    ):
        """
        Initialize Speech-to-Text system
        
        Args:
            model_size: Ignored (for Whisper compatibility)
            language: Target language for transcription
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            confidence_threshold: Minimum confidence to accept transcription
            device: Ignored (for Whisper compatibility)
        """
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("SpeechRecognition package not installed. Install with: pip install SpeechRecognition")
        
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice package not installed. Install with: pip install sounddevice")
        
        self.language = language.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.confidence_threshold = confidence_threshold
        
        # Validate language
        if self.language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported: {', '.join(self.SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Initialize Google Speech Recognizer
        self.recognizer = sr.Recognizer()
        
        # Adjust for ambient noise (makes it more sensitive)
        self.recognizer.energy_threshold = 300  # Lower = more sensitive
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8  # Seconds of silence to consider end
        
        # Recording state
        self.is_recording = False
        self.recorded_audio = None
    
    def capture_audio(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.01,
        silence_duration: float = 2.0,
        max_duration: float = 10.0
    ) -> np.ndarray:
        """
        Capture audio from microphone
        
        Args:
            duration: Fixed duration in seconds (if None, uses voice activity detection)
            silence_threshold: Amplitude threshold for silence detection
            silence_duration: Seconds of silence before stopping
            max_duration: Maximum recording duration
            
        Returns:
            NumPy array of audio samples
            
        Raises:
            AudioCaptureError: If audio capture fails
        """
        try:
            if duration is not None:
                # Fixed duration recording
                print(f"   Recording for {duration} seconds...")
                audio = sd.rec(
                    int(duration * self.sample_rate),
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='float32'
                )
                sd.wait()
                return audio.flatten()
            
            else:
                # Voice activity detection
                return self._capture_with_vad(
                    silence_threshold=silence_threshold,
                    silence_duration=silence_duration,
                    max_duration=max_duration
                )
        
        except Exception as e:
            raise AudioCaptureError(f"Failed to capture audio: {e}")
    
    def _capture_with_vad(
        self,
        silence_threshold: float,
        silence_duration: float,
        max_duration: float
    ) -> np.ndarray:
        """
        Capture audio with voice activity detection
        
        Args:
            silence_threshold: Threshold for silence
            silence_duration: Silence duration before stopping
            max_duration: Maximum duration
            
        Returns:
            Audio array
        """
        audio_chunks = []
        silence_samples = int(silence_duration * self.sample_rate)
        max_samples = int(max_duration * self.sample_rate)
        consecutive_silent_samples = 0
        total_samples = 0
        min_listen_samples = int(1.0 * self.sample_rate)  # Minimum 1 second
        chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
        
        print("   🎤 Listening... (speak clearly)")
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            ) as stream:
                
                while total_samples < max_samples:
                    chunk, overflowed = stream.read(chunk_size)
                    
                    if overflowed:
                        print("   ⚠️ Audio buffer overflow")
                    
                    audio_chunks.append(chunk.copy())
                    total_samples += len(chunk)
                    
                    # Check for silence
                    amplitude = np.abs(chunk).mean()
                    
                    if amplitude < silence_threshold:
                        consecutive_silent_samples += len(chunk)
                        # Allow silence-based stopping ONLY after minimum listen time
                        if (
                            total_samples >= min_listen_samples
                            and consecutive_silent_samples >= silence_samples
                        ):
                            print("   ✋ Silence detected, stopping...")
                            break
                    else:
                        consecutive_silent_samples = 0
                        # Show activity indicator
                        if total_samples % (self.sample_rate // 2) == 0:
                            print("   🔴 Recording...")
        
        except Exception as e:
            raise AudioCaptureError(f"Error during voice activity detection: {e}")
        
        # Concatenate chunks
        if not audio_chunks:
            raise AudioCaptureError("No audio captured")
        
        audio = np.concatenate(audio_chunks, axis=0).flatten()
        return audio
    
    def transcribe_audio(
        self,
        audio: np.ndarray,
        task: str = "transcribe"
    ) -> TranscriptionResult:
        """
        Transcribe audio using Google Speech Recognition
        
        Args:
            audio: Audio samples as NumPy array
            task: 'transcribe' or 'translate' (ignored, for compatibility)
            
        Returns:
            TranscriptionResult object
            
        Raises:
            TranscriptionFailedError: If transcription fails
            LowConfidenceError: If confidence too low
        """
        try:
            # Get language code
            language_code = self.SUPPORTED_LANGUAGES[self.language]
            
            # Calculate duration
            duration = len(audio) / self.sample_rate
            
            # Convert numpy array to AudioData
            # Google SR expects 16-bit PCM
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create AudioData object
            audio_data = sr.AudioData(
                audio_int16.tobytes(),
                sample_rate=self.sample_rate,
                sample_width=2  # 16-bit = 2 bytes
            )
            
            # Transcribe with Google
            print(f"   🔄 Transcribing in {self.language}...")
            try:
                text = self.recognizer.recognize_google(
                    audio_data,
                    language=language_code,
                    show_all=False  # Get best result only
                )
            except sr.UnknownValueError:
                # Google couldn't understand audio
                raise TranscriptionFailedError("Could not understand audio - please speak clearly")
            except sr.RequestError as e:
                # API unavailable
                raise TranscriptionFailedError(f"Google API error: {e}")
            
            # Google doesn't provide confidence scores in free tier
            # Estimate based on text length and quality
            confidence_score = self._estimate_confidence(text, duration)
            confidence_level = self._get_confidence_level(confidence_score)
            is_reliable = confidence_score >= self.confidence_threshold
            
            # Create result
            transcription_result = TranscriptionResult(
                transcribed_text=text.strip(),
                language=language_code,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                is_reliable=is_reliable,
                audio_duration=duration,
                model_used="google-speech-recognition",
                segments=None
            )
            
            # Flag low confidence
            if not is_reliable:
                raise LowConfidenceError(
                    f"Transcription confidence too low: {confidence_score:.2f} "
                    f"(threshold: {self.confidence_threshold:.2f}). "
                    f"Text: '{text}'"
                )
            
            return transcription_result
        
        except LowConfidenceError:
            raise
        
        except TranscriptionFailedError:
            raise
        
        except Exception as e:
            raise TranscriptionFailedError(f"Transcription failed: {e}")
    
    def _estimate_confidence(self, text: str, duration: float) -> float:
        """
        Estimate confidence score based on text and duration
        
        Args:
            text: Transcribed text
            duration: Audio duration
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Base confidence
        confidence = 0.7
        
        # Longer text = higher confidence
        word_count = len(text.split())
        if word_count >= 3:
            confidence += 0.15
        elif word_count >= 5:
            confidence += 0.20
        
        # Reasonable speaking rate
        if duration > 0:
            words_per_second = word_count / duration
            if 1.0 <= words_per_second <= 4.0:  # Normal speaking rate
                confidence += 0.10
        
        return min(confidence, 1.0)
    
    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """
        Get confidence level from score
        
        Args:
            score: Confidence score
            
        Returns:
            ConfidenceLevel enum
        """
        if score >= self.HIGH_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.HIGH
        elif score >= self.MEDIUM_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.MEDIUM
        elif score >= self.LOW_CONFIDENCE_THRESHOLD:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def capture_and_transcribe(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.005,
        silence_duration: float = 2.0,
        max_duration: float = 10.0
    ) -> TranscriptionResult:
        """
        Capture audio and transcribe in one step
        
        Args:
            duration: Recording duration (None for VAD)
            silence_threshold: Silence detection threshold
            silence_duration: Silence duration before stopping
            max_duration: Maximum recording duration
            
        Returns:
            TranscriptionResult
            
        Raises:
            AudioCaptureError: Audio capture failed
            TranscriptionFailedError: Transcription failed
            LowConfidenceError: Confidence too low
        """
        # Capture audio
        audio = self.capture_audio(
            duration=duration,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration
        )
        
        # Transcribe
        result = self.transcribe_audio(audio)
        
        return result
    
    def transcribe_audio_file(self, file_path: str) -> TranscriptionResult:
        """
        Transcribe audio from file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            TranscriptionResult
            
        Raises:
            TranscriptionFailedError: If transcription fails
        """
        try:
            with sr.AudioFile(file_path) as source:
                audio_data = self.recognizer.record(source)
            
            language_code = self.SUPPORTED_LANGUAGES[self.language]
            
            text = self.recognizer.recognize_google(
                audio_data,
                language=language_code
            )
            
            # Estimate duration from file
            duration = len(audio_data.frame_data) / audio_data.sample_rate / audio_data.sample_width
            
            confidence_score = self._estimate_confidence(text, duration)
            
            return TranscriptionResult(
                transcribed_text=text.strip(),
                language=language_code,
                confidence_score=confidence_score,
                confidence_level=self._get_confidence_level(confidence_score),
                is_reliable=confidence_score >= self.confidence_threshold,
                audio_duration=duration,
                model_used="google-speech-recognition"
            )
        
        except Exception as e:
            raise TranscriptionFailedError(f"Failed to transcribe file: {e}")
    
    def save_audio(self, audio: np.ndarray, file_path: str):
        """
        Save audio to file
        
        Args:
            audio: Audio array
            file_path: Output file path
        """
        try:
            # Ensure audio is in correct format
            audio_int16 = (audio * 32767).astype(np.int16)
            
            with wave.open(file_path, 'w') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_int16.tobytes())
        
        except Exception as e:
            raise AudioCaptureError(f"Failed to save audio: {e}")
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get audio device information"""
        try:
            devices = sd.query_devices()
            default_input = sd.query_devices(kind='input')
            
            return {
                "default_input_device": default_input,
                "all_devices": devices,
                "sample_rate": self.sample_rate,
                "channels": self.channels
            }
        except Exception as e:
            return {"error": str(e)}


# Convenience function
def transcribe_speech(
    language: str = "hindi",
    model_size: str = "base",  # Ignored
    duration: Optional[float] = None,
    confidence_threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Convenience function for one-shot transcription
    
    Args:
        language: Target language
        model_size: Ignored (for Whisper compatibility)
        duration: Recording duration (None for VAD)
        confidence_threshold: Minimum confidence threshold
        
    Returns:
        Transcription result dictionary
        
    Raises:
        AudioCaptureError: Audio capture failed
        TranscriptionFailedError: Transcription failed
        LowConfidenceError: Confidence too low
    """
    stt = SpeechToText(
        language=language,
        confidence_threshold=confidence_threshold
    )
    
    result = stt.capture_and_transcribe(duration=duration)
    
    return result.to_dict()