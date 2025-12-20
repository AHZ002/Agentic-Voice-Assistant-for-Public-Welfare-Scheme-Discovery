"""
Speech-to-Text (STT) Module

This module captures audio from the system microphone and transcribes
speech using OpenAI Whisper or equivalent models. Supports Indian languages.

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
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("Warning: sounddevice not installed. Audio capture will not work.")

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: whisper not installed. Transcription will not work.")

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
    Speech-to-Text transcription system using Whisper
    """
    
    # Supported Indian languages (Whisper language codes)
    SUPPORTED_LANGUAGES = {
        "marathi": "mr",
        "hindi": "hi",
        "telugu": "te",
        "tamil": "ta",
        "bengali": "bn",
        "gujarati": "gu",
        "kannada": "kn",
        "malayalam": "ml",
        "odia": "or",
        "punjabi": "pa",
        "urdu": "ur"
    }
    
    # Confidence thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.85
    MEDIUM_CONFIDENCE_THRESHOLD = 0.65
    LOW_CONFIDENCE_THRESHOLD = 0.40
    
    def __init__(
        self,
        model_size: str = "base",
        language: str = "marathi",
        sample_rate: int = 48000,
        channels: int = 1,
        confidence_threshold: float = 0.40,
        device: Optional[str] = None
    ):
        """
        Initialize Speech-to-Text system
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            language: Target language for transcription
            sample_rate: Audio sample rate in Hz
            channels: Number of audio channels (1 for mono)
            confidence_threshold: Minimum confidence to accept transcription
            device: Device for Whisper ('cpu' or 'cuda')
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("whisper package not installed. Install with: pip install openai-whisper")
        
        if not SOUNDDEVICE_AVAILABLE:
            raise ImportError("sounddevice package not installed. Install with: pip install sounddevice")
        
        self.model_size = model_size
        self.language = language.lower()
        self.sample_rate = sample_rate
        self.channels = channels
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Validate language
        if self.language not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' not supported. "
                f"Supported: {', '.join(self.SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Load Whisper model
        self.model = self._load_model()
        
        # Recording state
        self.is_recording = False
        self.recorded_audio = None
    
    def _load_model(self):
        """Load Whisper model"""
        try:
            model = whisper.load_model(self.model_size, device=self.device)
            return model
        except Exception as e:
            raise TranscriptionFailedError(f"Failed to load Whisper model: {e}")
    
    def capture_audio(
        self,
        duration: Optional[float] = None,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0
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
        min_listen_samples = int(2.5 * self.sample_rate)  # force 2.5 sec minimum listen
        chunk_size = int(0.1 * self.sample_rate)  # 100ms chunks
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32'
            ) as stream:
                
                while total_samples < max_samples:
                    chunk, overflowed = stream.read(chunk_size)
                    
                    if overflowed:
                        print("Warning: Audio buffer overflow")
                    
                    audio_chunks.append(chunk.copy())
                    total_samples += len(chunk)
                
                    # Check for silence
                    amplitude = np.abs(chunk).mean()
                    
                    # if amplitude < silence_threshold:
                    #     consecutive_silent_samples += len(chunk)
                    #     if consecutive_silent_samples >= silence_samples:
                    #         # Stop recording after silence
                    #         break
                    # else:
                    #     consecutive_silent_samples = 0

                    if amplitude < silence_threshold:
                        consecutive_silent_samples += len(chunk)
                        # Allow silence-based stopping ONLY after minimum listen time
                        if (
                            total_samples >= min_listen_samples
                            and consecutive_silent_samples >= silence_samples
                        ):
                            break
                    else:
                        consecutive_silent_samples = 0
                
                    
        
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
        Transcribe audio using Whisper
        
        Args:
            audio: Audio samples as NumPy array
            task: 'transcribe' or 'translate'
            
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
            
            # Transcribe with Whisper
            result = self.model.transcribe(
                audio,
                language=language_code,
                task=task,
                fp16=False,
                verbose=False
            )
            
            # Extract results
            text = result["text"].strip()
            detected_language = result.get("language", language_code)
            segments = result.get("segments", [])
            
            # Calculate confidence
            confidence_score = self._calculate_confidence(segments)
            confidence_level = self._get_confidence_level(confidence_score)
            is_reliable = confidence_score >= self.confidence_threshold
            
            # Create result
            transcription_result = TranscriptionResult(
                transcribed_text=text,
                language=detected_language,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                is_reliable=is_reliable,
                audio_duration=duration,
                model_used=f"whisper-{self.model_size}",
                segments=[
                    {
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"],
                        "confidence": seg.get("confidence", seg.get("no_speech_prob", 0.0))
                    }
                    for seg in segments
                ]
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
        
        except Exception as e:
            raise TranscriptionFailedError(f"Transcription failed: {e}")
    
    def _calculate_confidence(self, segments: list) -> float:
        """
        Calculate overall confidence from segments
        
        Args:
            segments: List of transcription segments
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not segments:
            return 0.0
        
        # Whisper doesn't always provide confidence scores
        # Use inverse of no_speech_prob as proxy
        confidences = []
        for seg in segments:
            if "confidence" in seg:
                confidences.append(seg["confidence"])
            elif "no_speech_prob" in seg:
                # Higher no_speech_prob means lower confidence
                confidences.append(1.0 - seg["no_speech_prob"])
            else:
                # Default medium confidence if no score available
                confidences.append(0.7)
        
        return np.mean(confidences) if confidences else 0.0
    
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
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0
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
            # Load audio using Whisper's built-in loader
            audio = whisper.load_audio(file_path)
            
            # Pad/trim to 30 seconds as Whisper expects
            audio = whisper.pad_or_trim(audio)
            
            # Transcribe
            result = self.transcribe_audio(audio)
            
            return result
        
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
    language: str = "marathi",
    model_size: str = "base",
    duration: Optional[float] = None,
    confidence_threshold: float = 0.40
) -> Dict[str, Any]:
    """
    Convenience function for one-shot transcription
    
    Args:
        language: Target language
        model_size: Whisper model size
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
        model_size=model_size,
        language=language,
        confidence_threshold=confidence_threshold
    )
    
    result = stt.capture_and_transcribe(duration=duration)
    
    return result.to_dict()