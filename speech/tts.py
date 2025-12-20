"""
Text-to-Speech (TTS) Module

This module converts text to speech in Indian languages using gTTS (Google TTS)
as the primary engine with fallback options. Plays audio through system speakers.

NO text generation
NO reasoning logic
NO user interaction
Pure text-to-speech only
"""

import os
import io
import tempfile
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import pygame
pygame.mixer.init()


try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not installed. Install with: pip install gTTS")

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Install with: pip install pygame")

try:
    from playsound import playsound
    PLAYSOUND_AVAILABLE = True
except ImportError:
    PLAYSOUND_AVAILABLE = False


class TTSError(Exception):
    """Base exception for TTS errors"""
    pass


class TTSEngineError(TTSError):
    """TTS engine failure"""
    pass


class AudioPlaybackError(TTSError):
    """Audio playback failure"""
    pass


class UnsupportedLanguageError(TTSError):
    """Language not supported"""
    pass


class TTSEngine(Enum):
    """TTS engine types"""
    GTTS = "gtts"
    PYGAME = "pygame"
    PLAYSOUND = "playsound"


@dataclass
class TTSResult:
    """TTS operation result"""
    success: bool
    text: str
    language: str
    engine_used: str
    audio_file: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "text": self.text,
            "language": self.language,
            "engine_used": self.engine_used,
            "audio_file": self.audio_file,
            "duration": self.duration,
            "error": self.error
        }


class TextToSpeech:
    """
    Text-to-Speech system for Indian languages
    """
    
    # Supported Indian languages (gTTS language codes)
    SUPPORTED_LANGUAGES = {
        "marathi": "mr",
        "hindi": "hi",
        "telugu": "te",
        "tamil": "ta",
        "bengali": "bn",
        "gujarati": "gu",
        "kannada": "kn",
        "malayalam": "ml",
        "urdu": "ur",
        "punjabi": "pa",
        "english": "en"
    }
    
    def __init__(
        self,
        language: str = "marathi",
        engine: str = "gtts",
        slow: bool = False,
        output_dir: Optional[str] = None
    ):
        """
        Initialize Text-to-Speech system
        
        Args:
            language: Target language for speech
            engine: TTS engine to use ('gtts', 'pygame', 'playsound')
            slow: Slow speech rate
            output_dir: Directory to save audio files (None for temp)
        """
        self.language = language.lower()
        self.engine = engine.lower()
        self.slow = slow
        self.output_dir = output_dir
        
        # Validate language
        if self.language not in self.SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError(
                f"Language '{language}' not supported. "
                f"Supported: {', '.join(self.SUPPORTED_LANGUAGES.keys())}"
            )
        
        # Check engine availability
        self._check_engine_availability()
        
        # Initialize pygame if needed
        if self.engine == "pygame" and PYGAME_AVAILABLE:
            pygame.mixer.init()
    
    def _check_engine_availability(self):
        """Check if required TTS engine is available"""
        if self.engine == "gtts" and not GTTS_AVAILABLE:
            raise TTSEngineError("gTTS not installed. Install with: pip install gTTS")
        
        if self.engine == "pygame" and not PYGAME_AVAILABLE:
            raise TTSEngineError("pygame not installed. Install with: pip install pygame")
        
        if self.engine == "playsound" and not PLAYSOUND_AVAILABLE:
            raise TTSEngineError("playsound not installed. Install with: pip install playsound")
    
    def synthesize_speech(
        self,
        text: str,
        save_file: bool = False,
        file_path: Optional[str] = None
    ) -> str:
        """
        Synthesize speech from text using gTTS
        
        Args:
            text: Text to convert to speech
            save_file: Whether to save to permanent file
            file_path: Custom file path (if save_file=True)
            
        Returns:
            Path to generated audio file
            
        Raises:
            TTSEngineError: If synthesis fails
        """
        if not text or not text.strip():
            raise TTSEngineError("Empty text provided")
        
        try:
            # Get language code
            lang_code = self.SUPPORTED_LANGUAGES[self.language]
            
            # Generate speech using gTTS
            tts = gTTS(text=text, lang=lang_code, slow=self.slow)
            
            # Determine output path
            if save_file and file_path:
                output_path = file_path
            elif save_file and self.output_dir:
                os.makedirs(self.output_dir, exist_ok=True)
                output_path = os.path.join(
                    self.output_dir,
                    f"tts_{hash(text) % 10**8}.mp3"
                )
            else:
                # Use temporary file
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3",
                    dir=self.output_dir
                )
                output_path = temp_file.name
                temp_file.close()
            
            # Save audio file
            tts.save(output_path)
            
            return output_path
        
        except Exception as e:
            raise TTSEngineError(f"Speech synthesis failed: {e}")
    
    def play_audio(self, audio_file: str):
        """
        Play audio file through system speakers
        
        Args:
            audio_file: Path to audio file
            
        Raises:
            AudioPlaybackError: If playback fails
        """
        if not os.path.exists(audio_file):
            raise AudioPlaybackError(f"Audio file not found: {audio_file}")
        
        try:
            if self.engine == "pygame" and PYGAME_AVAILABLE:
                self._play_with_pygame(audio_file)
            
            elif self.engine == "playsound" and PLAYSOUND_AVAILABLE:
                self._play_with_playsound(audio_file)
            
            else:
                # Default to pygame
                if PYGAME_AVAILABLE:
                    self._play_with_pygame(audio_file)
                elif PLAYSOUND_AVAILABLE:
                    self._play_with_playsound(audio_file)
                else:
                    raise AudioPlaybackError("No audio playback library available")
        
        except AudioPlaybackError:
            raise
        
        except Exception as e:
            raise AudioPlaybackError(f"Audio playback failed: {e}")
    
    def _play_with_pygame(self, audio_file: str):
        """Play audio using pygame"""
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to complete
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        
        except Exception as e:
            raise AudioPlaybackError(f"Pygame playback failed: {e}")
    
    def _play_with_playsound(self, audio_file: str):
        """Play audio using playsound"""
        try:
            playsound(audio_file)
        except Exception as e:
            raise AudioPlaybackError(f"Playsound playback failed: {e}")
    
    def speak(
        self,
        text: str,
        save_file: bool = False,
        file_path: Optional[str] = None,
        cleanup: bool = True
    ) -> TTSResult:
        """
        Convert text to speech and play it (synchronous)
        
        Args:
            text: Text to speak
            save_file: Whether to save audio file permanently
            file_path: Custom file path
            cleanup: Whether to delete temporary files after playback
            
        Returns:
            TTSResult object
        """
        audio_file = None
        
        try:
            # Synthesize speech
            audio_file = self.synthesize_speech(
                text=text,
                save_file=save_file,
                file_path=file_path
            )
            
            # Play audio
            self.play_audio(audio_file)
            
            # Get audio duration (approximate)
            duration = self._estimate_duration(text)
            
            result = TTSResult(
                success=True,
                text=text,
                language=self.language,
                engine_used=self.engine,
                audio_file=audio_file if save_file else None,
                duration=duration
            )
            
            # Cleanup temporary file
            if cleanup and not save_file and audio_file and os.path.exists(audio_file):
                try:
                    os.remove(audio_file)
                except:
                    pass
            
            return result
        
        except TTSEngineError as e:
            return TTSResult(
                success=False,
                text=text,
                language=self.language,
                engine_used=self.engine,
                error=f"TTS engine error: {str(e)}"
            )
        
        except AudioPlaybackError as e:
            return TTSResult(
                success=False,
                text=text,
                language=self.language,
                engine_used=self.engine,
                audio_file=audio_file,
                error=f"Audio playback error: {str(e)}"
            )
        
        except Exception as e:
            return TTSResult(
                success=False,
                text=text,
                language=self.language,
                engine_used=self.engine,
                error=f"Unexpected error: {str(e)}"
            )
    
    def speak_multiple(
        self,
        texts: list,
        pause_between: float = 0.5
    ) -> list:
        """
        Speak multiple texts in sequence
        
        Args:
            texts: List of text strings
            pause_between: Pause duration between texts (seconds)
            
        Returns:
            List of TTSResult objects
        """
        import time
        
        results = []
        for text in texts:
            result = self.speak(text)
            results.append(result)
            
            if pause_between > 0 and result.success:
                time.sleep(pause_between)
        
        return results
    
    def _estimate_duration(self, text: str) -> float:
        """
        Estimate audio duration based on text length
        
        Args:
            text: Input text
            
        Returns:
            Estimated duration in seconds
        """
        # Rough estimation: ~150 words per minute for normal speech
        # ~100 words per minute for slow speech
        words = len(text.split())
        wpm = 100 if self.slow else 150
        duration = (words / wpm) * 60
        return duration
    
    def save_speech(self, text: str, output_path: str) -> bool:
        """
        Generate speech and save to file without playing
        
        Args:
            text: Text to convert
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.synthesize_speech(text, save_file=True, file_path=output_path)
            return True
        except Exception as e:
            print(f"Error saving speech: {e}")
            return False
    
    def get_supported_languages(self) -> list:
        """Get list of supported languages"""
        return list(self.SUPPORTED_LANGUAGES.keys())
    
    def cleanup(self):
        """Cleanup resources"""
        if PYGAME_AVAILABLE and self.engine == "pygame":
            try:
                pygame.mixer.quit()
            except:
                pass


# Convenience functions
def speak_text(
    text: str,
    language: str = "marathi",
    slow: bool = False,
    engine: str = "gtts"
) -> Dict[str, Any]:
    """
    Convenience function for one-shot text-to-speech
    
    Args:
        text: Text to speak
        language: Target language
        slow: Slow speech rate
        engine: TTS engine
        
    Returns:
        Result dictionary
    """
    tts = TextToSpeech(language=language, engine=engine, slow=slow)
    result = tts.speak(text)
    return result.to_dict()


def save_speech_file(
    text: str,
    output_path: str,
    language: str = "marathi",
    slow: bool = False
) -> bool:
    """
    Save speech to file without playing
    
    Args:
        text: Text to convert
        output_path: Output file path
        language: Target language
        slow: Slow speech rate
        
    Returns:
        True if successful
    """
    tts = TextToSpeech(language=language, slow=slow)
    return tts.save_speech(text, output_path)


class TTSWithFallback:
    """
    TTS system with automatic fallback to alternative engines
    """
    
    def __init__(self, language: str = "marathi", slow: bool = False):
        """
        Initialize TTS with fallback support
        
        Args:
            language: Target language
            slow: Slow speech rate
        """
        self.language = language
        self.slow = slow
        self.engines = ["gtts", "pygame", "playsound"]
    
    def speak(self, text: str) -> TTSResult:
        """
        Speak text with automatic engine fallback
        
        Args:
            text: Text to speak
            
        Returns:
            TTSResult
        """
        last_error = None
        
        for engine in self.engines:
            try:
                tts = TextToSpeech(
                    language=self.language,
                    engine=engine,
                    slow=self.slow
                )
                result = tts.speak(text)
                
                if result.success:
                    return result
                
                last_error = result.error
            
            except Exception as e:
                last_error = str(e)
                continue
        
        # All engines failed
        return TTSResult(
            success=False,
            text=text,
            language=self.language,
            engine_used="none",
            error=f"All TTS engines failed. Last error: {last_error}"
        )


# Alias for backward compatibility
def text_to_speech(
    text: str,
    language: str = "marathi",
    play: bool = True,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Legacy function for text-to-speech
    
    Args:
        text: Text to convert
        language: Target language
        play: Whether to play audio
        save_path: Path to save audio file
        
    Returns:
        Result dictionary
    """
    tts = TextToSpeech(language=language)
    
    if play:
        result = tts.speak(text, save_file=(save_path is not None), file_path=save_path)
    else:
        if save_path:
            success = tts.save_speech(text, save_path)
            result = TTSResult(
                success=success,
                text=text,
                language=language,
                engine_used="gtts",
                audio_file=save_path if success else None
            )
        else:
            raise ValueError("Must either play audio or provide save_path")
    
    return result.to_dict()
