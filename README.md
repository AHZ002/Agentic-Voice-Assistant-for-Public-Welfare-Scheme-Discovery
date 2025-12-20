# Voice-First Agentic AI System for Government Schemes

A production-ready, voice-first agentic AI system that helps users discover and apply for government welfare schemes in Marathi (or other Indian languages).

## 🎯 Key Features

- ✅ **Voice-Only Interaction** - No text input, completely voice-driven
- ✅ **Native Language Support** - Operates entirely in Marathi
- ✅ **True Agentic Workflow** - Planner → Executor → Evaluator loop
- ✅ **Multi-Tool Integration** - Eligibility engine, scheme retriever, mock government API
- ✅ **Conversation Memory** - Tracks user profile across turns
- ✅ **Contradiction Handling** - Detects and resolves conflicting information
- ✅ **Failure Recovery** - Handles speech recognition errors, missing data, and tool failures

## 🏗️ Architecture

```
Voice Input (Microphone)
    ↓
Speech-to-Text (Whisper)
    ↓
LLM Intent Parser (Claude)
    ↓
State Machine → Planner → Executor → Evaluator
    ↓           ↓          ↓          ↓
    Memory ←────┴──────────┴──────────┘
    ↓
LLM Response Generator (Claude)
    ↓
Text-to-Speech (gTTS)
    ↓
Voice Output (Speakers)
```

### Core Components

**State Machine** (`agent/state_machine.py`)
- Manages agent states (IDLE, LISTENING, PLANNING, EXECUTING, EVALUATING, etc.)
- Enforces valid state transitions
- Tracks retry counts and error history

**Planner** (`agent/planner.py`)
- Decides next action based on user intent and context
- Handles missing information detection
- Prioritizes contradictions and critical tasks

**Executor** (`agent/executor.py`)
- Executes planned actions by calling appropriate tools
- Returns structured execution results
- No decision-making logic (pure execution)

**Evaluator** (`agent/evaluator.py`)
- Assesses execution results
- Determines success, failure, or need for clarification
- Provides recommendations for next steps

**Session Memory** (`memory/session_memory.py`)
- Stores user profile and conversation history
- Tracks explored and eligible schemes
- Records contradictions

**Contradiction Handler** (`memory/contradiction_handler.py`)
- Detects contradictions in user input
- Categorizes severity (critical, moderate, minor)
- Flags inconsistencies in user profile

**Speech-to-Text** (`speech/stt.py`)
- Captures audio from microphone
- Transcribes using OpenAI Whisper
- Voice activity detection (VAD)

**Text-to-Speech** (`speech/tts.py`)
- Converts Marathi text to speech using gTTS
- Plays audio through system speakers

**Tools**
- `tools/eligibility.py` - Evaluates user eligibility for schemes
- `tools/scheme_retriever.py` - Retrieves relevant schemes
- `tools/mock_gov_api.py` - Mock government application API

## 📦 Installation

### Prerequisites

- Python 3.8+
- Microphone (for voice input)
- Speakers/Headphones (for voice output)
- Anthropic API Key (free tier available)

### Step 1: Clone Repository

```bash
git clone https://github.com/AHZ002/Agentic-Voice-Assistant-for-Public-Welfare-Scheme-Discovery.git
cd voice_schemes_agent
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set API Key

**Windows:**
```cmd
set ANTHROPIC_API_KEY=your_key_here
```

**Linux/Mac:**
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Get your free API key at: https://console.anthropic.com/

### Step 4: Verify Installation

```bash
python -c "import whisper; import anthropic; import sounddevice; print('✅ All dependencies installed')"
```

## 🚀 Usage

### Run the Agent

```bash
python main.py
```

### Expected Flow

1. Agent greets in Marathi: "नमस्कार! मी सरकारी योजना सहाय्यक आहे..."
2. User speaks: Describe what you need (e.g., "मला शेतकऱ्यांसाठी योजना हवी आहे")
3. Agent processes: Transcribes → Parses intent → Plans → Executes → Evaluates
4. Agent responds in Marathi: Asks for missing info or presents results
5. Loop continues until task complete or user says "बंद करा"

### Example Interaction

```
🔊 Agent: नमस्कार! मी सरकारी योजना सहाय्यक आहे. तुम्हाला काय हवे आहे?

🎤 User: मला शेतकऱ्यांसाठी योजना हवी आहे

🔊 Agent: ठीक आहे. तुमचे वय काय आहे?

🎤 User: माझे वय ४५ वर्षे आहे

🔊 Agent: तुमचे वार्षिक उत्पन्न किती आहे?

🎤 User: दोन लाख रुपये

🔊 Agent: तुम्ही कोणत्या राज्यात राहता?

🎤 User: महाराष्ट्र

🔊 Agent: तुम्ही ३ योजनांसाठी पात्र आहात...
```

### Exit Commands

Say any of these to exit:
- "बंद करा" (stop)
- "थांबा" (pause)
- "बंद" (close)
- Or press `Ctrl+C`

## ⚙️ Configuration

Edit `AgentConfig` class in `main.py`:

```python
class AgentConfig:
    # Paths
    SCHEMES_FILE = r"C:\path\to\schemes.json"
    
    # Language
    LANGUAGE = "marathi"  # Change to: hindi, tamil, telugu, etc.
    
    # Speech settings
    STT_MODEL_SIZE = "base"  # Options: tiny, base, small, medium, large
    STT_CONFIDENCE_THRESHOLD = 0.40
    
    # Agent limits
    MAX_CONVERSATION_TURNS = 20
    MAX_RETRIES_PER_STATE = 3
```

## 📊 Session Logging

After each session, a JSON file is saved to `sessions/` directory:

```json
{
  "session_id": "session_abc123",
  "total_turns": 8,
  "user_profile": {
    "age": 45,
    "annual_income": 200000,
    "state": "maharashtra",
    "occupation": "farmer"
  },
  "eligible_schemes": ["PM-KISAN", "Krishi Sinchan Yojana"],
  "conversation_history": []
}
```

## 🧪 Testing

**Test Speech Recognition:**
```bash
python -c "from speech.stt import transcribe_speech; print(transcribe_speech(language='marathi', duration=5))"
```

**Test Speech Synthesis:**
```bash
python -c "from speech.tts import speak_text; speak_text('नमस्कार', language='marathi')"
```

**Test Tools:**
```bash
python -c "from tools.scheme_retriever import retrieve_schemes; print(retrieve_schemes(keywords=['farmer']))"
```

## 🐛 Troubleshooting

### "ANTHROPIC_API_KEY not found"
- Set the environment variable before running
- Verify with: `echo %ANTHROPIC_API_KEY%` (Windows) or `echo $ANTHROPIC_API_KEY` (Linux/Mac)

### "sounddevice not working"
Install PortAudio:
- **Windows:** `pip install sounddevice` should work
- **Linux:** `sudo apt-get install portaudio19-dev`
- **Mac:** `brew install portaudio`

### "Whisper model download failed"
- Models auto-download on first run
- Ensure internet connection
- Models saved to: `~/.cache/whisper/`

### "Low confidence errors"
- Speak clearly and closer to microphone
- Reduce background noise
- Lower `STT_CONFIDENCE_THRESHOLD` in config (not recommended below 0.3)

### "No audio playback"
- Check speaker/headphone connection
- Verify pygame installation: `python -c "import pygame; pygame.mixer.init()"`
- Try alternative TTS engine: Set `TTS_ENGINE = "playsound"` in config

## 📁 Project Structure

```
voice_schemes_agent/
├── main.py                    # Entry point
├── requirements.txt           # Dependencies
├── README.md                  # This file
│
├── agent/
│   ├── state_machine.py      # Finite state machine
│   ├── planner.py            # Action planner
│   ├── executor.py           # Action executor
│   └── evaluator.py          # Result evaluator
│
├── speech/
│   ├── stt.py                # Speech-to-Text (Whisper)
│   └── tts.py                # Text-to-Speech (gTTS)
│
├── memory/
│   ├── session_memory.py     # Session storage
│   └── contradiction_handler.py  # Contradiction detection
│
├── tools/
│   ├── eligibility.py        # Eligibility engine
│   ├── scheme_retriever.py   # Scheme search
│   └── mock_gov_api.py       # Mock application API
│
├── data/
│   └── schemes.json          # Scheme database
│
└── sessions/                 # Saved session logs
```

## 🔒 Privacy & Security

- All processing happens locally except LLM calls (Anthropic API)
- No user data is stored permanently unless you enable session saving
- Session logs contain conversation history - handle with care
- Microphone access required only during active listening

## 📝 License

[Add your license here]

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

**Built with ❤️ for accessible government services**
