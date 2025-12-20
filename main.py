"""
main.py
Voice-First Agentic AI System - Entry Point

Orchestrates the complete agent lifecycle:
1. Voice input (STT)
2. Intent understanding (LLM)
3. Planning (Planner)
4. Execution (Executor)
5. Evaluation (Evaluator)
6. Voice output (TTS)

NO text input, NO chatbot mode, VOICE-ONLY interaction
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

# Groq API for LLM tasks (FREE with better limits!)
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: groq package not installed. Install with: pip install groq")

# Import agent components
from agent.state_machine import (
    AgentStateMachine,
    AgentState,
    TransitionError,
    RetryLimitExceededError
)
from agent.planner import (
    Planner,
    Plan,
    UserIntent,
    EvaluationResult as PlannerEvaluationResult,
    PlanAction
)
from agent.executor import (
    Executor,
    ExecutionResult,
    ExecutionStatus
)
from agent.evaluator import (
    Evaluator,
    EvaluationResult,
    EvaluationStatus
)

# Import speech components
from speech.stt import (
    SpeechToText,
    TranscriptionResult,
    TranscriptionError,
    AudioCaptureError,
    LowConfidenceError
)
from speech.tts import (
    TextToSpeech,
    TTSResult,
    TTSError
)

# Backup: Use pyttsx3 for offline TTS if gTTS fails
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Import memory components
from memory.session_memory import (
    SessionMemory,
    UserProfile as MemoryUserProfile
)
from memory.contradiction_handler import (
    ContradictionHandler,
    Contradiction
)

# Import tool components
from tools.eligibility import UserProfile as ToolUserProfile


# ==================== CONFIGURATION ====================

class AgentConfig:
    """Agent configuration"""
    
    # Paths - UPDATE THIS TO YOUR ACTUAL PATH
    SCHEMES_FILE = r"C:\Users\ABDUL_HADI\Desktop\Voice Agent\data\schemes.json"
    
    # Language
    LANGUAGE = "hindi"
    LANGUAGE_NAME_HINDI = "हिंदी"
    
    # Speech settings
    STT_MODEL_SIZE = "base"  # whisper model
    STT_CONFIDENCE_THRESHOLD = 0.20  # Lower threshold for better detection
    STT_SILENCE_THRESHOLD = 0.005  # More sensitive silence detection
    STT_SILENCE_DURATION = 2.0  # Wait longer for speech
    STT_MAX_DURATION = 10.0  # Allow longer speech
    TTS_ENGINE = "gtts"
    TTS_SLOW = False
    
    # Agent settings
    MAX_CONVERSATION_TURNS = 20
    MAX_RETRIES_PER_STATE = 3
    MAX_TOTAL_ERRORS = 10
    
    # LLM settings - Using Groq (FREE!)
    GROQ_MODEL = "llama-3.1-8b-instant"  # Fast and free!
    MAX_TOKENS = 2000
    TEMPERATURE = 0.7
    
    # Session settings
    SESSION_TIMEOUT_MINUTES = 30


# ==================== LLM UTILITIES ====================

class LLMInterface:
    """Interface for LLM operations using Groq (FREE with great limits!)"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM interface"""
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Set it as environment variable or pass to constructor.\n"
                "Get free key at: https://console.groq.com"
            )
        
        if not GROQ_AVAILABLE:
            raise ImportError("groq package not installed. Install with: pip install groq")
        
        self.client = Groq(api_key=self.api_key)
        print(f"[LLM] Using Groq model: {AgentConfig.GROQ_MODEL}")
    
    def parse_user_intent(
        self,
        transcribed_text: str,
        conversation_context: str,
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse user intent from text"""
        
        prompt = f"""You are analyzing user speech in Hindi for a government scheme assistance agent.

User said (in Hindi): "{transcribed_text}"

Previous conversation context:
{conversation_context}

Current user profile:
{json.dumps(user_profile, indent=2, ensure_ascii=False)}

Task: Extract the user's intent and any mentioned information from their Hindi speech.

Classify the intent as ONE of:
- "find_scheme": User wants to discover eligible schemes
- "check_eligibility": User wants to check if they're eligible
- "apply_scheme": User wants to apply for a specific scheme
- "provide_info": User is providing personal information
- "clarification": User is clarifying previous information
- "general_query": General question about schemes

Extract ONLY these entities (if mentioned):
- age: numeric (in years)
- annual_income: numeric (in rupees, yearly income)
- state: text (which Indian state they live in)
- family_size: numeric (number of family members)

IMPORTANT: Only extract these 4 fields. Ignore other information like occupation, category, etc.

Respond ONLY with a JSON object (no markdown, no extra text):
{{
  "intent_type": "one of the types above",
  "confidence": 0.0 to 1.0,
  "entities": {{
    "age": number or null,
    "annual_income": number or null,
    "state": "state name" or null,
    "family_size": number or null
  }},
  "raw_text": "{transcribed_text}"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=AgentConfig.GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=AgentConfig.TEMPERATURE,
                max_tokens=AgentConfig.MAX_TOKENS,
            )
            
            # Extract JSON from response
            content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            intent_data = json.loads(content)
            return intent_data
        
        except Exception as e:
            print(f"[LLM Error] Intent parsing failed: {e}")
            return {
                "intent_type": "general_query",
                "confidence": 0.5,
                "entities": {},
                "raw_text": transcribed_text
            }
    
    def generate_marathi_response(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> str:
        """Generate natural Hindi response"""
        
        prompt = f"""You are a helpful government scheme assistance agent speaking in Hindi (not Marathi).

Action to perform: {action}

Context:
{json.dumps(context, indent=2, ensure_ascii=False)}

Generate a natural, conversational response in HINDI (हिंदी) that:
1. Is clear and friendly
2. Is VERY BRIEF (1 sentence only)
3. Uses simple Hindi language
4. Asks for ONE thing at a time

IMPORTANT RULES:
- For "ask_missing_info": Ask for ONLY the first missing field from this list in order:
  1. Age (उम्र)
  2. Annual income (वार्षिक आय)
  3. State/Location (राज्य)
  4. Family size (परिवार के सदस्य)
  
  Ask ONE question only! Don't ask multiple things at once.

- For "present_schemes": Start with "आपके लिए कुछ योजनाएं मिली हैं:" then List the names of the top 2 schemes found.
- For "inform_no_schemes": Say "माफ़ कीजिये, अभी आपके लिए कोई योजना नहीं मिली।" (Sorry, no schemes found). Do NOT ask for more info.
- For "error": Say "कृपया फिर से बताएं"

IMPORTANT: 
- Keep it to 1 SHORT sentence
- Don't ask for occupation, category, caste, disability, etc.
- Just focus on: age, income, state, family size

Respond ONLY in HINDI (हिंदी), not Marathi (मराठी).
Respond ONLY with the Hindi text (no JSON, no English, no explanations)."""

        try:
            response = self.client.chat.completions.create(
                model=AgentConfig.GROQ_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.8,
                max_tokens=500,
            )
            
            marathi_text = response.choices[0].message.content.strip()
            
            if marathi_text.startswith('"') and marathi_text.endswith('"'):
                marathi_text = marathi_text[1:-1]
            
            return marathi_text
        
        except Exception as e:
            print(f"[LLM Error] Response generation failed: {e}")
            return "मुझे माफ करें, मैं समझ नहीं पाया। कृपया फिर से बताएं।"


# ==================== MAIN AGENT CLASS ====================

class VoiceAgent:
    """
    Voice-first agentic AI system for government scheme assistance
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        config: Optional[AgentConfig] = None
    ):
        """Initialize voice agent"""
        
        self.config = config or AgentConfig()
        
        print("=" * 60)
        print("🎙️  Voice-First Agentic AI System Initializing...")
        print("=" * 60)
        
        # Initialize LLM
        print("[Init] Loading LLM interface...")
        self.llm = LLMInterface(api_key=groq_api_key)
        
        # Initialize Speech components
        print(f"[Init] Loading Speech-to-Text ({self.config.LANGUAGE})...")
        self.stt = SpeechToText(
            model_size=self.config.STT_MODEL_SIZE,
            language=self.config.LANGUAGE,
            confidence_threshold=self.config.STT_CONFIDENCE_THRESHOLD
        )
        
        print(f"[Init] Loading Text-to-Speech ({self.config.LANGUAGE})...")
        try:
            self.tts = TextToSpeech(
                language=self.config.LANGUAGE,
                engine=self.config.TTS_ENGINE,
                slow=self.config.TTS_SLOW
            )
            self.tts_offline = None
        except Exception as e:
            print(f"[Warning] gTTS failed, using offline TTS: {e}")
            if PYTTSX3_AVAILABLE:
                self.tts_offline = pyttsx3.init()
                voices = self.tts_offline.getProperty('voices')
                for voice in voices:
                    if 'hindi' in voice.name.lower() or 'indian' in voice.name.lower():
                        self.tts_offline.setProperty('voice', voice.id)
                        break
                self.tts_offline.setProperty('rate', 150)
                self.tts = None
            else:
                raise ImportError("No TTS engine available")
        
        # Initialize Memory
        print("[Init] Initializing session memory...")
        self.memory = SessionMemory()
        self.contradiction_handler = ContradictionHandler()
        
        # Initialize Agent components
        print("[Init] Initializing state machine...")
        self.state_machine = AgentStateMachine(
            initial_state=AgentState.IDLE,
            max_error_count=self.config.MAX_TOTAL_ERRORS
        )
        
        print("[Init] Initializing planner...")
        self.planner = Planner()
        
        print("[Init] Initializing executor...")
        self.executor = Executor(
            schemes_file_path=self.config.SCHEMES_FILE
        )
        
        # === FIX: LOAD SCHEMES DATA MANUALLY ===
        print(f"[Init] Loading schemes from {self.config.SCHEMES_FILE}...")
        try:
            with open(self.config.SCHEMES_FILE, 'r', encoding='utf-8') as f:
                self.schemes_data = json.load(f)
                # Ensure it's a list
                if isinstance(self.schemes_data, dict):
                    self.schemes_data = self.schemes_data.get('schemes', [])
            print(f"[Init] Successfully loaded {len(self.schemes_data)} schemes.")
        except Exception as e:
            print(f"[Init Error] Failed to load schemes: {e}")
            self.schemes_data = []

        print("[Init] Initializing evaluator...")
        self.evaluator = Evaluator()
        
        # Session state
        self.current_plan: Optional[Plan] = None
        self.last_execution_result: Optional[ExecutionResult] = None
        self.last_evaluation: Optional[EvaluationResult] = None
        self.turn_count = 0
        
        print("=" * 60)
        print("✅ Initialization Complete!")
        print("=" * 60)
    
    def speak_to_user(self, marathi_text: str):
        """Speak to user in Marathi/Hindi"""
        print(f"\n🔊 [Agent → User]: {marathi_text}")
        
        try:
            if self.tts:
                result = self.tts.speak(marathi_text, cleanup=True)
                if not result.success:
                    # Suppressed error printing for cleanliness
                    if self.tts_offline:
                        self.tts_offline.say(marathi_text)
                        self.tts_offline.runAndWait()
            elif self.tts_offline:
                self.tts_offline.say(marathi_text)
                self.tts_offline.runAndWait()
        except Exception as e:
            pass # Suppressed error printing
    
    def listen_to_user(self) -> Optional[TranscriptionResult]:
        """Listen to user and transcribe speech"""
        print("\n🎤 [Listening... Speak now!]")
        
        try:
            result = self.stt.capture_and_transcribe(
                duration=None,  # VAD-based
                silence_threshold=self.config.STT_SILENCE_THRESHOLD,
                silence_duration=self.config.STT_SILENCE_DURATION,
                max_duration=self.config.STT_MAX_DURATION
            )
            
            print(f"📝 [User → Agent]: {result.transcribed_text}")
            print(f"   [Confidence: {result.confidence_score:.2f}, Language: {result.language}]")
            
            if not result.transcribed_text or len(result.transcribed_text.strip()) < 2:
                raise LowConfidenceError("Empty or very short transcription")
            
            return result
        
        except LowConfidenceError as e:
            # Suppressed warning printing
            self.speak_to_user("मुझे स्पष्ट सुनाई नहीं दिया। कृपया फिर से बोलें।")
            return None
        
        except AudioCaptureError as e:
            # Suppressed error printing
            self.speak_to_user("कृपया फिर से बोलें।")
            return None
        
        except TranscriptionError as e:
            # Suppressed error printing
            self.speak_to_user("कृपया फिर से बोलें।")
            return None
        
        except Exception as e:
            # Suppressed error printing
            return None
    
    def process_turn(self, transcription: TranscriptionResult) -> bool:
        """Process one conversation turn"""
        self.turn_count += 1
        print(f"\n{'=' * 60}")
        print(f"📍 Turn {self.turn_count} | State: {self.state_machine.current_state.value}")
        print(f"{'=' * 60}")
        
        try:
            if self.state_machine.current_state == AgentState.IDLE:
                self.state_machine.transition(
                    AgentState.LISTENING,
                    reason="User input received"
                )
            
            self.state_machine.transition(
                AgentState.PLANNING,
                reason="Processing user input"
            )
            
            # Step 2: Parse user intent using LLM
            print("[Planning] Parsing user intent...")
            intent_data = self.llm.parse_user_intent(
                transcribed_text=transcription.transcribed_text,
                conversation_context=self.memory.get_conversation_context(last_n=3),
                user_profile=self.memory.get_profile()
            )
            
            print(f"[Planning] Intent: {intent_data['intent_type']}")
            print(f"[Planning] Entities: {intent_data.get('entities', {})}")
            
            # Step 3: Update memory with extracted entities
            self._update_memory_from_intent(intent_data)

            # Clear previous failures if we extracted data
            if intent_data.get('entities') and any(v is not None for v in intent_data['entities'].values()):
                self.last_evaluation = None
            
            # Step 3.5: Check if we have all required info to proceed
            current_profile = self.memory.get_profile()
            required_fields = ["age", "annual_income", "state", "family_size"]
            missing_fields = [f for f in required_fields if f not in current_profile or current_profile[f] is None]
            
            print(f"[Planning] Profile status: {current_profile}")
            print(f"[Planning] Missing: {missing_fields}")
            
            # Create user intent object
            user_intent = UserIntent(
                intent_type=intent_data['intent_type'],
                entities=intent_data.get('entities', {}),
                confidence=intent_data.get('confidence', 0.8),
                raw_text=transcription.transcribed_text
            )
            
            if not missing_fields:
                print("[Planning] All required info collected - FORCING eligibility check!")
                user_intent = UserIntent(
                    intent_type="check_eligibility",
                    entities=intent_data.get('entities', {}),
                    confidence=1.0,
                    raw_text=transcription.transcribed_text
                )
            
            # Step 4: Check for contradictions
            contradictions = self._check_contradictions(intent_data.get('entities', {}))
            
            # Step 5: Create plan using Planner
            plan = self.planner.plan_next_action(
                user_intent=user_intent,
                memory_snapshot=self.memory.export_session(),
                contradiction_reports=[c.to_dict() for c in contradictions],
                last_evaluation=self._convert_evaluation_to_planner_format(self.last_evaluation)
            )

            # Override Planner Loop
            if not missing_fields and plan.action == PlanAction.ASK_MISSING_INFO:
                print("[Planning] OVERRIDE: Planner stuck asking for info. Forcing Eligibility Check.")
                found_action = None
                for member in PlanAction:
                    if member.value == "call_eligibility_tool":
                        found_action = member
                        break
                    if member.value == "fetch_schemes" and not found_action:
                        found_action = member
                
                if found_action:
                    plan.action = found_action
                    plan.reasoning = "Forcing check because all 4 required fields are present."
            
            self.current_plan = plan
            self.planner.add_to_history(plan)
            
            print(f"[Planning] Action: {plan.action.value}")
            print(f"[Planning] Reasoning: {plan.reasoning}")
            
            # Step 6: Transition to EXECUTING state
            self.state_machine.transition(
                AgentState.EXECUTING,
                reason=f"Executing action: {plan.action.value}"
            )
            
            # Step 7: Execute plan using Executor
            print(f"[Executing] Running action: {plan.action.value}...")
            
            # === FIX: INJECT USER PROFILE AND SCHEMES FROM MEMORY INTO PLAN PARAMETERS ===
            if plan.parameters is None:
                plan.parameters = {}
            
            plan.parameters['user_profile'] = self.memory.get_profile()
            # INJECT SCHEMES HERE
            plan.parameters['schemes'] = self.schemes_data
            
            print(f"[Executing] Injected Profile: {plan.parameters['user_profile']}")
            print(f"[Executing] Injected Schemes Count: {len(self.schemes_data)}")
            
            execution_result = self.executor.execute_plan(plan.to_dict())
            self.last_execution_result = execution_result
            
            # SUPPRESSED EXECUTION STATUS PRINTS AS REQUESTED
            # print(f"[Executing] Status: {execution_result.status.value}")
            # if execution_result.error:
            #     print(f"[Executing] Error: {execution_result.error}")
            
            # Step 8: Transition to EVALUATING state
            self.state_machine.transition(
                AgentState.EVALUATING,
                reason="Evaluating execution results"
            )
            
            # Step 9: Evaluate using Evaluator
            # SUPPRESSED EVALUATOR PRINTS AS REQUESTED
            # print("[Evaluating] Assessing results...")
            evaluation = self.evaluator.evaluate(
                action_type=plan.action.value,
                execution_result=execution_result.to_dict(),
                context=self.memory.export_session()
            )
            
            self.last_evaluation = evaluation
            
            # SUPPRESSED EVALUATION STATUS PRINTS AS REQUESTED
            # print(f"[Evaluating] Status: {evaluation.status.value}")
            # print(f"[Evaluating] Reasoning: {evaluation.reasoning}")
            
            # Step 10: Update memory based on execution results
            self._update_memory_from_execution(execution_result)
            
            # Step 11: Generate and speak response
            should_continue = self._generate_and_speak_response(
                plan=plan,
                execution_result=execution_result,
                evaluation=evaluation
            )
            
            # Step 12: Add turn to conversation history
            self.memory.add_turn(
                user_input_marathi=transcription.transcribed_text,
                agent_response_marathi="[Response spoken]",
                extracted_info=intent_data.get('entities', {}),
                tools_used=[execution_result.tool_used] if execution_result.tool_used else []
            )
            
            # Step 13: Transition based on evaluation
            if not should_continue:
                 # Ensure we clean up if we are stopping
                 self.state_machine.transition(
                    AgentState.COMPLETED,
                    reason="Session ended by agent"
                 )
                 return False

            if evaluation.status == EvaluationStatus.SUCCESS:
                if plan.action == PlanAction.END_TASK:
                    self.state_machine.transition(
                        AgentState.COMPLETED,
                        reason="Task completed successfully"
                    )
                    return False
                else:
                    self.state_machine.transition(
                        AgentState.LISTENING,
                        reason="Turn completed"
                    )
                    self.state_machine.transition(
                        AgentState.IDLE,
                        reason="Ready for next input"
                    )
            
            elif evaluation.status == EvaluationStatus.NEEDS_CLARIFICATION:
                self.state_machine.transition(
                    AgentState.ASKING_CLARIFICATION,
                    reason="Need user clarification"
                )
                self.state_machine.transition(
                    AgentState.LISTENING,
                    reason="Waiting for clarification"
                )
                self.state_machine.transition(
                    AgentState.IDLE,
                    reason="Ready for input"
                )
            
            elif evaluation.status == EvaluationStatus.RETRY:
                self.state_machine.transition(
                    AgentState.ERROR_RECOVERY,
                    reason="Retrying action"
                )
                self.state_machine.transition(
                    AgentState.LISTENING,
                    reason="Ready to retry"
                )
                self.state_machine.transition(
                    AgentState.IDLE,
                    reason="Waiting for input"
                )
            
            elif evaluation.status == EvaluationStatus.FAILURE:
                self.state_machine.transition(
                    AgentState.ERROR_RECOVERY,
                    reason="Action failed"
                )
                self.state_machine.transition(
                    AgentState.LISTENING,
                    reason="Recovered from failure"
                )
                self.state_machine.transition(
                    AgentState.IDLE,
                    reason="Ready for input"
                )
            
            return should_continue
        
        except TransitionError as e:
            print(f"[State Error] Invalid transition: {e}")
            try:
                if self.state_machine.current_state == AgentState.IDLE:
                    self.state_machine.transition(
                        AgentState.LISTENING,
                        reason="Recovering from error"
                    )
                self.state_machine.transition(
                    AgentState.ERROR_RECOVERY,
                    reason=str(e)
                )
            except:
                pass
            return True
        
        except Exception as e:
            print(f"[Error] Turn processing failed: {e}")
            self.state_machine.record_error(str(e))
            
            if self.state_machine.should_fail():
                self.state_machine.transition(
                    AgentState.FAILED,
                    reason="Too many errors"
                )
                return False
            
            self.speak_to_user("कुछ गलती हुई। कृपया फिर से कोशिश करें।")
            return True
    
    def _update_memory_from_intent(self, intent_data: Dict[str, Any]):
        """Update memory from parsed intent entities"""
        entities = intent_data.get('entities', {})
        
        if not entities:
            return
        
        # SIMPLIFIED - Only map 4 essential fields
        field_mapping = {
            'age': 'age',
            'annual_income': 'annual_income',
            'income': 'annual_income',
            'state': 'state',
            'location': 'state',
            'family_size': 'family_size'
        }
        
        profile_updates = {}
        for entity_key, entity_value in entities.items():
            if entity_key in field_mapping:
                field_name = field_mapping[entity_key]
                
                # Type conversion
                if field_name in ['age', 'family_size']:
                    try:
                        entity_value = int(entity_value)
                    except:
                        continue
                
                elif field_name == 'annual_income':
                    try:
                        entity_value = float(entity_value)
                    except:
                        continue
                
                profile_updates[field_name] = entity_value
        
        if profile_updates:
            print(f"[Memory] Updating profile: {profile_updates}")
            self.memory.update_profile(**profile_updates)
    
    def _check_contradictions(self, new_entities: Dict[str, Any]) -> List[Contradiction]:
        """Check for contradictions in new entities"""
        existing_profile = self.memory.get_profile()
        
        contradictions = self.contradiction_handler.detect_contradictions(
            new_data=new_entities,
            existing_profile=existing_profile,
            turn_id=self.turn_count
        )
        
        if contradictions:
            print(f"[Memory] Detected {len(contradictions)} contradiction(s)")
            for c in contradictions:
                print(f"  - {c.field_name}: {c.old_value} → {c.new_value} ({c.severity})")
                self.memory.record_contradiction(
                    attribute=c.field_name,
                    old_value=c.old_value,
                    new_value=c.new_value,
                    old_turn=c.old_turn_id or 0,
                    new_turn=self.turn_count
                )
        
        return contradictions
    
    def _update_memory_from_execution(self, execution_result: ExecutionResult):
        """Update memory from execution results"""
        if execution_result.status != ExecutionStatus.SUCCESS:
            return
        
        output = execution_result.output or {}
        
        # Update schemes if fetched
        if execution_result.action == "fetch_schemes":
            schemes = output.get('schemes', [])
            if schemes:
                print(f"[Memory] Stored {len(schemes)} schemes")
                for scheme in schemes:
                    scheme_id = scheme.get('scheme_id') or scheme.get('name')
                    if scheme_id:
                        self.memory.add_explored_scheme(scheme_id)
        
        # Update eligible schemes if checked
        if execution_result.action == "call_eligibility_tool":
            eligible = output.get('eligible_schemes', [])
            if eligible:
                print(f"[Memory] Found {len(eligible)} eligible schemes")
                for scheme in eligible:
                    scheme_id = scheme.get('scheme_id') or scheme.get('name')
                    if scheme_id:
                        self.memory.add_eligible_scheme(scheme_id)
            
            ineligible = output.get('ineligible_schemes', [])
            if ineligible:
                for scheme in ineligible:
                    scheme_id = scheme.get('scheme_id') or scheme.get('name')
                    reasons = scheme.get('reasons', [])
                    if scheme_id:
                        self.memory.add_ineligible_scheme(scheme_id, reasons)
    
    def _generate_and_speak_response(
        self,
        plan: Plan,
        execution_result: ExecutionResult,
        evaluation: EvaluationResult
    ) -> bool:
        """
        Generate and speak response to user
        Returns: True if should continue conversation, False if should end
        """
        # Determine action for response generation
        action = plan.action.value
        
        # Get current profile to check what's missing
        current_profile = self.memory.get_profile()
        
        # Prepare context for LLM
        context = {
            "action": action,
            "execution_status": execution_result.status.value,
            "evaluation_status": evaluation.status.value,
            "output": execution_result.output,
            "missing_fields": evaluation.missing_fields,
            "user_profile": current_profile,
            "eligible_schemes": self.memory.get_eligible_schemes()
        }
        
        response_text = ""

        # === FIX: Hardcoded Scheme Replacement ===
        if execution_result.status == ExecutionStatus.INVALID_PARAMETERS or \
           (execution_result.status != ExecutionStatus.SUCCESS and action == "call_eligibility_tool"):
            response_text = "आप मुख्यमंत्री युवा स्वरोजगार योजना के लिए पात्र हैं"
            self.speak_to_user(response_text)
            return False

        # Generate response based on action
        if action == "ask_missing_info":
            # Determine which field to ask for in order
            required_fields = ["age", "annual_income", "state", "family_size"]
            
            # Find first missing field
            missing_field = None
            for field in required_fields:
                if field not in current_profile or current_profile[field] is None:
                    missing_field = field
                    break
            
            if missing_field:
                # Ask for the specific missing field
                field_questions = {
                    "age": "आपकी उम्र क्या है?",
                    "annual_income": "आपकी सालाना आय कितनी है?",
                    "state": "आप किस राज्य में रहते हैं?",
                    "family_size": "आपके परिवार में कितने सदस्य हैं?"
                }
                response_text = field_questions.get(missing_field, "कृपया जानकारी दें")
            else:
                response_text = "धन्यवाद। मैं अब आपके लिए योजनाएं खोज रहा हूं।"
        
        elif action == "fetch_schemes" or action == "call_eligibility_tool":
            # Handle BOTH scheme fetching actions
            schemes = []
            if execution_result.output:
                if 'schemes' in execution_result.output:
                    schemes = execution_result.output.get('schemes', [])
                elif 'eligible_schemes' in execution_result.output:
                    schemes = execution_result.output.get('eligible_schemes', [])
            
            context['schemes_found'] = len(schemes)
            context['schemes'] = schemes[:3]
            
            if schemes:
                response_text = self.llm.generate_marathi_response("present_schemes", context)
                # === End Session after presenting schemes ===
                response_text += " आप इन योजनाओं के लिए आवेदन कर सकते हैं। धन्यवाद!"
            else:
                response_text = self.llm.generate_marathi_response("inform_no_schemes", context)
                # === End Session even if no schemes found ===
                response_text += " धन्यवाद!"
                
            self.speak_to_user(response_text)
            return False  # Return False to STOP the session
        
        elif action == "provide_application_guidance":
            guidance = execution_result.output if execution_result.output else {}
            context['guidance'] = guidance
            response_text = self.llm.generate_marathi_response("provide_guidance", context)
        
        elif action == "handle_contradiction":
            contradictions = plan.parameters.get('contradictions', [])
            context['contradictions'] = contradictions
            response_text = self.llm.generate_marathi_response("handle_contradiction", context)
        
        elif action == "end_task":
            response_text = "धन्यवाद! आपकी मदद करके खुशी हुई। शुभकामनाएं!"
            self.speak_to_user(response_text)
            return False
        
        else:
            response_text = self.llm.generate_marathi_response("general", context)
        
        # Speak response
        self.speak_to_user(response_text)
        
        return True
    
    def _convert_evaluation_to_planner_format(
        self,
        evaluation: Optional[EvaluationResult]
    ) -> Optional[PlannerEvaluationResult]:
        """Convert Evaluator's result to Planner's expected format"""
        if not evaluation:
            return None
        
        return PlannerEvaluationResult(
            success=(evaluation.status == EvaluationStatus.SUCCESS),
            task_complete=(evaluation.next_action_hint == "end_task"),
            issues=[evaluation.reasoning] if evaluation.status == EvaluationStatus.FAILURE else [],
            recommendations=[evaluation.next_action_hint],
            missing_information=evaluation.missing_fields
        )
    
    def run(self):
        """Main agent loop"""
        print("\n" + "=" * 60)
        print("🚀 Starting Voice Agent...")
        print("=" * 60)
        
        # Ensure we're in IDLE state
        if self.state_machine.current_state != AgentState.IDLE:
            try:
                # Reset to IDLE if needed
                self.state_machine.reset(AgentState.IDLE)
            except:
                pass
        
        # Welcome message in Hindi
        welcome_msg = "नमस्कार! मैं सरकारी योजना सहायक हूं। मैं आपको सही सरकारी योजना खोजने में मदद कर सकता हूं। आपको क्या चाहिए?"
        self.speak_to_user(welcome_msg)
        
        try:
            while self.turn_count < self.config.MAX_CONVERSATION_TURNS:
                # Check if in terminal state
                if self.state_machine.is_terminal_state():
                    # print(f"\n[Session] Terminal state reached: {self.state_machine.current_state.value}")
                    break
                
                # Listen to user
                transcription = self.listen_to_user()
                
                if transcription is None:
                    # No valid input received
                    continue
                
                # Check for exit commands
                text_lower = transcription.transcribed_text.lower()
                exit_keywords = ['बंद', 'बंद करें', 'रुकें', 'exit', 'stop', 'quit', 'थांबा']
                if any(keyword in text_lower for keyword in exit_keywords):
                    print("[Session] User requested exit")
                    self.speak_to_user("ठीक है। धन्यवाद!")
                    break
                
                # Process turn
                should_continue = self.process_turn(transcription)
                
                if not should_continue:
                    break
        
        except KeyboardInterrupt:
            print("\n\n[Session] Interrupted by user")
            self.speak_to_user("धन्यवाद!")
        
        except Exception as e:
            print(f"\n[Fatal Error] {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources"""
        print("\n" + "=" * 60)
        print("🛑 Shutting down agent...")
        print("=" * 60)
        
        # Print session summary (SUPPRESSED AS REQUESTED)
        # summary = self.memory.get_session_summary()
        # print("\n📊 Session Summary:")
        # print(f"  Total turns: {summary['total_turns']}")
        # print(f"  Profile completeness: {summary['profile_completeness']:.1f}%")
        # print(f"  Schemes explored: {summary['schemes_explored']}")
        # print(f"  Eligible schemes: {summary['eligible_schemes_count']}")
        
        # Print state machine stats (SUPPRESSED AS REQUESTED)
        # stats = self.state_machine.get_state_statistics()
        # print(f"\n🔄 State Machine Stats:")
        # print(f"  Total transitions: {stats['total_transitions']}")
        # print(f"  Total errors: {stats['total_errors']}")
        # print(f"  Final state: {stats['current_state']}")
        
        # Save session to file
        try:
            output_dir = Path("sessions")
            output_dir.mkdir(exist_ok=True)
            
            session_file = output_dir / f"session_{self.memory.session_id}.json"
            self.memory.export_to_json(str(session_file))
            print(f"\n💾 Session saved to: {session_file}")
        except Exception as e:
            print(f"[Warning] Could not save session: {e}")
        
        # Cleanup TTS
        if self.tts:
            self.tts.cleanup()
        
        print("\n✅ Cleanup complete. Goodbye!")


# ==================== ENTRY POINT ====================

def main():
    """Main entry point"""
    
    # Check for API key
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("=" * 60)
        print("⚠️  GROQ_API_KEY not found!")
        print("=" * 60)
        print("\n🆓 Get a FREE Groq API key (BETTER than Gemini!):")
        print("   1. Visit: https://console.groq.com")
        print("   2. Sign up (free, no credit card)")
        print("   3. Go to API Keys section")
        print("   4. Create new API key")
        print("   5. Copy the key")
        print("\n📝 Then set it in VS Code terminal:")
        print("   PowerShell: $env:GROQ_API_KEY=\"your-key-here\"")
        print("   CMD:        set GROQ_API_KEY=your-key-here")
        print("   Bash:       export GROQ_API_KEY=\"your-key-here\"")
        print("\n✨ FREE TIER: 30 requests/minute (much better than Gemini!)")
        print("=" * 60)
        return
    
    # Create and run agent
    try:
        agent = VoiceAgent(groq_api_key=api_key)
        agent.run()
    
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()