"""
Voice-First Agentic AI System - Main Entry Point
-------------------------------------------------
Orchestrates the complete voice interaction loop with agentic decision-making.
All user-facing interaction happens in Marathi (can be configured to other languages).
"""

import sounddevice as sd
sd.default.device = (9, None)

import sys
import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

import logging

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')


# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import speech components
from speech.stt import SpeechToText
from speech.tts import TextToSpeech

# Import memory components
from memory.session_memory import SessionMemory
from memory.contradiction_handler import ContradictionHandler

# Import agent components
from agent.state_machine import AgentStateMachine, AgentState
from agent.planner import Planner
from agent.executor import Executor
from agent.evaluator import Evaluator, EvaluationStatus

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler('agent_interaction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class VoiceAgentOrchestrator:
    """
    Main orchestrator for voice-first agentic system.
    Manages the complete interaction lifecycle.
    """
    
    def __init__(self, target_language: str = "marathi"):
        """
        Initialize all components.
        
        Args:
            target_language: Native language for user interaction (default: marathi)
        """
        logger.info(f"Initializing Voice Agent with language: {target_language}")
        
        self.target_language = target_language
        self.session_id = f"session_{int(time.time())}"
        
        # Initialize speech components
        logger.info("Initializing speech components...")
        self.stt = SpeechToText(language=target_language)
        self.tts = TextToSpeech(language=target_language)
        
        # Initialize memory components
        logger.info("Initializing memory systems...")
        self.session_memory = SessionMemory()
        self.contradiction_handler = ContradictionHandler()

        
        # Initialize agent components
        logger.info("Initializing agent components...")
        self.state_machine = AgentStateMachine()
        self.planner = Planner()
        self.executor = Executor()
        self.evaluator = Evaluator()
        
        # Interaction state
        self.is_running = False
        self.turn_count = 0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 3
        
        logger.info("All components initialized successfully")
    
    def start(self):
        """Start the voice agent interaction loop"""
        logger.info("=" * 80)
        logger.info("VOICE AGENT STARTING")
        logger.info(f"Language: {self.target_language}")
        logger.info(f"Session ID: {self.session_id}")
        logger.info("=" * 80)
        
        self.is_running = True
        
        # Welcome message
        welcome_message = self._get_welcome_message()
        self._speak_to_user(welcome_message)
        logger.info(f"[AGENT → USER] {welcome_message}")
        
        # Main interaction loop
        try:
            while self.is_running:
                self._interaction_turn()
        except KeyboardInterrupt:
            logger.info("\n Keyboard interrupt received")
            self._graceful_shutdown()
        except Exception as e:
            logger.error(f" Critical error in main loop: {e}", exc_info=True)
            self._emergency_shutdown()
    
    def _interaction_turn(self):
        """Execute one complete interaction turn"""
        self.turn_count += 1
        logger.info(f"\n{'=' * 80}")
        logger.info(f"TURN {self.turn_count}")
        logger.info(f"Current State: {self.state_machine.current_state.value}")
        logger.info(f"{'=' * 80}")
        
        # Step 1: Listen to user
        user_speech = self._listen_to_user()
        
        if user_speech is None:
            logger.warning("No speech detected, prompting user again")
            self._handle_no_speech()
            return
        
        # Step 2: Check for exit intent
        if self._is_exit_intent(user_speech):
            logger.info("Exit intent detected")
            self._graceful_shutdown()
            return
        
        # Step 3: Plan next action
        plan = self._plan_action(user_speech)
        
        if plan is None:
            logger.error("Planner failed to generate action")
            self._handle_planning_failure()
            return
        
        logger.info(f"PLAN: {plan['action_type']} | Reasoning: {plan.get('reasoning', 'N/A')}")
        
        # Step 4: Check for contradictions (using planner-extracted data)
        contradiction_result = self._check_for_contradictions(plan)
        if contradiction_result["has_contradiction"]:
            logger.warning(f"Contradiction detected: {contradiction_result['message']}")
            response = self._handle_contradiction(contradiction_result)
            # Store turn with contradiction handling
            self.session_memory.add_turn(
                user_input_marathi=user_speech,
                agent_response_marathi=response,
                extracted_info=plan.get("extracted_data"),
                tools_used=["contradiction_handler"]
            )
            return
        
        # Step 5: Execute action
        execution_result = self._execute_action(plan)
        
        logger.info(f"EXECUTION: Status={execution_result.get('success', 'unknown')}")
        
        # Step 6: Evaluate results
        evaluation = self._evaluate_result(plan['action_type'], execution_result)
        
        logger.info(f"EVALUATION: {evaluation.status.value} | Confidence={evaluation.confidence:.2f}")
        logger.info(f" Reasoning: {evaluation.reasoning}")
        
        # Step 7: Handle evaluation outcome and get response
        agent_response = self._handle_evaluation(evaluation, plan, execution_result)
        
        # Step 8: Store complete turn in memory (SINGLE CALL)
        self.session_memory.add_turn(
            user_input_marathi=user_speech,
            agent_response_marathi=agent_response,
            extracted_info=plan.get("extracted_data"),
            tools_used=execution_result.get("tools_used")
        )
        
        # Step 9: Update state machine
        self._update_state(evaluation, plan)
        
        # Reset failure counter on success
        if evaluation.status == EvaluationStatus.SUCCESS:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
            
        # Check if too many failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Too many consecutive failures ({self.consecutive_failures})")
            self._handle_repeated_failures()
    
    def _listen_to_user(self) -> Optional[str]:
        """
        Listen to user speech and transcribe.
        
        Returns:
            Transcribed text or None if failed
        """
        logger.info("Listening for user input...")
        
        try:
            result = self.stt.capture_and_transcribe()
            # result = self.stt.capture_and_transcribe(duration=5)

            
            if result.success:
                transcription = result.transcription
                confidence = result.confidence
                
                logger.info(f"[USER] {transcription} (confidence: {confidence:.2f})")
                
                # Handle low confidence
                if confidence < 0.6:
                    logger.warning(f"Low confidence transcription: {confidence:.2f}")
                    self._handle_low_confidence(transcription, confidence)
                    return None
                
                return transcription
            else:
                error = result.get("error", "Unknown error")
                logger.error(f"STT failed: {error}")
                return None
                
        except Exception as e:
            logger.error(f"Exception during listening: {e}", exc_info=True)
            return None
    
    def _speak_to_user(self, text: str):
        """
        Speak to user via TTS.
        
        Args:
            text: Text to speak (in target language)
        """
        logger.info(f"Speaking to user: {text}")
        
        try:
            result = self.tts.speak(text)
            
            if result.success:
                logger.info("TTS successful")
            else:
                error = result.error if hasattr(result, "error") else "Unknown error"
                logger.error(f"TTS failed: {error}")
                # Fallback: at least log the intended message
                logger.warning(f"Intended message: {text}")
                
        except Exception as e:
            logger.error(f"Exception during speech: {e}", exc_info=True)
    
    def _check_for_contradictions(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if planner-extracted data contradicts stored profile.
        
        Args:
            plan: Plan dict potentially containing extracted structured data
            
        Returns:
            Dict with has_contradiction, message, and contradictions list
        """
        logger.info("Checking for contradictions...")
        
        try:
            # Extract structured data from plan (if available)
            new_data = plan.get("extracted_data", {})
            
            # Skip if no structured data available
            if not new_data or not isinstance(new_data, dict):
                logger.info("No structured data in plan, skipping contradiction check")
                return {"has_contradiction": False}
            
            # Get existing profile from memory
            existing_profile = self.session_memory.get_profile()
            
            # Skip if no existing profile
            if not existing_profile:
                logger.info("No existing profile, skipping contradiction check")
                return {"has_contradiction": False}
            
            # Detect contradictions
            contradictions = self.contradiction_handler.detect_contradictions(
                new_data=new_data,
                existing_profile=existing_profile,
                turn_id=self.turn_count
            )
            
            # Wrap result in expected format
            if contradictions:
                logger.warning(f"Found {len(contradictions)} contradiction(s)")
                for c in contradictions:
                    logger.warning(f"   - {c.field_name}: {c.old_value} → {c.new_value} (severity: {c.severity})")
                
                return {
                    "has_contradiction": True,
                    "message": "Contradictory information detected",
                    "contradictions": [c.to_dict() for c in contradictions],
                    "count": len(contradictions),
                    "critical_count": sum(1 for c in contradictions if c.severity == "critical")
                }
            else:
                logger.info("No contradictions detected")
                return {"has_contradiction": False}
            
        except Exception as e:
            logger.error(f"Error checking contradictions: {e}", exc_info=True)
            return {"has_contradiction": False, "error": str(e)}
    
    def _handle_contradiction(self, contradiction_result: Dict[str, Any]) -> str:
        """
        Handle detected contradiction.
        
        Returns:
            Agent response message
        """
        # Generate clarification message based on contradictions
        contradictions = contradiction_result.get("contradictions", [])
        
        response = "माफ करा, मला काही गोंधळ झाला आहे. कृपया पुन्हा स्पष्ट करा."
        
        if contradictions:
            # Use first critical or moderate contradiction for clarification
            critical_or_moderate = [
                c for c in contradictions 
                if c.get("severity") in ["critical", "moderate"]
            ]
            
            if critical_or_moderate:
                first = critical_or_moderate[0]
                field_name = first.get("field_name", "information")
                old_value = first.get("old_value")
                new_value = first.get("new_value")
                
                # Generate Marathi clarification
                response = f"माफ करा, तुम्ही आधी {field_name} बद्दल '{old_value}' सांगितले होते, आता '{new_value}' सांगितले आहे. कृपया खात्री करा कोणती माहिती बरोबर आहे?"
        
        self._speak_to_user(response)
        return response
    
    def _plan_action(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Generate action plan using planner.
        
        Returns:
            Plan dict or None if planning failed
        """
        logger.info("Planning next action...")
        
        try:
            # Get current context
            context = {
                "state": self.state_machine.current_state.value,
                "turn": self.turn_count,
                "user_profile": self.session_memory.get_profile(),
                "conversation_summary": self.session_memory.get_conversation_history(last_n=3)
            }
            
            plan = self.planner.plan(
                user_input=user_input,
                context=context
            )
            
            return plan
            
        except Exception as e:
            logger.error(f"Planning error: {e}", exc_info=True)
            return None
    
    def _execute_action(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute planned action using executor.
        
        Returns:
            Execution result
        """
        logger.info(f"Executing action: {plan['action_type']}")
        
        try:
            result = self.executor.execute(plan)
            return result
            
        except Exception as e:
            logger.error(f"Execution error: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "action_type": plan['action_type']
            }
    
    def _evaluate_result(
        self,
        action_type: str,
        execution_result: Dict[str, Any]
    ):
        """
        Evaluate execution result.
        
        Returns:
            EvaluationResult
        """
        logger.info("Evaluating execution result...")
        
        try:
            context = {
                "turn": self.turn_count,
                "state": self.state_machine.current_state.value,
                "user_profile": self.session_memory.get_profile()
            }
            
            evaluation = self.evaluator.evaluate(
                action_type=action_type,
                execution_result=execution_result,
                context=context
            )
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}", exc_info=True)
            # Return default failure evaluation
            from agent.evaluator import EvaluationResult, EvaluationStatus
            return EvaluationResult(
                status=EvaluationStatus.FAILURE,
                confidence=0.5,
                reasoning=f"Evaluation failed: {str(e)}",
                missing_fields=[],
                next_action_hint="replan",
                metadata={"error": str(e)}
            )
    
    def _handle_evaluation(
        self,
        evaluation,
        plan: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> str:
        """
        Handle evaluation outcome and respond to user.
        
        Returns:
            Agent response message
        """
        response = ""
        
        if evaluation.status == EvaluationStatus.SUCCESS:
            # Generate response from execution result
            response = execution_result.get("response", "")
            if response:
                self._speak_to_user(response)
            else:
                logger.warning("No response in successful execution result")
        
        elif evaluation.status == EvaluationStatus.NEEDS_CLARIFICATION:
            # Ask for missing information
            missing = evaluation.missing_fields
            response = self._generate_clarification_prompt(missing)
            self._speak_to_user(response)
        
        elif evaluation.status == EvaluationStatus.RETRY:
            logger.info("Retry needed, replanning...")
            response = "मला समजले नाही. कृपया दुसऱ्या शब्दांत सांगा."
            self._speak_to_user(response)
        
        elif evaluation.status == EvaluationStatus.FAILURE:
            # Inform user of failure
            response = self._generate_failure_message(evaluation)
            self._speak_to_user(response)
        
        return response
    
    def _update_state(self, evaluation, plan: Dict[str, Any]):
        """Update state machine based on evaluation"""
        try:
            # Determine event based on evaluation status
            if evaluation.status == EvaluationStatus.SUCCESS:
                event = self._map_plan_to_success_event(plan['action_type'])
            elif evaluation.status == EvaluationStatus.NEEDS_CLARIFICATION:
                event = "missing_info"
            elif evaluation.status == EvaluationStatus.RETRY:
                event = "retry"
            else:
                event = "error"
            
            # Transition state
            if event:
                success = self.state_machine.transition(event)
                if success:
                    logger.info(f"State transitioned to: {self.state_machine.current_state.value}")
                else:
                    logger.warning(f"State transition failed for event: {event}")
        
        except Exception as e:
            logger.error(f"State update error: {e}", exc_info=True)
    
    def _map_plan_to_success_event(self, action_type: str) -> str:
        """Map action type to state machine event"""
        mapping = {
            "eligibility_check": "eligibility_done",
            "scheme_details": "scheme_selected",
            "form_generation": "form_generated",
            "information_gathering": "info_gathered"
        }
        return mapping.get(action_type, "proceed")
    
    def _generate_clarification_prompt(self, missing_fields: list) -> str:
        """Generate clarification prompt in Marathi"""
        # This should ideally be handled by planner, but providing fallback
        if not missing_fields:
            return "मला काही अधिक माहिती हवी आहे. कृपया अधिक तपशील द्या."
        
        field_names_marathi = {
            "age": "वय",
            "income": "उत्पन्न",
            "location": "स्थान",
            "category": "वर्ग",
            "caste": "जात",
            "occupation": "व्यवसाय"
        }
        
        fields_str = ", ".join([field_names_marathi.get(f, f) for f in missing_fields[:3]])
        return f"कृपया आपली {fields_str} सांगा."
    
    def _generate_failure_message(self, evaluation) -> str:
        """Generate failure message in Marathi"""
        return "माफ करा, मला तुमची मदत करण्यात अडचण येत आहे. कृपया पुन्हा प्रयत्न करा."
    
    def _handle_no_speech(self):
        """Handle case where no speech was detected"""
        prompt = "मी ऐकू शकत नाही. कृपया पुन्हा बोला."
        self._speak_to_user(prompt)
    
    def _handle_low_confidence(self, transcription: str, confidence: float):
        """Handle low confidence speech recognition"""
        prompt = f"माफ करा, मला नीट समजले नाही. कृपया पुन्हा स्पष्टपणे बोला."
        self._speak_to_user(prompt)
    
    def _handle_planning_failure(self):
        """Handle case where planner fails"""
        self.consecutive_failures += 1
        message = "मला समजले नाही. कृपया दुसऱ्या शब्दांत सांगा."
        self._speak_to_user(message)
    
    def _handle_repeated_failures(self):
        """Handle repeated failures - escalate or exit"""
        message = "मला तुमची मदत करण्यात अडचण येत आहे. कृपया काही वेळाने पुन्हा प्रयत्न करा."
        self._speak_to_user(message)
        time.sleep(2)
        self._graceful_shutdown()
    
    def _is_exit_intent(self, text: str) -> bool:
        """Check if user wants to exit"""
        exit_phrases_marathi = [
            "बंद कर",
            "थांब",
            "exit",
            "quit",
            "बाय",
            "धन्यवाद",
            "पुरे झाले"
        ]
        
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in exit_phrases_marathi)
    
    def _get_welcome_message(self) -> str:
        """Get welcome message in target language"""
        messages = {
            "marathi": "नमस्कार! मी सरकारी योजना सहाय्यक आहे. मी तुम्हाला योग्य सरकारी योजना शोधण्यात मदत करू शकतो. तुम्हाला कशी मदत करू?",
            "hindi": "नमस्ते! मैं सरकारी योजना सहायक हूं। मैं आपको उपयुक्त सरकारी योजनाएं खोजने में मदद कर सकता हूं। मैं आपकी कैसे सहायता करूं?",
            "telugu": "నమస్కారం! నేను ప్రభుత్వ పథకాల సహాయకుడిని. మీకు తగిన ప్రభుత్వ పథకాలను కనుగొనడంలో నేను సహాయపడగలను. మీకు ఎలా సహాయం చేయాలి?"
        }
        return messages.get(self.target_language, messages["marathi"])
    
    def _graceful_shutdown(self):
        """Gracefully shutdown the agent"""
        logger.info("\n Initiating graceful shutdown...")
        
        goodbye_messages = {
            "marathi": "धन्यवाद! पुन्हा भेटू.",
            "hindi": "धन्यवाद! फिर मिलेंगे।",
            "telugu": "ధన్యవాదాలు! మళ్లీ కలుద్దాం."
        }
        
        goodbye = goodbye_messages.get(self.target_language, goodbye_messages["marathi"])
        self._speak_to_user(goodbye)
        
        # Save session
        try:
            session_file = f"session_{self.session_id}.json"
            self.session_memory.export_to_json(session_file)
            logger.info(f"✓ Session saved to {session_file}")
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
        
        # Cleanup
        self._cleanup()
        
        self.is_running = False
        logger.info("Shutdown complete")
    
    def _emergency_shutdown(self):
        """Emergency shutdown on critical error"""
        logger.error("EMERGENCY SHUTDOWN")
        
        try:
            self._speak_to_user("माफ करा, काही चूक झाली. मी बंद होत आहे.")
        except:
            pass
        
        self._cleanup()
        self.is_running = False
    
    def _cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up resources...")
        
        try:
            if hasattr(self.stt, 'cleanup'):
                self.stt.cleanup()
            if hasattr(self.tts, 'cleanup'):
                self.tts.cleanup()
            logger.info("✓ Cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


def main():
    """Main entry point"""
    print("\n" + "=" * 80)
    print("VOICE-FIRST AGENTIC AI SYSTEM")
    print("Government Schemes Assistant (सरकारी योजना सहाय्यक)")
    print("=" * 80 + "\n")
    
    # Configuration
    TARGET_LANGUAGE = "marathi"  # Change to: hindi, telugu, tamil, bengali, odia
    
    print(f"Target Language: {TARGET_LANGUAGE}")
    print("Starting voice agent...\n")
    
    # Create and start orchestrator
    orchestrator = VoiceAgentOrchestrator(target_language=TARGET_LANGUAGE)
    
    try:
        orchestrator.start()
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        print(f"\nFatal error occurred. Check logs for details.")
    
    print("\n" + "=" * 80)
    print("SESSION ENDED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()