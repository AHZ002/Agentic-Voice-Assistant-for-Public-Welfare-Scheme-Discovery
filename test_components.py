"""
test_components.py
Isolated tests for individual components
"""

import sys
import os

def test_stt():
    """Test Speech-to-Text"""
    print("\n" + "=" * 60)
    print("Testing Speech-to-Text (5 seconds)")
    print("=" * 60)
    print("Speak in Marathi when you see 'Listening...'")
    
    try:
        from speech.stt import SpeechToText
        
        stt = SpeechToText(
            model_size="base",
            language="marathi",
            confidence_threshold=0.40
        )
        
        result = stt.capture_and_transcribe(duration=5)
        
        print("\n✅ Transcription successful!")
        print(f"Text: {result.transcribed_text}")
        print(f"Language: {result.language}")
        print(f"Confidence: {result.confidence_score:.2f}")
        
    except Exception as e:
        print(f"\n❌ STT test failed: {e}")
        import traceback
        traceback.print_exc()

def test_tts():
    """Test Text-to-Speech"""
    print("\n" + "=" * 60)
    print("Testing Text-to-Speech")
    print("=" * 60)
    
    try:
        from speech.tts import TextToSpeech
        
        tts = TextToSpeech(language="marathi", engine="gtts")
        
        test_text = "नमस्कार! हे एक चाचणी आहे."
        print(f"Speaking: {test_text}")
        
        result = tts.speak(test_text)
        
        if result.success:
            print("\n✅ TTS successful!")
        else:
            print(f"\n❌ TTS failed: {result.error}")
    
    except Exception as e:
        print(f"\n❌ TTS test failed: {e}")
        import traceback
        traceback.print_exc()

def test_llm():
    """Test LLM integration"""
    print("\n" + "=" * 60)
    print("Testing LLM (Google Gemini API)")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set")
        return
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content("Say 'Hello' in Marathi")
        
        result = response.text
        print(f"\n✅ LLM response: {result}")
    
    except Exception as e:
        print(f"\n❌ LLM test failed: {e}")
        import traceback
        traceback.print_exc()

def test_state_machine():
    """Test state machine"""
    print("\n" + "=" * 60)
    print("Testing State Machine")
    print("=" * 60)
    
    try:
        from agent.state_machine import AgentStateMachine, AgentState
        
        sm = AgentStateMachine()
        
        print(f"Initial state: {sm.current_state.value}")
        
        # Test transition
        sm.transition(AgentState.LISTENING, reason="Test transition")
        print(f"After transition: {sm.current_state.value}")
        
        sm.transition(AgentState.PLANNING, reason="Test planning")
        print(f"After planning: {sm.current_state.value}")
        
        # Get stats
        stats = sm.get_state_statistics()
        print(f"\nTransitions: {stats['total_transitions']}")
        print(f"State path: {' → '.join(sm.get_state_path())}")
        
        print("\n✅ State machine working correctly")
    
    except Exception as e:
        print(f"\n❌ State machine test failed: {e}")
        import traceback
        traceback.print_exc()

def test_planner():
    """Test planner"""
    print("\n" + "=" * 60)
    print("Testing Planner")
    print("=" * 60)
    
    try:
        from agent.planner import Planner, UserIntent
        
        planner = Planner()
        
        # Create test intent
        intent = UserIntent(
            intent_type="find_scheme",
            entities={"occupation": "farmer"},
            confidence=0.9
        )
        
        # Create test memory
        memory = {
            "user_profile": {
                "age": 45,
                "state": "Maharashtra"
            }
        }
        
        plan = planner.plan_next_action(
            user_intent=intent,
            memory_snapshot=memory
        )
        
        print(f"\n✅ Plan created!")
        print(f"Action: {plan.action.value}")
        print(f"Priority: {plan.priority.value}")
        print(f"Reasoning: {plan.reasoning}")
    
    except Exception as e:
        print(f"\n❌ Planner test failed: {e}")
        import traceback
        traceback.print_exc()

def test_executor():
    """Test executor"""
    print("\n" + "=" * 60)
    print("Testing Executor")
    print("=" * 60)
    
    try:
        from agent.executor import Executor
        
        executor = Executor(
            schemes_file_path=r"C:\Users\ABDUL_HADI\Desktop\Voice Agent\data\schemes.json"
        )
        
        # Test scheme fetching
        plan = {
            "action": "fetch_schemes",
            "parameters": {
                "keywords": ["farmer"],
                "max_results": 5
            }
        }
        
        result = executor.execute_plan(plan)
        
        print(f"\n✅ Execution complete!")
        print(f"Status: {result.status.value}")
        print(f"Tool used: {result.tool_used}")
        
        if result.output:
            schemes_found = len(result.output.get('schemes', []))
            print(f"Schemes found: {schemes_found}")
    
    except Exception as e:
        print(f"\n❌ Executor test failed: {e}")
        import traceback
        traceback.print_exc()

def test_memory():
    """Test session memory"""
    print("\n" + "=" * 60)
    print("Testing Session Memory")
    print("=" * 60)
    
    try:
        from memory.session_memory import SessionMemory
        
        memory = SessionMemory()
        
        # Update profile
        memory.update_profile(age=45, state="Maharashtra", occupation="farmer")
        
        # Add a turn
        memory.add_turn(
            user_input_marathi="मला योजना हवी आहे",
            agent_response_marathi="तुमचे वय काय आहे?"
        )
        
        # Get summary
        summary = memory.get_session_summary()
        
        print(f"\n✅ Memory working!")
        print(f"Session ID: {memory.session_id}")
        print(f"Turns: {summary['total_turns']}")
        print(f"Profile completeness: {summary['profile_completeness']:.1f}%")
        print(f"Profile: {memory.get_profile()}")
    
    except Exception as e:
        print(f"\n❌ Memory test failed: {e}")
        import traceback
        traceback.print_exc()

def test_contradiction_handler():
    """Test contradiction handler"""
    print("\n" + "=" * 60)
    print("Testing Contradiction Handler")
    print("=" * 60)
    
    try:
        from memory.contradiction_handler import ContradictionHandler
        
        handler = ContradictionHandler()
        
        # Test data
        existing_profile = {
            "age": 45,
            "state": "Maharashtra"
        }
        
        new_data = {
            "age": 50,  # Contradiction!
            "occupation": "farmer"
        }
        
        contradictions = handler.detect_contradictions(
            new_data=new_data,
            existing_profile=existing_profile
        )
        
        print(f"\n✅ Contradiction detection working!")
        print(f"Contradictions found: {len(contradictions)}")
        
        for c in contradictions:
            print(f"  - {c.field_name}: {c.old_value} → {c.new_value} ({c.severity})")
    
    except Exception as e:
        print(f"\n❌ Contradiction handler test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run tests"""
    print("=" * 60)
    print("🧪 Component Testing Suite")
    print("=" * 60)
    
    tests = [
        ("State Machine", test_state_machine),
        ("Planner", test_planner),
        ("Executor", test_executor),
        ("Session Memory", test_memory),
        ("Contradiction Handler", test_contradiction_handler),
        ("LLM Integration", test_llm),
        ("Text-to-Speech", test_tts),
        ("Speech-to-Text", test_stt)  # Last because it requires user input
    ]
    
    print("\nAvailable tests:")
    for i, (name, _) in enumerate(tests, 1):
        print(f"{i}. {name}")
    print(f"{len(tests) + 1}. Run all tests")
    print("0. Exit")
    
    choice = input("\nSelect test (0-{}): ".format(len(tests) + 1))
    
    try:
        choice = int(choice)
        
        if choice == 0:
            print("Exiting...")
            return
        
        elif choice == len(tests) + 1:
            # Run all tests
            for name, test_func in tests:
                print(f"\n{'=' * 60}")
                print(f"Running: {name}")
                print(f"{'=' * 60}")
                test_func()
        
        elif 1 <= choice <= len(tests):
            name, test_func = tests[choice - 1]
            test_func()
        
        else:
            print("Invalid choice")
    
    except ValueError:
        print("Invalid input")
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\nTest error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()