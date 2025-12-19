# agent/evaluator.py

"""
Evaluator Module
----------------
Evaluates the outcomes of executor actions without side effects.
Does NOT call tools, speak to user, or modify memory.
Only inspects and classifies execution results.
"""

from typing import Dict, Any, List
from enum import Enum
from dataclasses import dataclass


class EvaluationStatus(Enum):
    """Possible evaluation outcomes"""
    SUCCESS = "success"
    NEEDS_CLARIFICATION = "needs_clarification"
    RETRY = "retry"
    FAILURE = "failure"


@dataclass
class EvaluationResult:
    """Structured evaluation output"""
    status: EvaluationStatus
    confidence: float  # 0.0 to 1.0
    reasoning: str
    missing_fields: List[str]
    next_action_hint: str  # Suggestion for planner
    metadata: Dict[str, Any]


class Evaluator:
    """
    Evaluates executor outcomes and classifies them.
    Pure evaluation logic - no side effects.
    """
    
    def __init__(self):
        self.evaluation_rules = self._load_evaluation_rules()
    
    def _load_evaluation_rules(self) -> Dict[str, Any]:
        """Define evaluation criteria for different action types"""
        return {
            "eligibility_check": {
                "required_fields": ["eligible_schemes", "user_profile"],
                "success_criteria": {
                    "min_schemes": 1,
                    "profile_completeness": 0.7
                }
            },
            "scheme_details": {
                "required_fields": ["scheme_info"],
                "success_criteria": {
                    "has_documents": True,
                    "has_benefits": True
                }
            },
            "form_generation": {
                "required_fields": ["form_data", "submission_url"],
                "success_criteria": {
                    "form_complete": True
                }
            },
            "information_gathering": {
                "required_fields": ["gathered_info"],
                "success_criteria": {
                    "info_extracted": True
                }
            }
        }
    
    def evaluate(
        self,
        action_type: str,
        execution_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """
        Main evaluation entry point.
        
        Args:
            action_type: Type of action that was executed
            execution_result: Output from executor
            context: Current conversation context
            
        Returns:
            EvaluationResult with status and recommendations
        """
        
        # Check for explicit tool failures first
        if self._is_tool_failure(execution_result):
            return self._handle_tool_failure(execution_result, action_type)
        
        # Route to specific evaluator based on action type
        if action_type == "eligibility_check":
            return self._evaluate_eligibility_check(execution_result, context)
        elif action_type == "scheme_details":
            return self._evaluate_scheme_details(execution_result, context)
        elif action_type == "form_generation":
            return self._evaluate_form_generation(execution_result, context)
        elif action_type == "information_gathering":
            return self._evaluate_information_gathering(execution_result, context)
        else:
            return self._evaluate_generic(execution_result, context)
    
    def _is_tool_failure(self, result: Dict[str, Any]) -> bool:
        """Detect explicit tool failures"""
        if "error" in result and result["error"]:
            return True
        if "success" in result and not result["success"]:
            return True
        if result.get("status") == "failed":
            return True
        return False
    
    def _handle_tool_failure(
        self,
        result: Dict[str, Any],
        action_type: str
    ) -> EvaluationResult:
        """Handle cases where tool execution failed"""
        error_msg = result.get("error", "Unknown tool error")
        
        # Determine if retry is possible
        is_retryable = self._is_retryable_error(error_msg)
        
        if is_retryable:
            return EvaluationResult(
                status=EvaluationStatus.RETRY,
                confidence=0.8,
                reasoning=f"Tool failure but retryable: {error_msg}",
                missing_fields=[],
                next_action_hint=f"retry_{action_type}",
                metadata={"error": error_msg, "retry_count": result.get("retry_count", 0)}
            )
        else:
            return EvaluationResult(
                status=EvaluationStatus.FAILURE,
                confidence=0.9,
                reasoning=f"Non-retryable tool failure: {error_msg}",
                missing_fields=[],
                next_action_hint="inform_user_of_failure",
                metadata={"error": error_msg}
            )
    
    def _is_retryable_error(self, error_msg: str) -> bool:
        """Determine if an error is worth retrying"""
        retryable_patterns = [
            "timeout",
            "connection",
            "network",
            "rate limit",
            "temporary"
        ]
        error_lower = error_msg.lower()
        return any(pattern in error_lower for pattern in retryable_patterns)
    
    def _evaluate_eligibility_check(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate eligibility check results"""
        
        eligible_schemes = result.get("eligible_schemes", [])
        user_profile = result.get("user_profile", {})
        missing_info = result.get("missing_information", [])
        
        # Case 1: No eligible schemes found
        if len(eligible_schemes) == 0:
            if len(missing_info) > 0:
                return EvaluationResult(
                    status=EvaluationStatus.NEEDS_CLARIFICATION,
                    confidence=0.85,
                    reasoning="No schemes found, but missing critical user information",
                    missing_fields=missing_info,
                    next_action_hint="gather_missing_info",
                    metadata={
                        "missing_count": len(missing_info),
                        "profile_completeness": self._calculate_profile_completeness(user_profile)
                    }
                )
            else:
                return EvaluationResult(
                    status=EvaluationStatus.SUCCESS,
                    confidence=0.7,
                    reasoning="Profile complete but no eligible schemes found",
                    missing_fields=[],
                    next_action_hint="inform_no_schemes",
                    metadata={"eligible_count": 0}
                )
        
        # Case 2: Schemes found but profile incomplete
        if len(missing_info) > 0:
            # Check if missing info is critical
            critical_missing = self._filter_critical_fields(missing_info)
            if len(critical_missing) > 0:
                return EvaluationResult(
                    status=EvaluationStatus.NEEDS_CLARIFICATION,
                    confidence=0.75,
                    reasoning=f"Found {len(eligible_schemes)} schemes but missing critical info",
                    missing_fields=critical_missing,
                    next_action_hint="gather_missing_info",
                    metadata={
                        "eligible_count": len(eligible_schemes),
                        "missing_critical": critical_missing
                    }
                )
        
        # Case 3: Success - schemes found and profile adequate
        return EvaluationResult(
            status=EvaluationStatus.SUCCESS,
            confidence=0.95,
            reasoning=f"Successfully found {len(eligible_schemes)} eligible schemes",
            missing_fields=[],
            next_action_hint="present_schemes",
            metadata={
                "eligible_count": len(eligible_schemes),
                "top_schemes": [s.get("name") for s in eligible_schemes[:3]]
            }
        )
    
    def _evaluate_scheme_details(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate scheme details retrieval"""
        
        scheme_info = result.get("scheme_info", {})
        
        if not scheme_info:
            return EvaluationResult(
                status=EvaluationStatus.FAILURE,
                confidence=0.9,
                reasoning="Scheme details retrieval returned empty",
                missing_fields=["scheme_info"],
                next_action_hint="retry_or_inform",
                metadata={}
            )
        
        # Check completeness of scheme information
        required_fields = ["name", "benefits", "eligibility", "documents"]
        missing = [f for f in required_fields if f not in scheme_info or not scheme_info[f]]
        
        if len(missing) > 2:
            return EvaluationResult(
                status=EvaluationStatus.RETRY,
                confidence=0.7,
                reasoning="Scheme details incomplete",
                missing_fields=missing,
                next_action_hint="retry_scheme_details",
                metadata={"missing_fields": missing}
            )
        
        return EvaluationResult(
            status=EvaluationStatus.SUCCESS,
            confidence=0.9,
            reasoning="Scheme details retrieved successfully",
            missing_fields=[],
            next_action_hint="present_scheme_details",
            metadata={"scheme_name": scheme_info.get("name")}
        )
    
    def _evaluate_form_generation(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate form generation results"""
        
        form_data = result.get("form_data", {})
        submission_url = result.get("submission_url")
        missing_fields = result.get("missing_fields", [])
        
        # Case 1: Missing user data for form
        if len(missing_fields) > 0:
            return EvaluationResult(
                status=EvaluationStatus.NEEDS_CLARIFICATION,
                confidence=0.85,
                reasoning=f"Cannot generate complete form, missing {len(missing_fields)} fields",
                missing_fields=missing_fields,
                next_action_hint="gather_form_info",
                metadata={"missing_count": len(missing_fields)}
            )
        
        # Case 2: Form generated but incomplete
        if not form_data or not submission_url:
            return EvaluationResult(
                status=EvaluationStatus.RETRY,
                confidence=0.7,
                reasoning="Form generation incomplete",
                missing_fields=["form_data" if not form_data else "submission_url"],
                next_action_hint="retry_form_generation",
                metadata={}
            )
        
        # Case 3: Success
        return EvaluationResult(
            status=EvaluationStatus.SUCCESS,
            confidence=0.95,
            reasoning="Form generated successfully",
            missing_fields=[],
            next_action_hint="present_form",
            metadata={
                "form_fields": len(form_data),
                "has_submission_url": bool(submission_url)
            }
        )
    
    def _evaluate_information_gathering(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Evaluate information gathering results"""
        
        gathered_info = result.get("gathered_info", {})
        extraction_success = result.get("extraction_success", False)
        
        if not extraction_success or not gathered_info:
            return EvaluationResult(
                status=EvaluationStatus.NEEDS_CLARIFICATION,
                confidence=0.8,
                reasoning="Failed to extract required information from user input",
                missing_fields=result.get("expected_fields", []),
                next_action_hint="rephrase_question",
                metadata={"extraction_attempted": True}
            )
        
        return EvaluationResult(
            status=EvaluationStatus.SUCCESS,
            confidence=0.85,
            reasoning="Information gathered successfully",
            missing_fields=[],
            next_action_hint="proceed_with_info",
            metadata={"gathered_fields": list(gathered_info.keys())}
        )
    
    def _evaluate_generic(
        self,
        result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> EvaluationResult:
        """Fallback evaluation for unknown action types"""
        
        if result.get("success", False):
            return EvaluationResult(
                status=EvaluationStatus.SUCCESS,
                confidence=0.6,
                reasoning="Generic action completed",
                missing_fields=[],
                next_action_hint="continue",
                metadata=result
            )
        else:
            return EvaluationResult(
                status=EvaluationStatus.FAILURE,
                confidence=0.6,
                reasoning="Generic action failed",
                missing_fields=[],
                next_action_hint="replan",
                metadata=result
            )
    
    def _calculate_profile_completeness(self, profile: Dict[str, Any]) -> float:
        """Calculate how complete a user profile is"""
        critical_fields = ["age", "income", "location", "category"]
        filled = sum(1 for f in critical_fields if profile.get(f))
        return filled / len(critical_fields)
    
    def _filter_critical_fields(self, fields: List[str]) -> List[str]:
        """Filter to only critical missing fields"""
        critical = {"age", "income", "location", "category", "caste"}
        return [f for f in fields if f in critical]
    
    def should_retry(self, evaluation: EvaluationResult, retry_count: int) -> bool:
        """
        Decide if action should be retried based on evaluation.
        
        Args:
            evaluation: The evaluation result
            retry_count: Number of times action has been retried
            
        Returns:
            bool: Whether to retry
        """
        if evaluation.status != EvaluationStatus.RETRY:
            return False
        
        # Max retry limits
        max_retries = 3
        return retry_count < max_retries
    
    def get_failure_summary(self, evaluation: EvaluationResult) -> Dict[str, Any]:
        """
        Generate a summary for failure cases.
        Used by planner to decide next steps.
        """
        return {
            "status": evaluation.status.value,
            "reason": evaluation.reasoning,
            "missing_fields": evaluation.missing_fields,
            "recommended_action": evaluation.next_action_hint,
            "confidence": evaluation.confidence,
            "metadata": evaluation.metadata
        }