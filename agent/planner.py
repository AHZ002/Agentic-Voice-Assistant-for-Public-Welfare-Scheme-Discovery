"""
Agent Action Planner

This module decides the next action for the agent based on user intent,
session memory, contradictions, and evaluation results. It outputs structured
plans without executing tools or interacting with the user.

Planner DOES NOT:
- Call tools
- Speak to user
- Modify memory
- Use hard-coded responses
"""

from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import re


class PlanAction(Enum):
    """Available planner actions"""
    ASK_MISSING_INFO = "ask_missing_info"
    CALL_ELIGIBILITY_TOOL = "call_eligibility_tool"
    FETCH_SCHEMES = "fetch_schemes"
    HANDLE_CONTRADICTION = "handle_contradiction"
    PROVIDE_APPLICATION_GUIDANCE = "provide_application_guidance"
    END_TASK = "end_task"


class PlanPriority(Enum):
    """Plan priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Plan:
    """Structured plan output"""
    action: PlanAction
    priority: PlanPriority
    reasoning: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    required_info: List[str] = field(default_factory=list)
    alternative_actions: List[PlanAction] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action.value,
            "priority": self.priority.value,
            "reasoning": self.reasoning,
            "parameters": self.parameters,
            "required_info": self.required_info,
            "alternative_actions": [a.value for a in self.alternative_actions],
            "confidence": self.confidence
        }


@dataclass
class UserIntent:
    """Parsed user intent"""
    intent_type: str  # e.g., "find_scheme", "apply_scheme", "check_eligibility"
    entities: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    raw_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "intent_type": self.intent_type,
            "entities": self.entities,
            "confidence": self.confidence,
            "raw_text": self.raw_text
        }


@dataclass
class EvaluationResult:
    """Result from evaluator"""
    success: bool
    task_complete: bool
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    missing_information: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "task_complete": self.task_complete,
            "issues": self.issues,
            "recommendations": self.recommendations,
            "missing_information": self.missing_information
        }


class Planner:
    """
    Agent action planner - decides next action based on context
    """
    
    # Required fields for eligibility check
    REQUIRED_ELIGIBILITY_FIELDS = [
        "age", "income", "state", "category"
    ]
    
    # Optional but useful fields
    OPTIONAL_ELIGIBILITY_FIELDS = [
        "gender", "occupation", "disability", "bpl_card", 
        "farmer", "widow", "minority"
    ]
    
    # Intent type mappings
    INTENT_ACTIONS = {
        "find_scheme": PlanAction.FETCH_SCHEMES,
        "check_eligibility": PlanAction.CALL_ELIGIBILITY_TOOL,
        "apply_scheme": PlanAction.PROVIDE_APPLICATION_GUIDANCE,
        "ask_about_scheme": PlanAction.FETCH_SCHEMES,
        "general_query": PlanAction.ASK_MISSING_INFO
    }
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.6,
        max_missing_fields: int = 3
    ):
        """
        Initialize planner
        
        Args:
            min_confidence_threshold: Minimum confidence to proceed
            max_missing_fields: Maximum missing fields before asking
        """
        self.min_confidence_threshold = min_confidence_threshold
        self.max_missing_fields = max_missing_fields
        
        self.plan_history: List[Plan] = []
    
    def plan_next_action(
        self,
        user_intent: Optional[UserIntent] = None,
        memory_snapshot: Optional[Dict[str, Any]] = None,
        contradiction_reports: Optional[List[Dict[str, Any]]] = None,
        last_evaluation: Optional[EvaluationResult] = None,
        current_context: Optional[Dict[str, Any]] = None
    ) -> Plan:
        """
        Plan the next action for the agent
        
        Args:
            user_intent: Parsed user intent
            memory_snapshot: Current session memory
            contradiction_reports: Any contradictions detected
            last_evaluation: Result from last evaluation
            current_context: Additional context
            
        Returns:
            Plan object with next action
        """
        # Initialize context
        memory = memory_snapshot or {}
        contradictions = contradiction_reports or []
        evaluation = last_evaluation
        context = current_context or {}
        
        # Priority 1: Handle contradictions first
        if contradictions:
            return self._plan_contradiction_handling(contradictions, memory)
        
        # Priority 2: Check if task is complete
        if evaluation and evaluation.task_complete:
            return self._plan_task_completion(evaluation, memory)
        
        # Priority 3: Handle evaluation issues
        if evaluation and evaluation.issues:
            return self._plan_issue_resolution(evaluation, memory)
        
        # Priority 4: Check for missing critical information
        missing_fields = self._identify_missing_fields(memory)
        if missing_fields:
            critical_missing = [
                f for f in missing_fields 
                if f in self.REQUIRED_ELIGIBILITY_FIELDS
            ]
            if critical_missing or len(missing_fields) > self.max_missing_fields:
                return self._plan_information_gathering(
                    missing_fields, 
                    user_intent, 
                    memory
                )
        
        # Priority 5: Process user intent
        if user_intent:
            return self._plan_from_intent(user_intent, memory)
        
        # Priority 6: Continue workflow if schemes exist
        if self._has_schemes_data(memory):
            if not self._has_eligibility_results(memory):
                return self._plan_eligibility_check(memory)
            else:
                return self._plan_application_guidance(memory)
        
        # Default: Ask for more information
        return self._plan_information_gathering(
            missing_fields or ["user_requirement"],
            user_intent,
            memory
        )
    
    def _plan_contradiction_handling(
        self,
        contradictions: List[Dict[str, Any]],
        memory: Dict[str, Any]
    ) -> Plan:
        """Plan to handle contradictions"""
        contradiction_fields = [c.get("field", "unknown") for c in contradictions]
        
        return Plan(
            action=PlanAction.HANDLE_CONTRADICTION,
            priority=PlanPriority.CRITICAL,
            reasoning=(
                f"Detected contradictions in: {', '.join(contradiction_fields)}. "
                "Must resolve before proceeding."
            ),
            parameters={
                "contradictions": contradictions,
                "fields_to_clarify": contradiction_fields
            },
            required_info=contradiction_fields,
            confidence=1.0
        )
    
    def _plan_task_completion(
        self,
        evaluation: EvaluationResult,
        memory: Dict[str, Any]
    ) -> Plan:
        """Plan task completion"""
        return Plan(
            action=PlanAction.END_TASK,
            priority=PlanPriority.HIGH,
            reasoning="Task successfully completed. All requirements satisfied.",
            parameters={
                "completion_status": "success",
                "evaluation": evaluation.to_dict() if evaluation else {}
            },
            confidence=1.0
        )
    
    def _plan_issue_resolution(
        self,
        evaluation: EvaluationResult,
        memory: Dict[str, Any]
    ) -> Plan:
        """Plan to resolve evaluation issues"""
        issues = evaluation.issues
        missing_info = evaluation.missing_information
        
        # Determine if issues require information or re-execution
        if missing_info:
            return Plan(
                action=PlanAction.ASK_MISSING_INFO,
                priority=PlanPriority.HIGH,
                reasoning=(
                    f"Evaluation found missing information: {', '.join(missing_info)}"
                ),
                parameters={
                    "missing_fields": missing_info,
                    "issues": issues
                },
                required_info=missing_info,
                confidence=0.9
            )
        
        # Issues without missing info - may need re-planning
        return Plan(
            action=PlanAction.ASK_MISSING_INFO,
            priority=PlanPriority.MEDIUM,
            reasoning=f"Evaluation issues detected: {', '.join(issues)}",
            parameters={
                "issues": issues,
                "clarification_needed": True
            },
            required_info=["clarification"],
            alternative_actions=[PlanAction.FETCH_SCHEMES],
            confidence=0.7
        )
    
    def _plan_information_gathering(
        self,
        missing_fields: List[str],
        user_intent: Optional[UserIntent],
        memory: Dict[str, Any]
    ) -> Plan:
        """Plan to gather missing information"""
        # Prioritize critical fields
        critical_missing = [
            f for f in missing_fields 
            if f in self.REQUIRED_ELIGIBILITY_FIELDS
        ]
        
        fields_to_ask = critical_missing if critical_missing else missing_fields
        
        return Plan(
            action=PlanAction.ASK_MISSING_INFO,
            priority=PlanPriority.HIGH if critical_missing else PlanPriority.MEDIUM,
            reasoning=(
                f"Missing required information: {', '.join(fields_to_ask)}. "
                "Cannot proceed without this data."
            ),
            parameters={
                "missing_fields": fields_to_ask,
                "all_missing": missing_fields,
                "context": "eligibility_check"
            },
            required_info=fields_to_ask,
            confidence=0.95
        )
    
    def _plan_from_intent(
        self,
        user_intent: UserIntent,
        memory: Dict[str, Any]
    ) -> Plan:
        """Plan action based on user intent"""
        intent_type = user_intent.intent_type
        
        # Map intent to action
        action = self.INTENT_ACTIONS.get(
            intent_type,
            PlanAction.ASK_MISSING_INFO
        )
        
        # Special handling for specific intents
        if intent_type == "find_scheme":
            return self._plan_scheme_search(user_intent, memory)
        
        elif intent_type == "check_eligibility":
            return self._plan_eligibility_check(memory, user_intent)
        
        elif intent_type == "apply_scheme":
            return self._plan_application_guidance(memory, user_intent)
        
        # Generic intent handling
        return Plan(
            action=action,
            priority=PlanPriority.MEDIUM,
            reasoning=f"User intent detected: {intent_type}",
            parameters={
                "intent": user_intent.to_dict(),
                "entities": user_intent.entities
            },
            confidence=user_intent.confidence
        )
    
    def _plan_scheme_search(
        self,
        user_intent: UserIntent,
        memory: Dict[str, Any]
    ) -> Plan:
        """Plan scheme search action"""
        # Extract search parameters from intent
        entities = user_intent.entities
        keywords = entities.get("keywords", [])
        category = entities.get("category")
        
        # Check if we have enough info to search
        if not keywords and not category:
            # Extract from memory
            user_profile = memory.get("user_profile", {})
            if user_profile:
                # Use profile attributes as search hints
                keywords = self._extract_search_keywords_from_profile(user_profile)
        
        return Plan(
            action=PlanAction.FETCH_SCHEMES,
            priority=PlanPriority.HIGH,
            reasoning="User wants to find schemes. Searching based on available criteria.",
            parameters={
                "keywords": keywords,
                "category": category,
                "use_profile": bool(memory.get("user_profile"))
            },
            confidence=0.85 if keywords or category else 0.6
        )
    
    def _plan_eligibility_check(
        self,
        memory: Dict[str, Any],
        user_intent: Optional[UserIntent] = None
    ) -> Plan:
        """Plan eligibility check action"""
        user_profile = memory.get("user_profile", {})
        schemes = memory.get("schemes", [])
        
        # Verify we have required data
        missing = self._identify_missing_fields(memory)
        if missing:
            return self._plan_information_gathering(missing, user_intent, memory)
        
        if not schemes:
            return Plan(
                action=PlanAction.FETCH_SCHEMES,
                priority=PlanPriority.HIGH,
                reasoning="Need to fetch schemes before checking eligibility",
                parameters={"for_eligibility_check": True},
                alternative_actions=[PlanAction.ASK_MISSING_INFO],
                confidence=0.9
            )
        
        return Plan(
            action=PlanAction.CALL_ELIGIBILITY_TOOL,
            priority=PlanPriority.HIGH,
            reasoning="User profile and schemes available. Ready to check eligibility.",
            parameters={
                "user_profile": user_profile,
                "schemes": schemes,
                "check_all": True
            },
            confidence=0.95
        )
    
    def _plan_application_guidance(
        self,
        memory: Dict[str, Any],
        user_intent: Optional[UserIntent] = None
    ) -> Plan:
        """Plan application guidance action"""
        eligible_schemes = memory.get("eligible_schemes", [])
        
        if not eligible_schemes:
            return Plan(
                action=PlanAction.CALL_ELIGIBILITY_TOOL,
                priority=PlanPriority.HIGH,
                reasoning="Need to check eligibility before providing application guidance",
                parameters={},
                confidence=0.9
            )
        
        # Select scheme to guide
        selected_scheme = self._select_scheme_for_guidance(
            eligible_schemes,
            user_intent
        )
        
        return Plan(
            action=PlanAction.PROVIDE_APPLICATION_GUIDANCE,
            priority=PlanPriority.HIGH,
            reasoning=(
                f"User eligible for {len(eligible_schemes)} scheme(s). "
                "Providing application guidance."
            ),
            parameters={
                "eligible_schemes": eligible_schemes,
                "selected_scheme": selected_scheme,
                "user_wants_application": True
            },
            confidence=0.9
        )
    
    def _identify_missing_fields(
        self,
        memory: Dict[str, Any]
    ) -> List[str]:
        """Identify missing fields in user profile"""
        user_profile = memory.get("user_profile", {})
        
        missing = []
        for field in self.REQUIRED_ELIGIBILITY_FIELDS:
            if field not in user_profile or user_profile[field] is None:
                missing.append(field)
        
        return missing
    
    def _has_schemes_data(self, memory: Dict[str, Any]) -> bool:
        """Check if schemes data exists in memory"""
        schemes = memory.get("schemes", [])
        return len(schemes) > 0
    
    def _has_eligibility_results(self, memory: Dict[str, Any]) -> bool:
        """Check if eligibility results exist"""
        return "eligible_schemes" in memory or "eligibility_results" in memory
    
    def _extract_search_keywords_from_profile(
        self,
        user_profile: Dict[str, Any]
    ) -> List[str]:
        """Extract search keywords from user profile"""
        keywords = []
        
        # Add category
        if "category" in user_profile:
            keywords.append(user_profile["category"])
        
        # Add occupation
        if "occupation" in user_profile:
            keywords.append(user_profile["occupation"])
        
        # Add special attributes
        if user_profile.get("farmer"):
            keywords.append("farmer")
        
        if user_profile.get("disability"):
            keywords.append("disability")
        
        if user_profile.get("widow"):
            keywords.append("widow")
        
        return keywords
    
    def _select_scheme_for_guidance(
        self,
        eligible_schemes: List[Dict[str, Any]],
        user_intent: Optional[UserIntent]
    ) -> Optional[Dict[str, Any]]:
        """Select best scheme for application guidance"""
        if not eligible_schemes:
            return None
        
        # If user specified a scheme
        if user_intent and "scheme_name" in user_intent.entities:
            scheme_name = user_intent.entities["scheme_name"]
            for scheme in eligible_schemes:
                if scheme_name.lower() in scheme.get("name", "").lower():
                    return scheme
        
        # Return highest match score
        return max(
            eligible_schemes,
            key=lambda s: s.get("match_score", 0),
            default=eligible_schemes[0]
        )
    
    def get_plan_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get planning history"""
        history = [p.to_dict() for p in self.plan_history]
        if limit:
            return history[-limit:]
        return history
    
    def add_to_history(self, plan: Plan):
        """Add plan to history"""
        self.plan_history.append(plan)
    
    def clear_history(self):
        """Clear planning history"""
        self.plan_history.clear()


# Convenience function
def plan_action(
    user_intent: Optional[Dict[str, Any]] = None,
    memory_snapshot: Optional[Dict[str, Any]] = None,
    contradiction_reports: Optional[List[Dict[str, Any]]] = None,
    last_evaluation: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for planning
    
    Args:
        user_intent: User intent dictionary
        memory_snapshot: Memory snapshot
        contradiction_reports: Contradiction reports
        last_evaluation: Last evaluation result
        
    Returns:
        Plan dictionary
    """
    planner = Planner()
    
    # Convert dictionaries to objects if needed
    intent_obj = None
    if user_intent:
        intent_obj = UserIntent(
            intent_type=user_intent.get("intent_type", "general_query"),
            entities=user_intent.get("entities", {}),
            confidence=user_intent.get("confidence", 1.0),
            raw_text=user_intent.get("raw_text")
        )
    
    eval_obj = None
    if last_evaluation:
        eval_obj = EvaluationResult(
            success=last_evaluation.get("success", False),
            task_complete=last_evaluation.get("task_complete", False),
            issues=last_evaluation.get("issues", []),
            recommendations=last_evaluation.get("recommendations", []),
            missing_information=last_evaluation.get("missing_information", [])
        )
    
    plan = planner.plan_next_action(
        user_intent=intent_obj,
        memory_snapshot=memory_snapshot,
        contradiction_reports=contradiction_reports,
        last_evaluation=eval_obj
    )
    
    planner.add_to_history(plan)
    
    return plan.to_dict()