"""
session_memory.py
Session-level memory storage for voice-based agentic system.
Stores user profile, conversation history, and session state.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
import json


@dataclass
class UserProfile:
    """User profile attributes for eligibility determination."""
    age: Optional[int] = None
    annual_income: Optional[float] = None
    location: Optional[str] = None
    district: Optional[str] = None
    state: Optional[str] = None
    gender: Optional[str] = None
    category: Optional[str] = None  # SC, ST, OBC, General, etc.
    caste_certificate: Optional[bool] = None
    disability_status: Optional[bool] = None
    disability_percentage: Optional[int] = None
    land_ownership: Optional[bool] = None
    land_area_acres: Optional[float] = None
    family_size: Optional[int] = None
    bpl_status: Optional[bool] = None
    has_bank_account: Optional[bool] = None
    aadhaar_linked: Optional[bool] = None
    occupation: Optional[str] = None
    residence_duration_years: Optional[int] = None
    residence_type: Optional[str] = None  # Rural, Urban
    ration_card_type: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def update(self, **kwargs) -> None:
        """Update profile attributes."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)


@dataclass
class ConversationTurn:
    """Single turn in conversation history."""
    turn_id: int
    timestamp: str
    user_input_marathi: str
    agent_response_marathi: str
    user_input_english: Optional[str] = None
    agent_response_english: Optional[str] = None
    extracted_info: Dict[str, Any] = field(default_factory=dict)
    tools_used: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert turn to dictionary."""
        return asdict(self)


class SessionMemory:
    """
    Session memory manager for voice-based agent.
    Maintains user profile and conversation history.
    """
    
    def __init__(self):
        """Initialize empty session memory."""
        self.session_id: str = self._generate_session_id()
        self.session_start: str = datetime.now().isoformat()
        self.user_profile: UserProfile = UserProfile()
        self.conversation_history: List[ConversationTurn] = []
        self.extracted_facts: Dict[str, Any] = {}
        self.contradictions: List[Dict[str, Any]] = []
        self.turn_counter: int = 0
        self.current_schemes_explored: List[str] = []
        self.eligible_schemes: List[str] = []
        self.ineligible_schemes: List[Dict[str, Any]] = []
        self.application_status: Dict[str, str] = {}
        
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        from uuid import uuid4
        return f"session_{uuid4().hex[:12]}"
    
    # ==================== USER PROFILE METHODS ====================
    
    def update_profile(self, **kwargs) -> None:
        """
        Update user profile attributes.
        
        Args:
            **kwargs: Profile attributes to update
        """
        self.user_profile.update(**kwargs)
        
    def get_profile(self) -> Dict[str, Any]:
        """
        Get current user profile.
        
        Returns:
            Dictionary of profile attributes
        """
        return self.user_profile.to_dict()
    
    def get_profile_attribute(self, attribute: str) -> Any:
        """
        Get specific profile attribute.
        
        Args:
            attribute: Name of the attribute
            
        Returns:
            Attribute value or None
        """
        return getattr(self.user_profile, attribute, None)
    
    def has_profile_attribute(self, attribute: str) -> bool:
        """
        Check if profile attribute is set.
        
        Args:
            attribute: Name of the attribute
            
        Returns:
            True if attribute exists and is not None
        """
        value = getattr(self.user_profile, attribute, None)
        return value is not None
    
    def get_missing_profile_attributes(self, required_attributes: List[str]) -> List[str]:
        """
        Get list of missing required attributes.
        
        Args:
            required_attributes: List of attribute names
            
        Returns:
            List of missing attribute names
        """
        return [
            attr for attr in required_attributes 
            if not self.has_profile_attribute(attr)
        ]
    
    # ==================== CONVERSATION HISTORY METHODS ====================
    
    def add_turn(
        self,
        user_input_marathi: str,
        agent_response_marathi: str,
        user_input_english: Optional[str] = None,
        agent_response_english: Optional[str] = None,
        extracted_info: Optional[Dict[str, Any]] = None,
        tools_used: Optional[List[str]] = None
    ) -> int:
        """
        Add new conversation turn.
        
        Args:
            user_input_marathi: User's input in Marathi
            agent_response_marathi: Agent's response in Marathi
            user_input_english: Optional English translation
            agent_response_english: Optional English translation
            extracted_info: Optional extracted information
            tools_used: Optional list of tools used
            
        Returns:
            Turn ID
        """
        self.turn_counter += 1
        
        turn = ConversationTurn(
            turn_id=self.turn_counter,
            timestamp=datetime.now().isoformat(),
            user_input_marathi=user_input_marathi,
            user_input_english=user_input_english,
            agent_response_marathi=agent_response_marathi,
            agent_response_english=agent_response_english,
            extracted_info=extracted_info or {},
            tools_used=tools_used or []
        )
        
        self.conversation_history.append(turn)
        return self.turn_counter
    
    def get_conversation_history(self, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history.
        
        Args:
            last_n: Optional number of recent turns to retrieve
            
        Returns:
            List of conversation turns as dictionaries
        """
        history = self.conversation_history
        if last_n is not None:
            history = history[-last_n:]
        return [turn.to_dict() for turn in history]
    
    def get_last_turn(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent conversation turn.
        
        Returns:
            Last turn as dictionary or None
        """
        if self.conversation_history:
            return self.conversation_history[-1].to_dict()
        return None
    
    def get_conversation_context(self, last_n: int = 3) -> str:
        """
        Get formatted conversation context for prompt.
        
        Args:
            last_n: Number of recent turns to include
            
        Returns:
            Formatted context string
        """
        recent_turns = self.conversation_history[-last_n:]
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_input_marathi}")
            context_parts.append(f"Agent: {turn.agent_response_marathi}")
        
        return "\n".join(context_parts)
    
    # ==================== EXTRACTED FACTS METHODS ====================
    
    def add_fact(self, key: str, value: Any, source_turn: Optional[int] = None) -> None:
        """
        Add extracted fact to memory.
        
        Args:
            key: Fact identifier
            value: Fact value
            source_turn: Optional turn ID where fact was extracted
        """
        self.extracted_facts[key] = {
            "value": value,
            "source_turn": source_turn or self.turn_counter,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_fact(self, key: str) -> Any:
        """
        Get extracted fact value.
        
        Args:
            key: Fact identifier
            
        Returns:
            Fact value or None
        """
        fact_data = self.extracted_facts.get(key)
        return fact_data["value"] if fact_data else None
    
    def has_fact(self, key: str) -> bool:
        """
        Check if fact exists.
        
        Args:
            key: Fact identifier
            
        Returns:
            True if fact exists
        """
        return key in self.extracted_facts
    
    def get_all_facts(self) -> Dict[str, Any]:
        """
        Get all extracted facts.
        
        Returns:
            Dictionary of all facts
        """
        return self.extracted_facts.copy()
    
    # ==================== CONTRADICTION TRACKING ====================
    
    def record_contradiction(
        self,
        attribute: str,
        old_value: Any,
        new_value: Any,
        old_turn: int,
        new_turn: int
    ) -> None:
        """
        Record contradiction between two values.
        
        Args:
            attribute: Attribute name
            old_value: Previous value
            new_value: New conflicting value
            old_turn: Turn where old value was stated
            new_turn: Turn where new value was stated
        """
        self.contradictions.append({
            "attribute": attribute,
            "old_value": old_value,
            "new_value": new_value,
            "old_turn": old_turn,
            "new_turn": new_turn,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_contradictions(self) -> List[Dict[str, Any]]:
        """
        Get all recorded contradictions.
        
        Returns:
            List of contradiction records
        """
        return self.contradictions.copy()
    
    def has_contradictions(self) -> bool:
        """
        Check if any contradictions exist.
        
        Returns:
            True if contradictions exist
        """
        return len(self.contradictions) > 0
    
    # ==================== SCHEME TRACKING ====================
    
    def add_explored_scheme(self, scheme_id: str) -> None:
        """Add scheme to explored list."""
        if scheme_id not in self.current_schemes_explored:
            self.current_schemes_explored.append(scheme_id)
    
    def add_eligible_scheme(self, scheme_id: str) -> None:
        """Add scheme to eligible list."""
        if scheme_id not in self.eligible_schemes:
            self.eligible_schemes.append(scheme_id)
    
    def add_ineligible_scheme(self, scheme_id: str, reasons: List[str]) -> None:
        """Add scheme to ineligible list with reasons."""
        self.ineligible_schemes.append({
            "scheme_id": scheme_id,
            "reasons": reasons,
            "timestamp": datetime.now().isoformat()
        })
    
    def set_application_status(self, scheme_id: str, status: str) -> None:
        """
        Set application status for a scheme.
        
        Args:
            scheme_id: Scheme identifier
            status: Status string (e.g., 'initiated', 'completed', 'pending')
        """
        self.application_status[scheme_id] = status
    
    def get_application_status(self, scheme_id: str) -> Optional[str]:
        """Get application status for a scheme."""
        return self.application_status.get(scheme_id)
    
    def get_eligible_schemes(self) -> List[str]:
        """Get list of eligible schemes."""
        return self.eligible_schemes.copy()
    
    def get_ineligible_schemes(self) -> List[Dict[str, Any]]:
        """Get list of ineligible schemes with reasons."""
        return self.ineligible_schemes.copy()
    
    # ==================== SESSION EXPORT/IMPORT ====================
    
    def export_session(self) -> Dict[str, Any]:
        """
        Export complete session state.
        
        Returns:
            Dictionary containing all session data
        """
        return {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "user_profile": self.user_profile.to_dict(),
            "conversation_history": self.get_conversation_history(),
            "extracted_facts": self.extracted_facts,
            "contradictions": self.contradictions,
            "current_schemes_explored": self.current_schemes_explored,
            "eligible_schemes": self.eligible_schemes,
            "ineligible_schemes": self.ineligible_schemes,
            "application_status": self.application_status,
            "turn_counter": self.turn_counter
        }
    
    def export_to_json(self, filepath: str) -> None:
        """
        Export session to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.export_session(), f, ensure_ascii=False, indent=2)
    
    def clear_session(self) -> None:
        """Clear all session data and reset."""
        self.__init__()
    
    # ==================== UTILITY METHODS ====================
    
    def get_session_summary(self) -> Dict[str, Any]:
        """
        Get high-level session summary.
        
        Returns:
            Summary dictionary
        """
        return {
            "session_id": self.session_id,
            "session_start": self.session_start,
            "total_turns": self.turn_counter,
            "profile_completeness": self._calculate_profile_completeness(),
            "schemes_explored": len(self.current_schemes_explored),
            "eligible_schemes_count": len(self.eligible_schemes),
            "ineligible_schemes_count": len(self.ineligible_schemes),
            "contradictions_count": len(self.contradictions),
            "has_profile_data": bool(self.user_profile.to_dict())
        }
    
    def _calculate_profile_completeness(self) -> float:
        """
        Calculate profile completeness percentage.
        
        Returns:
            Completeness as percentage (0-100)
        """
        all_attributes = [
            attr for attr in dir(self.user_profile)
            if not attr.startswith('_') and not callable(getattr(self.user_profile, attr))
        ]
        
        filled_attributes = sum(
            1 for attr in all_attributes
            if getattr(self.user_profile, attr, None) is not None
        )
        
        if not all_attributes:
            return 0.0
        
        return (filled_attributes / len(all_attributes)) * 100