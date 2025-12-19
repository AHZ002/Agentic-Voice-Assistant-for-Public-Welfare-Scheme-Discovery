"""
Finite State Machine for Voice-Based Service Agent

This module implements a deterministic state machine to manage agent states
and transitions. It tracks state history, supports retries, and enforces
valid state transitions.

NO user interaction
NO LLM calls
NO business logic
Pure state management only
"""

from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class AgentState(Enum):
    """Agent state enumeration"""
    IDLE = "idle"
    LISTENING = "listening"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    ASKING_CLARIFICATION = "asking_clarification"
    ERROR_RECOVERY = "error_recovery"
    COMPLETED = "completed"
    FAILED = "failed"


class TransitionError(Exception):
    """Invalid state transition error"""
    pass


class RetryLimitExceededError(Exception):
    """Retry limit exceeded"""
    pass


@dataclass
class StateTransition:
    """Record of a state transition"""
    from_state: AgentState
    to_state: AgentState
    timestamp: str
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "metadata": self.metadata
        }


@dataclass
class StateContext:
    """Context information for current state"""
    data: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "data": self.data,
            "retry_count": self.retry_count,
            "error_count": self.error_count,
            "last_error": self.last_error
        }


class AgentStateMachine:
    """
    Finite State Machine for agent state management
    """
    
    # Define valid state transitions
    VALID_TRANSITIONS: Dict[AgentState, Set[AgentState]] = {
        AgentState.IDLE: {
            AgentState.LISTENING,
            AgentState.COMPLETED,
            AgentState.FAILED
        },
        AgentState.LISTENING: {
            AgentState.PLANNING,
            AgentState.ASKING_CLARIFICATION,
            AgentState.ERROR_RECOVERY,
            AgentState.IDLE
        },
        AgentState.PLANNING: {
            AgentState.EXECUTING,
            AgentState.ASKING_CLARIFICATION,
            AgentState.ERROR_RECOVERY,
            AgentState.LISTENING
        },
        AgentState.EXECUTING: {
            AgentState.EVALUATING,
            AgentState.ERROR_RECOVERY,
            AgentState.PLANNING
        },
        AgentState.EVALUATING: {
            AgentState.COMPLETED,
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.ASKING_CLARIFICATION,
            AgentState.ERROR_RECOVERY
        },
        AgentState.ASKING_CLARIFICATION: {
            AgentState.LISTENING,
            AgentState.ERROR_RECOVERY,
            AgentState.FAILED
        },
        AgentState.ERROR_RECOVERY: {
            AgentState.LISTENING,
            AgentState.PLANNING,
            AgentState.EXECUTING,
            AgentState.ASKING_CLARIFICATION,
            AgentState.FAILED
        },
        AgentState.COMPLETED: {
            AgentState.IDLE,
            AgentState.LISTENING
        },
        AgentState.FAILED: {
            AgentState.IDLE,
            AgentState.ERROR_RECOVERY
        }
    }
    
    # Maximum retry counts per state
    DEFAULT_MAX_RETRIES = {
        AgentState.LISTENING: 3,
        AgentState.PLANNING: 3,
        AgentState.EXECUTING: 5,
        AgentState.EVALUATING: 3,
        AgentState.ERROR_RECOVERY: 3,
        AgentState.ASKING_CLARIFICATION: 2
    }
    
    def __init__(
        self,
        initial_state: AgentState = AgentState.IDLE,
        max_retries: Optional[Dict[AgentState, int]] = None,
        max_error_count: int = 10
    ):
        """
        Initialize state machine
        
        Args:
            initial_state: Starting state
            max_retries: Custom retry limits per state
            max_error_count: Maximum total errors before failure
        """
        self.current_state = initial_state
        self.previous_state: Optional[AgentState] = None
        
        # State history
        self.state_history: List[StateTransition] = []
        
        # Context per state
        self.context: Dict[AgentState, StateContext] = {
            state: StateContext() for state in AgentState
        }
        
        # Retry configuration
        self.max_retries = max_retries or self.DEFAULT_MAX_RETRIES.copy()
        self.max_error_count = max_error_count
        
        # Counters
        self.total_transitions = 0
        self.total_errors = 0
        
        # Lock for thread safety (optional)
        self._locked = False
    
    def can_transition(self, to_state: AgentState) -> bool:
        """
        Check if transition to target state is valid
        
        Args:
            to_state: Target state
            
        Returns:
            True if transition is valid
        """
        valid_next_states = self.VALID_TRANSITIONS.get(self.current_state, set())
        return to_state in valid_next_states
    
    def get_valid_transitions(self) -> Set[AgentState]:
        """
        Get set of valid transitions from current state
        
        Returns:
            Set of valid next states
        """
        return self.VALID_TRANSITIONS.get(self.current_state, set()).copy()
    
    def transition(
        self,
        to_state: AgentState,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Transition to a new state
        
        Args:
            to_state: Target state
            reason: Reason for transition
            metadata: Additional metadata
            
        Returns:
            True if transition successful
            
        Raises:
            TransitionError: If transition is invalid
        """
        # Check if transition is valid
        if not self.can_transition(to_state):
            raise TransitionError(
                f"Invalid transition from {self.current_state.value} "
                f"to {to_state.value}. "
                f"Valid transitions: {[s.value for s in self.get_valid_transitions()]}"
            )
        
        # Record transition
        transition = StateTransition(
            from_state=self.current_state,
            to_state=to_state,
            timestamp=datetime.now().isoformat(),
            reason=reason,
            metadata=metadata or {}
        )
        
        self.state_history.append(transition)
        
        # Update states
        self.previous_state = self.current_state
        self.current_state = to_state
        
        # Increment counter
        self.total_transitions += 1
        
        # Reset retry count when moving to different state
        if self.previous_state != self.current_state:
            self.context[self.current_state].retry_count = 0
        
        return True
    
    def retry_current_state(self, reason: Optional[str] = None) -> bool:
        """
        Retry current state (transition to same state)
        
        Args:
            reason: Reason for retry
            
        Returns:
            True if retry allowed
            
        Raises:
            RetryLimitExceededError: If retry limit exceeded
        """
        # Check retry limit
        max_retries = self.max_retries.get(self.current_state, 3)
        current_retries = self.context[self.current_state].retry_count
        
        if current_retries >= max_retries:
            raise RetryLimitExceededError(
                f"Retry limit exceeded for state {self.current_state.value}: "
                f"{current_retries}/{max_retries}"
            )
        
        # Increment retry count
        self.context[self.current_state].retry_count += 1
        
        # Record transition to same state
        transition = StateTransition(
            from_state=self.current_state,
            to_state=self.current_state,
            timestamp=datetime.now().isoformat(),
            reason=f"retry_{current_retries + 1}: {reason or 'retry'}",
            metadata={"retry_count": current_retries + 1}
        )
        
        self.state_history.append(transition)
        self.total_transitions += 1
        
        return True
    
    def record_error(
        self,
        error_message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record an error in current state
        
        Args:
            error_message: Error description
            metadata: Additional error metadata
        """
        self.context[self.current_state].error_count += 1
        self.context[self.current_state].last_error = error_message
        self.total_errors += 1
        
        # Store error in context
        if "errors" not in self.context[self.current_state].data:
            self.context[self.current_state].data["errors"] = []
        
        self.context[self.current_state].data["errors"].append({
            "message": error_message,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        })
    
    def should_fail(self) -> bool:
        """
        Check if agent should transition to FAILED state
        
        Returns:
            True if failure conditions met
        """
        return self.total_errors >= self.max_error_count
    
    def get_state_context(self, state: Optional[AgentState] = None) -> StateContext:
        """
        Get context for a state
        
        Args:
            state: State to get context for (None for current state)
            
        Returns:
            StateContext object
        """
        target_state = state or self.current_state
        return self.context[target_state]
    
    def set_context_data(self, key: str, value: Any):
        """
        Set data in current state context
        
        Args:
            key: Data key
            value: Data value
        """
        self.context[self.current_state].data[key] = value
    
    def get_context_data(self, key: str, default: Any = None) -> Any:
        """
        Get data from current state context
        
        Args:
            key: Data key
            default: Default value if key not found
            
        Returns:
            Data value
        """
        return self.context[self.current_state].data.get(key, default)
    
    def clear_context(self, state: Optional[AgentState] = None):
        """
        Clear context for a state
        
        Args:
            state: State to clear (None for current state)
        """
        target_state = state or self.current_state
        self.context[target_state] = StateContext()
    
    def get_transition_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get state transition history
        
        Args:
            limit: Maximum number of recent transitions (None for all)
            
        Returns:
            List of transition dictionaries
        """
        history = [t.to_dict() for t in self.state_history]
        
        if limit:
            return history[-limit:]
        return history
    
    def get_state_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about state usage
        
        Returns:
            Dictionary with statistics
        """
        state_counts = {}
        state_durations = {}
        
        for transition in self.state_history:
            state = transition.from_state.value
            state_counts[state] = state_counts.get(state, 0) + 1
        
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "total_transitions": self.total_transitions,
            "total_errors": self.total_errors,
            "state_counts": state_counts,
            "retry_counts": {
                state.value: context.retry_count
                for state, context in self.context.items()
                if context.retry_count > 0
            },
            "error_counts": {
                state.value: context.error_count
                for state, context in self.context.items()
                if context.error_count > 0
            }
        }
    
    def reset(self, initial_state: AgentState = AgentState.IDLE):
        """
        Reset state machine to initial state
        
        Args:
            initial_state: State to reset to
        """
        self.current_state = initial_state
        self.previous_state = None
        self.state_history.clear()
        
        # Reset all contexts
        for state in AgentState:
            self.context[state] = StateContext()
        
        self.total_transitions = 0
        self.total_errors = 0
    
    def is_terminal_state(self) -> bool:
        """
        Check if current state is terminal
        
        Returns:
            True if in COMPLETED or FAILED state
        """
        return self.current_state in {AgentState.COMPLETED, AgentState.FAILED}
    
    def get_state_path(self) -> List[str]:
        """
        Get path of states traversed
        
        Returns:
            List of state names in order
        """
        path = []
        for transition in self.state_history:
            if not path or path[-1] != transition.from_state.value:
                path.append(transition.from_state.value)
            path.append(transition.to_state.value)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_path = []
        for state in path:
            if state not in seen:
                seen.add(state)
                unique_path.append(state)
        
        return unique_path
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export state machine state to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value if self.previous_state else None,
            "total_transitions": self.total_transitions,
            "total_errors": self.total_errors,
            "is_terminal": self.is_terminal_state(),
            "state_path": self.get_state_path(),
            "recent_transitions": self.get_transition_history(limit=10),
            "statistics": self.get_state_statistics(),
            "context": {
                state.value: context.to_dict()
                for state, context in self.context.items()
                if context.data or context.retry_count > 0 or context.error_count > 0
            }
        }


# Convenience function
def create_state_machine(
    initial_state: AgentState = AgentState.IDLE,
    max_retries: Optional[Dict[str, int]] = None
) -> AgentStateMachine:
    """
    Create a new state machine instance
    
    Args:
        initial_state: Starting state
        max_retries: Custom retry limits
        
    Returns:
        AgentStateMachine instance
    """
    # Convert string keys to AgentState if needed
    retry_config = None
    if max_retries:
        retry_config = {
            AgentState[k.upper()] if isinstance(k, str) else k: v
            for k, v in max_retries.items()
        }
    
    return AgentStateMachine(
        initial_state=initial_state,
        max_retries=retry_config
    )