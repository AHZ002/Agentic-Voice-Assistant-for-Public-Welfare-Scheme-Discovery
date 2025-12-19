"""
Agent Action Executor

This module executes actions decided by the planner. It calls appropriate
tools based on the plan and returns structured execution results.

Executor DOES NOT:
- Decide what to do (planner's job)
- Speak to user
- Maintain state
- Implement retry/recovery logic
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import tools
try:
    from tools.eligibility import (
        EligibilityEngine,
        UserProfile,
        EligibilityCriteria,
        evaluate_eligibility
    )
    ELIGIBILITY_AVAILABLE = True
except ImportError:
    ELIGIBILITY_AVAILABLE = False

try:
    from tools.scheme_retriever import (
        SchemeRetriever,
        retrieve_schemes
    )
    RETRIEVER_AVAILABLE = True
except ImportError:
    RETRIEVER_AVAILABLE = False

try:
    from tools.mock_gov_api import (
        MockGovAPI,
        submit_application,
        check_application_status
    )
    GOV_API_AVAILABLE = True
except ImportError:
    GOV_API_AVAILABLE = False


class ExecutionStatus(Enum):
    """Execution status enumeration"""
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    TOOL_ERROR = "tool_error"
    INVALID_PARAMETERS = "invalid_parameters"


@dataclass
class ExecutionResult:
    """Structured execution result"""
    action: str
    status: ExecutionStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tool_used: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "action": self.action,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "tool_used": self.tool_used,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


class ToolExecutionError(Exception):
    """Tool execution error"""
    pass


class Executor:
    """
    Stateless action executor - executes plans by calling tools
    """
    
    def __init__(
        self,
        schemes_file_path: Optional[str] = None,
        gov_api_instance: Optional[Any] = None
    ):
        """
        Initialize executor
        
        Args:
            schemes_file_path: Path to schemes.json file
            gov_api_instance: Optional MockGovAPI instance
        """
        self.schemes_file_path = schemes_file_path
        self.gov_api = gov_api_instance
        
        # Initialize tools lazily
        self._eligibility_engine = None
        self._scheme_retriever = None
        
        # Check tool availability
        self._check_tool_availability()
    
    def _check_tool_availability(self):
        """Check which tools are available"""
        if not ELIGIBILITY_AVAILABLE:
            print("Warning: Eligibility tool not available")
        
        if not RETRIEVER_AVAILABLE:
            print("Warning: Scheme retriever not available")
        
        if not GOV_API_AVAILABLE:
            print("Warning: Government API not available")
    
    def _get_eligibility_engine(self) -> EligibilityEngine:
        """Get or create eligibility engine"""
        if self._eligibility_engine is None:
            if not ELIGIBILITY_AVAILABLE:
                raise ToolExecutionError("Eligibility engine not available")
            self._eligibility_engine = EligibilityEngine()
        return self._eligibility_engine
    
    def _get_scheme_retriever(self) -> SchemeRetriever:
        """Get or create scheme retriever"""
        if self._scheme_retriever is None:
            if not RETRIEVER_AVAILABLE:
                raise ToolExecutionError("Scheme retriever not available")
            self._scheme_retriever = SchemeRetriever(self.schemes_file_path)
        return self._scheme_retriever
    
    def _get_gov_api(self) -> MockGovAPI:
        """Get or create government API instance"""
        if self.gov_api is None:
            if not GOV_API_AVAILABLE:
                raise ToolExecutionError("Government API not available")
            self.gov_api = MockGovAPI()
        return self.gov_api
    
    def execute_plan(self, plan: Dict[str, Any]) -> ExecutionResult:
        """
        Execute a plan from the planner
        
        Args:
            plan: Plan dictionary with 'action' and 'parameters'
            
        Returns:
            ExecutionResult object
        """
        action = plan.get("action")
        parameters = plan.get("parameters", {})
        
        if not action:
            return ExecutionResult(
                action="unknown",
                status=ExecutionStatus.INVALID_PARAMETERS,
                error="No action specified in plan"
            )
        
        # Route to appropriate executor
        start_time = datetime.now()
        
        try:
            if action == "fetch_schemes":
                result = self._execute_fetch_schemes(parameters)
            
            elif action == "call_eligibility_tool":
                result = self._execute_eligibility_check(parameters)
            
            elif action == "provide_application_guidance":
                result = self._execute_application_guidance(parameters)
            
            elif action == "ask_missing_info":
                result = self._execute_ask_missing_info(parameters)
            
            elif action == "handle_contradiction":
                result = self._execute_handle_contradiction(parameters)
            
            elif action == "end_task":
                result = self._execute_end_task(parameters)
            
            else:
                result = ExecutionResult(
                    action=action,
                    status=ExecutionStatus.INVALID_PARAMETERS,
                    error=f"Unknown action: {action}"
                )
            
            # Calculate execution time
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            
            return result
        
        except Exception as e:
            return ExecutionResult(
                action=action,
                status=ExecutionStatus.TOOL_ERROR,
                error=f"Execution error: {str(e)}",
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    def _execute_fetch_schemes(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute scheme retrieval
        
        Args:
            parameters: Dict with keywords, category, tags, etc.
            
        Returns:
            ExecutionResult
        """
        try:
            retriever = self._get_scheme_retriever()
            
            keywords = parameters.get("keywords", [])
            category = parameters.get("category")
            tags = parameters.get("tags", [])
            max_results = parameters.get("max_results", 10)
            
            # Determine search method
            if keywords:
                results = retriever.search_by_keywords(keywords, max_results)
            elif tags:
                results = retriever.search_by_tags(tags, max_results=max_results)
            elif category:
                results = retriever.search_by_category(category, max_results)
            else:
                # Get all schemes
                all_schemes = retriever.get_all_schemes()
                results = all_schemes[:max_results]
                # Convert to proper format
                if results and not isinstance(results[0], dict):
                    results = [r.to_dict() if hasattr(r, 'to_dict') else r for r in results]
            
            # Convert results to dictionaries
            if results and hasattr(results[0], 'to_dict'):
                schemes = [r.to_dict() for r in results]
            else:
                schemes = results
            
            return ExecutionResult(
                action="fetch_schemes",
                status=ExecutionStatus.SUCCESS,
                output={
                    "schemes": schemes,
                    "total_found": len(schemes),
                    "search_criteria": {
                        "keywords": keywords,
                        "category": category,
                        "tags": tags
                    }
                },
                tool_used="scheme_retriever",
                metadata={"retrieval_method": "keyword" if keywords else "category" if category else "all"}
            )
        
        except Exception as e:
            return ExecutionResult(
                action="fetch_schemes",
                status=ExecutionStatus.TOOL_ERROR,
                error=f"Scheme retrieval failed: {str(e)}",
                tool_used="scheme_retriever"
            )
    
    def _execute_eligibility_check(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute eligibility check
        
        Args:
            parameters: Dict with user_profile and schemes
            
        Returns:
            ExecutionResult
        """
        try:
            engine = self._get_eligibility_engine()
            
            user_profile_data = parameters.get("user_profile", {})
            schemes = parameters.get("schemes", [])
            
            if not user_profile_data:
                return ExecutionResult(
                    action="call_eligibility_tool",
                    status=ExecutionStatus.INVALID_PARAMETERS,
                    error="No user profile provided",
                    tool_used="eligibility_engine"
                )
            
            if not schemes:
                return ExecutionResult(
                    action="call_eligibility_tool",
                    status=ExecutionStatus.INVALID_PARAMETERS,
                    error="No schemes provided for eligibility check",
                    tool_used="eligibility_engine"
                )
            
            # Create user profile
            user_profile = UserProfile(**user_profile_data)
            
            # Run eligibility check
            results = engine.evaluate_multiple_schemes(user_profile, schemes)
            
            return ExecutionResult(
                action="call_eligibility_tool",
                status=ExecutionStatus.SUCCESS,
                output=results,
                tool_used="eligibility_engine",
                metadata={
                    "total_schemes_checked": results.get("total_evaluated", 0),
                    "eligible_count": len(results.get("eligible_schemes", [])),
                    "profile_completeness": results.get("user_profile_completeness", 0)
                }
            )
        
        except Exception as e:
            return ExecutionResult(
                action="call_eligibility_tool",
                status=ExecutionStatus.TOOL_ERROR,
                error=f"Eligibility check failed: {str(e)}",
                tool_used="eligibility_engine"
            )
    
    def _execute_application_guidance(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute application guidance (may involve gov API)
        
        Args:
            parameters: Dict with eligible_schemes, selected_scheme
            
        Returns:
            ExecutionResult
        """
        try:
            eligible_schemes = parameters.get("eligible_schemes", [])
            selected_scheme = parameters.get("selected_scheme")
            submit_application_flag = parameters.get("submit_application", False)
            
            if not eligible_schemes:
                return ExecutionResult(
                    action="provide_application_guidance",
                    status=ExecutionStatus.INVALID_PARAMETERS,
                    error="No eligible schemes provided",
                    tool_used="none"
                )
            
            # Prepare guidance output
            guidance = {
                "eligible_schemes": eligible_schemes,
                "recommended_scheme": selected_scheme or eligible_schemes[0],
                "total_eligible": len(eligible_schemes)
            }
            
            # If application submission requested
            if submit_application_flag and selected_scheme:
                gov_api = self._get_gov_api()
                user_data = parameters.get("user_profile", {})
                
                # Submit application
                response = gov_api.submit_application(
                    scheme_id=selected_scheme.get("scheme_id"),
                    user_data=user_data
                )
                
                guidance["application_submitted"] = response.success
                guidance["application_response"] = response.to_dict()
            
            return ExecutionResult(
                action="provide_application_guidance",
                status=ExecutionStatus.SUCCESS,
                output=guidance,
                tool_used="gov_api" if submit_application_flag else "none",
                metadata={
                    "guidance_type": "application_submission" if submit_application_flag else "information"
                }
            )
        
        except Exception as e:
            return ExecutionResult(
                action="provide_application_guidance",
                status=ExecutionStatus.TOOL_ERROR,
                error=f"Application guidance failed: {str(e)}",
                tool_used="gov_api"
            )
    
    def _execute_ask_missing_info(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute ask missing info (no tool call, just structure output)
        
        Args:
            parameters: Dict with missing_fields
            
        Returns:
            ExecutionResult
        """
        missing_fields = parameters.get("missing_fields", [])
        all_missing = parameters.get("all_missing", missing_fields)
        context = parameters.get("context", "general")
        
        return ExecutionResult(
            action="ask_missing_info",
            status=ExecutionStatus.SUCCESS,
            output={
                "missing_fields": missing_fields,
                "all_missing_fields": all_missing,
                "context": context,
                "action_required": "gather_information"
            },
            tool_used="none",
            metadata={
                "field_count": len(missing_fields),
                "context": context
            }
        )
    
    def _execute_handle_contradiction(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute contradiction handling (no tool call)
        
        Args:
            parameters: Dict with contradictions, fields_to_clarify
            
        Returns:
            ExecutionResult
        """
        contradictions = parameters.get("contradictions", [])
        fields = parameters.get("fields_to_clarify", [])
        
        return ExecutionResult(
            action="handle_contradiction",
            status=ExecutionStatus.SUCCESS,
            output={
                "contradictions": contradictions,
                "fields_to_clarify": fields,
                "action_required": "resolve_contradictions"
            },
            tool_used="none",
            metadata={
                "contradiction_count": len(contradictions),
                "fields_count": len(fields)
            }
        )
    
    def _execute_end_task(self, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute task completion (no tool call)
        
        Args:
            parameters: Dict with completion_status, evaluation
            
        Returns:
            ExecutionResult
        """
        completion_status = parameters.get("completion_status", "success")
        evaluation = parameters.get("evaluation", {})
        
        return ExecutionResult(
            action="end_task",
            status=ExecutionStatus.SUCCESS,
            output={
                "task_complete": True,
                "completion_status": completion_status,
                "evaluation": evaluation
            },
            tool_used="none",
            metadata={
                "completion_status": completion_status
            }
        )
    
    def execute_tool_directly(
        self,
        tool_name: str,
        method_name: str,
        **kwargs
    ) -> ExecutionResult:
        """
        Execute a tool method directly (for testing or special cases)
        
        Args:
            tool_name: Name of tool ('eligibility', 'retriever', 'gov_api')
            method_name: Method to call
            **kwargs: Method arguments
            
        Returns:
            ExecutionResult
        """
        try:
            if tool_name == "eligibility":
                engine = self._get_eligibility_engine()
                method = getattr(engine, method_name)
                output = method(**kwargs)
            
            elif tool_name == "retriever":
                retriever = self._get_scheme_retriever()
                method = getattr(retriever, method_name)
                output = method(**kwargs)
            
            elif tool_name == "gov_api":
                api = self._get_gov_api()
                method = getattr(api, method_name)
                output = method(**kwargs)
            
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Convert output to dict if possible
            if hasattr(output, 'to_dict'):
                output = output.to_dict()
            
            return ExecutionResult(
                action=f"{tool_name}.{method_name}",
                status=ExecutionStatus.SUCCESS,
                output=output,
                tool_used=tool_name
            )
        
        except Exception as e:
            return ExecutionResult(
                action=f"{tool_name}.{method_name}",
                status=ExecutionStatus.TOOL_ERROR,
                error=str(e),
                tool_used=tool_name
            )
    
    def get_tool_status(self) -> Dict[str, bool]:
        """
        Get availability status of all tools
        
        Returns:
            Dict mapping tool names to availability
        """
        return {
            "eligibility_engine": ELIGIBILITY_AVAILABLE,
            "scheme_retriever": RETRIEVER_AVAILABLE,
            "gov_api": GOV_API_AVAILABLE
        }


# Convenience function
def execute_action(
    action: str,
    parameters: Dict[str, Any],
    schemes_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function for executing actions
    
    Args:
        action: Action name
        parameters: Action parameters
        schemes_file_path: Path to schemes file
        
    Returns:
        Execution result dictionary
    """
    executor = Executor(schemes_file_path=schemes_file_path)
    
    plan = {
        "action": action,
        "parameters": parameters
    }
    
    result = executor.execute_plan(plan)
    return result.to_dict()