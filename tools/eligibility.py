"""
Deterministic Eligibility Engine for Government Welfare Schemes

This module provides rule-based eligibility evaluation for government schemes.
It accepts structured user profile data and scheme criteria, then returns
structured eligibility results with clear rejection reasons.

NO ML/LLM usage - Pure logic-based evaluation
NO user-facing text generation
NO interactive questions
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class EligibilityStatus(Enum):
    """Eligibility status enumeration"""
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    PARTIAL = "partial"  # Missing information for complete evaluation


@dataclass
class UserProfile:
    """Structured user profile data"""
    age: Optional[int] = None
    income: Optional[float] = None  # Annual income in INR
    state: Optional[str] = None
    gender: Optional[str] = None  # "male", "female", "other"
    category: Optional[str] = None  # "general", "obc", "sc", "st", "ews"
    occupation: Optional[str] = None
    disability: Optional[bool] = None
    bpl_card: Optional[bool] = None  # Below Poverty Line card holder
    farmer: Optional[bool] = None
    widow: Optional[bool] = None
    minority: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing fields"""
        return [k for k, v in self.__dict__.items() if v is None]


@dataclass
class EligibilityCriteria:
    """Structured scheme eligibility criteria"""
    min_age: Optional[int] = None
    max_age: Optional[int] = None
    max_income: Optional[float] = None  # Annual income ceiling
    min_income: Optional[float] = None  # Annual income floor
    allowed_states: Optional[List[str]] = None  # None means all states
    allowed_genders: Optional[List[str]] = None  # None means all genders
    allowed_categories: Optional[List[str]] = None  # None means all categories
    required_occupation: Optional[str] = None
    requires_disability: Optional[bool] = None
    requires_bpl_card: Optional[bool] = None
    requires_farmer: Optional[bool] = None
    requires_widow: Optional[bool] = None
    requires_minority: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert criteria to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() if v is not None
        }


@dataclass
class EligibilityResult:
    """Structured eligibility evaluation result"""
    scheme_id: str
    scheme_name: str
    status: EligibilityStatus
    reasons: List[str]  # Rejection reasons or eligibility confirmations
    missing_info: List[str]  # Fields needed for complete evaluation
    match_score: float  # 0.0 to 1.0 - how well user matches criteria
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            "scheme_id": self.scheme_id,
            "scheme_name": self.scheme_name,
            "status": self.status.value,
            "reasons": self.reasons,
            "missing_info": self.missing_info,
            "match_score": self.match_score
        }


class EligibilityEngine:
    """
    Deterministic rule-based eligibility evaluation engine
    """
    
    def __init__(self):
        """Initialize the eligibility engine"""
        self.evaluation_history = []
    
    def evaluate_scheme(
        self,
        user_profile: UserProfile,
        scheme_id: str,
        scheme_name: str,
        criteria: EligibilityCriteria
    ) -> EligibilityResult:
        """
        Evaluate user eligibility for a single scheme
        
        Args:
            user_profile: User's profile data
            scheme_id: Unique scheme identifier
            scheme_name: Human-readable scheme name
            criteria: Scheme eligibility criteria
            
        Returns:
            EligibilityResult with detailed evaluation
        """
        reasons = []
        missing_info = []
        checks_passed = 0
        total_checks = 0
        
        # Age check
        if criteria.min_age is not None or criteria.max_age is not None:
            total_checks += 1
            if user_profile.age is None:
                missing_info.append("age")
            else:
                if criteria.min_age is not None and user_profile.age < criteria.min_age:
                    reasons.append(f"age_below_minimum: {user_profile.age} < {criteria.min_age}")
                elif criteria.max_age is not None and user_profile.age > criteria.max_age:
                    reasons.append(f"age_above_maximum: {user_profile.age} > {criteria.max_age}")
                else:
                    checks_passed += 1
        
        # Income check
        if criteria.max_income is not None or criteria.min_income is not None:
            total_checks += 1
            if user_profile.income is None:
                missing_info.append("income")
            else:
                if criteria.max_income is not None and user_profile.income > criteria.max_income:
                    reasons.append(f"income_exceeds_limit: {user_profile.income} > {criteria.max_income}")
                elif criteria.min_income is not None and user_profile.income < criteria.min_income:
                    reasons.append(f"income_below_minimum: {user_profile.income} < {criteria.min_income}")
                else:
                    checks_passed += 1
        
        # State check
        if criteria.allowed_states is not None:
            total_checks += 1
            if user_profile.state is None:
                missing_info.append("state")
            else:
                if user_profile.state.lower() in [s.lower() for s in criteria.allowed_states]:
                    checks_passed += 1
                else:
                    reasons.append(f"state_not_allowed: {user_profile.state} not in {criteria.allowed_states}")
        
        # Gender check
        if criteria.allowed_genders is not None:
            total_checks += 1
            if user_profile.gender is None:
                missing_info.append("gender")
            else:
                if user_profile.gender.lower() in [g.lower() for g in criteria.allowed_genders]:
                    checks_passed += 1
                else:
                    reasons.append(f"gender_not_allowed: {user_profile.gender} not in {criteria.allowed_genders}")
        
        # Category check (caste/reservation category)
        if criteria.allowed_categories is not None:
            total_checks += 1
            if user_profile.category is None:
                missing_info.append("category")
            else:
                if user_profile.category.lower() in [c.lower() for c in criteria.allowed_categories]:
                    checks_passed += 1
                else:
                    reasons.append(f"category_not_allowed: {user_profile.category} not in {criteria.allowed_categories}")
        
        # Occupation check
        if criteria.required_occupation is not None:
            total_checks += 1
            if user_profile.occupation is None:
                missing_info.append("occupation")
            else:
                if user_profile.occupation.lower() == criteria.required_occupation.lower():
                    checks_passed += 1
                else:
                    reasons.append(f"occupation_mismatch: required {criteria.required_occupation}, got {user_profile.occupation}")
        
        # Disability check
        if criteria.requires_disability is not None:
            total_checks += 1
            if user_profile.disability is None:
                missing_info.append("disability")
            else:
                if user_profile.disability == criteria.requires_disability:
                    checks_passed += 1
                else:
                    reasons.append(f"disability_requirement_not_met: required {criteria.requires_disability}")
        
        # BPL card check
        if criteria.requires_bpl_card is not None:
            total_checks += 1
            if user_profile.bpl_card is None:
                missing_info.append("bpl_card")
            else:
                if user_profile.bpl_card == criteria.requires_bpl_card:
                    checks_passed += 1
                else:
                    reasons.append(f"bpl_card_requirement_not_met: required {criteria.requires_bpl_card}")
        
        # Farmer check
        if criteria.requires_farmer is not None:
            total_checks += 1
            if user_profile.farmer is None:
                missing_info.append("farmer")
            else:
                if user_profile.farmer == criteria.requires_farmer:
                    checks_passed += 1
                else:
                    reasons.append(f"farmer_requirement_not_met: required {criteria.requires_farmer}")
        
        # Widow check
        if criteria.requires_widow is not None:
            total_checks += 1
            if user_profile.widow is None:
                missing_info.append("widow")
            else:
                if user_profile.widow == criteria.requires_widow:
                    checks_passed += 1
                else:
                    reasons.append(f"widow_requirement_not_met: required {criteria.requires_widow}")
        
        # Minority check
        if criteria.requires_minority is not None:
            total_checks += 1
            if user_profile.minority is None:
                missing_info.append("minority")
            else:
                if user_profile.minority == criteria.requires_minority:
                    checks_passed += 1
                else:
                    reasons.append(f"minority_requirement_not_met: required {criteria.requires_minority}")
        
        # Calculate match score
        if total_checks == 0:
            match_score = 1.0  # No criteria means everyone is eligible
        else:
            match_score = checks_passed / total_checks
        
        # Determine status
        if missing_info:
            status = EligibilityStatus.PARTIAL
            reasons.append(f"incomplete_profile: missing {', '.join(missing_info)}")
        elif match_score == 1.0:
            status = EligibilityStatus.ELIGIBLE
            reasons = ["all_criteria_met"]
        else:
            status = EligibilityStatus.INELIGIBLE
        
        result = EligibilityResult(
            scheme_id=scheme_id,
            scheme_name=scheme_name,
            status=status,
            reasons=reasons,
            missing_info=missing_info,
            match_score=match_score
        )
        
        # Store in history
        self.evaluation_history.append(result.to_dict())
        
        return result
    
    def evaluate_multiple_schemes(
        self,
        user_profile: UserProfile,
        schemes: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate user eligibility across multiple schemes
        
        Args:
            user_profile: User's profile data
            schemes: List of scheme dictionaries with 'id', 'name', and 'criteria'
            
        Returns:
            Dictionary with eligible_schemes, ineligible_schemes, and partial_schemes
        """
        eligible = []
        ineligible = []
        partial = []
        
        for scheme in schemes:
            # Extract scheme data
            scheme_id = scheme.get("id", "unknown")
            scheme_name = scheme.get("name", "Unknown Scheme")
            criteria_data = scheme.get("criteria", {})
            
            # Convert criteria dict to EligibilityCriteria object
            criteria = EligibilityCriteria(**criteria_data)
            
            # Evaluate
            result = self.evaluate_scheme(
                user_profile=user_profile,
                scheme_id=scheme_id,
                scheme_name=scheme_name,
                criteria=criteria
            )
            
            # Categorize
            if result.status == EligibilityStatus.ELIGIBLE:
                eligible.append(result.to_dict())
            elif result.status == EligibilityStatus.PARTIAL:
                partial.append(result.to_dict())
            else:
                ineligible.append(result.to_dict())
        
        # Sort by match score (descending)
        eligible.sort(key=lambda x: x["match_score"], reverse=True)
        partial.sort(key=lambda x: x["match_score"], reverse=True)
        ineligible.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "eligible_schemes": eligible,
            "ineligible_schemes": ineligible,
            "partial_schemes": partial,
            "total_evaluated": len(schemes),
            "user_profile_completeness": self._calculate_profile_completeness(user_profile)
        }
    
    def _calculate_profile_completeness(self, profile: UserProfile) -> float:
        """
        Calculate how complete the user profile is
        
        Args:
            profile: User profile
            
        Returns:
            Completeness score (0.0 to 1.0)
        """
        total_fields = len(profile.__dataclass_fields__)
        filled_fields = sum(1 for v in profile.__dict__.values() if v is not None)
        return filled_fields / total_fields if total_fields > 0 else 0.0
    
    def get_evaluation_history(self) -> List[Dict[str, Any]]:
        """Get history of all evaluations"""
        return self.evaluation_history.copy()
    
    def clear_history(self):
        """Clear evaluation history"""
        self.evaluation_history.clear()


# Helper function for quick evaluation
def evaluate_eligibility(
    user_data: Dict[str, Any],
    schemes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Convenience function for one-shot eligibility evaluation
    
    Args:
        user_data: Dictionary with user profile fields
        schemes: List of scheme dictionaries
        
    Returns:
        Evaluation results dictionary
    """
    # Create user profile
    profile = UserProfile(**user_data)
    
    # Create engine and evaluate
    engine = EligibilityEngine()
    results = engine.evaluate_multiple_schemes(profile, schemes)
    
    return results