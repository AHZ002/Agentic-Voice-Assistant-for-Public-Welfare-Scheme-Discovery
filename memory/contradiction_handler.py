"""
contradiction_handler.py
Detects contradictions between new user input and existing session memory.
Pure detection layer - does not resolve or modify memory.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Contradiction:
    """Represents a detected contradiction."""
    field_name: str
    old_value: Any
    new_value: Any
    old_turn_id: Optional[int]
    severity: str  # 'critical', 'moderate', 'minor'
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "field_name": self.field_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "old_turn_id": self.old_turn_id,
            "severity": self.severity,
            "timestamp": self.timestamp
        }


class ContradictionHandler:
    """
    Detects contradictions between new data and session memory.
    Does not resolve conflicts or modify memory.
    """
    
    # Fields that should never contradict (critical)
    CRITICAL_FIELDS = {
        'age', 'gender', 'category', 'state', 'district', 
        'disability_status', 'bpl_status', 'caste_certificate'
    }
    
    # Fields that may change but are worth flagging (moderate)
    MODERATE_FIELDS = {
        'annual_income', 'occupation', 'land_ownership', 
        'land_area_acres', 'family_size', 'has_bank_account',
        'aadhaar_linked', 'location', 'residence_type'
    }
    
    # Fields that can change freely (minor)
    MINOR_FIELDS = {
        'residence_duration_years', 'ration_card_type'
    }
    
    # Tolerance thresholds for numeric fields
    NUMERIC_TOLERANCES = {
        'age': 0,  # Age should not change in a session
        'annual_income': 0,  # Income should not change
        'land_area_acres': 0.5,  # Allow minor measurement differences
        'family_size': 0,  # Family size should not change
        'disability_percentage': 5,  # Allow minor assessment differences
        'residence_duration_years': 1  # Allow rounding differences
    }
    
    def __init__(self):
        """Initialize contradiction handler."""
        pass
    
    def detect_contradictions(
        self,
        new_data: Dict[str, Any],
        existing_profile: Dict[str, Any],
        turn_id: Optional[int] = None
    ) -> List[Contradiction]:
        """
        Detect contradictions between new data and existing profile.
        
        Args:
            new_data: Dictionary of new field values
            existing_profile: Dictionary of existing profile values
            turn_id: Optional turn ID for tracking
            
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        for field_name, new_value in new_data.items():
            # Skip None values in new data
            if new_value is None:
                continue
            
            # Check if field exists in existing profile
            if field_name not in existing_profile:
                continue
            
            old_value = existing_profile[field_name]
            
            # Skip if old value is None (no contradiction)
            if old_value is None:
                continue
            
            # Check for contradiction
            if self._is_contradiction(field_name, old_value, new_value):
                severity = self._determine_severity(field_name)
                
                contradiction = Contradiction(
                    field_name=field_name,
                    old_value=old_value,
                    new_value=new_value,
                    old_turn_id=turn_id,
                    severity=severity,
                    timestamp=datetime.now().isoformat()
                )
                
                contradictions.append(contradiction)
        
        return contradictions
    
    def _is_contradiction(
        self,
        field_name: str,
        old_value: Any,
        new_value: Any
    ) -> bool:
        """
        Determine if two values contradict each other.
        
        Args:
            field_name: Name of the field
            old_value: Existing value
            new_value: New value
            
        Returns:
            True if values contradict
        """
        # Handle numeric fields with tolerance
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            return self._is_numeric_contradiction(field_name, old_value, new_value)
        
        # Handle boolean fields
        if isinstance(old_value, bool) and isinstance(new_value, bool):
            return old_value != new_value
        
        # Handle string fields (case-insensitive comparison)
        if isinstance(old_value, str) and isinstance(new_value, str):
            return self._is_string_contradiction(old_value, new_value)
        
        # Handle different types as contradiction
        if type(old_value) != type(new_value):
            return True
        
        # Default comparison
        return old_value != new_value
    
    def _is_numeric_contradiction(
        self,
        field_name: str,
        old_value: float,
        new_value: float
    ) -> bool:
        """
        Check numeric contradiction with tolerance.
        
        Args:
            field_name: Field name
            old_value: Old numeric value
            new_value: New numeric value
            
        Returns:
            True if contradiction exists
        """
        tolerance = self.NUMERIC_TOLERANCES.get(field_name, 0)
        difference = abs(old_value - new_value)
        return difference > tolerance
    
    def _is_string_contradiction(
        self,
        old_value: str,
        new_value: str
    ) -> bool:
        """
        Check string contradiction (case-insensitive, whitespace-normalized).
        
        Args:
            old_value: Old string value
            new_value: New string value
            
        Returns:
            True if contradiction exists
        """
        old_normalized = old_value.strip().lower()
        new_normalized = new_value.strip().lower()
        
        # Check exact match
        if old_normalized == new_normalized:
            return False
        
        # Check common variations
        variations = self._get_common_variations(old_normalized)
        if new_normalized in variations:
            return False
        
        return True
    
    def _get_common_variations(self, value: str) -> List[str]:
        """
        Get common variations of a string value.
        
        Args:
            value: Original value
            
        Returns:
            List of common variations
        """
        variations = [value]
        
        # Common abbreviations and variations
        variation_map = {
            'male': ['m', 'पुरुष', 'purush'],
            'female': ['f', 'स्त्री', 'stri', 'mahila', 'महिला'],
            'general': ['gen', 'सामान्य', 'samanya'],
            'obc': ['इतर मागासवर्ग', 'other backward class'],
            'sc': ['अनुसूचित जाती', 'scheduled caste'],
            'st': ['अनुसूचित जमाती', 'scheduled tribe'],
            'rural': ['ग्रामीण', 'grameen', 'village'],
            'urban': ['शहरी', 'shahari', 'city'],
            'yes': ['y', 'होय', 'hoy', 'true'],
            'no': ['n', 'नाही', 'nahi', 'false'],
            'maharashtra': ['mh', 'महाराष्ट्र', 'maha']
        }
        
        for key, var_list in variation_map.items():
            if value in var_list or value == key:
                variations.extend(var_list)
                variations.append(key)
        
        return list(set(variations))
    
    def _determine_severity(self, field_name: str) -> str:
        """
        Determine severity of contradiction.
        
        Args:
            field_name: Field name
            
        Returns:
            Severity level: 'critical', 'moderate', or 'minor'
        """
        if field_name in self.CRITICAL_FIELDS:
            return 'critical'
        elif field_name in self.MODERATE_FIELDS:
            return 'moderate'
        else:
            return 'minor'
    
    def check_profile_consistency(
        self,
        profile: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Check for logical inconsistencies within a profile.
        
        Args:
            profile: User profile dictionary
            
        Returns:
            List of inconsistency reports
        """
        inconsistencies = []
        
        # Check age vs category logic
        if profile.get('age') and profile.get('age') < 18:
            if profile.get('occupation') in ['government employee', 'retired']:
                inconsistencies.append({
                    "type": "logical_inconsistency",
                    "fields": ['age', 'occupation'],
                    "description": "Age less than 18 but occupation suggests adult employment",
                    "severity": "moderate"
                })
        
        # Check disability percentage without disability status
        if profile.get('disability_percentage') and not profile.get('disability_status'):
            inconsistencies.append({
                "type": "logical_inconsistency",
                "fields": ['disability_status', 'disability_percentage'],
                "description": "Disability percentage provided but disability_status is False or missing",
                "severity": "moderate"
            })
        
        # Check land area without land ownership
        if profile.get('land_area_acres') and not profile.get('land_ownership'):
            inconsistencies.append({
                "type": "logical_inconsistency",
                "fields": ['land_ownership', 'land_area_acres'],
                "description": "Land area provided but land_ownership is False or missing",
                "severity": "moderate"
            })
        
        # Check income consistency with BPL status
        if profile.get('annual_income') and profile.get('bpl_status'):
            if profile['annual_income'] > 100000 and profile['bpl_status'] is True:
                inconsistencies.append({
                    "type": "logical_inconsistency",
                    "fields": ['annual_income', 'bpl_status'],
                    "description": "High income with BPL status marked as True",
                    "severity": "critical"
                })
        
        # Check bank account for Aadhaar linking
        if profile.get('aadhaar_linked') and not profile.get('has_bank_account'):
            inconsistencies.append({
                "type": "logical_inconsistency",
                "fields": ['has_bank_account', 'aadhaar_linked'],
                "description": "Aadhaar linked but no bank account",
                "severity": "minor"
            })
        
        return inconsistencies
    
    def compare_field_history(
        self,
        field_name: str,
        values_with_turns: List[Tuple[Any, int]]
    ) -> List[Dict[str, Any]]:
        """
        Compare historical values of a single field across multiple turns.
        
        Args:
            field_name: Field to analyze
            values_with_turns: List of (value, turn_id) tuples
            
        Returns:
            List of contradiction reports
        """
        contradictions = []
        
        if len(values_with_turns) < 2:
            return contradictions
        
        # Compare each consecutive pair
        for i in range(len(values_with_turns) - 1):
            old_value, old_turn = values_with_turns[i]
            new_value, new_turn = values_with_turns[i + 1]
            
            if self._is_contradiction(field_name, old_value, new_value):
                contradictions.append({
                    "field_name": field_name,
                    "old_value": old_value,
                    "old_turn": old_turn,
                    "new_value": new_value,
                    "new_turn": new_turn,
                    "severity": self._determine_severity(field_name)
                })
        
        return contradictions
    
    def generate_contradiction_report(
        self,
        contradictions: List[Contradiction]
    ) -> Dict[str, Any]:
        """
        Generate structured report of contradictions.
        
        Args:
            contradictions: List of contradiction objects
            
        Returns:
            Structured report dictionary
        """
        if not contradictions:
            return {
                "has_contradictions": False,
                "total_count": 0,
                "by_severity": {"critical": 0, "moderate": 0, "minor": 0},
                "contradictions": []
            }
        
        # Count by severity
        severity_counts = {"critical": 0, "moderate": 0, "minor": 0}
        for c in contradictions:
            severity_counts[c.severity] += 1
        
        # Sort by severity
        severity_order = {"critical": 0, "moderate": 1, "minor": 2}
        sorted_contradictions = sorted(
            contradictions,
            key=lambda x: severity_order[x.severity]
        )
        
        return {
            "has_contradictions": True,
            "total_count": len(contradictions),
            "by_severity": severity_counts,
            "contradictions": [c.to_dict() for c in sorted_contradictions],
            "fields_affected": list(set(c.field_name for c in contradictions)),
            "requires_resolution": severity_counts["critical"] > 0
        }
    
    def get_contradiction_summary_text(
        self,
        contradictions: List[Contradiction]
    ) -> str:
        """
        Generate human-readable summary of contradictions.
        
        Args:
            contradictions: List of contradictions
            
        Returns:
            Summary text
        """
        if not contradictions:
            return "No contradictions detected."
        
        report = self.generate_contradiction_report(contradictions)
        
        summary_parts = [
            f"Detected {report['total_count']} contradiction(s):"
        ]
        
        if report['by_severity']['critical'] > 0:
            summary_parts.append(f"- {report['by_severity']['critical']} critical")
        if report['by_severity']['moderate'] > 0:
            summary_parts.append(f"- {report['by_severity']['moderate']} moderate")
        if report['by_severity']['minor'] > 0:
            summary_parts.append(f"- {report['by_severity']['minor']} minor")
        
        summary_parts.append(f"\nFields affected: {', '.join(report['fields_affected'])}")
        
        return "\n".join(summary_parts)