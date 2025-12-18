"""
Mock Government Application API

This module simulates a government welfare scheme application API.
It intentionally includes failures, timeouts, and edge cases to test
agent recovery and failure handling capabilities.

NO user interaction
NO real API calls
Intentionally unreliable for testing purposes
"""

import time
import random
import uuid
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum


class ApplicationStatus(Enum):
    """Application status enumeration"""
    SUBMITTED = "submitted"
    PENDING_DOCUMENTS = "pending_documents"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"
    TIMEOUT = "timeout"


class APIErrorType(Enum):
    """API error types"""
    NETWORK_TIMEOUT = "network_timeout"
    SERVER_ERROR = "server_error"
    INVALID_REQUEST = "invalid_request"
    MISSING_FIELDS = "missing_fields"
    INVALID_SCHEME_ID = "invalid_scheme_id"
    DUPLICATE_APPLICATION = "duplicate_application"
    SERVICE_UNAVAILABLE = "service_unavailable"
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"


@dataclass
class APIResponse:
    """Structured API response"""
    success: bool
    status_code: int
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "success": self.success,
            "status_code": self.status_code,
            "data": self.data,
            "error": self.error,
            "error_type": self.error_type,
            "message": self.message,
            "timestamp": self.timestamp
        }


@dataclass
class Application:
    """Application record"""
    application_id: str
    scheme_id: str
    user_data: Dict[str, Any]
    status: ApplicationStatus
    submitted_at: str
    documents_submitted: List[str] = field(default_factory=list)
    required_documents: List[str] = field(default_factory=list)
    review_notes: Optional[str] = None
    expected_completion_date: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "application_id": self.application_id,
            "scheme_id": self.scheme_id,
            "user_data": self.user_data,
            "status": self.status.value,
            "submitted_at": self.submitted_at,
            "documents_submitted": self.documents_submitted,
            "required_documents": self.required_documents,
            "review_notes": self.review_notes,
            "expected_completion_date": self.expected_completion_date
        }


class MockGovAPI:
    """
    Simulated Government Application API with intentional failures
    """
    
    # Document requirements by category
    COMMON_DOCUMENTS = [
        "aadhaar_card",
        "pan_card",
        "address_proof",
        "income_certificate",
        "photograph"
    ]
    
    CATEGORY_SPECIFIC_DOCS = {
        "farmer": ["land_ownership_certificate", "kisan_credit_card"],
        "student": ["school_id_card", "fee_receipt", "marks_card"],
        "senior_citizen": ["age_proof", "pension_passbook"],
        "disability": ["disability_certificate", "medical_report"],
        "women": ["self_declaration"],
        "bpl": ["bpl_card", "ration_card"]
    }
    
    def __init__(
        self,
        failure_rate: float = 0.3,
        timeout_rate: float = 0.15,
        enable_random_delays: bool = True,
        max_delay_seconds: float = 2.0
    ):
        """
        Initialize mock API
        
        Args:
            failure_rate: Probability of random failures (0.0 to 1.0)
            timeout_rate: Probability of timeouts (0.0 to 1.0)
            enable_random_delays: Whether to simulate network delays
            max_delay_seconds: Maximum delay in seconds
        """
        self.failure_rate = failure_rate
        self.timeout_rate = timeout_rate
        self.enable_random_delays = enable_random_delays
        self.max_delay_seconds = max_delay_seconds
        
        # In-memory storage
        self.applications: Dict[str, Application] = {}
        self.request_count = 0
        self.rate_limit_threshold = 10  # requests per minute
        self.rate_limit_window: List[float] = []
    
    def _simulate_network_delay(self):
        """Simulate random network delay"""
        if self.enable_random_delays:
            delay = random.uniform(0.1, self.max_delay_seconds)
            time.sleep(delay)
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded"""
        current_time = time.time()
        
        # Remove timestamps older than 1 minute
        self.rate_limit_window = [
            t for t in self.rate_limit_window
            if current_time - t < 60
        ]
        
        # Check threshold
        if len(self.rate_limit_window) >= self.rate_limit_threshold:
            return True
        
        # Add current request
        self.rate_limit_window.append(current_time)
        return False
    
    def _should_fail(self) -> bool:
        """Determine if this request should fail"""
        return random.random() < self.failure_rate
    
    def _should_timeout(self) -> bool:
        """Determine if this request should timeout"""
        return random.random() < self.timeout_rate
    
    def _generate_random_error(self) -> APIResponse:
        """Generate a random error response"""
        error_types = [
            (APIErrorType.SERVER_ERROR, 500, "Internal server error occurred"),
            (APIErrorType.SERVICE_UNAVAILABLE, 503, "Service temporarily unavailable"),
            (APIErrorType.NETWORK_TIMEOUT, 408, "Request timeout"),
        ]
        
        error_type, status_code, message = random.choice(error_types)
        
        return APIResponse(
            success=False,
            status_code=status_code,
            error=error_type.value,
            error_type=error_type.value,
            message=message
        )
    
    def _get_required_documents(
        self,
        scheme_id: str,
        user_data: Dict[str, Any]
    ) -> List[str]:
        """Determine required documents based on scheme and user profile"""
        required = self.COMMON_DOCUMENTS.copy()
        
        # Add category-specific documents
        category = user_data.get("category", "").lower()
        if category in self.CATEGORY_SPECIFIC_DOCS:
            required.extend(self.CATEGORY_SPECIFIC_DOCS[category])
        
        # Add based on user attributes
        if user_data.get("farmer"):
            required.extend(self.CATEGORY_SPECIFIC_DOCS.get("farmer", []))
        
        if user_data.get("disability"):
            required.extend(self.CATEGORY_SPECIFIC_DOCS.get("disability", []))
        
        if user_data.get("bpl_card"):
            required.extend(self.CATEGORY_SPECIFIC_DOCS.get("bpl", []))
        
        # Remove duplicates
        return list(set(required))
    
    def submit_application(
        self,
        scheme_id: str,
        user_data: Dict[str, Any]
    ) -> APIResponse:
        """
        Submit a new application
        
        Args:
            scheme_id: Scheme identifier
            user_data: User profile data
            
        Returns:
            APIResponse with application details or error
        """
        self.request_count += 1
        self._simulate_network_delay()
        
        # Check rate limit
        if self._check_rate_limit():
            return APIResponse(
                success=False,
                status_code=429,
                error=APIErrorType.RATE_LIMIT_EXCEEDED.value,
                error_type=APIErrorType.RATE_LIMIT_EXCEEDED.value,
                message="Too many requests. Please try again later."
            )
        
        # Random timeout
        if self._should_timeout():
            time.sleep(5)  # Simulate long timeout
            return APIResponse(
                success=False,
                status_code=408,
                error=APIErrorType.NETWORK_TIMEOUT.value,
                error_type=APIErrorType.NETWORK_TIMEOUT.value,
                message="Request timed out. Please retry."
            )
        
        # Random failure
        if self._should_fail():
            return self._generate_random_error()
        
        # Validate input
        if not scheme_id:
            return APIResponse(
                success=False,
                status_code=400,
                error=APIErrorType.INVALID_SCHEME_ID.value,
                error_type=APIErrorType.INVALID_SCHEME_ID.value,
                message="Scheme ID is required"
            )
        
        required_fields = ["name", "age", "state"]
        missing_fields = [f for f in required_fields if f not in user_data]
        
        if missing_fields:
            return APIResponse(
                success=False,
                status_code=400,
                error=APIErrorType.MISSING_FIELDS.value,
                error_type=APIErrorType.MISSING_FIELDS.value,
                message=f"Missing required fields: {', '.join(missing_fields)}"
            )
        
        # Check for duplicate (10% chance)
        if random.random() < 0.1:
            return APIResponse(
                success=False,
                status_code=409,
                error=APIErrorType.DUPLICATE_APPLICATION.value,
                error_type=APIErrorType.DUPLICATE_APPLICATION.value,
                message="Application already exists for this scheme"
            )
        
        # Create application
        application_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
        required_docs = self._get_required_documents(scheme_id, user_data)
        
        expected_date = datetime.now() + timedelta(days=random.randint(7, 30))
        
        application = Application(
            application_id=application_id,
            scheme_id=scheme_id,
            user_data=user_data,
            status=ApplicationStatus.PENDING_DOCUMENTS,
            submitted_at=datetime.now().isoformat(),
            required_documents=required_docs,
            expected_completion_date=expected_date.strftime("%Y-%m-%d")
        )
        
        self.applications[application_id] = application
        
        return APIResponse(
            success=True,
            status_code=201,
            data=application.to_dict(),
            message="Application submitted successfully"
        )
    
    def get_application_status(self, application_id: str) -> APIResponse:
        """
        Get application status
        
        Args:
            application_id: Application identifier
            
        Returns:
            APIResponse with application status
        """
        self.request_count += 1
        self._simulate_network_delay()
        
        # Check rate limit
        if self._check_rate_limit():
            return APIResponse(
                success=False,
                status_code=429,
                error=APIErrorType.RATE_LIMIT_EXCEEDED.value,
                error_type=APIErrorType.RATE_LIMIT_EXCEEDED.value,
                message="Too many requests"
            )
        
        # Random failure
        if self._should_fail():
            return self._generate_random_error()
        
        # Find application
        if application_id not in self.applications:
            return APIResponse(
                success=False,
                status_code=404,
                error="not_found",
                message=f"Application {application_id} not found"
            )
        
        application = self.applications[application_id]
        
        # Randomly update status (simulate processing)
        self._update_application_status(application)
        
        return APIResponse(
            success=True,
            status_code=200,
            data=application.to_dict(),
            message="Status retrieved successfully"
        )
    
    def submit_documents(
        self,
        application_id: str,
        documents: List[str]
    ) -> APIResponse:
        """
        Submit required documents
        
        Args:
            application_id: Application identifier
            documents: List of document identifiers
            
        Returns:
            APIResponse
        """
        self.request_count += 1
        self._simulate_network_delay()
        
        # Random failure
        if self._should_fail():
            return self._generate_random_error()
        
        if application_id not in self.applications:
            return APIResponse(
                success=False,
                status_code=404,
                error="not_found",
                message="Application not found"
            )
        
        application = self.applications[application_id]
        
        # Add documents
        application.documents_submitted.extend(documents)
        application.documents_submitted = list(set(application.documents_submitted))
        
        # Check if all documents submitted
        missing_docs = [
            doc for doc in application.required_documents
            if doc not in application.documents_submitted
        ]
        
        if not missing_docs:
            application.status = ApplicationStatus.UNDER_REVIEW
            message = "All documents submitted. Application under review."
        else:
            message = f"Documents received. Still pending: {', '.join(missing_docs)}"
        
        return APIResponse(
            success=True,
            status_code=200,
            data=application.to_dict(),
            message=message
        )
    
    def _update_application_status(self, application: Application):
        """Randomly update application status to simulate processing"""
        if application.status == ApplicationStatus.PENDING_DOCUMENTS:
            # Stay in pending documents
            pass
        elif application.status == ApplicationStatus.UNDER_REVIEW:
            # 30% chance to move to approved/rejected
            rand = random.random()
            if rand < 0.2:
                application.status = ApplicationStatus.APPROVED
                application.review_notes = "Application approved after verification"
            elif rand < 0.3:
                application.status = ApplicationStatus.REJECTED
                application.review_notes = "Application rejected due to incomplete information"
        elif application.status in [ApplicationStatus.APPROVED, ApplicationStatus.REJECTED]:
            # Final states - don't change
            pass
    
    def get_required_documents(self, scheme_id: str) -> APIResponse:
        """
        Get list of required documents for a scheme
        
        Args:
            scheme_id: Scheme identifier
            
        Returns:
            APIResponse with document list
        """
        self.request_count += 1
        self._simulate_network_delay()
        
        # Random failure
        if self._should_fail():
            return self._generate_random_error()
        
        # Return common documents (simplified)
        return APIResponse(
            success=True,
            status_code=200,
            data={
                "scheme_id": scheme_id,
                "required_documents": self.COMMON_DOCUMENTS
            },
            message="Document requirements retrieved"
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get API statistics"""
        return {
            "total_requests": self.request_count,
            "total_applications": len(self.applications),
            "status_breakdown": self._get_status_breakdown(),
            "failure_rate": self.failure_rate,
            "timeout_rate": self.timeout_rate
        }
    
    def _get_status_breakdown(self) -> Dict[str, int]:
        """Get breakdown of applications by status"""
        breakdown = {}
        for app in self.applications.values():
            status = app.status.value
            breakdown[status] = breakdown.get(status, 0) + 1
        return breakdown
    
    def reset(self):
        """Reset API state (for testing)"""
        self.applications.clear()
        self.request_count = 0
        self.rate_limit_window.clear()


# Convenience functions
def submit_application(
    scheme_id: str,
    user_data: Dict[str, Any],
    api_instance: Optional[MockGovAPI] = None
) -> Dict[str, Any]:
    """
    Convenience function for submitting application
    
    Args:
        scheme_id: Scheme ID
        user_data: User profile data
        api_instance: Optional API instance (creates new if None)
        
    Returns:
        Response dictionary
    """
    if api_instance is None:
        api_instance = MockGovAPI()
    
    response = api_instance.submit_application(scheme_id, user_data)
    return response.to_dict()


def check_application_status(
    application_id: str,
    api_instance: Optional[MockGovAPI] = None
) -> Dict[str, Any]:
    """
    Convenience function for checking status
    
    Args:
        application_id: Application ID
        api_instance: Optional API instance
        
    Returns:
        Response dictionary
    """
    if api_instance is None:
        api_instance = MockGovAPI()
    
    response = api_instance.get_application_status(application_id)
    return response.to_dict()