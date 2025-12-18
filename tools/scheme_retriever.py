"""
Government Schemes Retrieval Tool

This module provides retrieval functionality for government welfare schemes
from a JSON data source. It supports keyword-based, intent-based, and
tag-based searching without making eligibility decisions.

NO agent logic
NO eligibility evaluation
NO user interaction
NO hard-coded schemes
"""

import json
import os
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class SchemeSearchResult:
    """Structured search result for a scheme"""
    scheme_id: str
    name: str
    description: str
    category: str
    relevance_score: float
    matched_keywords: List[str]
    tags: List[str]
    criteria: Dict[str, Any]
    benefits: Optional[str] = None
    application_process: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scheme_id": self.scheme_id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "relevance_score": self.relevance_score,
            "matched_keywords": self.matched_keywords,
            "tags": self.tags,
            "criteria": self.criteria,
            "benefits": self.benefits,
            "application_process": self.application_process
        }


class SchemeRetriever:
    """
    Retrieval tool for government welfare schemes
    """
    
    def __init__(self, schemes_file_path: Optional[str] = None):
        """
        Initialize the scheme retriever
        
        Args:
            schemes_file_path: Path to schemes.json file
                              If None, defaults to data/schemes.json
        """
        if schemes_file_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent
            schemes_file_path = project_root / "data" / "schemes.json"
        
        self.schemes_file_path = Path(schemes_file_path)
        self.schemes_data: List[Dict[str, Any]] = []
        self.indexed_schemes: Dict[str, List[str]] = {}  # keyword -> [scheme_ids]
        
        self._load_schemes()
        self._build_index()
    
    def _load_schemes(self):
        """Load schemes from JSON file"""
        if not self.schemes_file_path.exists():
            raise FileNotFoundError(
                f"Schemes file not found: {self.schemes_file_path}"
            )
        
        try:
            with open(self.schemes_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                self.schemes_data = data
            elif isinstance(data, dict) and "schemes" in data:
                self.schemes_data = data["schemes"]
            else:
                raise ValueError("Invalid schemes.json structure")
                
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in schemes file: {e}")
    
    def _build_index(self):
        """Build keyword index for fast retrieval"""
        self.indexed_schemes = {}
        
        for scheme in self.schemes_data:
            scheme_id = scheme.get("id", "")
            
            # Extract searchable text
            searchable_text = self._extract_searchable_text(scheme)
            
            # Tokenize and normalize
            keywords = self._tokenize_and_normalize(searchable_text)
            
            # Index each keyword
            for keyword in keywords:
                if keyword not in self.indexed_schemes:
                    self.indexed_schemes[keyword] = []
                if scheme_id not in self.indexed_schemes[keyword]:
                    self.indexed_schemes[keyword].append(scheme_id)
    
    def _extract_searchable_text(self, scheme: Dict[str, Any]) -> str:
        """Extract all searchable text from a scheme"""
        text_parts = []
        
        # Name and description
        text_parts.append(scheme.get("name", ""))
        text_parts.append(scheme.get("description", ""))
        
        # Category
        text_parts.append(scheme.get("category", ""))
        
        # Tags
        tags = scheme.get("tags", [])
        if isinstance(tags, list):
            text_parts.extend(tags)
        
        # Benefits
        text_parts.append(scheme.get("benefits", ""))
        
        # Keywords field (if exists)
        keywords = scheme.get("keywords", [])
        if isinstance(keywords, list):
            text_parts.extend(keywords)
        
        return " ".join(str(part) for part in text_parts if part)
    
    def _tokenize_and_normalize(self, text: str) -> List[str]:
        """
        Tokenize and normalize text for searching
        
        Args:
            text: Input text
            
        Returns:
            List of normalized tokens
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split into tokens
        tokens = text.split()
        
        # Remove very short tokens (length < 2)
        tokens = [t for t in tokens if len(t) >= 2]
        
        return tokens
    
    def search_by_keywords(
        self,
        keywords: List[str],
        max_results: int = 10
    ) -> List[SchemeSearchResult]:
        """
        Search schemes by keywords
        
        Args:
            keywords: List of search keywords
            max_results: Maximum number of results to return
            
        Returns:
            List of SchemeSearchResult objects sorted by relevance
        """
        if not keywords:
            return []
        
        # Normalize keywords
        normalized_keywords = []
        for kw in keywords:
            normalized_keywords.extend(self._tokenize_and_normalize(kw))
        
        # Score schemes
        scheme_scores: Dict[str, Dict[str, Any]] = {}
        
        for keyword in normalized_keywords:
            scheme_ids = self.indexed_schemes.get(keyword, [])
            for scheme_id in scheme_ids:
                if scheme_id not in scheme_scores:
                    scheme_scores[scheme_id] = {
                        "score": 0,
                        "matched_keywords": []
                    }
                scheme_scores[scheme_id]["score"] += 1
                scheme_scores[scheme_id]["matched_keywords"].append(keyword)
        
        # Convert to results
        results = []
        for scheme in self.schemes_data:
            scheme_id = scheme.get("id", "")
            if scheme_id in scheme_scores:
                score_data = scheme_scores[scheme_id]
                relevance_score = score_data["score"] / len(normalized_keywords)
                
                result = SchemeSearchResult(
                    scheme_id=scheme_id,
                    name=scheme.get("name", ""),
                    description=scheme.get("description", ""),
                    category=scheme.get("category", ""),
                    relevance_score=relevance_score,
                    matched_keywords=list(set(score_data["matched_keywords"])),
                    tags=scheme.get("tags", []),
                    criteria=scheme.get("criteria", {}),
                    benefits=scheme.get("benefits"),
                    application_process=scheme.get("application_process")
                )
                results.append(result)
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:max_results]
    
    def search_by_tags(
        self,
        tags: List[str],
        match_all: bool = False,
        max_results: int = 10
    ) -> List[SchemeSearchResult]:
        """
        Search schemes by tags
        
        Args:
            tags: List of tags to search for
            match_all: If True, scheme must have all tags. If False, any tag matches
            max_results: Maximum number of results
            
        Returns:
            List of matching schemes
        """
        if not tags:
            return []
        
        # Normalize tags
        normalized_tags = [tag.lower().strip() for tag in tags]
        
        results = []
        for scheme in self.schemes_data:
            scheme_tags = scheme.get("tags", [])
            if not scheme_tags:
                continue
            
            # Normalize scheme tags
            scheme_tags_normalized = [t.lower().strip() for t in scheme_tags]
            
            # Check match
            matched_tags = [t for t in normalized_tags if t in scheme_tags_normalized]
            
            if match_all:
                # All tags must match
                if len(matched_tags) == len(normalized_tags):
                    relevance_score = 1.0
                else:
                    continue
            else:
                # Any tag matches
                if not matched_tags:
                    continue
                relevance_score = len(matched_tags) / len(normalized_tags)
            
            result = SchemeSearchResult(
                scheme_id=scheme.get("id", ""),
                name=scheme.get("name", ""),
                description=scheme.get("description", ""),
                category=scheme.get("category", ""),
                relevance_score=relevance_score,
                matched_keywords=matched_tags,
                tags=scheme_tags,
                criteria=scheme.get("criteria", {}),
                benefits=scheme.get("benefits"),
                application_process=scheme.get("application_process")
            )
            results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:max_results]
    
    def search_by_category(
        self,
        category: str,
        max_results: int = 10
    ) -> List[SchemeSearchResult]:
        """
        Search schemes by category
        
        Args:
            category: Category name
            max_results: Maximum number of results
            
        Returns:
            List of matching schemes
        """
        category_normalized = category.lower().strip()
        
        results = []
        for scheme in self.schemes_data:
            scheme_category = scheme.get("category", "").lower().strip()
            
            if category_normalized in scheme_category or scheme_category in category_normalized:
                result = SchemeSearchResult(
                    scheme_id=scheme.get("id", ""),
                    name=scheme.get("name", ""),
                    description=scheme.get("description", ""),
                    category=scheme.get("category", ""),
                    relevance_score=1.0,
                    matched_keywords=[category],
                    tags=scheme.get("tags", []),
                    criteria=scheme.get("criteria", {}),
                    benefits=scheme.get("benefits"),
                    application_process=scheme.get("application_process")
                )
                results.append(result)
        
        return results[:max_results]
    
    def get_scheme_by_id(self, scheme_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific scheme by ID
        
        Args:
            scheme_id: Scheme identifier
            
        Returns:
            Scheme dictionary or None if not found
        """
        for scheme in self.schemes_data:
            if scheme.get("id") == scheme_id:
                return scheme
        return None
    
    def get_all_schemes(self) -> List[Dict[str, Any]]:
        """
        Get all schemes
        
        Returns:
            List of all scheme dictionaries
        """
        return self.schemes_data.copy()
    
    def get_all_categories(self) -> List[str]:
        """
        Get list of all unique categories
        
        Returns:
            List of category names
        """
        categories = set()
        for scheme in self.schemes_data:
            category = scheme.get("category", "").strip()
            if category:
                categories.add(category)
        return sorted(list(categories))
    
    def get_all_tags(self) -> List[str]:
        """
        Get list of all unique tags
        
        Returns:
            List of tags
        """
        tags = set()
        for scheme in self.schemes_data:
            scheme_tags = scheme.get("tags", [])
            tags.update(scheme_tags)
        return sorted(list(tags))
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get retrieval statistics
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_schemes": len(self.schemes_data),
            "total_categories": len(self.get_all_categories()),
            "total_tags": len(self.get_all_tags()),
            "indexed_keywords": len(self.indexed_schemes),
            "schemes_file": str(self.schemes_file_path)
        }
    
    def reload_schemes(self):
        """Reload schemes from file and rebuild index"""
        self._load_schemes()
        self._build_index()


# Convenience function for quick retrieval
def retrieve_schemes(
    keywords: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    category: Optional[str] = None,
    schemes_file_path: Optional[str] = None,
    max_results: int = 10
) -> List[Dict[str, Any]]:
    """
    Convenience function for one-shot scheme retrieval
    
    Args:
        keywords: Search keywords
        tags: Search tags
        category: Category filter
        schemes_file_path: Path to schemes.json
        max_results: Maximum results
        
    Returns:
        List of matching scheme dictionaries
    """
    retriever = SchemeRetriever(schemes_file_path)
    
    if keywords:
        results = retriever.search_by_keywords(keywords, max_results)
    elif tags:
        results = retriever.search_by_tags(tags, max_results=max_results)
    elif category:
        results = retriever.search_by_category(category, max_results)
    else:
        # Return all schemes
        all_schemes = retriever.get_all_schemes()
        return all_schemes[:max_results]
    
    return [r.to_dict() for r in results]