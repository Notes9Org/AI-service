"""Validation functions for normalize node output."""
from typing import Tuple, List
from agents.contracts.normalized import NormalizedQuery
from agents.graph.state import AgentState


def validate_normalized_output(
    normalized: NormalizedQuery,
    request: dict
) -> Tuple[bool, List[str]]:
    """
    Validate normalized output against invariants.
    
    Args:
        normalized: NormalizedQuery object to validate
        request: Original request dict from state
        
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues: List[str] = []

    # Domain/scope invariants (single source of truth)
    if not normalized.in_scope:
        if not normalized.out_of_scope_reason or not str(normalized.out_of_scope_reason).strip():
            issues.append("in_scope is false but out_of_scope_reason is empty")
    else:
        if normalized.intent not in ("aggregate", "search", "hybrid"):
            issues.append("in_scope is true but intent must be one of aggregate, search, hybrid")

    # Intent/context invariants only when in-scope
    if normalized.in_scope:
        if normalized.intent == "aggregate":
            if not normalized.context.get("requires_aggregation"):
                issues.append(
                    "Intent is 'aggregate' but context.requires_aggregation is not True"
                )
        if normalized.intent == "search":
            if not normalized.context.get("requires_semantic_search"):
                issues.append(
                    "Intent is 'search' but context.requires_semantic_search is not True"
                )
        if normalized.intent == "hybrid":
            if not normalized.context.get("requires_aggregation"):
                issues.append(
                    "Intent is 'hybrid' but context.requires_aggregation is not True"
                )
            if not normalized.context.get("requires_semantic_search"):
                issues.append(
                    "Intent is 'hybrid' but context.requires_semantic_search is not True"
                )
    
    # Normalized query not empty
    if not normalized.normalized_query or not normalized.normalized_query.strip():
        issues.append("normalized_query is empty")
    
    if not isinstance(normalized.entities, dict):
        issues.append("entities is not a dict")
    
    if not isinstance(normalized.context, dict):
        issues.append("context is not a dict")
    
    return len(issues) == 0, issues
