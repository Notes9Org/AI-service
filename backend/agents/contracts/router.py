"""Router decision contract for agent schemas."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Literal, List, Dict, Any, Optional

RouteKind = Literal["in_scope", "out_of_scope"]


class RouterDecision(BaseModel):
    """Router decision schema for agent execution and tool selection."""
    tools: List[Literal["sql", "rag"]] = Field(
        default_factory=list,
        description="Selected tools: ['sql'], ['rag'], ['sql', 'rag'], or [] when route is out_of_scope."
    )
    route: RouteKind = Field(
        default="in_scope",
        description="'in_scope' for normal tool routing; 'out_of_scope' for revert (no tools)."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for routing decision."
    )
    reasoning: str = Field(
        ...,
        description="Human readable explanation of routing decision."
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional constraints for the routing decision like date ranges, filters, limits, etc."
    )

    @field_validator("tools")
    @classmethod
    def validate_tools(cls, v: List[str]) -> List[str]:
        """Validate tools value: only 'sql' and 'rag', or empty when route is out_of_scope."""
        allowed_tools = ["sql", "rag"]
        if v and not all(t in allowed_tools for t in v):
            raise ValueError(f"Invalid tools: {v}. Must be subset of: {allowed_tools}")
        return v

    @model_validator(mode="after")
    def validate_route_and_tools(self) -> "RouterDecision":
        """When route is out_of_scope, tools must be empty; when in_scope, tools must be non-empty."""
        if self.route == "out_of_scope":
            if self.tools:
                raise ValueError("When route is out_of_scope, tools must be empty")
        else:
            if not self.tools:
                raise ValueError("When route is in_scope, tools must be non-empty")
        return self