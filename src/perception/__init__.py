"""
INIDS Perception Layer - Makes the system's reasoning transparent.

Modules:
- attack_story: Constructs narrative timelines of attack progression
- confidence_breakdown: Explains model decision factors
- live_pulse: Real-time animated metrics dashboard
"""

from .attack_story import AttackStoryEngine
from .confidence_breakdown import ConfidenceBreakdownEngine
from .live_pulse import LiveSystemPulse

__all__ = [
    "AttackStoryEngine",
    "ConfidenceBreakdownEngine",
    "LiveSystemPulse",
]
