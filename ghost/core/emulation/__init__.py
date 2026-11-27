"""
src/core/emulation — OpenAI parameter and agentic tool-calling emulation engine.
Provides request transpilation, prompt steering, and streaming response interception.
"""
from .pipeline import EmulationPipeline

__all__ = ["EmulationPipeline"]
