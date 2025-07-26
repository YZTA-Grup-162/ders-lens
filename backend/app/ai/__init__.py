"""
AI module initialization
"""
from .attention_detector import (AttentionData, AttentionDetector,
                                 AttentionLevel, DistractionType)
__all__ = ['AttentionDetector', 'AttentionLevel', 'DistractionType', 'AttentionData']