"""
Hindi-English Book Translation System
A comprehensive system for translating books from Hindi to English
"""

__version__ = "1.0.0"
__author__ = "Sid Dani"

# Import main components
from .translation_engine import TranslationEngine, TranslationConfig
from .document_processor import DocumentProcessor, DocumentSegment
from .cost_meter import CostMeter
from .quality_assurance import QualityAssurance, QAConfig
from .latex_output import LaTeXOutputGenerator
from .main_controller import BookTranslationController

__all__ = [
    "TranslationEngine",
    "TranslationConfig",
    "DocumentProcessor",
    "DocumentSegment",
    "CostMeter",
    "QualityAssurance",
    "QAConfig",
    "LaTeXOutputGenerator",
    "BookTranslationController",
] 