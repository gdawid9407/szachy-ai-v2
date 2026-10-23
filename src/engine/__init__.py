"""Moduł silnika reguł szachowych."""

from src.engine.board_manager import BoardManager
from src.engine.fen_parser import FenParser, FenValidationResult

__all__ = ["BoardManager", "FenParser", "FenValidationResult"]
