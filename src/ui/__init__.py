"""
src/ui/__init__.py

Pakiet interfejsu użytkownika (UI), prezentacji planszy oraz menu konsolowego.
"""

from src.ui.board_view import BoardView
from src.ui.menu import GameConfig, GameMode, InGameAction, Menu, ParsedCommand

__all__ = [
    "BoardView",
    "Menu",
    "GameMode",
    "InGameAction",
    "ParsedCommand",
    "GameConfig",
]
