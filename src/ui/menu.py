"""
src/ui/menu.py

Moduł obsługi menu gry, konfiguracji rozgrywki oraz parsowania komend gracza.
Obsługuje:
- Tryby rozgrywki: Human vs Human, Human vs AI, AI vs AI.
- Wybór strony przed grą (Białe, Czarne, Losowo).
- Dynamiczną zmianę strony (switch side) w trakcie rozgrywki.
- Cofanie ruchów (back move / undo).
- Przegląd historii wykonanych ruchów.
- Wczytywanie własnych pozycji szachowych w formacie FEN (z walidacją FenParser).
- Bezpieczne wyjście (exit) oraz system pomocy (help).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Callable, Final, List, Optional, Tuple
import chess

from src.engine.board_manager import BoardManager
from src.engine.fen_parser import FenParser, FenValidationResult


# Kolory i formatowanie ANSI
ANSI_RESET: Final[str] = "\033[0m"
ANSI_BOLD: Final[str] = "\033[1m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_GREEN: Final[str] = "\033[92m"
ANSI_YELLOW: Final[str] = "\033[93m"
ANSI_RED: Final[str] = "\033[91m"


class GameMode(str, Enum):
    """Dostępne tryby rozgrywki."""
    HUMAN_VS_HUMAN = "human_vs_human"
    HUMAN_VS_AI = "human_vs_ai"
    AI_VS_AI = "ai_vs_ai"


class InGameAction(str, Enum):
    """Akcje i polecenia wydawane przez użytkownika podczas partii."""
    MOVE = "move"
    BACK = "back"
    HISTORY = "history"
    SWITCH_SIDE = "switch_side"
    LOAD_FEN = "load_fen"
    RESTART = "restart"
    HELP = "help"
    EXIT = "exit"


@dataclass(frozen=True)
class ParsedCommand:
    """Reprezentacja sparsowanego polecenia lub ruchu gracza."""
    action: InGameAction
    payload: Optional[str] = None


@dataclass
class GameConfig:
    """Kompletna konfiguracja rozpoczęcia partii."""
    mode: GameMode = GameMode.HUMAN_VS_HUMAN
    human_color: chess.Color = chess.WHITE
    ai_tier: int = 0
    ai_white_tier: int = 0
    ai_black_tier: int = 0
    initial_fen: Optional[str] = None


class Menu:
    """
    Zarządza interfejsem tekstowym menu, konfiguracją gry i interpretacją komend.
    """

    def __init__(self, use_colors: bool = True) -> None:
        self.use_colors: bool = use_colors

    # --------------------------------------------------------------------------
    # Formatowanie widoków menu i komunikatów (Czyste funkcje)
    # --------------------------------------------------------------------------

    def format_main_menu(self) -> str:
        """Zwraca sformatowany tekst głównego menu wyboru trybu rozgrywki."""
        lines = [
            "============================================================",
            "                 SZACHY AI v2 - MENU GŁÓWNE                 ",
            "============================================================",
            "  1. Człowiek vs Człowiek (Human vs Human)",
            "  2. Człowiek vs Sztuczna Inteligencja (Human vs AI)",
            "  3. Mecz AI vs AI (AI vs AI)",
            "  4. Wczytaj pozycję z FEN (chess.com / Lichess)",
            "  0. Wyjście z programu (Exit)",
            "============================================================",
        ]
        return "\n".join(lines)

    def format_side_selection(self) -> str:
        """Zwraca menu wyboru strony przed rozpoczęciem gry."""
        lines = [
            "------------------------------------------------------------",
            "  Wybierz kolor bierek:",
            "  1. Białe (White - pierwszy ruch)",
            "  2. Czarne (Black)",
            "  3. Losowo (Random)",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_ai_tier_selection(self) -> str:
        """Zwraca menu wyboru poziomu trudności / modelu AI."""
        lines = [
            "------------------------------------------------------------",
            "  Wybierz poziom AI:",
            "  0. Poziom 0: AI bez wiedzy (losowe, legalne posunięcia)",
            "  1. Poziom 1: AI z gier historycznych (Supervised - wkrótce)",
            "  2. Poziom 2: AI z gier własnych (Self-Play - wkrótce)",
            "  3. Poziom 3: Połączenie wszystkiego (Hybrid Engine - wkrótce)",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_help(self) -> str:
        """Zwraca listę komend dostępnych podczas partii."""
        lines = [
            "======================= POMOC / KOMENDY =====================",
            "  Ruch szachowy : Wpisz ruch w formacie UCI (np. e2e4, g1f3)",
            "                  Promocja piona: dopisz figurę (np. e7e8q)",
            "  back / undo   : Cofnij ostatnie posunięcie",
            "  history       : Wyświetl pełną historię ruchów w partii",
            "  switch        : Przejmij stronę przeciwnika (w trybie vs AI)",
            "  fen <kod_fen> : Wczytaj nowe ustawienie planszy z FEN",
            "  restart       : Zresetuj planszę do pozycji początkowej",
            "  help / ?      : Wyświetl tę pomoc",
            "  exit / quit   : Zakończ bieżącą rozgrywkę",
            "============================================================",
        ]
        return "\n".join(lines)

    def format_history(self, board_manager: BoardManager) -> str:
        """Zwraca sformatowaną historię ruchów partii w standardowej notacji."""
        formatted = board_manager.get_formatted_history()
        if not formatted:
            return "Historia ruchów: brak (pozycja początkowa)."

        lines = ["--- HISTORIA ROZGRYWKI ---"]
        lines.extend(formatted)
        lines.append("---------------------------")
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Parsowanie wejścia gracza w trakcie rozgrywki
    # --------------------------------------------------------------------------

    def parse_in_game_input(self, raw_input: str) -> ParsedCommand:
        """
        Analizuje łańcuch wpisany przez użytkownika i klasyfikuje go jako
        ruch szachowy lub komendę sterującą.

        Args:
            raw_input: Ciąg wprowadzony przez użytkownika w konsoli.

        Returns:
            Instancja ParsedCommand zawierająca InGameAction i ewentualny parametr.
        """
        cleaned = raw_input.strip()
        lower = cleaned.lower()

        if not lower:
            return ParsedCommand(action=InGameAction.HELP)

        # Komendy sterujące
        if lower in ("exit", "quit", "q"):
            return ParsedCommand(action=InGameAction.EXIT)
        if lower in ("back", "undo", "b"):
            return ParsedCommand(action=InGameAction.BACK)
        if lower in ("history", "hist", "h"):
            return ParsedCommand(action=InGameAction.HISTORY)
        if lower in ("switch", "flip", "switch_side"):
            return ParsedCommand(action=InGameAction.SWITCH_SIDE)
        if lower in ("restart", "reset"):
            return ParsedCommand(action=InGameAction.RESTART)
        if lower in ("help", "?", "pomoc"):
            return ParsedCommand(action=InGameAction.HELP)

        # Załadowanie FEN (np. "fen rnbqkbnr/..." lub "load_fen ...")
        if lower.startswith("fen ") or lower.startswith("load_fen "):
            parts = cleaned.split(maxsplit=1)
            fen_str = parts[1].strip() if len(parts) > 1 else ""
            return ParsedCommand(action=InGameAction.LOAD_FEN, payload=fen_str)

        # Domyślnie traktowane jako próba wykonania ruchu (UCI)
        return ParsedCommand(action=InGameAction.MOVE, payload=cleaned)

    # --------------------------------------------------------------------------
    # Interaktywne procedury dialogowe (wspierające wstrzykiwanie input_func)
    # --------------------------------------------------------------------------

    def prompt_game_mode(self, input_func: Callable[[str], str] = input) -> Optional[GameMode]:
        """Pyta użytkownika o wybór trybu gry."""
        while True:
            print(self.format_main_menu())
            choice = input_func("Wybierz opcję (0-4): ").strip()
            if choice == "1":
                return GameMode.HUMAN_VS_HUMAN
            if choice == "2":
                return GameMode.HUMAN_VS_AI
            if choice == "3":
                return GameMode.AI_VS_AI
            if choice == "4":
                return None  # Sygnał załadowania FEN przed wyborem trybu
            if choice in ("0", "exit", "quit", "q"):
                return None
            print("Niepoprawny wybór! Wpisz cyfrę od 0 do 4.")

    def prompt_side_selection(
        self,
        input_func: Callable[[str], str] = input,
    ) -> chess.Color:
        """Pyta gracza o wybór koloru bierek (Białe / Czarne / Losowo)."""
        while True:
            print(self.format_side_selection())
            choice = input_func("Wybierz stronę (1-3) [Domyślnie: 1 - Białe]: ").strip()
            if choice in ("1", "w", "white", "biale", ""):
                return chess.WHITE
            if choice in ("2", "b", "black", "czarne"):
                return chess.BLACK
            if choice in ("3", "r", "random", "losowo"):
                return chess.WHITE if random.choice([True, False]) else chess.BLACK
            print("Niepoprawna opcja! Wpisz 1, 2 lub 3.")

    def prompt_ai_tier(
        self,
        player_label: str = "AI",
        input_func: Callable[[str], str] = input,
    ) -> int:
        """Pyta o poziom trudności AI (obecnie dostępny Tier 0)."""
        while True:
            print(self.format_ai_tier_selection())
            choice = input_func(f"Wybierz poziom dla {player_label} (0-3) [Domyślnie: 0]: ").strip()
            if choice in ("0", ""):
                return 0
            if choice in ("1", "2", "3"):
                print("Poziomy 1-3 są w trakcie implementacji w kolejnych fazach projektu.")
                print("Ustawiono Poziom 0 (AI bez wiedzy).")
                return 0
            print("Niepoprawny poziom! Wybierz 0.")

    def prompt_fen(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[str]:
        """Pobiera i waliduje łańcuch FEN wprowadzony przez użytkownika."""
        while True:
            fen_input = input_func("Wklej kod pozycji FEN (lub 'anuluj'): ").strip()
            if fen_input.lower() in ("anuluj", "cancel", "c", "exit", ""):
                return None

            result: FenValidationResult = FenParser.validate(fen_input)
            if result.is_valid and result.cleaned_fen is not None:
                print(f"Pozycja FEN poprawna: {result.cleaned_fen}")
                return result.cleaned_fen

            print(f"Błąd walidacji FEN: {result.error_message}")
            print("Spróbuj ponownie lub wpisz 'anuluj'.")

    def prompt_full_setup(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[GameConfig]:
        """
        Przeprowadza pełną procedurę konfiguracyjną nowej partii.
        """
        mode = self.prompt_game_mode(input_func=input_func)
        if mode is None:
            # Użytkownik mógł wybrać opcję 4 (wczytaj FEN) lub wyjście
            return None

        initial_fen: Optional[str] = None
        human_color: chess.Color = chess.WHITE
        ai_tier: int = 0
        ai_white_tier: int = 0
        ai_black_tier: int = 0

        if mode == GameMode.HUMAN_VS_HUMAN:
            pass  # Domyślne ustawienia
        elif mode == GameMode.HUMAN_VS_AI:
            human_color = self.prompt_side_selection(input_func=input_func)
            ai_tier = self.prompt_ai_tier(player_label="Przeciwnika AI", input_func=input_func)
            if human_color == chess.WHITE:
                ai_black_tier = ai_tier
            else:
                ai_white_tier = ai_tier
        elif mode == GameMode.AI_VS_AI:
            ai_white_tier = self.prompt_ai_tier(player_label="Białego AI", input_func=input_func)
            ai_black_tier = self.prompt_ai_tier(player_label="Czarnego AI", input_func=input_func)

        return GameConfig(
            mode=mode,
            human_color=human_color,
            ai_tier=ai_tier,
            ai_white_tier=ai_white_tier,
            ai_black_tier=ai_black_tier,
            initial_fen=initial_fen,
        )

