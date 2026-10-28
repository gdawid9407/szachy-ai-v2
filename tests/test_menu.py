"""
tests/test_menu.py

Zestaw testów jednostkowych dla modułu menu i obsługi komend Menu (Zadanie 2.3).
Weryfikuje:
- Formatowanie menu głównego, wyboru strony, poziomów AI, pomocy i historii
- Precyzyjne parsowanie komend w trakcie rozgrywki (exit, back, history, switch, fen, restart, move)
- Działanie procedur wyboru trybu gry (Human vs Human, Human vs AI, AI vs AI)
- Działanie procedur wyboru koloru bierek (Białe, Czarne, Losowo)
- Pobieranie i walidację FEN przez menu
- Konfigurację parametrów partii w GameConfig
"""

import io
import unittest
from unittest.mock import patch
import chess

from src.engine.board_manager import BoardManager
from src.ui.menu import GameConfig, GameMode, InGameAction, Menu, ParsedCommand


class TestMenu(unittest.TestCase):

    def setUp(self) -> None:
        self.menu = Menu(use_colors=False)
        self.bm = BoardManager()

    def test_format_main_menu(self) -> None:
        """Sprawdza obecność kluczowych opcji w tekście menu głównego."""
        text = self.menu.format_main_menu()
        self.assertIn("Human vs Human", text)
        self.assertIn("Human vs AI", text)
        self.assertIn("AI vs AI", text)
        self.assertIn("FEN", text)
        self.assertIn("Exit", text)

    def test_format_help(self) -> None:
        """Sprawdza obecność komend sterujących w tekście pomocy."""
        text = self.menu.format_help()
        self.assertIn("back / undo", text)
        self.assertIn("history", text)
        self.assertIn("switch", text)
        self.assertIn("fen", text)
        self.assertIn("restart", text)
        self.assertIn("exit", text)

    def test_format_history(self) -> None:
        """Weryfikuje formatowanie historii pustej i po wykonaniu ruchów."""
        # Pusta historia
        self.assertIn("brak", self.menu.format_history(self.bm))

        # Historia po posunięciach
        self.bm.push_uci("e2e4")
        self.bm.push_uci("e7e5")
        text = self.menu.format_history(self.bm)
        self.assertIn("1. e4", text)
        self.assertIn("e5", text)

    def test_parse_in_game_commands(self) -> None:
        """Testuje parsowanie komend tekstowych wprowadzanych podczas gry."""
        # Wyjście
        self.assertEqual(self.menu.parse_in_game_input("exit").action, InGameAction.EXIT)
        self.assertEqual(self.menu.parse_in_game_input("QUIT").action, InGameAction.EXIT)
        self.assertEqual(self.menu.parse_in_game_input("q").action, InGameAction.EXIT)

        # Cofanie (Back move)
        self.assertEqual(self.menu.parse_in_game_input("back").action, InGameAction.BACK)
        self.assertEqual(self.menu.parse_in_game_input("undo").action, InGameAction.BACK)
        self.assertEqual(self.menu.parse_in_game_input("b").action, InGameAction.BACK)

        # Historia
        self.assertEqual(self.menu.parse_in_game_input("history").action, InGameAction.HISTORY)
        self.assertEqual(self.menu.parse_in_game_input("hist").action, InGameAction.HISTORY)

        # Zmiana strony (Switch side)
        self.assertEqual(self.menu.parse_in_game_input("switch").action, InGameAction.SWITCH_SIDE)
        self.assertEqual(self.menu.parse_in_game_input("flip").action, InGameAction.SWITCH_SIDE)

        # Restart
        self.assertEqual(self.menu.parse_in_game_input("restart").action, InGameAction.RESTART)
        self.assertEqual(self.menu.parse_in_game_input("reset").action, InGameAction.RESTART)

        # Pomoc
        self.assertEqual(self.menu.parse_in_game_input("help").action, InGameAction.HELP)
        self.assertEqual(self.menu.parse_in_game_input("?").action, InGameAction.HELP)
        self.assertEqual(self.menu.parse_in_game_input("").action, InGameAction.HELP)

    def test_parse_fen_command(self) -> None:
        """Weryfikuje parsowanie komendy wczytania FEN z parametrem."""
        fen_str = "8/8/8/8/8/8/8/8 w - - 0 1"
        cmd = self.menu.parse_in_game_input(f"fen {fen_str}")
        self.assertEqual(cmd.action, InGameAction.LOAD_FEN)
        self.assertEqual(cmd.payload, fen_str)

    def test_parse_move_command(self) -> None:
        """Standardowe posunięcia szachowe są klasyfikowane jako MOVE."""
        cmd1 = self.menu.parse_in_game_input("e2e4")
        self.assertEqual(cmd1.action, InGameAction.MOVE)
        self.assertEqual(cmd1.payload, "e2e4")

        cmd2 = self.menu.parse_in_game_input("e7e8q")
        self.assertEqual(cmd2.action, InGameAction.MOVE)
        self.assertEqual(cmd2.payload, "e7e8q")

    def test_prompt_game_mode(self) -> None:
        """Sprawdza wybór trybów gry za pomocą zamockowanego wejścia."""
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(self.menu.prompt_game_mode(input_func=lambda _: "1"), GameMode.HUMAN_VS_HUMAN)
            self.assertEqual(self.menu.prompt_game_mode(input_func=lambda _: "2"), GameMode.HUMAN_VS_AI)
            self.assertEqual(self.menu.prompt_game_mode(input_func=lambda _: "3"), GameMode.AI_VS_AI)
            self.assertIsNone(self.menu.prompt_game_mode(input_func=lambda _: "0"))

    def test_prompt_side_selection(self) -> None:
        """Sprawdza wybór koloru bierek przez gracza."""
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(self.menu.prompt_side_selection(input_func=lambda _: "1"), chess.WHITE)
            self.assertEqual(self.menu.prompt_side_selection(input_func=lambda _: "2"), chess.BLACK)
            self.assertIn(self.menu.prompt_side_selection(input_func=lambda _: "3"), [chess.WHITE, chess.BLACK])

    def test_prompt_ai_tier(self) -> None:
        """Sprawdza wybór poziomu AI (Tier 0)."""
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(self.menu.prompt_ai_tier(input_func=lambda _: "0"), 0)
            self.assertEqual(self.menu.prompt_ai_tier(input_func=lambda _: ""), 0)

    def test_prompt_fen_validation(self) -> None:
        """Sprawdza pobieranie poprawnego FEN oraz anulowanie."""
        with patch("sys.stdout", new=io.StringIO()):
            valid_fen = chess.STARTING_FEN
            loaded = self.menu.prompt_fen(input_func=lambda _: valid_fen)
            self.assertEqual(loaded, valid_fen)

            cancelled = self.menu.prompt_fen(input_func=lambda _: "anuluj")
            self.assertIsNone(cancelled)

    def test_prompt_full_setup_human_vs_human(self) -> None:
        """Weryfikuje konfigurację trybu Human vs Human."""
        with patch("sys.stdout", new=io.StringIO()):
            config = self.menu.prompt_full_setup(input_func=lambda _: "1")
            self.assertIsNotNone(config)
            self.assertEqual(config.mode, GameMode.HUMAN_VS_HUMAN)

    def test_prompt_full_setup_human_vs_ai(self) -> None:
        """Weryfikuje konfigurację trybu Human vs AI."""
        with patch("sys.stdout", new=io.StringIO()):
            inputs = iter(["2", "1", "0"])
            config = self.menu.prompt_full_setup(input_func=lambda _: next(inputs))
            self.assertIsNotNone(config)
            self.assertEqual(config.mode, GameMode.HUMAN_VS_AI)
            self.assertEqual(config.human_color, chess.WHITE)
            self.assertEqual(config.ai_black_tier, 0)


if __name__ == "__main__":
    unittest.main()
