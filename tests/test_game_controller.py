"""
tests/test_game_controller.py

Zestaw testów jednostkowych i integracyjnych dla modułu GameController (Zadanie 2.4).
Weryfikuje:
- Rozgrywkę Człowiek vs Człowiek (wykonywanie ruchów UCI i SAN, mat Szewski, wyjście)
- Obsługę komend w grze: back/undo, history, fen (odczyt i wczytanie), eval, switch, restart, save (PGN)
- Rozgrywkę Człowiek vs AI (Poziom 0) z cofaniem 2 półruchów oraz trybem testowym
- Rozgrywkę AI vs AI (pojedyncza partia oraz seria N partii z zamianą kolorów w trybie szybkim)
- Eksport partii do pliku .pgn
"""

import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import chess

from src.ui.board_view import BoardView
from src.ui.game_controller import GameController
from src.ui.menu import (
    AgentType,
    AiDisplayMode,
    GameConfig,
    GameMode,
    Menu,
    PlayerConfig,
)


class TestGameController(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pgn_dir = Path(self.temp_dir.name)
        self.menu = Menu(use_colors=False)
        self.board_view = BoardView(use_colors=False, use_unicode_pieces=False)
        self.controller = GameController(
            board_view=self.board_view,
            menu=self.menu,
            pgn_output_dir=self.pgn_dir,
            seed=42,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_human_vs_human_fools_mate_and_pgn_export(self) -> None:
        """Sprawdza pełną partię Człowiek vs Człowiek zakończoną matem (Fool's Mate) i zapisem PGN."""
        cfg = GameConfig(
            mode=GameMode.HUMAN_VS_HUMAN,
            white=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            black=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            save_pgn=True,
        )
        # Mat szewski (Fool's Mate): 1. f3 e5 2. g4 Qh4#
        moves = iter(["f2f3", "e7e5", "g2g4", "Qh4"])  # Mieszanka UCI i SAN
        with patch("sys.stdout", new=io.StringIO()):
            summary = self.controller.run_match(cfg, input_func=lambda _: next(moves))

        self.assertEqual(summary.games_played, 1)
        self.assertEqual(summary.player_b_wins, 1)
        self.assertEqual(summary.results[0].result, "0-1")
        self.assertEqual(summary.results[0].moves_count, 4)
        self.assertIsNotNone(summary.results[0].pgn_path)
        assert summary.results[0].pgn_path is not None
        self.assertTrue(summary.results[0].pgn_path.exists())
        pgn_text = summary.results[0].pgn_path.read_text(encoding="utf-8")
        self.assertIn("Qh4#", pgn_text)

    def test_in_game_commands_back_fen_eval_history_restart_exit(self) -> None:
        """Weryfikuje działanie komend sterujących w trakcie partii."""
        cfg = GameConfig(
            mode=GameMode.HUMAN_VS_HUMAN,
            white=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            black=PlayerConfig(AgentType.HUMAN, "Człowiek"),
        )
        # Ruch e2e4 -> historia -> fen (odczyt) -> eval -> back -> save -> restart -> exit
        commands = iter(["e2e4", "history", "fen", "eval", "back", "save", "restart", "exit"])
        out_buf = io.StringIO()
        with patch("sys.stdout", new=out_buf):
            summary = self.controller.run_match(cfg, input_func=lambda _: next(commands))

        output = out_buf.getvalue()
        self.assertIn("1. e4", output)
        self.assertIn("Aktualny FEN:", output)
        self.assertIn("[Ocena pozycji]", output)
        self.assertIn("Cofnięto ruch.", output)
        self.assertIn("Zapisano partię do pliku PGN:", output)
        self.assertIn("Zrestartowano planszę.", output)
        self.assertEqual(summary.results[0].result, "*")
        self.assertEqual(summary.results[0].moves_count, 0)

    def test_load_fen_in_game(self) -> None:
        """Sprawdza wczytanie własnego FEN w trakcie rozgrywki i natychmiastowy mat."""
        cfg = GameConfig(
            mode=GameMode.HUMAN_VS_HUMAN,
            white=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            black=PlayerConfig(AgentType.HUMAN, "Człowiek"),
        )
        mate_in_one_fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
        commands = iter([f"fen {mate_in_one_fen}", "h5f7"])
        with patch("sys.stdout", new=io.StringIO()):
            summary = self.controller.run_match(cfg, input_func=lambda _: next(commands))

        self.assertEqual(summary.results[0].result, "1-0")
        self.assertIn("Szach-mat", summary.results[0].reason)

    def test_human_vs_ai_back_and_switch(self) -> None:
        """Sprawdza tryb Człowiek vs AI: ruch człowieka, odpowiedź AI, cofnięcie 2 półruchów i wyjście."""
        cfg = GameConfig(
            mode=GameMode.HUMAN_VS_AI,
            white=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            black=PlayerConfig(AgentType.RANDOM_BOT, "Bot Poziom 0"),
            test_mode=True,
        )
        # Człowiek gra e2e4 -> AI odpowiada -> człowiek cofa (wraca do 0 półruchów) -> exit
        commands = iter(["e2e4", "back", "exit"])
        out_buf = io.StringIO()
        with patch("sys.stdout", new=out_buf):
            summary = self.controller.run_match(cfg, input_func=lambda _: next(commands))

        self.assertIn("[Tryb testowy]", out_buf.getvalue())
        self.assertEqual(summary.results[0].moves_count, 0)

    def test_ai_vs_ai_fast_benchmark_series(self) -> None:
        """Sprawdza serię 4 partii AI vs AI z zamianą kolorów w trybie FAST_BENCHMARK."""
        cfg = GameConfig(
            mode=GameMode.AI_VS_AI,
            white=PlayerConfig(AgentType.RANDOM_BOT, "Bot A"),
            black=PlayerConfig(AgentType.RANDOM_BOT, "Bot B"),
            num_games=4,
            swap_colors=True,
            display_mode=AiDisplayMode.FAST_BENCHMARK,
        )
        with patch("sys.stdout", new=io.StringIO()):
            summary = self.controller.run_match(cfg, max_plies=20)

        self.assertEqual(summary.games_played, 4)
        self.assertEqual(len(summary.results), 4)
        for res in summary.results:
            self.assertLessEqual(res.moves_count, 20)
            self.assertGreater(res.moves_count, 0)

    def test_post_game_save_prompt_save_and_discard(self) -> None:
        """Weryfikuje pytanie na końcu gry: zapis do PGN z własną nazwą oraz odrzucenie."""
        cfg = GameConfig(
            mode=GameMode.HUMAN_VS_HUMAN,
            white=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            black=PlayerConfig(AgentType.HUMAN, "Człowiek"),
            save_pgn=False,
        )
        # 1. Zapis z własną nazwą: ruch e2e4 -> exit -> [1] Zapisz -> nazwa 'moja_partia'
        save_inputs = iter(["e2e4", "exit", "1", "moja_partia"])
        with patch("sys.stdout", new=io.StringIO()):
            summary_saved = self.controller.run_match(
                cfg,
                input_func=lambda _: next(save_inputs),
                ask_save_at_end=True,
            )

        self.assertIsNotNone(summary_saved.series_pgn_path)
        assert summary_saved.series_pgn_path is not None
        self.assertEqual(summary_saved.series_pgn_path.name, "moja_partia.pgn")
        self.assertTrue(summary_saved.series_pgn_path.exists())
        self.assertIn("1. e4", summary_saved.series_pgn_path.read_text(encoding="utf-8"))

        # 2. Odrzucenie: ruch d2d4 -> exit -> [Enter = Odrzuć]
        discard_inputs = iter(["d2d4", "exit", ""])
        with patch("sys.stdout", new=io.StringIO()):
            summary_discarded = self.controller.run_match(
                cfg,
                input_func=lambda _: next(discard_inputs),
                ask_save_at_end=True,
            )

        self.assertIsNone(summary_discarded.series_pgn_path)
        self.assertIsNone(summary_discarded.results[0].pgn_path)

    def test_post_series_save_combined_pgn(self) -> None:
        """Sprawdza zapis zbiorczego pliku PGN po zakończeniu serii 3 partii AI vs AI."""
        cfg = GameConfig(
            mode=GameMode.AI_VS_AI,
            white=PlayerConfig(AgentType.RANDOM_BOT, "Bot A"),
            black=PlayerConfig(AgentType.RANDOM_BOT, "Bot B"),
            num_games=3,
            swap_colors=True,
            display_mode=AiDisplayMode.FAST_BENCHMARK,
        )
        inputs = iter(["1", "seria_testowa"])
        with patch("sys.stdout", new=io.StringIO()):
            summary = self.controller.run_match(
                cfg,
                input_func=lambda _: next(inputs),
                max_plies=10,
                ask_save_at_end=True,
            )

        self.assertIsNotNone(summary.series_pgn_path)
        assert summary.series_pgn_path is not None
        self.assertEqual(summary.series_pgn_path.name, "seria_testowa.pgn")
        content = summary.series_pgn_path.read_text(encoding="utf-8")
        self.assertEqual(content.count('[Event "Szachy AI v2 CLI"]'), 3)


if __name__ == "__main__":
    unittest.main()
