"""
tests/test_menu.py

Zestaw testów jednostkowych dla rozbudowanego modułu menu (Kreator Wyboru).
Weryfikuje:
- Formatowanie Menu Głównego (GRA / NAUKA / STAN I METRYKI), trybów gry, podsumowania i pomocy
- Parsowanie komend w trakcie rozgrywki (w tym SHOW_FEN, LOAD_FEN, EVAL, SAVE_PGN)
- Wspólny Kreator Agenta AI (Bot bazowy, Silnik UCI z Elo/Głębokością/Czasem, Własna Sieć AI)
- Pełny przepływ Kreatora Wyboru dla wszystkich 4 trybów gry oraz cofanie [0] Powrót
- Presety FEN oraz walidację ręcznego FEN
- Szkielet Kreatora Treningu (NAUKA) oraz skanowanie katalogu models/ (STAN I METRYKI)
"""

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import chess

from src.engine.board_manager import BoardManager
from src.ui.menu import (
    AgentType,
    AiDisplayMode,
    FEN_PRESETS,
    GameConfig,
    GameMode,
    InGameAction,
    MainMenuAction,
    Menu,
    NeuralSearchMode,
    PlayerConfig,
    TrainingMethod,
    TrainingStartMode,
    UciLimitMode,
)


class TestMenu(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.models_dir = Path(self.temp_dir.name)
        self.menu = Menu(use_colors=False, models_dir=self.models_dir)
        self.bm = BoardManager()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # --------------------------------------------------------------------------
    # Testy formatowania widoków
    # --------------------------------------------------------------------------

    def test_format_main_menu(self) -> None:
        """Sprawdza obecność 3 głównych gałęzi i wyjścia w Menu Głównym."""
        text = self.menu.format_main_menu()
        self.assertIn("[1] GRA", text)
        self.assertIn("[2] NAUKA", text)
        self.assertIn("[3] STAN I METRYKI", text)
        self.assertIn("[0] WYJŚCIE", text)

    def test_format_play_mode_menu(self) -> None:
        """Sprawdza obecność wszystkich 4 trybów gry w gałęzi GRA."""
        text = self.menu.format_play_mode_menu()
        self.assertIn("Człowiek vs AI", text)
        self.assertIn("Człowiek vs Człowiek", text)
        self.assertIn("AI vs AI", text)
        self.assertIn("Konfigurator własny", text)

    def test_format_help(self) -> None:
        """Sprawdza obecność komend sterujących (w tym nowych) w tekście pomocy."""
        text = self.menu.format_help()
        self.assertIn("back / undo", text)
        self.assertIn("history", text)
        self.assertIn("switch", text)
        self.assertIn("fen", text)
        self.assertIn("eval / hint", text)
        self.assertIn("save", text)
        self.assertIn("restart", text)
        self.assertIn("exit", text)

    def test_format_history(self) -> None:
        """Weryfikuje formatowanie historii pustej i po wykonaniu ruchów."""
        self.assertIn("brak", self.menu.format_history(self.bm))
        self.bm.push_uci("e2e4")
        self.bm.push_uci("e7e5")
        text = self.menu.format_history(self.bm)
        self.assertIn("1. e4", text)
        self.assertIn("e5", text)

    def test_format_status_and_metrics_with_models(self) -> None:
        """Weryfikuje skanowanie linii treningowych i metadata.json w models/."""
        line_dir = self.models_dir / "hybrid_v1"
        line_dir.mkdir()
        (line_dir / "iter_010.pt").write_bytes(b"fake-ckpt")
        (line_dir / "metadata.json").write_text(
            json.dumps({"parent": "internet_v1/epoch_020.pt", "latest_loss": 0.42}),
            encoding="utf-8",
        )

        text = self.menu.format_status_and_metrics()
        self.assertIn("hybrid_v1", text)
        self.assertIn("1 checkpointów", text)
        self.assertIn("internet_v1/epoch_020.pt", text)
        self.assertIn("0.42", text)

    # --------------------------------------------------------------------------
    # Testy parsowania komend w trakcie gry
    # --------------------------------------------------------------------------

    def test_parse_in_game_commands(self) -> None:
        """Testuje parsowanie komend tekstowych wprowadzanych podczas gry."""
        self.assertEqual(self.menu.parse_in_game_input("exit").action, InGameAction.EXIT)
        self.assertEqual(self.menu.parse_in_game_input("QUIT").action, InGameAction.EXIT)
        self.assertEqual(self.menu.parse_in_game_input("q").action, InGameAction.EXIT)

        self.assertEqual(self.menu.parse_in_game_input("back").action, InGameAction.BACK)
        self.assertEqual(self.menu.parse_in_game_input("undo").action, InGameAction.BACK)
        self.assertEqual(self.menu.parse_in_game_input("b").action, InGameAction.BACK)

        self.assertEqual(self.menu.parse_in_game_input("history").action, InGameAction.HISTORY)
        self.assertEqual(self.menu.parse_in_game_input("hist").action, InGameAction.HISTORY)

        self.assertEqual(self.menu.parse_in_game_input("switch").action, InGameAction.SWITCH_SIDE)
        self.assertEqual(self.menu.parse_in_game_input("flip").action, InGameAction.SWITCH_SIDE)

        self.assertEqual(self.menu.parse_in_game_input("eval").action, InGameAction.EVAL)
        self.assertEqual(self.menu.parse_in_game_input("hint").action, InGameAction.EVAL)

        self.assertEqual(self.menu.parse_in_game_input("save").action, InGameAction.SAVE_PGN)
        self.assertEqual(self.menu.parse_in_game_input("pgn").action, InGameAction.SAVE_PGN)

        self.assertEqual(self.menu.parse_in_game_input("restart").action, InGameAction.RESTART)
        self.assertEqual(self.menu.parse_in_game_input("reset").action, InGameAction.RESTART)

        self.assertEqual(self.menu.parse_in_game_input("help").action, InGameAction.HELP)
        self.assertEqual(self.menu.parse_in_game_input("?").action, InGameAction.HELP)
        self.assertEqual(self.menu.parse_in_game_input("").action, InGameAction.HELP)

    def test_parse_fen_commands(self) -> None:
        """Rozróżnia samo 'fen' (SHOW_FEN) od 'fen <kod>' (LOAD_FEN)."""
        show_cmd = self.menu.parse_in_game_input("fen")
        self.assertEqual(show_cmd.action, InGameAction.SHOW_FEN)
        self.assertIsNone(show_cmd.payload)

        fen_str = "8/8/8/8/8/8/8/8 w - - 0 1"
        load_cmd = self.menu.parse_in_game_input(f"fen {fen_str}")
        self.assertEqual(load_cmd.action, InGameAction.LOAD_FEN)
        self.assertEqual(load_cmd.payload, fen_str)

    def test_parse_move_command(self) -> None:
        """Standardowe posunięcia szachowe są klasyfikowane jako MOVE."""
        cmd1 = self.menu.parse_in_game_input("e2e4")
        self.assertEqual(cmd1.action, InGameAction.MOVE)
        self.assertEqual(cmd1.payload, "e2e4")

        cmd2 = self.menu.parse_in_game_input("e7e8q")
        self.assertEqual(cmd2.action, InGameAction.MOVE)
        self.assertEqual(cmd2.payload, "e7e8q")

    # --------------------------------------------------------------------------
    # Testy Menu Głównego i Wspólnego Kreatora Agenta AI
    # --------------------------------------------------------------------------

    def test_prompt_main_menu(self) -> None:
        with patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(self.menu.prompt_main_menu(input_func=lambda _: "1"), MainMenuAction.PLAY)
            self.assertEqual(self.menu.prompt_main_menu(input_func=lambda _: "2"), MainMenuAction.TRAIN)
            self.assertEqual(self.menu.prompt_main_menu(input_func=lambda _: "3"), MainMenuAction.STATUS)
            self.assertEqual(self.menu.prompt_main_menu(input_func=lambda _: "0"), MainMenuAction.EXIT)

    def test_prompt_agent_random_and_uci(self) -> None:
        """Testuje wybór Bota bazowego oraz Silnika UCI (Elo / Głębokość)."""
        with patch("sys.stdout", new=io.StringIO()):
            bot = self.menu.prompt_agent_config(input_func=lambda _: "1")
            self.assertIsNotNone(bot)
            assert bot is not None
            self.assertEqual(bot.agent_type, AgentType.RANDOM_BOT)

            # UCI: opcja 2 -> silnik 'stockfish' -> limit Elo (1) -> 1800
            uci_inputs = iter(["2", "stockfish", "1", "1800"])
            uci = self.menu.prompt_agent_config(input_func=lambda _: next(uci_inputs))
            self.assertIsNotNone(uci)
            assert uci is not None
            self.assertEqual(uci.agent_type, AgentType.UCI_ENGINE)
            self.assertEqual(uci.uci_engine_name, "stockfish")
            self.assertEqual(uci.uci_limit_mode, UciLimitMode.ELO)
            self.assertEqual(uci.uci_limit_value, 1800)

    def test_prompt_agent_neural_net(self) -> None:
        """Testuje wybór Własnej Sieci AI, gdy istnieją zapisane checkpointy."""
        line_dir = self.models_dir / "internet_v1"
        line_dir.mkdir()
        (line_dir / "epoch_005.pt").write_bytes(b"ckpt1")
        (line_dir / "epoch_020.pt").write_bytes(b"ckpt2")

        with patch("sys.stdout", new=io.StringIO()):
            # opcja 3 (Sieć) -> linia 1 -> checkpoint 2 -> MCTS (1) -> 200 symulacji
            inputs = iter(["3", "1", "2", "1", "200"])
            nn = self.menu.prompt_agent_config(input_func=lambda _: next(inputs))
            self.assertIsNotNone(nn)
            assert nn is not None
            self.assertEqual(nn.agent_type, AgentType.NEURAL_NET)
            self.assertEqual(nn.model_line, "internet_v1")
            self.assertEqual(nn.checkpoint_name, "epoch_020.pt")
            self.assertEqual(nn.search_mode, NeuralSearchMode.MCTS)
            self.assertEqual(nn.mcts_simulations, 200)

    # --------------------------------------------------------------------------
    # Testy pełnego Kreatora Wyboru (GRA + NAUKA)
    # --------------------------------------------------------------------------

    def test_game_wizard_human_vs_ai_defaults(self) -> None:
        """Sprawdza szybkie przejście kreatora Człowiek vs AI z domyślnymi [Enter]."""
        with patch("sys.stdout", new=io.StringIO()):
            # Tryb 1 (Human vs AI) -> kolor [Enter=Białe] -> agent [Enter=Bot 0]
            # -> tryb testowy [Enter=Nie] -> pozycja [Enter=Start] -> potwierdzenie [Enter=Start]
            inputs = iter(["1", "", "", "", "", ""])
            cfg = self.menu.prompt_game_wizard(input_func=lambda _: next(inputs))
            self.assertIsNotNone(cfg)
            assert cfg is not None
            self.assertEqual(cfg.mode, GameMode.HUMAN_VS_AI)
            self.assertEqual(cfg.white.agent_type, AgentType.HUMAN)
            self.assertEqual(cfg.black.agent_type, AgentType.RANDOM_BOT)
            self.assertEqual(cfg.human_color, chess.WHITE)
            self.assertFalse(cfg.test_mode)
            self.assertIsNone(cfg.initial_fen)

    def test_game_wizard_human_vs_human_with_assist_and_preset(self) -> None:
        """Sprawdza Człowiek vs Człowiek z asystą w tle i presetem FEN."""
        with patch("sys.stdout", new=io.StringIO()):
            # Tryb 2 -> asysta 't' -> agent '1' (Bot 0) -> pozycja '2' (Presety)
            # -> preset '2' (Końcówka K+W) -> potwierdzenie [Enter]
            inputs = iter(["2", "t", "1", "2", "2", ""])
            cfg = self.menu.prompt_game_wizard(input_func=lambda _: next(inputs))
            self.assertIsNotNone(cfg)
            assert cfg is not None
            self.assertEqual(cfg.mode, GameMode.HUMAN_VS_HUMAN)
            self.assertIsNotNone(cfg.assist_agent)
            assert cfg.assist_agent is not None
            self.assertEqual(cfg.assist_agent.agent_type, AgentType.RANDOM_BOT)
            self.assertEqual(cfg.initial_fen, FEN_PRESETS[1][1])

    def test_game_wizard_ai_vs_ai_series(self) -> None:
        """Sprawdza konfigurację serii N partii AI vs AI z zamianą kolorów."""
        with patch("sys.stdout", new=io.StringIO()):
            # Tryb 3 -> Białe Bot (1) -> Czarne Bot (1) -> 10 partii -> zamiana kolorów [Enter=Tak]
            # -> wyświetlanie [Enter=3 Fast benchmark] -> pozycja [Enter=Start] -> potwierdzenie [Enter]
            inputs = iter(["3", "1", "1", "10", "", "", "", ""])
            cfg = self.menu.prompt_game_wizard(input_func=lambda _: next(inputs))
            self.assertIsNotNone(cfg)
            assert cfg is not None
            self.assertEqual(cfg.mode, GameMode.AI_VS_AI)
            self.assertEqual(cfg.num_games, 10)
            self.assertTrue(cfg.swap_colors)
            self.assertEqual(cfg.display_mode, AiDisplayMode.FAST_BENCHMARK)

    def test_game_wizard_back_navigation(self) -> None:
        """Sprawdza, że [0] cofa krok i pozwala bezpiecznie wrócić do Menu Głównego."""
        with patch("sys.stdout", new=io.StringIO()):
            # Tryb 1 -> kolor '0' (powrót do wyboru trybu) -> tryb '0' (powrót do Menu Głównego)
            inputs = iter(["1", "0", "0"])
            cfg = self.menu.prompt_game_wizard(input_func=lambda _: next(inputs))
            self.assertIsNone(cfg)

    def test_training_wizard(self) -> None:
        """Sprawdza działanie szkieletu Kreatora Treningu w gałęzi NAUKA."""
        with patch("sys.stdout", new=io.StringIO()):
            # Metoda 2 (Self-Play) -> start 3 (Rozgałęź) -> nazwa 'hybryda_v1'
            inputs = iter(["2", "3", "hybryda_v1"])
            tcfg = self.menu.prompt_training_wizard(input_func=lambda _: next(inputs))
            self.assertIsNotNone(tcfg)
            assert tcfg is not None
            self.assertEqual(tcfg.method, TrainingMethod.SELF_PLAY_RL)
            self.assertEqual(tcfg.start_mode, TrainingStartMode.BRANCH_CHECKPOINT)
            self.assertEqual(tcfg.target_line_name, "hybryda_v1")

    def test_prompt_fen_validation(self) -> None:
        """Sprawdza pobieranie poprawnego FEN oraz anulowanie."""
        with patch("sys.stdout", new=io.StringIO()):
            valid_fen = chess.STARTING_FEN
            loaded = self.menu.prompt_fen(input_func=lambda _: valid_fen)
            self.assertEqual(loaded, valid_fen)

            cancelled = self.menu.prompt_fen(input_func=lambda _: "anuluj")
            self.assertIsNone(cancelled)


if __name__ == "__main__":
    unittest.main()
