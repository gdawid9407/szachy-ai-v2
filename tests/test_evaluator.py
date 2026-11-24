"""
tests/test_evaluator.py

Zestaw testów jednostkowych dla modułu ewaluatora i silnika decyzyjnego (Zadanie 2.2).
Weryfikuje:
- Poprawność metadanych Poziomu 0 AI (tier=0, version_id='v0.0-random')
- Bezwzględną legalność wybieranych posunięć (python-chess)
- Obsługę wejścia typu BoardManager oraz chess.Board
- Prawidłowe zachowanie w pozycjach terminalnych (brak ruchów przy macie i pacie)
- Determinizm generatora przy zadanym ziarnie losowości (seed)
- Ewaluację pozycji standardowej i pozycji mata
- Zgodność przeliczania wskaźnika szans na zwycięstwo (get_win_percentage)
"""

import unittest
import chess

from src.engine.board_manager import BoardManager
from src.search.evaluator import RandomEvaluator, Tier0RandomAI, BaseEvaluator


class TestRandomEvaluator(unittest.TestCase):

    def setUp(self) -> None:
        self.bm = BoardManager()
        self.evaluator = RandomEvaluator(seed=42)

    def test_inheritance_and_metadata(self) -> None:
        """Sprawdza czy RandomEvaluator dziedziczy po BaseEvaluator oraz ma poprawne metadane."""
        self.assertIsInstance(self.evaluator, BaseEvaluator)
        self.assertEqual(self.evaluator.tier, 0)
        self.assertEqual(self.evaluator.version_id, "v0.0-random")
        self.assertIn("Poziom 0", self.evaluator.name)
        self.assertIs(RandomEvaluator, Tier0RandomAI)

    def test_initial_board_evaluation(self) -> None:
        """W pozycji startowej AI bez wiedzy zwraca 0.0 (50% szans)."""
        val = self.evaluator.evaluate(self.bm.board)
        self.assertEqual(val, 0.0)

        win_pct = self.evaluator.get_win_percentage(self.bm.board)
        self.assertEqual(win_pct, 50.0)

    def test_checkmate_evaluation(self) -> None:
        """W pozycji mata ewaluator zwraca -1.0 (gdy białe zamatowane) lub +1.0 (gdy czarne)."""
        # Mat głupca: 1. f3 e5 2. g4 Qh4# (ruch białych, białe w macie)
        fools_mate_fen = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        board_white_mated = chess.Board(fools_mate_fen)
        self.assertEqual(self.evaluator.evaluate(board_white_mated), -1.0)
        self.assertEqual(self.evaluator.get_win_percentage(board_white_mated), 0.0)

        # Mat szewski: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# (ruch czarnych, czarne w macie)
        scholars_mate_fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
        board_black_mated = chess.Board(scholars_mate_fen)
        self.assertEqual(self.evaluator.evaluate(board_black_mated), 1.0)
        self.assertEqual(self.evaluator.get_win_percentage(board_black_mated), 100.0)

    def test_select_move_legality(self) -> None:
        """Sprawdza czy wybrany ruch zawsze należy do puli legalnych ruchów."""
        move = self.evaluator.select_move(self.bm)
        self.assertIsNotNone(move)
        self.assertIn(move, self.bm.get_legal_moves())
        self.assertTrue(self.bm.board.is_legal(move))

    def test_select_move_from_chess_board(self) -> None:
        """Sprawdza czy select_move działa bezpośrednio z obiektem chess.Board."""
        board = chess.Board()
        move = self.evaluator.select_move(board)
        self.assertIsNotNone(move)
        self.assertIn(move, list(board.legal_moves))

    def test_select_move_invalid_type_raises_error(self) -> None:
        """Przekazanie nieobsługiwanego typu powinno zgłosić TypeError."""
        with self.assertRaises(TypeError):
            self.evaluator.select_move("niepoprawny_obiekt")  # type: ignore

    def test_select_move_terminal_position(self) -> None:
        """W pozycji mata lub pata select_move zwraca None."""
        # Pozycja pata
        fen_pat = "k7/8/1Q6/8/8/8/8/K7 b - - 0 1"
        bm_pat = BoardManager(fen=fen_pat)
        self.assertIsNone(self.evaluator.select_move(bm_pat))

        # Pozycja mata
        fen_mat = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        bm_mat = BoardManager(fen=fen_mat)
        self.assertIsNone(self.evaluator.select_move(bm_mat))

    def test_deterministic_seed(self) -> None:
        """Dwa ewaluatory z tym samym ziarnem generują identyczny ciąg ruchów."""
        ai1 = RandomEvaluator(seed=123)
        ai2 = RandomEvaluator(seed=123)

        board1 = chess.Board()
        board2 = chess.Board()

        for _ in range(15):
            move1 = ai1.select_move(board1)
            move2 = ai2.select_move(board2)
            self.assertEqual(move1, move2)
            if move1 is None or move2 is None:
                break
            board1.push(move1)
            board2.push(move2)

    def test_simulated_game_integrity(self) -> None:
        """Symulacja 40 półruchów losowej gry - każdy ruch musi być w 100% legalny."""
        bm = BoardManager()
        evaluator = RandomEvaluator(seed=999)

        for _ in range(40):
            if bm.is_game_over():
                break
            move = evaluator.select_move(bm)
            self.assertIsNotNone(move)
            self.assertTrue(bm.board.is_legal(move))
            self.assertTrue(bm.push_move(move))


if __name__ == "__main__":
    unittest.main()

