"""
tests/test_board_manager.py

Kompleksowy zestaw testów jednostkowych dla modułu BoardManager (Zadanie 1.1 / 1.3).
Weryfikuje:
- Poprawność inicjalizacji i resetu planszy
- Legalność ruchów (UCI i chess.Move)
- Odrzucanie błędnych ruchów i niepoprawnych formatów
- Historię ruchów (SAN, UCI, formatted history z uwzględnieniem FEN)
- Funkcjonalność Back Move (pop_move)
- Specjalne ruchy szachowe: roszada, bicie w przelocie (en passant), promocja
- Warunki zakończenia gry: mat, pat, brak materiału, 50 ruchów, 3-krotne powtórzenie
- Podświetlenie ostatniego ruchu (last_move)
- Kopiowanie głębokie (copy)
- Walidację i ładowanie pozycji FEN
"""

import unittest
import chess
from src.engine.board_manager import BoardManager


class TestBoardManager(unittest.TestCase):

    def setUp(self) -> None:
        self.bm = BoardManager()

    def test_initial_state(self) -> None:
        """Weryfikuje domyślny stan początkowy planszy."""
        self.assertEqual(self.bm.get_fen(), chess.STARTING_FEN)
        self.assertEqual(self.bm.turn, chess.WHITE)
        self.assertEqual(self.bm.turn_name, "Białe")
        self.assertEqual(self.bm.fullmove_number, 1)
        self.assertEqual(self.bm.halfmove_clock, 0)
        self.assertFalse(self.bm.is_check)
        self.assertFalse(self.bm.is_game_over())
        self.assertIsNone(self.bm.last_move)
        self.assertEqual(len(self.bm), 0)
        self.assertEqual(self.bm.get_san_history(), [])
        self.assertEqual(self.bm.get_uci_history(), [])
        self.assertEqual(self.bm.get_formatted_history(), [])
        self.assertIsNone(self.bm.get_game_over_reason())

    def test_push_and_history(self) -> None:
        """Sprawdza wykonywanie ruchów i aktualizację historii SAN/UCI/formatted."""
        self.assertTrue(self.bm.push_uci("e2e4"))
        self.assertEqual(self.bm.turn, chess.BLACK)
        self.assertEqual(self.bm.turn_name, "Czarne")
        self.assertEqual(len(self.bm), 1)
        self.assertEqual(self.bm.last_move, chess.Move.from_uci("e2e4"))
        self.assertEqual(self.bm.get_san_history(), ["e4"])
        self.assertEqual(self.bm.get_uci_history(), ["e2e4"])
        self.assertEqual(self.bm.get_formatted_history(), ["1. e4"])

        self.assertTrue(self.bm.push_uci("e7e5"))
        self.assertEqual(self.bm.turn, chess.WHITE)
        self.assertEqual(self.bm.fullmove_number, 2)
        self.assertEqual(self.bm.get_formatted_history(), ["1. e4 e5"])

        self.assertTrue(self.bm.push_uci("g1f3"))
        self.assertEqual(self.bm.get_formatted_history(), ["1. e4 e5", "2. Nf3"])

    def test_illegal_moves(self) -> None:
        """Weryfikuje odrzucanie ruchów niedozwolonych przez przepisy oraz błędnych ciągów UCI."""
        # Ruch nielegalny dla bieżącej strony
        self.assertFalse(self.bm.push_uci("e7e5"))  # Ruch czarnych, gdy zaczynają białe
        self.assertFalse(self.bm.push_uci("e2e5"))  # Skok piona o 3 pola
        # Niepoprawna składnia UCI
        self.assertFalse(self.bm.push_uci("invalid"))
        self.assertFalse(self.bm.push_uci(""))
        self.assertFalse(self.bm.push_uci(None))  # type: ignore

    def test_back_move(self) -> None:
        """Weryfikuje cofanie posunięć (Back Move)."""
        self.bm.push_uci("e2e4")
        self.bm.push_uci("e7e5")
        self.bm.push_uci("g1f3")

        popped = self.bm.pop_move()
        self.assertEqual(popped, chess.Move.from_uci("g1f3"))
        self.assertEqual(self.bm.turn, chess.WHITE)
        self.assertEqual(self.bm.last_move, chess.Move.from_uci("e7e5"))
        self.assertEqual(self.bm.get_san_history(), ["e4", "e5"])
        self.assertEqual(self.bm.get_formatted_history(), ["1. e4 e5"])

        self.bm.pop_move()
        self.bm.pop_move()
        self.assertEqual(len(self.bm), 0)
        self.assertIsNone(self.bm.last_move)
        self.assertIsNone(self.bm.pop_move())  # Próba cofnięcia na pustej planszy

    def test_castling(self) -> None:
        """Sprawdza roszadę krótką i długą oraz ich cofanie."""
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1g1"]
        for m in moves:
            self.assertTrue(self.bm.push_uci(m))
        self.assertEqual(self.bm.get_san_history()[-1], "O-O")
        self.assertEqual(self.bm.board.piece_at(chess.G1), chess.Piece(chess.KING, chess.WHITE))
        self.assertEqual(self.bm.board.piece_at(chess.F1), chess.Piece(chess.ROOK, chess.WHITE))

        self.bm.pop_move()
        self.assertEqual(self.bm.board.piece_at(chess.E1), chess.Piece(chess.KING, chess.WHITE))
        self.assertEqual(self.bm.board.piece_at(chess.H1), chess.Piece(chess.ROOK, chess.WHITE))

    def test_en_passant(self) -> None:
        """Sprawdza bicie w przelocie oraz jego cofnięcie."""
        moves = ["e2e4", "a7a6", "e4e5", "d7d5"]
        for m in moves:
            self.assertTrue(self.bm.push_uci(m))
        self.assertIn("e5d6", self.bm.get_legal_moves_uci())
        self.assertTrue(self.bm.push_uci("e5d6"))
        self.assertEqual(self.bm.get_san_history()[-1], "exd6")
        self.assertIsNone(self.bm.board.piece_at(chess.D5))

        self.bm.pop_move()
        self.assertEqual(self.bm.board.piece_at(chess.D5), chess.Piece(chess.PAWN, chess.BLACK))

    def test_pawn_promotion(self) -> None:
        """Sprawdza promocję piona na hetmana i podpromocję na skoczka."""
        fen = "8/4P3/8/8/8/8/8/4K2k w - - 0 1"
        bm = BoardManager(fen)
        self.assertTrue(bm.push_uci("e7e8q"))
        self.assertEqual(bm.board.piece_at(chess.E8), chess.Piece(chess.QUEEN, chess.WHITE))

        bm.pop_move()
        self.assertTrue(bm.push_uci("e7e8n"))
        self.assertEqual(bm.board.piece_at(chess.E8), chess.Piece(chess.KNIGHT, chess.WHITE))

    def test_checkmate(self) -> None:
        """Sprawdza detekcję mata i komunikat o zwycięstwie."""
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            self.bm.push_uci(m)
        self.assertTrue(self.bm.is_check)
        self.assertTrue(self.bm.is_checkmate())
        self.assertTrue(self.bm.is_game_over())
        self.assertEqual(self.bm.get_game_result(), "0-1")
        self.assertIn("Czarne", self.bm.get_game_over_reason() or "")

    def test_stalemate(self) -> None:
        """Sprawdza detekcję pata."""
        bm = BoardManager("5k2/5P2/5K2/8/8/8/8/8 b - - 0 1")
        self.assertTrue(bm.is_stalemate())
        self.assertTrue(bm.is_game_over())
        self.assertEqual(bm.get_game_result(), "1/2-1/2")
        self.assertIn("pat", bm.get_game_over_reason() or "")

    def test_insufficient_material(self) -> None:
        """Sprawdza remis z braku materiału (Król przeciw Królowi)."""
        bm = BoardManager("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
        self.assertTrue(bm.is_insufficient_material())
        self.assertTrue(bm.is_game_over())
        self.assertEqual(bm.get_game_result(), "1/2-1/2")
        self.assertIn("niewystarczającego materiału", bm.get_game_over_reason() or "")

    def test_fifty_moves_rule(self) -> None:
        """Sprawdza zasadę 50 ruchów."""
        self.bm.board.halfmove_clock = 100
        self.assertTrue(self.bm.can_claim_fifty_moves())
        self.assertTrue(self.bm.is_game_over())
        self.assertIn("50 ruchów", self.bm.get_game_over_reason() or "")

    def test_threefold_repetition(self) -> None:
        """Sprawdza trzykrotne powtórzenie pozycji."""
        for m in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]:
            self.bm.push_uci(m)
        self.assertTrue(self.bm.can_claim_threefold_repetition())
        self.assertTrue(self.bm.is_game_over())
        self.assertIn("trzykrotnego powtórzenia", self.bm.get_game_over_reason() or "")

    def test_copy(self) -> None:
        """Weryfikuje, że copy() tworzy w pełni niezależną głęboką kopię."""
        self.bm.push_uci("e2e4")
        cpy = self.bm.copy()
        self.assertEqual(cpy.get_fen(), self.bm.get_fen())
        self.assertEqual(cpy.get_san_history(), self.bm.get_san_history())

        cpy.push_uci("e7e5")
        self.assertNotEqual(len(cpy), len(self.bm))
        self.assertNotEqual(cpy.get_fen(), self.bm.get_fen())

    def test_fen_validation_and_formatted_history(self) -> None:
        """Sprawdza walidację FEN oraz formatowanie historii przy niestandardowej pozycji startowej."""
        # Prawidłowy FEN z ruchem czarnych
        fen_black = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        bm = BoardManager(fen_black)
        self.assertTrue(bm.push_uci("e7e5"))
        self.assertTrue(bm.push_uci("g1f3"))
        # Czarne zagrały jako pierwsze w tej pozycji -> format: '1... e5', '2. Nf3'
        self.assertEqual(bm.get_formatted_history(), ["1... e5", "2. Nf3"])

        # Prawidłowy FEN od ruchu 15
        fen_move_15 = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 15"
        bm15 = BoardManager(fen_move_15)
        bm15.push_uci("d2d3")
        bm15.push_uci("f8c5")
        self.assertEqual(bm15.get_formatted_history(), ["15. d3 Bc5"])

        # Błędny FEN - odrzucenie w set_fen
        self.assertFalse(self.bm.set_fen("niepoprawny_fen"))
        self.assertFalse(self.bm.set_fen("8/8/8/8/8/8/8/8 w - - 0 1"))  # brak króli
        # Błędny FEN - wyjątek w konstruktorze
        with self.assertRaises(ValueError):
            BoardManager("niepoprawny_fen")
        with self.assertRaises(ValueError):
            BoardManager("8/8/8/8/8/8/8/8 w - - 0 1")


if __name__ == "__main__":
    unittest.main()
