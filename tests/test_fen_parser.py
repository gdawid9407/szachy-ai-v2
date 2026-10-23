"""
tests/test_fen_parser.py

Kompleksowy zestaw testów jednostkowych dla modułu FenParser (Zadanie 1.2).
Weryfikuje:
- Poprawność klasycznego FEN początkowego i pozycji ze zbędnymi spacjami
- Poprawność pozycji gry środkowej i końcówek
- Odrzucanie nieprawidłowych typów i pustych łańcuchów
- Weryfikację liczby segmentów (musi być dokładnie 6)
- Błędy w segmencie bierek (liczba wierszy, suma pól w wierszu, nielegalne znaki)
- Błędy w segmencie strony na posunięciu ('w' / 'b')
- Błędy w segmencie praw do roszady ('KQkq' / '-')
- Błędy w segmencie pola bicia w przelocie (en passant)
- Błędy w licznikach ruchów (halfmove i fullmove)
- Błędy semantyczne szachowe (brak króla, piony na liniach skrajnych, opposite check, pusta plansza)
- Metodę parse_to_board() oraz is_valid()
"""

import unittest
import chess
from src.engine.fen_parser import FenParser, FenValidationResult


class TestFenParser(unittest.TestCase):

    def test_starting_fen(self) -> None:
        """Weryfikuje klasyczną pozycję startową."""
        res = FenParser.validate(FenParser.STARTING_FEN)
        self.assertTrue(res.is_valid)
        self.assertIsNone(res.error_message)
        self.assertEqual(res.cleaned_fen, chess.STARTING_FEN)
        self.assertTrue(FenParser.is_valid(FenParser.STARTING_FEN))

    def test_whitespace_normalization(self) -> None:
        """Weryfikuje usuwanie nadmiarowych spacji z początku, końca i między segmentami."""
        fen_with_spaces = "   rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR   b   KQkq   e3   0   1   "
        res = FenParser.validate(fen_with_spaces)
        self.assertTrue(res.is_valid)
        self.assertEqual(res.cleaned_fen, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

    def test_valid_custom_positions(self) -> None:
        """Weryfikuje poprawne pozycje z partii i końcówek."""
        positions = [
            "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQK2R w KQkq - 4 5",
            "8/5k2/8/8/8/8/5K2/8 w - - 0 1",
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
            "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
        ]
        for fen in positions:
            res = FenParser.validate(fen)
            self.assertTrue(res.is_valid, f"Pozycja powinna być poprawna: {fen}")
            board = FenParser.parse_to_board(fen)
            self.assertIsInstance(board, chess.Board)

    def test_invalid_types_and_empty(self) -> None:
        """Odrzucanie typów innych niż str oraz pustych ciągów."""
        self.assertFalse(FenParser.validate(None).is_valid)  # type: ignore
        self.assertFalse(FenParser.validate(123).is_valid)  # type: ignore
        self.assertFalse(FenParser.validate("").is_valid)
        self.assertFalse(FenParser.validate("   ").is_valid)

    def test_segment_count(self) -> None:
        """Weryfikacja liczby segmentów (musi być dokładnie 6)."""
        # 1 segment
        res1 = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        self.assertFalse(res1.is_valid)
        self.assertIn("6 segmentów", res1.error_message or "")

        # 5 segmentów (brak fullmove_number)
        res5 = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0")
        self.assertFalse(res5.is_valid)
        self.assertIn("6 segmentów", res5.error_message or "")

        # 7 segmentów
        res7 = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra")
        self.assertFalse(res7.is_valid)
        self.assertIn("6 segmentów", res7.error_message or "")

    def test_piece_placement_syntax(self) -> None:
        """Błędy składniowe w ułożeniu bierek."""
        # 7 wierszy zamiast 8
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("dokładnie 8 wierszy", res.error_message or "")

        # Wiersz o sumie pól 9
        res = FenParser.validate("rnbqkbnr/ppppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("nieprawidłową łączną liczbę pól", res.error_message or "")

        # Niedozwolony znak 'X'
        res = FenParser.validate("rnbqkbnr/pppppppX/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("niedozwolony znak", res.error_message or "")

    def test_active_color(self) -> None:
        """Błędy w polu strony na posunięciu."""
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("Niepoprawne oznaczenie strony", res.error_message or "")

    def test_castling_rights(self) -> None:
        """Błędy w oznaczeniu praw do roszad."""
        # Znak niedozwolony
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQx - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("Niepoprawny znak w prawach do roszady", res.error_message or "")

        # Zduplikowany znak
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KKqk - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("Zduplikowany znak", res.error_message or "")

    def test_en_passant(self) -> None:
        """Błędy w polu bicia w przelocie."""
        # Zła długość
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("bicia w przelocie", res.error_message or "")

        # Niepoprawna linia dla białych (musi być 6)
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e3 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("pole bicia w przelocie musi leżeć na 6. linii", res.error_message or "")

        # Niepoprawna linia dla czarnych (musi być 3)
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq e6 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("pole bicia w przelocie musi leżeć na 3. linii", res.error_message or "")

    def test_move_counters(self) -> None:
        """Błędy w licznikach ruchów."""
        # Ujemny halfmove
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - -1 1")
        self.assertFalse(res.is_valid)
        self.assertIn("Licznik półruchów", res.error_message or "")

        # Zerowy fullmove
        res = FenParser.validate("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 0")
        self.assertFalse(res.is_valid)
        self.assertIn("Numer pełnego ruchu", res.error_message or "")

    def test_chess_semantics(self) -> None:
        """Błędy semantyczne reguł szachowych."""
        # Pusta plansza
        res = FenParser.validate("8/8/8/8/8/8/8/8 w - - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("pusta", res.error_message or "")

        # Brak białego króla
        res = FenParser.validate("4k3/8/8/8/8/8/8/8 w - - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("Brak białego króla", res.error_message or "")

        # Brak czarnego króla
        res = FenParser.validate("8/8/8/8/8/8/8/4K3 w - - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("Brak czarnego króla", res.error_message or "")

        # Piony na 8. linii
        res = FenParser.validate("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")  # to jest 7. linia - ok
        self.assertTrue(res.is_valid)
        res = FenParser.validate("4k1P1/8/8/8/8/8/8/4K3 w - - 0 1")  # pion na 8. linii
        self.assertFalse(res.is_valid)
        self.assertIn("liniach skrajnych", res.error_message or "")

        # Opposite check (czarne mają ruch, ale biały król jest szachowany)
        # Biały król na e1, czarna wieża na e8 bez przeszkód, kolej czarnych
        res = FenParser.validate("4r3/8/8/8/8/8/8/4K2k b - - 0 1")
        self.assertFalse(res.is_valid)
        self.assertIn("znajduje się pod szachem", res.error_message or "")

    def test_parse_to_board_exception(self) -> None:
        """parse_to_board zgłasza ValueError z polskim komunikatem błędu."""
        with self.assertRaises(ValueError) as ctx:
            FenParser.parse_to_board("niepoprawny fen")
        self.assertIn("Błąd notacji FEN", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
