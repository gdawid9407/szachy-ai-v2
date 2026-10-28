"""
tests/test_board_view.py

Zestaw testów jednostkowych dla modułu wizualizacji szachownicy BoardView (Zadanie 2.1).
Weryfikuje:
- Poprawność renderowania planszy w orientacji domyślnej (perspektywa Białych)
- Poprawność odwracania planszy (perspektywa Czarnych)
- Wsparcie dla symboli szachowych Unicode oraz znaków ASCII
- Działanie z włączonym i wyłączonym formatowaniem ANSI
- Wyróżnianie ostatnio wykonanego ruchu (last_move)
- Precyzję formatowania paska predykcji wygranej (Evaluation Bar)
- Wykrywanie i prezentację stanów partii (w toku, szach, mat, pat, remisy)
- Opcję renderowania samej planszy bez panelu bocznego
"""

import unittest
import chess

from src.engine.board_manager import BoardManager
from src.ui.board_view import BoardView, UNICODE_PIECES, ASCII_PIECES


class TestBoardView(unittest.TestCase):

    def setUp(self) -> None:
        self.bm = BoardManager()
        self.view_unicode = BoardView(use_colors=False, use_unicode_pieces=True)
        self.view_ascii = BoardView(use_colors=False, use_unicode_pieces=False)
        self.view_colored = BoardView(use_colors=True, use_unicode_pieces=True)

    def test_initial_render_white_perspective(self) -> None:
        """Sprawdza renderowanie planszy początkowej z perspektywy białych."""
        rendered = self.view_unicode.render(self.bm, perspective=chess.WHITE)
        self.assertIsInstance(rendered, str)
        self.assertIn("a   b   c   d   e   f   g   h", rendered)
        self.assertIn("8 │", rendered)
        self.assertIn("1 │", rendered)

        # Sprawdzenie obecności białych i czarnych bierek Unicode
        self.assertIn(UNICODE_PIECES["R"], rendered)
        self.assertIn(UNICODE_PIECES["r"], rendered)
        self.assertIn(UNICODE_PIECES["K"], rendered)
        self.assertIn(UNICODE_PIECES["k"], rendered)

    def test_render_black_perspective(self) -> None:
        """Sprawdza odwrócenie planszy (perspektywa czarnych: linia 1 u góry, linia 8 u dołu)."""
        rendered = self.view_unicode.render(self.bm, perspective=chess.BLACK)
        self.assertIn("h   g   f   e   d   c   b   a", rendered)

        lines = rendered.splitlines()
        # Wiersz 1 powinien pojawić się przed wierszem 8
        first_rank_pos = next(i for i, line in enumerate(lines) if "1 │" in line)
        eighth_rank_pos = next(i for i, line in enumerate(lines) if "8 │" in line)
        self.assertLess(first_rank_pos, eighth_rank_pos)

    def test_ascii_pieces_fallback(self) -> None:
        """Weryfikuje poprawność renderowania przy użyciu znaków ASCII."""
        rendered = self.view_ascii.render(self.bm)
        self.assertIn("P", rendered)
        self.assertIn("p", rendered)
        self.assertIn("K", rendered)
        self.assertIn("k", rendered)
        # Żadne symbole Unicode nie powinny wystąpić na szachownicy
        self.assertNotIn("♙", rendered)
        self.assertNotIn("♟", rendered)

    def test_ansi_color_stripping_when_disabled(self) -> None:
        """Gdy use_colors=False, wynik nie może zawierać sekwencji ucieczki ANSI."""
        rendered = self.view_unicode.render(self.bm, eval_value=0.5)
        self.assertNotIn("\033[", rendered)

    def test_ansi_color_presence_when_enabled(self) -> None:
        """Gdy use_colors=True, wynik powinien zawierać kody formatowania ANSI."""
        rendered = self.view_colored.render(self.bm, eval_value=0.5)
        self.assertIn("\033[", rendered)

    def test_eval_bar_formatting(self) -> None:
        """Weryfikuje pasek predykcji wygranej dla różnych wartości wskaźnika."""
        # 1. Stan neutralny (brak oceny)
        bar_none = self.view_unicode.format_eval_bar(None)
        self.assertIn("Białe: 50.0%", bar_none)
        self.assertIn("50.0% :Czarne", bar_none)

        # 2. Pewna wygrana białych (+1.0) -> 100%
        bar_white = self.view_unicode.format_eval_bar(1.0)
        self.assertIn("100.0%", bar_white)
        self.assertIn("0.0%", bar_white)
        self.assertIn("█" * self.view_unicode.eval_bar_width, bar_white)

        # 3. Pewna wygrana czarnych (-1.0) -> 0% dla białych
        bar_black = self.view_unicode.format_eval_bar(-1.0)
        self.assertIn("0.0%", bar_black)
        self.assertIn("100.0%", bar_black)
        self.assertIn("░" * self.view_unicode.eval_bar_width, bar_black)

        # 4. Równowaga (0.0) -> 50%
        bar_zero = self.view_unicode.format_eval_bar(0.0)
        self.assertIn("50.0%", bar_zero)

        # 5. Ochrona przed przekroczeniem zakresu (clamping)
        bar_overflow = self.view_unicode.format_eval_bar(5.0)
        self.assertIn("100.0%", bar_overflow)

    def test_last_move_highlighting(self) -> None:
        """Sprawdza podświetlenie ostatniego ruchu po wykonaniu posunięcia."""
        self.bm.push_uci("e2e4")
        rendered_colored = self.view_colored.render(self.bm)
        self.assertIn("e2e4", rendered_colored)

        rendered_plain = self.view_unicode.render(self.bm)
        self.assertIn("e4 (e2e4)", rendered_plain)

    def test_check_status_display(self) -> None:
        """Sprawdza czy informacja o szachu pojawia się na panelu bocznym."""
        # Ustawienie pozycji z szachem: 1. e4 e5 2. Qh5 Nc6 3. Qxf7# (ale tu sprawdzimy szach)
        # Prosty szach: Biały król na e1, Czarny hetman na e2 (sprawdzający)
        fen_check = "rnb1kbnr/pppp1ppp/8/4p3/5PPq/8/PPPPP2P/RNBQKBNR w KQkq - 1 3"  # Fool's mate (szach-mat)
        bm_checkmate = BoardManager(fen=fen_check)
        rendered = self.view_unicode.render(bm_checkmate)
        self.assertIn("SZACH-MAT", rendered)

    def test_stalemate_status_display(self) -> None:
        """Sprawdza wyświetlanie statusu pata."""
        fen_stalemate = "k7/8/1Q6/8/8/8/8/K7 b - - 0 1"
        bm_pat = BoardManager(fen=fen_stalemate)
        rendered = self.view_unicode.render(bm_pat)
        self.assertIn("PAT", rendered)

    def test_render_without_side_panel(self) -> None:
        """Sprawdza renderowanie samej planszy bez panelu bocznego."""
        rendered = self.view_unicode.render(self.bm, include_side_panel=False)
        self.assertNotIn("[ STAN ROZGRYWKI ]", rendered)
        self.assertNotIn("[ WSKAŹNIK PREDYKCJI WYGRANEJ ]", rendered)
        self.assertIn("a   b   c   d   e   f   g   h", rendered)

    def test_display_callable(self) -> None:
        """Upewnia się, że metoda display nie rzuca wyjątków i wysyła dane do stdout."""
        import io
        from unittest.mock import patch

        with patch("sys.stdout", new=io.StringIO()) as fake_out:
            self.view_unicode.display(self.bm, clear_screen=False)
            output = fake_out.getvalue()
            self.assertIn("STAN ROZGRYWKI", output)


if __name__ == "__main__":
    unittest.main()
