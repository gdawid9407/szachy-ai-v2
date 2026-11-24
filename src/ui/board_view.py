"""
src/ui/board_view.py

Moduł odpowiedzialny za renderowanie szachownicy, informacji o stanie partii
oraz wskaźnika predykcji wygranej (Evaluation Bar) w interfejsie terminalowym.
Zapewnia czytelną prezentację planszy, obsługę perspektywy (Białe/Czarne),
podświetlanie ostatniego posunięcia oraz formatowanie wizualne ANSI.
"""

from __future__ import annotations

import re
from typing import Dict, Final, List, Optional
import chess

from src.config import value_to_win_percentage
from src.engine.board_manager import BoardManager


# Kody ANSI formatowania tekstu i tła
ANSI_RESET: Final[str] = "\033[0m"
ANSI_BOLD: Final[str] = "\033[1m"
ANSI_DIM: Final[str] = "\033[2m"

# Kolory bierek
ANSI_WHITE_PIECE: Final[str] = "\033[97m"      # Jasny biały
ANSI_BLACK_PIECE: Final[str] = "\033[36m"      # Turkusowy/Cyan (wyraźny kontrast w dark mode)

# Podświetlenie ostatniego ruchu
ANSI_LAST_MOVE_SRC: Final[str] = "\033[48;5;238m"   # Ciemnoszary podkład dla pola źródłowego
ANSI_LAST_MOVE_DST: Final[str] = "\033[48;5;242m"   # Jaśniejszy szary podkład dla pola docelowego
ANSI_CHECK_ALERT: Final[str] = "\033[41m\033[97m"   # Czerwone tło dla króla w szachu

# Symbole bierek Unicode
UNICODE_PIECES: Final[Dict[str, str]] = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}

# Symbole bierek ASCII (fallback)
ASCII_PIECES: Final[Dict[str, str]] = {
    "P": "P", "N": "N", "B": "B", "R": "R", "Q": "Q", "K": "K",
    "p": "p", "n": "n", "b": "b", "r": "r", "q": "q", "k": "k",
}


class BoardView:
    """
    Klasa odpowiedzialna za wizualizację szachownicy i stanu rozgrywki w konsoli.
    
    Cechy:
    - Duża, przejrzysta szachownica z ramkami tabeli i oznaczeniami pól (a-h, 1-8).
    - Obsługa perspektywy widoku (Białe od dołu / Czarne od dołu).
    - Opcjonalne podświetlanie ostatniego ruchu (ANSI).
    - Boczny panel telemetryczny: kolej ruchu, numer posunięcia, status szacha/mata.
    - Pasek predykcji wygranej (Evaluation Bar) wyliczany na podstawie wartości głowicy Value.
    """

    def __init__(
        self,
        use_colors: bool = True,
        use_unicode_pieces: bool = True,
        eval_bar_width: int = 16,
    ) -> None:
        """
        Inicjalizuje renderer widoku szachownicy.

        Args:
            use_colors: Czy używać kodów ANSI do kolorowania i wyróżniania pól.
            use_unicode_pieces: Czy używać symboli szachowych Unicode czy znaków ASCII.
            eval_bar_width: Szerokość graficznego paska wskaźnika predykcji wygranej.
        """
        self.use_colors: bool = use_colors
        self.use_unicode_pieces: bool = use_unicode_pieces
        self.eval_bar_width: int = max(8, eval_bar_width)
        self._piece_map = UNICODE_PIECES if use_unicode_pieces else ASCII_PIECES

    # --------------------------------------------------------------------------
    # Formatowanie paska ewaluacji (Evaluation Bar)
    # --------------------------------------------------------------------------

    def format_eval_bar(self, value: Optional[float] = None) -> str:
        """
        Tworzy tekstowy pasek postępu odzwierciedlający szanse na wygraną.

        Args:
            value: Wynik oceny z zakresu [-1.0, 1.0] (lub None, gdy brak wyceny AI).

        Returns:
            Sformatowany ciąg znaków przedstawiający pasek szans obu stron.
        """
        if value is None:
            # Wskaźnik neutralny / brak aktywnego modelu
            half = self.eval_bar_width // 2
            bar = "░" * half + "░" * (self.eval_bar_width - half)
            return f"Białe: 50.0% [{bar}] 50.0% :Czarne"

        white_pct = value_to_win_percentage(value)
        black_pct = 100.0 - white_pct

        # Obliczenie proporcji wypełnienia paska
        white_blocks = round((white_pct / 100.0) * self.eval_bar_width)
        white_blocks = max(0, min(self.eval_bar_width, white_blocks))
        black_blocks = self.eval_bar_width - white_blocks

        bar = "█" * white_blocks + "░" * black_blocks

        if self.use_colors:
            return (
                f"{ANSI_BOLD}Białe: {white_pct:5.1f}%{ANSI_RESET} "
                f"[{ANSI_WHITE_PIECE}{'█' * white_blocks}{ANSI_RESET}"
                f"{ANSI_BLACK_PIECE}{'░' * black_blocks}{ANSI_RESET}] "
                f"{ANSI_BOLD}{black_pct:5.1f}% :Czarne{ANSI_RESET}"
            )

        return f"Białe: {white_pct:5.1f}% [{bar}] {black_pct:5.1f}% :Czarne"

    # --------------------------------------------------------------------------
    # Renderowanie pojedynczego pola szachownicy
    # --------------------------------------------------------------------------

    def _render_square(
        self,
        board: chess.Board,
        square: chess.Square,
        last_move: Optional[chess.Move],
    ) -> str:
        """
        Renderuje zawartość pojedynczego pola (szerokość 3 znaki).
        """
        piece = board.piece_at(square)
        piece_char = self._piece_map.get(piece.symbol(), " ") if piece else " "

        if not self.use_colors:
            return f" {piece_char} "

        # Zastosowanie kolorów bierek
        if piece is not None:
            color_code = ANSI_WHITE_PIECE if piece.color == chess.WHITE else ANSI_BLACK_PIECE
            styled_piece = f"{ANSI_BOLD}{color_code}{piece_char}{ANSI_RESET}"
        else:
            styled_piece = " "

        # Sprawdzenie wyróżnienia szacha dla króla
        is_checked_king = False
        if board.is_check() and piece is not None:
            if piece.piece_type == chess.KING and piece.color == board.turn:
                is_checked_king = True

        # Podświetlenie ostatniego ruchu lub szacha
        if is_checked_king:
            return f"{ANSI_CHECK_ALERT} {piece_char} {ANSI_RESET}"
        if last_move is not None:
            if square == last_move.to_square:
                return f"{ANSI_LAST_MOVE_DST} {styled_piece}{ANSI_LAST_MOVE_DST} {ANSI_RESET}"
            if square == last_move.from_square:
                return f"{ANSI_LAST_MOVE_SRC} {styled_piece}{ANSI_LAST_MOVE_SRC} {ANSI_RESET}"

        return f" {styled_piece} "

    # --------------------------------------------------------------------------
    # Renderowanie kompletnego widoku
    # --------------------------------------------------------------------------

    def render(
        self,
        board_manager: BoardManager,
        perspective: chess.Color = chess.WHITE,
        eval_value: Optional[float] = None,
        last_move: Optional[chess.Move] = None,
        include_side_panel: bool = True,
    ) -> str:
        """
        Generuje sformatowany tekstowy widok szachownicy wraz z panelem bocznym.

        Args:
            board_manager: Instancja BoardManager ze stanem gry.
            perspective: Kolor gracza określający orientację planszy (Białe od dołu / Czarne od dołu).
            eval_value: Wartość oceny pozycji [-1.0, 1.0] z głowicy Value sieci neuronowej.
            last_move: Ostatnio wykonany ruch do podświetlenia (jeśli None, próbuje pobrać z historii).
            include_side_panel: Czy dołączyć boczny panel telemetryczny.

        Returns:
            Wieloliniowy łańcuch znaków gotowy do wyświetlenia w konsoli.
        """
        board = board_manager.board

        # Jeśli nie przekazano last_move wprost, sprawdź historię ruchu
        if last_move is None and board.move_stack:
            last_move = board.peek()

        # Ustalenie kolejności wierszy i kolumn według perspektywy
        ranks = range(7, -1, -1) if perspective == chess.WHITE else range(0, 8)
        files = range(0, 8) if perspective == chess.WHITE else range(7, -1, -1)

        # Generowanie linii bocznego panelu (8 wierszy odpowiadających 8 poziomom planszy)
        side_lines = self._generate_side_panel_lines(
            board_manager=board_manager,
            eval_value=eval_value,
            last_move=last_move,
        ) if include_side_panel else [""] * 8

        # Budowanie nagłówka kolumn
        col_labels = [chess.FILE_NAMES[f] for f in files]
        col_header = "     " + "   ".join(col_labels) + "    "
        divider_top = "   ┌" + "───┬" * 7 + "───┐"
        divider_mid = "   ├" + "───┼" * 7 + "───┤"
        divider_bot = "   └" + "───┴" * 7 + "───┘"

        lines: List[str] = [col_header, divider_top]

        for idx, r in enumerate(ranks):
            rank_label = str(r + 1)
            row_cells: List[str] = []

            for f in files:
                square = chess.square(f, r)
                cell_str = self._render_square(board, square, last_move)
                row_cells.append(cell_str)

            board_row = f" {rank_label} │" + "│".join(row_cells) + f"│ {rank_label}"
            side_info = f"   {side_lines[idx]}" if include_side_panel and side_lines[idx] else ""
            lines.append(board_row + side_info)

            if idx < 7:
                lines.append(divider_mid)

        lines.append(divider_bot)
        lines.append(col_header)

        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Budowanie linii bocznego panelu informacyjnego
    # --------------------------------------------------------------------------

    def _generate_side_panel_lines(
        self,
        board_manager: BoardManager,
        eval_value: Optional[float],
        last_move: Optional[chess.Move],
    ) -> List[str]:
        """
        Przygotowuje 8 linii informacji telemetrycznych do wyświetlenia obok szachownicy.
        """
        board = board_manager.board

        # 1. Określenie statusu partii
        if board.is_checkmate():
            status = f"{ANSI_BOLD}\033[91mSZACH-MAT! Wygrywają { 'Czarne' if board.turn == chess.WHITE else 'Białe' }{ANSI_RESET}" if self.use_colors else "SZACH-MAT!"
        elif board.is_stalemate():
            status = f"{ANSI_BOLD}\033[93mPAT - Remis{ANSI_RESET}" if self.use_colors else "PAT - Remis"
        elif board.is_insufficient_material():
            status = "Remis (niewystarczający materiał)"
        elif board.can_claim_fifty_moves():
            status = "Remis (zasada 50 ruchów)"
        elif board.can_claim_threefold_repetition():
            status = "Remis (trzykrotne powtórzenie)"
        elif board.is_check():
            status = f"{ANSI_BOLD}\033[91mSZACH!{ANSI_RESET}" if self.use_colors else "SZACH!"
        else:
            status = "W toku"

        # 2. Informacja o ostatnim ruchu
        if last_move is not None:
            last_move_uci = last_move.uci()
            history = board_manager.get_san_history()
            last_move_san = history[-1] if history else last_move_uci
            move_info = f"{last_move_san} ({last_move_uci})"
        else:
            move_info = "Brak (pozycja początkowa)"

        # 3. Informacja o kolejce ruchu
        turn_str = "Białe" if board.turn == chess.WHITE else "Czarne"
        turn_styled = f"{ANSI_BOLD}{turn_str}{ANSI_RESET}" if self.use_colors else turn_str

        # 4. Ocena wartości
        eval_bar_str = self.format_eval_bar(eval_value)
        val_display = f"{eval_value:+.2f}" if eval_value is not None else "brak oceny"

        # Mapowanie na 8 wierszy
        lines = [
            f"┌── [ STAN ROZGRYWKI ] ───────────────────────",
            f"│ Posunięcie : #{board_manager.fullmove_number} | Ruch: {turn_styled}",
            f"│ Ostatni ruch: {move_info}",
            f"│ Status      : {status}",
            f"├── [ WSKAŹNIK PREDYKCJI WYGRANEJ ] ──────────",
            f"│ Ocena Value: {val_display}",
            f"│ {eval_bar_str}",
            f"└─────────────────────────────────────────────",
        ]
        return lines

    # --------------------------------------------------------------------------
    # Metoda wyświetlająca bezpośrednio w strumieniu konsoli
    # --------------------------------------------------------------------------

    def display(
        self,
        board_manager: BoardManager,
        perspective: chess.Color = chess.WHITE,
        eval_value: Optional[float] = None,
        last_move: Optional[chess.Move] = None,
        clear_screen: bool = False,
    ) -> None:
        """
        Wyświetla szachownicę w terminalu z opcjonalnym czyszczeniem ekranu.
        """
        if clear_screen:
            # Sekwencja ANSI czyszcząca ekran i ustawiająca kursor w lewym górnym rogu
            print("\033[2J\033[H", end="")
        output = self.render(
            board_manager=board_manager,
            perspective=perspective,
            eval_value=eval_value,
            last_move=last_move,
            include_side_panel=True,
        )
        print(output)
