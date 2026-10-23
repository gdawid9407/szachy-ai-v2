"""
src/engine/board_manager.py

Moduł zarządzający stanem planszy szachowej, legalnością posunięć oraz historią rozgrywki.
Pełni rolę fasady (Facade) izolującej bibliotekę python-chess od reszty systemu.
"""

from __future__ import annotations

from typing import List, Optional
import chess

from src.engine.fen_parser import FenParser


class BoardManager:
    """
    Zarządza stanem planszy, walidacją i wykonywaniem ruchów oraz historią gry.
    
    Wykorzystuje bibliotekę python-chess jako wyłączny autorytet w zakresie
    zasad gry w szachy (legalność ruchów, roszady, bicie w przelocie, mat, pat).
    """

    def __init__(self, fen: Optional[str] = None) -> None:
        """
        Inicjalizuje nową planszę.
        
        Args:
            fen: Opcjonalny łańcuch FEN definiujący pozycję początkową.
                 W przypadku braku, ustawiana jest klasyczna pozycja startowa.
        """
        self._board: chess.Board = chess.Board()
        self._san_history: List[str] = []

        if fen is not None:
            if not self.set_fen(fen):
                raise ValueError(f"Niepoprawny format FEN: '{fen}'")

    # --------------------------------------------------------------------------
    # Właściwości stanu planszy
    # --------------------------------------------------------------------------

    @property
    def board(self) -> chess.Board:
        """Zwraca bezpośredni dostęp do wewnętrznego obiektu chess.Board (np. dla enkodera)."""
        return self._board

    @property
    def turn(self) -> chess.Color:
        """Zwraca stronę na posunięciu: chess.WHITE (True) lub chess.BLACK (False)."""
        return self._board.turn

    @property
    def turn_name(self) -> str:
        """Zwraca czytelną nazwę strony wykonującej ruch ('Białe' / 'Czarne')."""
        return "Białe" if self._board.turn == chess.WHITE else "Czarne"

    @property
    def fullmove_number(self) -> int:
        """Zwraca bieżący numer pełnego ruchu partii."""
        return self._board.fullmove_number

    @property
    def halfmove_clock(self) -> int:
        """Zwraca liczbę półruchów od ostatniego pchnięcia piona lub zbicia figury."""
        return self._board.halfmove_clock

    @property
    def is_check(self) -> bool:
        """Zwraca True, jeśli król strony na posunięciu jest szachowany."""
        return self._board.is_check()

    @property
    def last_move(self) -> Optional[chess.Move]:
        """Zwraca ostatnio wykonany ruch lub None, jeśli na planszy nie wykonano żadnego posunięcia."""
        if self._board.move_stack:
            return self._board.peek()
        return None

    # --------------------------------------------------------------------------
    # Zarządzanie pozycją i FEN
    # --------------------------------------------------------------------------

    def reset(self) -> None:
        """Resetuje planszę do klasycznej pozycji początkowej i czyści historię."""
        self._board.reset()
        self._san_history.clear()

    def set_fen(self, fen: str) -> bool:
        """
        Ustawia planszę w pozycji zdefiniowanej przez kod FEN.
        
        Args:
            fen: Łańcuch znaków w formacie FEN.
            
        Returns:
            True jeśli FEN był poprawny i został załadowany, False w przeciwnym razie.
        """
        result = FenParser.validate(fen)
        if not result.is_valid:
            return False

        assert result.cleaned_fen is not None
        self._board = chess.Board(result.cleaned_fen)
        self._san_history.clear()
        return True

    def get_fen(self) -> str:
        """Zwraca aktualną pozycję w notacji FEN."""
        return self._board.fen()

    # --------------------------------------------------------------------------
    # Walidacja i wykonywanie ruchów
    # --------------------------------------------------------------------------

    def is_legal(self, move: chess.Move) -> bool:
        """Sprawdza, czy dany ruch jest legalny w bieżącej pozycji."""
        return move in self._board.legal_moves

    def get_legal_moves(self) -> List[chess.Move]:
        """Zwraca listę wszystkich legalnych ruchów w bieżącej pozycji."""
        return list(self._board.legal_moves)

    def get_legal_moves_uci(self) -> List[str]:
        """Zwraca listę legalnych ruchów jako łańcuchy znaków UCI (np. ['e2e4', 'g1f3'])."""
        return [move.uci() for move in self._board.legal_moves]

    def push_move(self, move: chess.Move) -> bool:
        """
        Wykonuje ruch na planszy, o ile jest w pełni legalny.
        
        Args:
            move: Obiekt chess.Move do wykonania.
            
        Returns:
            True jeśli ruch był legalny i został wykonany, False w przeciwnym razie.
        """
        if not self.is_legal(move):
            return False

        # Zapisujemy notację SAN zanim wykonamy ruch na planszy
        san_str = self._board.san(move)
        self._board.push(move)
        self._san_history.append(san_str)
        return True

    def push_uci(self, uci_str: str) -> bool:
        """
        Wykonuje ruch zadany w formacie UCI (np. 'e2e4', 'e7e8q').
        
        Args:
            uci_str: Notacja tekstowa ruchu UCI.
            
        Returns:
            True jeśli ruch był poprawny syntaktycznie, legalny i wykonany; False w p.p.
        """
        try:
            move = chess.Move.from_uci(uci_str.strip().lower())
        except (ValueError, AttributeError):
            return False

        return self.push_move(move)

    def pop_move(self) -> Optional[chess.Move]:
        """
        Cofa ostatnio wykonany ruch (funkcjonalność Back Move).
        
        Returns:
            Cofnięty obiekt chess.Move lub None, jeśli na planszy nie wykonano żadnego ruchu.
        """
        if len(self._board.move_stack) == 0:
            return None

        popped_move = self._board.pop()
        if self._san_history:
            self._san_history.pop()

        return popped_move

    # --------------------------------------------------------------------------
    # Weryfikacja stanu gry i zakończenia partii
    # --------------------------------------------------------------------------

    def is_game_over(self) -> bool:
        """Sprawdza, czy partia została zakończona (mat, pat, remis)."""
        return self._board.is_game_over(claim_draw=True)

    def is_checkmate(self) -> bool:
        """Zwraca True, jeśli na planszy jest mat."""
        return self._board.is_checkmate()

    def is_stalemate(self) -> bool:
        """Zwraca True, jeśli na planszy jest pat."""
        return self._board.is_stalemate()

    def is_insufficient_material(self) -> bool:
        """Zwraca True, jeśli materiał na planszy uniemożliwia danie mata."""
        return self._board.is_insufficient_material()

    def can_claim_fifty_moves(self) -> bool:
        """Zwraca True, jeśli spełniona jest zasada 50 ruchów."""
        return self._board.can_claim_fifty_moves()

    def can_claim_threefold_repetition(self) -> bool:
        """Zwraca True, jeśli pozycja powtórzyła się po raz trzeci."""
        return self._board.can_claim_threefold_repetition()

    def get_game_result(self) -> str:
        """
        Zwraca formalny wynik partii:
        '1-0' (wygrana białych), '0-1' (wygrana czarnych), '1/2-1/2' (remis), '*' (w toku).
        """
        return self._board.result(claim_draw=True)

    def get_game_over_reason(self) -> Optional[str]:
        """
        Zwraca szczegółowy, polskojęzyczny powód zakończenia gry.
        Jeśli partia nadal trwa, zwraca None.
        """
        outcome = self._board.outcome(claim_draw=True)
        if outcome is None:
            return None

        if outcome.termination == chess.Termination.CHECKMATE:
            winner = "Białe" if outcome.winner == chess.WHITE else "Czarne"
            return f"Szach-mat. Wygrywają {winner}."
        if outcome.termination == chess.Termination.STALEMATE:
            return "Remis przez pat (brak legalnych ruchów)."
        if outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
            return "Remis z powodu niewystarczającego materiału do mata."
        if outcome.termination in (chess.Termination.FIFTY_MOVES, chess.Termination.SEVENTYFIVE_MOVES):
            return "Remis z powodu zasady 50 ruchów."
        if outcome.termination in (chess.Termination.THREEFOLD_REPETITION, chess.Termination.FIVEFOLD_REPETITION):
            return "Remis z powodu trzykrotnego powtórzenia pozycji."

        return "Partia zakończona remisem."

    # --------------------------------------------------------------------------
    # Historia rozgrywki
    # --------------------------------------------------------------------------

    def get_san_history(self) -> List[str]:
        """Zwraca listę wszystkich ruchów partii w notacji SAN (np. ['e4', 'e5', 'Nf3'])."""
        return list(self._san_history)

    def get_uci_history(self) -> List[str]:
        """Zwraca listę wszystkich ruchów partii w notacji UCI (np. ['e2e4', 'e7e5'])."""
        return [move.uci() for move in self._board.move_stack]

    def get_formatted_history(self) -> List[str]:
        """
        Zwraca sformatowaną historię partii w czytelnej notacji z numeracją:
        np. ['1. e4 e5', '2. Nf3 Nc6', '3. Bb5'].
        Obsługuje poprawnie pozycje startowe FEN z ruchem czarnych (np. ['1... e5', '2. Nf3 Nc6'])
        oraz partie rozpoczynane od późniejszych numerów ruchów.
        """
        lines: List[str] = []
        if not self._san_history:
            return lines

        root = self._board.root()
        start_move = root.fullmove_number
        is_white_first = (root.turn == chess.WHITE)

        idx = 0
        current_move = start_move

        if not is_white_first:
            lines.append(f"{current_move}... {self._san_history[0]}")
            idx = 1
            current_move += 1

        while idx < len(self._san_history):
            white_move = self._san_history[idx]
            if idx + 1 < len(self._san_history):
                black_move = self._san_history[idx + 1]
                lines.append(f"{current_move}. {white_move} {black_move}")
                idx += 2
            else:
                lines.append(f"{current_move}. {white_move}")
                idx += 1
            current_move += 1

        return lines

    # --------------------------------------------------------------------------
    # Metody pomocnicze
    # --------------------------------------------------------------------------

    def copy(self) -> BoardManager:
        """Tworzy niezależną, głęboką kopię obiektu BoardManager."""
        new_manager = BoardManager()
        new_manager._board = self._board.copy()
        new_manager._san_history = list(self._san_history)
        return new_manager

    def __len__(self) -> int:
        """Zwraca całkowitą liczbę wykonanych półruchów (ply)."""
        return len(self._board.move_stack)

    def __str__(self) -> str:
        """Zwraca tekstową reprezentację szachownicy."""
        return str(self._board)

    def __repr__(self) -> str:
        return f"BoardManager(fen='{self.get_fen()}', plies={len(self)})"
