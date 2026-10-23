"""
src/engine/fen_parser.py

Moduł odpowiedzialny za walidację, sanityzację i parsowanie notacji FEN (Forsyth-Edwards Notation).
Zapewnia dwuetapową weryfikację:
1. Walidację syntaktyczną (poprawność 6 segmentów, struktura wierszy, dozwolone znaki).
2. Walidację semantyczną (legalność szachowa: obecność królów, brak pionów na skrajnych liniach, brak opposite check).
Zwraca szczegółowe, polskojęzyczne komunikaty diagnostyczne przy błędnych danych wejściowych.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, List, Optional, Set
import chess


# Dopuszczalne litery figur w FEN (wielkie = białe, małe = czarne)
ALLOWED_PIECE_CHARS: Final[Set[str]] = set("pnbrqkPNBRQK")

# Dopuszczalne litery praw do roszady
ALLOWED_CASTLING_CHARS: Final[Set[str]] = set("KQkq")


@dataclass(frozen=True)
class FenValidationResult:
    """
    Niezmienny obiekt reprezentujący wynik walidacji łańcucha FEN.
    
    Attributes:
        is_valid: True jeśli FEN jest w 100% poprawny syntaktycznie i szachowo; False w p.p.
        error_message: Szczegółowy opis błędu w języku polskim lub None, gdy FEN jest poprawny.
        cleaned_fen: Znormalizowany łańcuch FEN (usunięte zbędne spacje) lub None przy błędzie.
    """
    is_valid: bool
    error_message: Optional[str] = None
    cleaned_fen: Optional[str] = None

    def __bool__(self) -> bool:
        return self.is_valid


class FenParser:
    """
    Parser i walidator ciągów notacji FEN.
    
    Wykorzystuje bibliotekę python-chess do głębokiej weryfikacji semantycznej,
    uzupełniając ją o szczegółową analizę składniową i polskojęzyczną diagnostykę.
    """

    STARTING_FEN: Final[str] = chess.STARTING_FEN

    # --------------------------------------------------------------------------
    # Główne metody publiczne
    # --------------------------------------------------------------------------

    @classmethod
    def validate(cls, fen: str) -> FenValidationResult:
        """
        Przeprowadza pełną, dwuetapową walidację ciągu FEN.
        
        Args:
            fen: Łańcuch znaków reprezentujący pozycję FEN.
            
        Returns:
            FenValidationResult zawierający informację o poprawności, znormalizowany FEN
            oraz czytelny komunikat błędu w przypadku niepowodzenia.
        """
        if not isinstance(fen, str):
            return FenValidationResult(
                is_valid=False,
                error_message="Nieprawidłowy typ danych: oczekiwano ciągu znaków (str)."
            )

        cleaned = " ".join(fen.strip().split())
        if not cleaned:
            return FenValidationResult(
                is_valid=False,
                error_message="Łańcuch FEN jest pusty."
            )

        parts = cleaned.split(" ")
        if len(parts) != 6:
            return FenValidationResult(
                is_valid=False,
                error_message=(
                    f"Łańcuch FEN musi zawierać dokładnie 6 segmentów oddzielonych spacjami. "
                    f"Znaleziono {len(parts)} segmentów."
                )
            )

        piece_placement, active_color, castling, en_passant, halfmove, fullmove = parts

        # 1. Walidacja rozmieszczenia bierek
        piece_error = cls._validate_piece_placement(piece_placement)
        if piece_error is not None:
            return FenValidationResult(is_valid=False, error_message=piece_error)

        # 2. Walidacja strony na posunięciu
        if active_color not in ("w", "b"):
            return FenValidationResult(
                is_valid=False,
                error_message=(
                    f"Niepoprawne oznaczenie strony na posunięciu: '{active_color}'. "
                    f"Dozwolone wartości to 'w' (Białe) lub 'b' (Czarne)."
                )
            )

        # 3. Walidacja praw do roszady
        castling_error = cls._validate_castling_rights(castling)
        if castling_error is not None:
            return FenValidationResult(is_valid=False, error_message=castling_error)

        # 4. Walidacja pola bicia w przelocie (En Passant)
        ep_error = cls._validate_en_passant(en_passant, active_color)
        if ep_error is not None:
            return FenValidationResult(is_valid=False, error_message=ep_error)

        # 5. Walidacja licznika półruchów (Halfmove Clock)
        if not halfmove.isdigit():
            return FenValidationResult(
                is_valid=False,
                error_message=(
                    f"Licznik półruchów musi być nieujemną liczbą całkowitą (podano '{halfmove}')."
                )
            )

        # 6. Walidacja numeru pełnego ruchu (Fullmove Number)
        if not fullmove.isdigit() or int(fullmove) < 1:
            return FenValidationResult(
                is_valid=False,
                error_message=(
                    f"Numer pełnego ruchu partii musi być liczbą całkowitą większą lub równą 1 "
                    f"(podano '{fullmove}')."
                )
            )

        # 7. Walidacja semantyczna (reguły szachowe w python-chess)
        semantic_error = cls._validate_chess_semantics(cleaned)
        if semantic_error is not None:
            return FenValidationResult(is_valid=False, error_message=semantic_error)

        return FenValidationResult(is_valid=True, cleaned_fen=cleaned)

    @classmethod
    def is_valid(cls, fen: str) -> bool:
        """Szybka metoda pomocnicza zwracająca True jeśli FEN jest poprawny."""
        return cls.validate(fen).is_valid

    @classmethod
    def parse_to_board(cls, fen: str) -> chess.Board:
        """
        Parsuje ciąg FEN do instancji chess.Board.
        
        Args:
            fen: Łańcuch FEN do sparsowania.
            
        Returns:
            Zainicjalizowany obiekt chess.Board.
            
        Raises:
            ValueError: Jeśli FEN jest niepoprawny syntaktycznie lub semantycznie.
        """
        result = cls.validate(fen)
        if not result.is_valid:
            raise ValueError(f"Błąd notacji FEN: {result.error_message}")

        assert result.cleaned_fen is not None
        return chess.Board(result.cleaned_fen)

    # --------------------------------------------------------------------------
    # Prywatne metody walidacji szczegółowej
    # --------------------------------------------------------------------------

    @classmethod
    def _validate_piece_placement(cls, placement: str) -> Optional[str]:
        """Sprawdza składnię rozmieszczenia bierek (wiersze 8-1)."""
        ranks = placement.split("/")
        if len(ranks) != 8:
            return (
                f"Rozmieszczenie bierek musi zawierać dokładnie 8 wierszy oddzielonych znakiem '/'. "
                f"Znaleziono {len(ranks)} wierszy."
            )

        for rank_idx, rank in enumerate(ranks):
            rank_num = 8 - rank_idx
            cols_count = 0

            for char in rank:
                if char.isdigit():
                    digit = int(char)
                    if digit < 1 or digit > 8:
                        return f"Wiersz {rank_num}: niepoprawna cyfra pustych pól '{char}'."
                    cols_count += digit
                elif char in ALLOWED_PIECE_CHARS:
                    cols_count += 1
                else:
                    return f"Wiersz {rank_num}: niedozwolony znak figury '{char}'."

            if cols_count != 8:
                return (
                    f"Wiersz {rank_num} zawiera nieprawidłową łączną liczbę pól: {cols_count} "
                    f"(każdy wiersz musi liczyć dokładnie 8 pól)."
                )

        return None

    @classmethod
    def _validate_castling_rights(cls, castling: str) -> Optional[str]:
        """Sprawdza poprawność pola praw do roszady."""
        if castling == "-":
            return None

        seen: Set[str] = set()
        for char in castling:
            if char not in ALLOWED_CASTLING_CHARS:
                return (
                    f"Niepoprawny znak w prawach do roszady: '{char}'. "
                    f"Dozwolone są wyłącznie litery 'K', 'Q', 'k', 'q' lub myślnik '-'."
                )
            if char in seen:
                return f"Zduplikowany znak w prawach do roszady: '{char}'."
            seen.add(char)

        return None

    @classmethod
    def _validate_en_passant(cls, ep_str: str, active_color: str) -> Optional[str]:
        """Sprawdza składnię pola bicia w przelocie."""
        if ep_str == "-":
            return None

        if len(ep_str) != 2:
            return f"Niepoprawne pole bicia w przelocie: '{ep_str}'. Oczekiwano 2 znaków (np. 'e3')."

        col, rank = ep_str[0], ep_str[1]
        if col not in "abcdefgh":
            return f"Niepoprawna kolumna w polu bicia w przelocie: '{col}'."

        # Jeśli białe są na posunięciu, poprzedni ruch wykonały czarne (pchnięcie z 7 na 5 linię -> pole bicia to 6 linia)
        # Jeśli czarne są na posunięciu, poprzedni ruch wykonały białe (pchnięcie z 2 na 4 linię -> pole bicia to 3 linia)
        if active_color == "w" and rank != "6":
            return (
                f"Niepoprawna linia bicia w przelocie: '{ep_str}'. Gdy białe są na posunięciu, "
                f"pole bicia w przelocie musi leżeć na 6. linii."
            )
        if active_color == "b" and rank != "3":
            return (
                f"Niepoprawna linia bicia w przelocie: '{ep_str}'. Gdy czarne są na posunięciu, "
                f"pole bicia w przelocie musi leżeć na 3. linii."
            )

        return None

    @classmethod
    def _validate_chess_semantics(cls, cleaned_fen: str) -> Optional[str]:
        """Weryfikuje legalność szachową na podstawie bitmaski statusu w python-chess."""
        try:
            board = chess.Board(cleaned_fen)
        except ValueError as exc:
            return f"Błąd interpretacji FEN przez silnik szachowy: {exc}"

        status = board.status()
        if status == chess.STATUS_VALID:
            return None

        # Mapowanie flag błędów szachowych na komunikaty w języku polskim
        messages: List[str] = []

        if status & chess.STATUS_EMPTY:
            messages.append("Szachownica jest pusta.")
        if status & chess.STATUS_NO_WHITE_KING:
            messages.append("Brak białego króla na planszy.")
        if status & chess.STATUS_NO_BLACK_KING:
            messages.append("Brak czarnego króla na planszy.")
        if status & chess.STATUS_TOO_MANY_KINGS:
            messages.append("Nieprawidłowa liczba królów na planszy (musi być dokładnie jeden każdego koloru).")
        if status & chess.STATUS_PAWNS_ON_BACKRANK:
            messages.append("Piony nie mogą znajdować się na liniach skrajnych (1. lub 8. wiersz).")
        if status & chess.STATUS_OPPOSITE_CHECK:
            messages.append("Król strony niebędącej na posunięciu znajduje się pod szachem (niemożliwa pozycja).")
        if status & chess.STATUS_TOO_MANY_WHITE_PIECES:
            messages.append("Zbyt wiele białych bierek na planszy (maksymalnie 16).")
        if status & chess.STATUS_TOO_MANY_BLACK_PIECES:
            messages.append("Zbyt wiele czarnych bierek na planszy (maksymalnie 16).")
        if status & chess.STATUS_TOO_MANY_WHITE_PAWNS:
            messages.append("Zbyt wiele białych pionów (maksymalnie 8).")
        if status & chess.STATUS_TOO_MANY_BLACK_PAWNS:
            messages.append("Zbyt wiele czarnych pionów (maksymalnie 8).")
        if status & chess.STATUS_BAD_CASTLING_RIGHTS:
            messages.append("Prawa do roszady są niezgodne z pozycją króla lub wież.")
        if status & chess.STATUS_INVALID_EP_SQUARE:
            messages.append("Pole bicia w przelocie jest niepoprawne w bieżącym układzie bierek.")
        if status & (chess.STATUS_TOO_MANY_CHECKERS | chess.STATUS_IMPOSSIBLE_CHECK):
            messages.append("Pozycja zawiera niemożliwy stan szacha (zbyt wielu atakujących).")

        if messages:
            return " ".join(messages)

        return "Pozycja narusza reguły gry w szachy."
