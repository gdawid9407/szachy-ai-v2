"""
src/search/evaluator.py

Moduł oceny pozycji szachowej oraz wyboru posunięć przez silnik AI.
W Fazie 2 (Część 1) implementuje bazową klasę BaseEvaluator oraz Poziom 0 AI
(Tier 0: AI bez wiedzy), który wybiera losowe posunięcia spośród ruchów
dozwolonych przez reguły gry (python-chess).

Zgodnie z zasadą separacji reguł od AI, wybór ruchu odbywa się wyłącznie
z puli legalnych ruchów dostarczanych przez silnik szachowy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import List, Optional, Union
import chess

from src.config import value_to_win_percentage
from src.engine.board_manager import BoardManager


class BaseEvaluator(ABC):
    """
    Abstrakcyjna klasa bazowa dla ewaluatorów pozycji i silników decyzyjnych AI.
    
    Definiuje jednolity interfejs dla wszystkich poziomów zaawansowania AI:
    - Poziom 0 (RandomEvaluator): losowe legalne posunięcia (Faza 2)
    - Poziom 1 (SupervisedEvaluator): sieć uczona na danych historycznych (Faza 5)
    - Poziom 2 (SelfPlayEvaluator): sieć uczona metodą self-play (Faza 5)
    - Poziom 3 (HybridEvaluator): sieć + przeszukiwanie drzewa gry MCTS/Alpha-Beta (Faza 6)
    """

    @property
    @abstractmethod
    def tier(self) -> int:
        """Zwraca poziom zaawansowania AI (0, 1, 2, 3)."""
        pass

    @property
    @abstractmethod
    def version_id(self) -> str:
        """Zwraca identyfikator wersji modelu (np. 'v0.0-random')."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Zwraca czytelną nazwę poziomu AI."""
        pass

    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        """
        Dokonuje statycznej oceny pozycji szachowej.

        Args:
            board: Obiekt chess.Board reprezentujący ocenianą pozycję.

        Returns:
            Wartość zmiennoprzecinkowa z zakresu [-1.0, 1.0], gdzie:
             1.0 oznacza wygraną białych,
            -1.0 oznacza wygraną czarnych,
             0.0 oznacza pozycję równą / remisową.
        """
        pass

    @abstractmethod
    def select_move(
        self,
        board_input: Union[BoardManager, chess.Board],
    ) -> Optional[chess.Move]:
        """
        Wybiera ruch dla strony znajdującej się aktualnie na posunięciu.

        Args:
            board_input: Obiekt BoardManager lub chess.Board.

        Returns:
            Instancja chess.Move reprezentująca wybrany legalny ruch,
            lub None w przypadku gdy partia jest zakończona (brak legalnych ruchów).
        """
        pass

    def get_win_percentage(self, board: chess.Board) -> float:
        """
        Zwraca szansę na wygraną białych w procentach [0.0%, 100.0%].

        Args:
            board: Oceniana pozycja szachowa.
        """
        val = self.evaluate(board)
        return value_to_win_percentage(val)


class RandomEvaluator(BaseEvaluator):
    """
    Poziom 0 AI (Tier 0: AI bez wiedzy).
    
    Charakterystyka:
    - Brak sieci neuronowej (0 parametrów, 0 głębokości przeszukiwania).
    - Zwraca neutralną ocenę pozycji (0.0 -> 50% szans dla obu stron).
    - Dokonuje wyboru posunięcia w sposób pseudolosowy, wyłącznie spośród 
      ruchów legalnych wygenerowanych przez python-chess.
    - Zapewnia opcjonalne ziarno losowości (seed) dla pełnej deterministyczności testów.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Inicjalizuje losowy silnik AI.

        Args:
            seed: Opcjonalne ziarno generatora liczb pseudolosowych.
        """
        self._seed: Optional[int] = seed
        self._rng: random.Random = random.Random(seed)

    @property
    def tier(self) -> int:
        return 0

    @property
    def version_id(self) -> str:
        return "v0.0-random"

    @property
    def name(self) -> str:
        return "Poziom 0 (AI bez wiedzy - ruchy losowe)"

    def set_seed(self, seed: Optional[int]) -> None:
        """Ustawia nowe ziarno generatora losowego."""
        self._seed = seed
        self._rng = random.Random(seed)

    def evaluate(self, board: chess.Board) -> float:
        """
        Zwraca stałą wartość 0.0 (neutralna ocena pozycji, brak wiedzy szachowej).
        W przypadku mata zwraca deterministycznie 1.0 (wygrana białych) lub -1.0 (wygrana czarnych).
        """
        if board.is_checkmate():
            # Jeśli mat i ruch białych -> wygrały czarne (-1.0); jeśli ruch czarnych -> wygrały białe (1.0)
            return -1.0 if board.turn == chess.WHITE else 1.0
        return 0.0

    def select_move(
        self,
        board_input: Union[BoardManager, chess.Board],
    ) -> Optional[chess.Move]:
        """
        Wybiera losowy ruch z puli legalnych ruchów.

        Args:
            board_input: Instancja BoardManager lub bezpośredni obiekt chess.Board.

        Returns:
            Wybrany legalny ruch (chess.Move) lub None, jeśli brak dostępnych ruchów.
        """
        if isinstance(board_input, BoardManager):
            legal_moves = board_input.get_legal_moves()
        elif isinstance(board_input, chess.Board):
            legal_moves = list(board_input.legal_moves)
        else:
            raise TypeError(
                f"Oczekiwano BoardManager lub chess.Board, otrzymano: {type(board_input).__name__}"
            )

        if not legal_moves:
            return None

        return self._rng.choice(legal_moves)


# Alias semantyczny dla Tier 0 AI
Tier0RandomAI = RandomEvaluator

