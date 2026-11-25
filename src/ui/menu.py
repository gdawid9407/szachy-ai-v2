"""
src/ui/menu.py

Moduł obsługi rozbudowanego menu gry (Kreator Wyboru), konfiguracji rozgrywki,
szkieletu treningu, podglądu stanu/metryk oraz parsowania komend gracza.

Architektura menu (zgodna z docelową specyfikacją):
- Menu Główne: [1] GRA | [2] NAUKA | [3] STAN I METRYKI | [0] WYJŚCIE
- Wspólny Kreator Agenta AI (Bot bazowy Poziom 0 / Silnik UCI / Własna Sieć AI)
- Tryby gry: Człowiek vs AI, Człowiek vs Człowiek, AI vs AI (1 partia / seria N partii),
  Konfigurator własny (dowolna para + własny FEN / presety FEN).
- Ekran podsumowania przed startem, domyślne wartości pod [Enter], [0] Powrót na każdym kroku.
- Komendy w trakcie rozgrywki: ruch UCI, back/undo, history, switch, fen (odczyt i zapis),
  eval/hint, save, restart, help, exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import random
import shutil
from typing import Callable, Dict, Final, List, Optional, Tuple
import chess

from src.config import MODELS_DIR, get_device_name
from src.engine.board_manager import BoardManager
from src.engine.fen_parser import FenParser, FenValidationResult


# Kolory i formatowanie ANSI
ANSI_RESET: Final[str] = "\033[0m"
ANSI_BOLD: Final[str] = "\033[1m"
ANSI_CYAN: Final[str] = "\033[36m"
ANSI_GREEN: Final[str] = "\033[92m"
ANSI_YELLOW: Final[str] = "\033[93m"
ANSI_RED: Final[str] = "\033[91m"


# Wbudowane presety pozycji startowych (FEN) do szybkiego testowania
FEN_PRESETS: Final[List[Tuple[str, str]]] = [
    (
        "Pozycja początkowa (Klasyczne szachy)",
        chess.STARTING_FEN,
    ),
    (
        "Końcówka: Król i Wieża vs Król (ćwiczenie mata)",
        "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
    ),
    (
        "Mat w 2 ruchach (prosta pozycja testowa)",
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    ),
    (
        "Obrona Sycylijska (po 1. e4 c5 2. Nf3 d6)",
        "rnbqkbnr/pp2pppp/3p4/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 3",
    ),
]


# ==============================================================================
# 1. TYPY WYLICZENIOWE (ENUMS) I KONTRAKTY KONFIGURACYJNE (DATACLASSES)
# ==============================================================================


class MainMenuAction(str, Enum):
    """Główne gałęzie menu początkowego."""
    PLAY = "play"
    TRAIN = "train"
    STATUS = "status"
    EXIT = "exit"


class GameMode(str, Enum):
    """Dostępne tryby rozgrywki w gałęzi GRA."""
    HUMAN_VS_AI = "human_vs_ai"
    HUMAN_VS_HUMAN = "human_vs_human"
    AI_VS_AI = "ai_vs_ai"
    CUSTOM = "custom"


class AgentType(str, Enum):
    """Rodzaj gracza (człowiek lub jeden z typów AI)."""
    HUMAN = "human"
    RANDOM_BOT = "random_bot"
    UCI_ENGINE = "uci_engine"
    NEURAL_NET = "neural_net"


class UciLimitMode(str, Enum):
    """Sposób ograniczania siły silnika zewnętrznego UCI."""
    ELO = "elo"
    DEPTH = "depth"
    TIME_MS = "time_ms"


class NeuralSearchMode(str, Enum):
    """Sposób wybierania ruchu przez własną sieć neuronową."""
    RAW_POLICY = "raw_policy"
    MCTS = "mcts"


class AiDisplayMode(str, Enum):
    """Tryb prezentacji partii w meczach AI vs AI."""
    LIVE_DELAY = "live_delay"
    STEP_BY_STEP = "step_by_step"
    FAST_BENCHMARK = "fast_benchmark"


class TrainingMethod(str, Enum):
    """Dostępne metody uczenia w gałęzi NAUKA."""
    SUPERVISED_PGN = "supervised_pgn"
    SELF_PLAY_RL = "self_play_rl"
    UCI_DISTILLATION = "uci_distillation"
    CURRICULUM_EPD = "curriculum_epd"


class TrainingStartMode(str, Enum):
    """Punkt startowy wag sieci w procesie treningu."""
    FROM_SCRATCH = "from_scratch"
    RESUME_LINE = "resume_line"
    BRANCH_CHECKPOINT = "branch_checkpoint"


class InGameAction(str, Enum):
    """Akcje i polecenia wydawane przez użytkownika podczas partii."""
    MOVE = "move"
    BACK = "back"
    HISTORY = "history"
    SWITCH_SIDE = "switch_side"
    SHOW_FEN = "show_fen"
    LOAD_FEN = "load_fen"
    EVAL = "eval"
    SAVE_PGN = "save_pgn"
    RESTART = "restart"
    HELP = "help"
    EXIT = "exit"


@dataclass(frozen=True)
class ParsedCommand:
    """Reprezentacja sparsowanego polecenia lub ruchu gracza."""
    action: InGameAction
    payload: Optional[str] = None


@dataclass
class PlayerConfig:
    """
    Uniwersalna konfiguracja pojedynczego gracza (Białych, Czarnych lub Asystenta).
    """
    agent_type: AgentType = AgentType.HUMAN
    name: str = "Człowiek"

    # Parametry silnika zewnętrznego UCI (np. Stockfish)
    uci_engine_name: str = "stockfish"
    uci_limit_mode: UciLimitMode = UciLimitMode.ELO
    uci_limit_value: int = 1500

    # Parametry Własnej Sieci AI
    model_line: Optional[str] = None
    checkpoint_name: Optional[str] = None
    search_mode: NeuralSearchMode = NeuralSearchMode.MCTS
    mcts_simulations: int = 100
    temperature: float = 0.1

    def describe(self) -> str:
        """Zwraca czytelny, jednoliniowy opis konfiguracji gracza."""
        if self.agent_type == AgentType.HUMAN:
            return "Człowiek"
        if self.agent_type == AgentType.RANDOM_BOT:
            return "Bot bazowy (Poziom 0 - losowe ruchy)"
        if self.agent_type == AgentType.UCI_ENGINE:
            unit_map = {
                UciLimitMode.ELO: f"Elo {self.uci_limit_value}",
                UciLimitMode.DEPTH: f"Głębokość {self.uci_limit_value}",
                UciLimitMode.TIME_MS: f"Czas {self.uci_limit_value} ms",
            }
            return f"Silnik UCI ({self.uci_engine_name}, {unit_map[self.uci_limit_mode]})"
        if self.agent_type == AgentType.NEURAL_NET:
            line = self.model_line or "brak linii"
            ckpt = self.checkpoint_name or "najnowszy"
            if self.search_mode == NeuralSearchMode.MCTS:
                mode_desc = f"MCTS {self.mcts_simulations} sym., temp={self.temperature}"
            else:
                mode_desc = f"Raw Policy, temp={self.temperature}"
            return f"Własna Sieć ({line}/{ckpt}, {mode_desc})"
        return self.name


@dataclass
class GameConfig:
    """Kompletna, uniwersalna konfiguracja rozgrywki zwracana przez Kreator Wyboru."""
    mode: GameMode = GameMode.HUMAN_VS_HUMAN
    white: PlayerConfig = field(default_factory=lambda: PlayerConfig(AgentType.HUMAN, "Człowiek"))
    black: PlayerConfig = field(default_factory=lambda: PlayerConfig(AgentType.HUMAN, "Człowiek"))
    initial_fen: Optional[str] = None

    # Parametry serii i wyświetlania (szczególnie AI vs AI)
    num_games: int = 1
    swap_colors: bool = False
    display_mode: AiDisplayMode = AiDisplayMode.LIVE_DELAY
    move_delay_sec: float = 0.5

    # Opcje wsparcia i analizy
    test_mode: bool = False
    assist_agent: Optional[PlayerConfig] = None
    save_pgn: bool = False

    @property
    def human_color(self) -> Optional[chess.Color]:
        """Zwraca kolor człowieka w trybie Człowiek vs AI (lub None dla innych trybów)."""
        if self.white.agent_type == AgentType.HUMAN and self.black.agent_type != AgentType.HUMAN:
            return chess.WHITE
        if self.black.agent_type == AgentType.HUMAN and self.white.agent_type != AgentType.HUMAN:
            return chess.BLACK
        return None


@dataclass
class TrainingConfig:
    """Konfiguracja procesu uczenia zwracana przez gałąź NAUKA."""
    method: TrainingMethod = TrainingMethod.SUPERVISED_PGN
    start_mode: TrainingStartMode = TrainingStartMode.FROM_SCRATCH
    target_line_name: str = "nowa_linia_v1"
    base_line: Optional[str] = None
    base_checkpoint: Optional[str] = None
    source_path: Optional[str] = None
    min_elo: int = 1800
    mcts_simulations: int = 100
    temperature: float = 1.0


# ==============================================================================
# 2. KLASA MENU (KREATOR WYBORU + PARSER KOMEND)
# ==============================================================================


class Menu:
    """
    Zarządza interfejsem tekstowym menu (Kreator Wyboru), podglądem stanu modeli
    oraz interpretacją poleceń wprowadzanych w trakcie partii.
    """

    def __init__(
        self,
        use_colors: bool = True,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        self.use_colors: bool = use_colors
        self.models_dir: Path = models_dir

    # --------------------------------------------------------------------------
    # Skanowanie katalogu modeli i diagnostyka środowiska
    # --------------------------------------------------------------------------

    def discover_model_lines(self) -> List[Dict[str, object]]:
        """
        Skanuje katalog models/ w poszukiwaniu linii treningowych (podkatalogów
        zawierających metadata.json lub pliki checkpointów .pt).
        """
        lines_info: List[Dict[str, object]] = []
        if not self.models_dir.exists():
            return lines_info

        for entry in sorted(self.models_dir.iterdir()):
            if not entry.is_dir() or entry.name.startswith(".") or entry.name == "checkpoints":
                continue

            checkpoints = sorted(
                p.name for p in entry.glob("*.pt") if not p.name.startswith("._")
            )
            meta_file = entry / "metadata.json"
            metadata: Dict[str, object] = {}
            if meta_file.exists():
                try:
                    metadata = json.loads(meta_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    metadata = {}

            if checkpoints or meta_file.exists():
                lines_info.append(
                    {
                        "name": entry.name,
                        "checkpoints": checkpoints,
                        "parent": metadata.get("parent"),
                        "source": metadata.get("source"),
                        "latest_loss": metadata.get("latest_loss"),
                    }
                )
        return lines_info

    def detect_uci_engines(self) -> List[Tuple[str, str]]:
        """Wykrywa zainstalowane w systemie silniki UCI (np. Stockfish)."""
        found: List[Tuple[str, str]] = []
        for candidate in ("stockfish", "lc0", "komodo"):
            path = shutil.which(candidate)
            if path is not None:
                found.append((candidate, path))
        return found

    # --------------------------------------------------------------------------
    # Formatowanie widoków menu i komunikatów (Czyste funkcje)
    # --------------------------------------------------------------------------

    def format_main_menu(self) -> str:
        """Zwraca sformatowany tekst Menu Głównego (Poziom 0)."""
        lines = [
            "============================================================",
            "                 SZACHY AI v2 - MENU GŁÓWNE                 ",
            "============================================================",
            "  [1] GRA               (Rozgrywka i testy, bez zmiany wag)",
            "  [2] NAUKA             (Trening sieci i nowe wersje)",
            "  [3] STAN I METRYKI    (Modele, linie treningowe, sprzęt)",
            "  [0] WYJŚCIE           (Zamknij program)",
            "============================================================",
        ]
        return "\n".join(lines)

    def format_play_mode_menu(self) -> str:
        """Zwraca ekran wyboru trybu rozgrywki (Gałąź [1] GRA)."""
        lines = [
            "------------------------------------------------------------",
            "  GRA — Wybierz tryb rozgrywki:",
            "  [1] Człowiek vs AI",
            "  [2] Człowiek vs Człowiek",
            "  [3] AI vs AI (Pojedyncza partia lub Seria N partii)",
            "  [4] Konfigurator własny (dowolna para + własny FEN)",
            "  [0] Powrót do Menu Głównego",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_agent_selection(self, role_label: str = "Agenta AI", allow_human: bool = False) -> str:
        """Zwraca ekran Wspólnego Kreatora Agenta AI."""
        lines = [
            "------------------------------------------------------------",
            f"  Wybór gracza ({role_label}):",
            "  [1] Bot bazowy (Poziom 0 - losowe, legalne ruchy)",
            "  [2] Silnik zewnętrzny (UCI, np. Stockfish)",
            "  [3] Własna Sieć AI (linia treningowa + checkpoint)",
        ]
        if allow_human:
            lines.append("  [4] Człowiek (gracz lokalny)")
        lines.extend(
            [
                "  [0] Powrót",
                "------------------------------------------------------------",
            ]
        )
        return "\n".join(lines)

    def format_side_selection(self) -> str:
        """Zwraca menu wyboru koloru bierek przed rozpoczęciem gry."""
        lines = [
            "------------------------------------------------------------",
            "  Wybierz kolor bierek człowieka:",
            "  [1] Białe (pierwszy ruch) [Domyślny]",
            "  [2] Czarne",
            "  [3] Losowo",
            "  [0] Powrót",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_fen_selection(self) -> str:
        """Zwraca menu wyboru pozycji początkowej (FEN)."""
        lines = [
            "------------------------------------------------------------",
            "  Pozycja początkowa:",
            "  [1] Standardowa pozycja startowa (klasyczne szachy) [Domyślna]",
            "  [2] Wklej własny kod FEN (np. z Lichess / Chess.com)",
            "  [3] Wybierz z gotowych pozycji testowych (Presety)",
            "  [0] Powrót",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_fen_presets(self) -> str:
        """Zwraca listę wbudowanych pozycji startowych FEN."""
        lines = [
            "------------------------------------------------------------",
            "  Gotowe pozycje testowe (Presety FEN):",
        ]
        for idx, (label, _) in enumerate(FEN_PRESETS, start=1):
            lines.append(f"  [{idx}] {label}")
        lines.extend(
            [
                "  [0] Powrót",
                "------------------------------------------------------------",
            ]
        )
        return "\n".join(lines)

    def format_game_summary(self, config: GameConfig) -> str:
        """Zwraca ekran podsumowania konfiguracji przed uruchomieniem rozgrywki."""
        fen_desc = config.initial_fen if config.initial_fen else "Pozycja startowa (klasyczna)"
        lines = [
            "============================================================",
            "               PODSUMOWANIE KONFIGURACJI GRY                ",
            "============================================================",
            f"  Tryb          : {config.mode.value}",
            f"  Białe         : {config.white.describe()}",
            f"  Czarne        : {config.black.describe()}",
            f"  Pozycja (FEN) : {fen_desc}",
        ]
        if config.num_games > 1 or config.mode == GameMode.AI_VS_AI:
            swap_str = "TAK" if config.swap_colors else "NIE"
            lines.append(f"  Liczba partii : {config.num_games} (Zamiana kolorów: {swap_str})")
            lines.append(f"  Wyświetlanie  : {config.display_mode.value}")
        if config.test_mode:
            lines.append("  Tryb testowy  : WŁĄCZONY (podgląd ocen / cofanie ruchów)")
        if config.assist_agent is not None:
            lines.append(f"  Asysta w tle  : {config.assist_agent.describe()}")
        if config.save_pgn:
            lines.append("  Zapis PGN     : TAK")
        lines.extend(
            [
                "============================================================",
                "  [Enter] Rozpocznij grę   |   [0] Anuluj i wróć",
            ]
        )
        return "\n".join(lines)

    def format_training_method_menu(self) -> str:
        """Zwraca ekran wyboru metody treningu (Gałąź [2] NAUKA)."""
        lines = [
            "------------------------------------------------------------",
            "  NAUKA — Wybierz metodę treningu sieci:",
            "  [1] Uczenie nadzorowane (Bazy PGN z internetu)",
            "  [2] Self-Play RL (Własna Sieć vs Własna Sieć)",
            "  [3] Destylacja wiedzy (Nauka od silnika zewnętrznego UCI)",
            "  [4] Trening fazowy / Curriculum (Zagadki taktyczne EPD)",
            "  [0] Powrót do Menu Głównego",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_training_start_menu(self) -> str:
        """Zwraca ekran wyboru punktu startowego wag sieci."""
        lines = [
            "------------------------------------------------------------",
            "  Wybierz punkt startowy modelu:",
            "  [1] Nowa sieć od zera [Domyślny]",
            "  [2] Wznów istniejącą linię treningową",
            "  [3] Rozgałęź z istniejącego checkpointu (nowa linia hybrydowa)",
            "  [0] Powrót",
            "------------------------------------------------------------",
        ]
        return "\n".join(lines)

    def format_status_and_metrics(self) -> str:
        """Zwraca raport diagnostyczny środowiska oraz zapisanych linii modeli."""
        device = get_device_name().upper()
        uci_engines = self.detect_uci_engines()
        model_lines = self.discover_model_lines()

        lines = [
            "============================================================",
            "              STAN ŚRODOWISKA I METRYKI SIECI               ",
            "============================================================",
            f"  Akcelerator obliczeń : {device}",
        ]
        if uci_engines:
            engines_str = ", ".join(f"{name} ({path})" for name, path in uci_engines)
            lines.append(f"  Wykryte silniki UCI  : {engines_str}")
        else:
            lines.append("  Wykryte silniki UCI  : brak w PATH (zainstaluj np. Stockfish)")

        lines.append("------------------------------------------------------------")
        lines.append(f"  Katalog modeli       : {self.models_dir}")
        if not model_lines:
            lines.append("  Zapisane linie AI    : brak (sieć nie była jeszcze trenowana)")
        else:
            lines.append(f"  Zapisane linie AI ({len(model_lines)}):")
            for info in model_lines:
                ckpts = info["checkpoints"]
                ckpt_count = len(ckpts) if isinstance(ckpts, list) else 0
                parent_str = f" | parent: {info['parent']}" if info.get("parent") else ""
                loss_str = f" | loss: {info['latest_loss']}" if info.get("latest_loss") is not None else ""
                lines.append(f"    - {info['name']}: {ckpt_count} checkpointów{parent_str}{loss_str}")
        lines.append("============================================================")
        return "\n".join(lines)

    def format_help(self) -> str:
        """Zwraca listę komend dostępnych podczas partii."""
        lines = [
            "======================= POMOC / KOMENDY =====================",
            "  Ruch szachowy : Wpisz ruch w formacie UCI (np. e2e4, g1f3)",
            "                  Promocja piona: dopisz figurę (np. e7e8q)",
            "  back / undo   : Cofnij ostatnie posunięcie",
            "  history       : Wyświetl pełną historię ruchów w partii",
            "  switch        : Przejmij stronę przeciwnika (w trybie vs AI)",
            "  fen           : Wyświetl aktualny kod FEN pozycji",
            "  fen <kod_fen> : Wczytaj nowe ustawienie planszy z FEN",
            "  eval / hint   : Poproś o ocenę pozycji i podpowiedź ruchu",
            "  save          : Zapisz bieżącą partię do pliku PGN",
            "  restart       : Zresetuj planszę do pozycji początkowej",
            "  help / ?      : Wyświetl tę pomoc",
            "  exit / quit   : Zakończ bieżącą rozgrywkę",
            "============================================================",
        ]
        return "\n".join(lines)

    def format_history(self, board_manager: BoardManager) -> str:
        """Zwraca sformatowaną historię ruchów partii w standardowej notacji."""
        formatted = board_manager.get_formatted_history()
        if not formatted:
            return "Historia ruchów: brak (pozycja początkowa)."

        lines = ["--- HISTORIA ROZGRYWKI ---"]
        lines.extend(formatted)
        lines.append("---------------------------")
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # Parsowanie wejścia gracza w trakcie rozgrywki
    # --------------------------------------------------------------------------

    def parse_in_game_input(self, raw_input: str) -> ParsedCommand:
        """
        Analizuje łańcuch wpisany przez użytkownika i klasyfikuje go jako
        ruch szachowy lub komendę sterującą.
        """
        cleaned = raw_input.strip()
        lower = cleaned.lower()

        if not lower:
            return ParsedCommand(action=InGameAction.HELP)

        if lower in ("exit", "quit", "q"):
            return ParsedCommand(action=InGameAction.EXIT)
        if lower in ("back", "undo", "b"):
            return ParsedCommand(action=InGameAction.BACK)
        if lower in ("history", "hist", "h"):
            return ParsedCommand(action=InGameAction.HISTORY)
        if lower in ("switch", "flip", "switch_side"):
            return ParsedCommand(action=InGameAction.SWITCH_SIDE)
        if lower in ("eval", "hint", "ocena", "podpowiedz"):
            return ParsedCommand(action=InGameAction.EVAL)
        if lower in ("save", "save_pgn", "pgn"):
            return ParsedCommand(action=InGameAction.SAVE_PGN)
        if lower in ("restart", "reset"):
            return ParsedCommand(action=InGameAction.RESTART)
        if lower in ("help", "?", "pomoc"):
            return ParsedCommand(action=InGameAction.HELP)

        # Samo "fen" -> wyświetlenie bieżącego FEN
        if lower == "fen":
            return ParsedCommand(action=InGameAction.SHOW_FEN)

        # Załadowanie FEN (np. "fen rnbqkbnr/..." lub "load_fen ...")
        if lower.startswith("fen ") or lower.startswith("load_fen "):
            parts = cleaned.split(maxsplit=1)
            fen_str = parts[1].strip() if len(parts) > 1 else ""
            return ParsedCommand(action=InGameAction.LOAD_FEN, payload=fen_str)

        # Domyślnie traktowane jako próba wykonania ruchu (UCI)
        return ParsedCommand(action=InGameAction.MOVE, payload=cleaned)

    # --------------------------------------------------------------------------
    # Pomocnicze dialogi wejścia (z obsługą domyślnych wartości i [0] Powrót)
    # --------------------------------------------------------------------------

    def _prompt_yes_no(
        self,
        question: str,
        default: bool = False,
        input_func: Callable[[str], str] = input,
    ) -> Optional[bool]:
        """Pyta o decyzję Tak/Nie. Zwraca None, jeśli użytkownik wpisze '0' (Powrót)."""
        hint = "[T/n]" if default else "[t/N]"
        while True:
            ans = input_func(f"{question} {hint} (0 = powrót): ").strip().lower()
            if ans == "0":
                return None
            if ans == "":
                return default
            if ans in ("t", "tak", "y", "yes"):
                return True
            if ans in ("n", "nie", "no"):
                return False
            print("Wpisz 't' (tak), 'n' (nie) lub '0' (powrót).")

    def _prompt_positive_int(
        self,
        prompt_text: str,
        default: int,
        min_val: int = 1,
        max_val: Optional[int] = None,
        input_func: Callable[[str], str] = input,
    ) -> Optional[int]:
        """Pobiera liczbę całkowitą z zakresu lub zwraca None przy '0' (Powrót)."""
        while True:
            raw = input_func(f"{prompt_text} [Domyślnie: {default}] (0 = powrót): ").strip()
            if raw == "0":
                return None
            if raw == "":
                return default
            if raw.isdigit():
                value = int(raw)
                if value >= min_val and (max_val is None or value <= max_val):
                    return value
            bounds = f"od {min_val} do {max_val}" if max_val is not None else f">= {min_val}"
            print(f"Niepoprawna liczba! Podaj wartość {bounds}.")

    # --------------------------------------------------------------------------
    # Interaktywne procedury Kreatora Wyboru
    # --------------------------------------------------------------------------

    def prompt_main_menu(self, input_func: Callable[[str], str] = input) -> MainMenuAction:
        """Wyświetla Menu Główne (Poziom 0) i zwraca wybraną gałąź."""
        while True:
            print(self.format_main_menu())
            choice = input_func("Wybierz opcję (0-3): ").strip().lower()
            if choice == "1":
                return MainMenuAction.PLAY
            if choice == "2":
                return MainMenuAction.TRAIN
            if choice == "3":
                return MainMenuAction.STATUS
            if choice in ("0", "exit", "quit", "q"):
                return MainMenuAction.EXIT
            print("Niepoprawny wybór! Wpisz cyfrę od 0 do 3.")

    def prompt_game_mode(self, input_func: Callable[[str], str] = input) -> Optional[GameMode]:
        """Pyta o tryb rozgrywki w gałęzi [1] GRA. Zwraca None przy [0] Powrót."""
        while True:
            print(self.format_play_mode_menu())
            choice = input_func("Wybierz tryb gry (0-4): ").strip().lower()
            if choice == "1":
                return GameMode.HUMAN_VS_AI
            if choice == "2":
                return GameMode.HUMAN_VS_HUMAN
            if choice == "3":
                return GameMode.AI_VS_AI
            if choice == "4":
                return GameMode.CUSTOM
            if choice in ("0", "back", "powrot"):
                return None
            print("Niepoprawny wybór! Wpisz cyfrę od 0 do 4.")

    def prompt_side_selection(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[chess.Color]:
        """Pyta gracza o kolor bierek. Zwraca None przy [0] Powrót."""
        while True:
            print(self.format_side_selection())
            choice = input_func("Wybierz kolor (0-3) [Enter = Białe]: ").strip().lower()
            if choice == "0":
                return None
            if choice in ("1", "w", "white", "biale", ""):
                return chess.WHITE
            if choice in ("2", "b", "black", "czarne"):
                return chess.BLACK
            if choice in ("3", "r", "random", "losowo"):
                return chess.WHITE if random.choice([True, False]) else chess.BLACK
            print("Niepoprawna opcja! Wpisz 0, 1, 2 lub 3.")

    def prompt_agent_config(
        self,
        role_label: str = "Agenta AI",
        allow_human: bool = False,
        input_func: Callable[[str], str] = input,
    ) -> Optional[PlayerConfig]:
        """
        Wspólny Kreator Agenta AI (używany we wszystkich trybach gry i asyście).
        Zwraca PlayerConfig lub None przy wyborze [0] Powrót.
        """
        while True:
            print(self.format_agent_selection(role_label=role_label, allow_human=allow_human))
            max_opt = "4" if allow_human else "3"
            choice = input_func(
                f"Wybierz dla {role_label} (0-{max_opt}) [Enter = Bot bazowy]: "
            ).strip()

            if choice == "0":
                return None
            if choice in ("1", ""):
                return PlayerConfig(
                    agent_type=AgentType.RANDOM_BOT,
                    name="Bot bazowy (Poziom 0)",
                )
            if choice == "2":
                uci_cfg = self._prompt_uci_config(input_func=input_func)
                if uci_cfg is not None:
                    return uci_cfg
                continue
            if choice == "3":
                nn_cfg = self._prompt_neural_config(input_func=input_func)
                if nn_cfg is not None:
                    return nn_cfg
                continue
            if allow_human and choice == "4":
                return PlayerConfig(agent_type=AgentType.HUMAN, name="Człowiek")

            print(f"Niepoprawny wybór! Wpisz cyfrę od 0 do {max_opt}.")

    def _prompt_uci_config(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[PlayerConfig]:
        """Podkreator ustawień silnika zewnętrznego UCI (Stockfish itp.)."""
        detected = self.detect_uci_engines()
        default_engine = detected[0][0] if detected else "stockfish"
        engine_raw = input_func(
            f"Nazwa lub ścieżka silnika UCI [Enter = {default_engine}] (0 = powrót): "
        ).strip()
        if engine_raw == "0":
            return None
        engine_name = engine_raw if engine_raw else default_engine

        while True:
            print("  Tryb limitu siły UCI: [1] Elo [Domyślny] | [2] Głębokość | [3] Czas (ms) | [0] Powrót")
            mode_raw = input_func("Wybierz tryb limitu (0-3) [Enter = Elo]: ").strip()
            if mode_raw == "0":
                return None
            if mode_raw in ("1", ""):
                val = self._prompt_positive_int(
                    "Docelowe Elo silnika", default=1500, min_val=600, max_val=3500, input_func=input_func
                )
                if val is None:
                    continue
                return PlayerConfig(
                    agent_type=AgentType.UCI_ENGINE,
                    name=f"UCI:{engine_name}",
                    uci_engine_name=engine_name,
                    uci_limit_mode=UciLimitMode.ELO,
                    uci_limit_value=val,
                )
            if mode_raw == "2":
                val = self._prompt_positive_int(
                    "Głębokość przeszukiwania (depth)", default=8, min_val=1, max_val=30, input_func=input_func
                )
                if val is None:
                    continue
                return PlayerConfig(
                    agent_type=AgentType.UCI_ENGINE,
                    name=f"UCI:{engine_name}",
                    uci_engine_name=engine_name,
                    uci_limit_mode=UciLimitMode.DEPTH,
                    uci_limit_value=val,
                )
            if mode_raw == "3":
                val = self._prompt_positive_int(
                    "Czas na ruch w milisekundach", default=500, min_val=50, max_val=30000, input_func=input_func
                )
                if val is None:
                    continue
                return PlayerConfig(
                    agent_type=AgentType.UCI_ENGINE,
                    name=f"UCI:{engine_name}",
                    uci_engine_name=engine_name,
                    uci_limit_mode=UciLimitMode.TIME_MS,
                    uci_limit_value=val,
                )
            print("Niepoprawny wybór! Wpisz 0, 1, 2 lub 3.")

    def _prompt_neural_config(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[PlayerConfig]:
        """Podkreator wyboru linii treningowej, checkpointu i parametrów Własnej Sieci AI."""
        lines = self.discover_model_lines()
        if not lines:
            print("Brak zapisanych modeli własnej sieci w katalogu models/.")
            print("Wytrenuj pierwszy model w Fazie 5 lub wybierz Bota bazowego / Silnik UCI.")
            return None

        print("  Dostępne linie treningowe:")
        for idx, info in enumerate(lines, start=1):
            print(f"  [{idx}] {info['name']}")
        line_idx = self._prompt_positive_int(
            "Wybierz numer linii", default=1, min_val=1, max_val=len(lines), input_func=input_func
        )
        if line_idx is None:
            return None
        chosen_line = lines[line_idx - 1]
        line_name = str(chosen_line["name"])

        ckpts = chosen_line["checkpoints"]
        ckpt_list: List[str] = ckpts if isinstance(ckpts, list) else []
        checkpoint_name: Optional[str] = ckpt_list[-1] if ckpt_list else None
        if len(ckpt_list) > 1:
            print("  Dostępne checkpointy:")
            for idx, ckpt in enumerate(ckpt_list, start=1):
                print(f"  [{idx}] {ckpt}")
            ckpt_idx = self._prompt_positive_int(
                "Wybierz numer checkpointu",
                default=len(ckpt_list),
                min_val=1,
                max_val=len(ckpt_list),
                input_func=input_func,
            )
            if ckpt_idx is None:
                return None
            checkpoint_name = ckpt_list[ckpt_idx - 1]

        print("  Tryb wyboru ruchu: [1] MCTS [Domyślny] | [2] Raw Policy | [0] Powrót")
        mode_raw = input_func("Wybierz tryb (0-2) [Enter = MCTS]: ").strip()
        if mode_raw == "0":
            return None
        search_mode = NeuralSearchMode.RAW_POLICY if mode_raw == "2" else NeuralSearchMode.MCTS

        sims = 100
        if search_mode == NeuralSearchMode.MCTS:
            sims_val = self._prompt_positive_int(
                "Liczba symulacji MCTS", default=100, min_val=1, max_val=10000, input_func=input_func
            )
            if sims_val is None:
                return None
            sims = sims_val

        return PlayerConfig(
            agent_type=AgentType.NEURAL_NET,
            name=f"Sieć:{line_name}",
            model_line=line_name,
            checkpoint_name=checkpoint_name,
            search_mode=search_mode,
            mcts_simulations=sims,
        )

    def prompt_fen(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[str]:
        """Pobiera i waliduje ręcznie wklejony łańcuch FEN."""
        while True:
            fen_input = input_func("Wklej kod pozycji FEN (lub '0' / 'anuluj'): ").strip()
            if fen_input.lower() in ("0", "anuluj", "cancel", "c", "exit", ""):
                return None

            result: FenValidationResult = FenParser.validate(fen_input)
            if result.is_valid and result.cleaned_fen is not None:
                print(f"Pozycja FEN poprawna: {result.cleaned_fen}")
                return result.cleaned_fen

            print(f"Błąd walidacji FEN: {result.error_message}")
            print("Spróbuj ponownie lub wpisz '0', aby anulować.")

    def prompt_initial_position(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Tuple[bool, Optional[str]]:
        """
        Pyta o pozycję początkową (Domyślna / Własny FEN / Presety).
        Zwraca (ok, fen_or_none). Jeśli ok=False, użytkownik wybrał [0] Powrót.
        """
        while True:
            print(self.format_fen_selection())
            choice = input_func("Wybierz pozycję (0-3) [Enter = 1]: ").strip()
            if choice == "0":
                return False, None
            if choice in ("1", ""):
                return True, None
            if choice == "2":
                custom_fen = self.prompt_fen(input_func=input_func)
                if custom_fen is not None:
                    return True, custom_fen
                continue
            if choice == "3":
                print(self.format_fen_presets())
                preset_idx = self._prompt_positive_int(
                    "Wybierz numer presetu",
                    default=1,
                    min_val=1,
                    max_val=len(FEN_PRESETS),
                    input_func=input_func,
                )
                if preset_idx is None:
                    continue
                _, preset_fen = FEN_PRESETS[preset_idx - 1]
                return True, None if preset_fen == chess.STARTING_FEN else preset_fen
            print("Niepoprawny wybór! Wpisz 0, 1, 2 lub 3 (albo naciśnij Enter).")

    def prompt_ai_display_mode(
        self,
        num_games: int,
        input_func: Callable[[str], str] = input,
    ) -> Optional[Tuple[AiDisplayMode, float]]:
        """Pyta o sposób wyświetlania partii w meczach AI vs AI."""
        default_choice = "3" if num_games > 1 else "1"
        while True:
            print("  Tryb wyświetlania AI vs AI:")
            print("  [1] Na żywo z pauzą (0.5s na ruch)")
            print("  [2] Krok po kroku (kolejny ruch po naciśnięciu Enter)")
            print("  [3] Szybki benchmark (bez rysowania planszy, tylko wynik)")
            print("  [0] Powrót")
            choice = input_func(
                f"Wybierz tryb (0-3) [Enter = {default_choice}]: "
            ).strip()
            if choice == "0":
                return None
            effective = choice if choice else default_choice
            if effective == "1":
                return AiDisplayMode.LIVE_DELAY, 0.5
            if effective == "2":
                return AiDisplayMode.STEP_BY_STEP, 0.0
            if effective == "3":
                return AiDisplayMode.FAST_BENCHMARK, 0.0
            print("Niepoprawny wybór! Wpisz 0, 1, 2 lub 3.")

    def prompt_game_wizard(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[GameConfig]:
        """
        Pełny Kreator Wyboru dla gałęzi [1] GRA.
        Zwraca gotowy obiekt GameConfig lub None, jeśli użytkownik wrócił do Menu Głównego.
        """
        while True:
            mode = self.prompt_game_mode(input_func=input_func)
            if mode is None:
                return None

            config = self._run_mode_wizard(mode=mode, input_func=input_func)
            if config is None:
                continue

            print(self.format_game_summary(config))
            confirm = input_func("Potwierdzenie [Enter = Start / 0 = Anuluj]: ").strip()
            if confirm == "0":
                continue
            return config

    def _run_mode_wizard(
        self,
        mode: GameMode,
        input_func: Callable[[str], str] = input,
    ) -> Optional[GameConfig]:
        """Zbiera parametry specyficzne dla wybranego trybu rozgrywki."""
        human_cfg = PlayerConfig(agent_type=AgentType.HUMAN, name="Człowiek")

        if mode == GameMode.HUMAN_VS_AI:
            color = self.prompt_side_selection(input_func=input_func)
            if color is None:
                return None
            ai_cfg = self.prompt_agent_config(role_label="Przeciwnika AI", input_func=input_func)
            if ai_cfg is None:
                return None
            test_mode = self._prompt_yes_no("Włączyć tryb testowy (podgląd ocen / cofanie)?", default=False, input_func=input_func)
            if test_mode is None:
                return None
            pos_ok, initial_fen = self.prompt_initial_position(input_func=input_func)
            if not pos_ok:
                return None
            white = human_cfg if color == chess.WHITE else ai_cfg
            black = ai_cfg if color == chess.WHITE else human_cfg
            return GameConfig(
                mode=mode,
                white=white,
                black=black,
                initial_fen=initial_fen,
                test_mode=test_mode,
            )

        if mode == GameMode.HUMAN_VS_HUMAN:
            use_assist = self._prompt_yes_no("Włączyć cichą asystę AI w tle?", default=False, input_func=input_func)
            if use_assist is None:
                return None
            assist_agent: Optional[PlayerConfig] = None
            if use_assist:
                assist_agent = self.prompt_agent_config(role_label="Asystenta AI", input_func=input_func)
                if assist_agent is None:
                    return None
            pos_ok, initial_fen = self.prompt_initial_position(input_func=input_func)
            if not pos_ok:
                return None
            return GameConfig(
                mode=mode,
                white=human_cfg,
                black=human_cfg,
                initial_fen=initial_fen,
                assist_agent=assist_agent,
            )

        if mode == GameMode.AI_VS_AI:
            white_ai = self.prompt_agent_config(role_label="Białego AI", input_func=input_func)
            if white_ai is None:
                return None
            black_ai = self.prompt_agent_config(role_label="Czarnego AI", input_func=input_func)
            if black_ai is None:
                return None
            num_games = self._prompt_positive_int(
                "Liczba partii w meczu", default=1, min_val=1, max_val=1000, input_func=input_func
            )
            if num_games is None:
                return None
            swap_colors = False
            if num_games > 1:
                swap_ans = self._prompt_yes_no("Zamieniać kolory co partię?", default=True, input_func=input_func)
                if swap_ans is None:
                    return None
                swap_colors = swap_ans
            display_res = self.prompt_ai_display_mode(num_games=num_games, input_func=input_func)
            if display_res is None:
                return None
            display_mode, delay = display_res
            pos_ok, initial_fen = self.prompt_initial_position(input_func=input_func)
            if not pos_ok:
                return None
            return GameConfig(
                mode=mode,
                white=white_ai,
                black=black_ai,
                initial_fen=initial_fen,
                num_games=num_games,
                swap_colors=swap_colors,
                display_mode=display_mode,
                move_delay_sec=delay,
            )

        # GameMode.CUSTOM (Konfigurator własny - pełna swoboda)
        white_any = self.prompt_agent_config(
            role_label="Białych", allow_human=True, input_func=input_func
        )
        if white_any is None:
            return None
        black_any = self.prompt_agent_config(
            role_label="Czarnych", allow_human=True, input_func=input_func
        )
        if black_any is None:
            return None
        pos_ok, initial_fen = self.prompt_initial_position(input_func=input_func)
        if not pos_ok:
            return None
        test_mode = self._prompt_yes_no("Włączyć tryb testowy?", default=False, input_func=input_func)
        if test_mode is None:
            return None
        save_pgn = self._prompt_yes_no("Zapisać wynikową partię do pliku PGN?", default=False, input_func=input_func)
        if save_pgn is None:
            return None
        return GameConfig(
            mode=GameMode.CUSTOM,
            white=white_any,
            black=black_any,
            initial_fen=initial_fen,
            test_mode=test_mode,
            save_pgn=save_pgn,
        )

    def prompt_training_wizard(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[TrainingConfig]:
        """
        Szkielet Kreatora Treningu dla gałęzi [2] NAUKA.
        Zwraca TrainingConfig lub None przy wyborze [0] Powrót.
        """
        method_map = {
            "1": TrainingMethod.SUPERVISED_PGN,
            "2": TrainingMethod.SELF_PLAY_RL,
            "3": TrainingMethod.UCI_DISTILLATION,
            "4": TrainingMethod.CURRICULUM_EPD,
        }
        while True:
            print(self.format_training_method_menu())
            m_choice = input_func("Wybierz metodę nauki (0-4): ").strip()
            if m_choice == "0":
                return None
            if m_choice not in method_map:
                print("Niepoprawny wybór! Wpisz cyfrę od 0 do 4.")
                continue
            method = method_map[m_choice]

            print(self.format_training_start_menu())
            s_choice = input_func("Wybierz punkt startowy (0-3) [Enter = Od zera]: ").strip()
            if s_choice == "0":
                continue
            start_map = {
                "": TrainingStartMode.FROM_SCRATCH,
                "1": TrainingStartMode.FROM_SCRATCH,
                "2": TrainingStartMode.RESUME_LINE,
                "3": TrainingStartMode.BRANCH_CHECKPOINT,
            }
            if s_choice not in start_map:
                print("Niepoprawny wybór punktu startowego.")
                continue
            start_mode = start_map[s_choice]

            line_name = input_func(
                f"Nazwa docelowej linii treningowej [Enter = {method.value}_v1] (0 = powrót): "
            ).strip()
            if line_name == "0":
                continue
            if not line_name:
                line_name = f"{method.value}_v1"

            return TrainingConfig(
                method=method,
                start_mode=start_mode,
                target_line_name=line_name,
            )

    def prompt_full_setup(
        self,
        input_func: Callable[[str], str] = input,
    ) -> Optional[GameConfig]:
        """Alias uruchamiający bezpośrednio Kreator Wyboru rozgrywki."""
        return self.prompt_game_wizard(input_func=input_func)
