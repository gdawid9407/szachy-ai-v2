"""
src/config.py

Centralny moduł konfiguracyjny projektu Szachy AI v2.
Zawiera:
- Definicje ścieżek projektowych w oparciu o pathlib.Path (brak sztywnych ścieżek).
- Niezmienniki techniczne reprezentacji szachownicy i przestrzeni akcji.
- Dynamiczną detekcję akceleratora sprzętowego (CUDA -> MPS -> CPU).
- Hiperparametry architektury sieci neuronowej Dual-Head ResNet.
- Ustawienia strumieniowania danych binarnych (numpy.memmap).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final, Tuple


# ==============================================================================
# 1. ŚCIEŻKI PROJEKTOWE (Pathlib)
# ==============================================================================

# Główny katalog projektu (szachy ai v2/)
ROOT_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Katalog kodu źródłowego
SRC_DIR: Final[Path] = ROOT_DIR / "src"

# Katalogi danych (surowe PGN oraz przetworzone binarne memmap)
DATA_DIR: Final[Path] = ROOT_DIR / "data"
DATA_RAW_DIR: Final[Path] = DATA_DIR / "raw"
DATA_PROCESSED_DIR: Final[Path] = DATA_DIR / "processed"

# Katalogi modeli i wag sieci
MODELS_DIR: Final[Path] = ROOT_DIR / "models"
CHECKPOINTS_DIR: Final[Path] = MODELS_DIR / "checkpoints"
MODEL_REGISTRY_FILE: Final[Path] = MODELS_DIR / "registry.md"

# Katalog konfiguracji i specyfikacji
CONFIGS_DIR: Final[Path] = ROOT_DIR / "configs"
PROJECT_INSTRUCTIONS_FILE: Final[Path] = CONFIGS_DIR / "PROJECT_INSTRUCTIONS.md"
CONTEXT_FILE: Final[Path] = CONFIGS_DIR / "CONTEXT.md"


def ensure_directories() -> None:
    """Tworzy wymagane katalogi projektu, jeśli jeszcze nie istnieją."""
    for directory in (DATA_RAW_DIR, DATA_PROCESSED_DIR, CHECKPOINTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# 2. DETEKCJA SPRZĘTU (CUDA -> MPS -> CPU)
# ==============================================================================

def get_device_name() -> str:
    """
    Automatycznie wykrywa najszybsze dostępne urządzenie obliczeniowe.
    Priorytet:
    1. CUDA (NVIDIA GPU - PC Windows 11 / RTX 3070)
    2. MPS (Apple Silicon Metal - MacBook M2)
    3. CPU (Uniwersalny fallback)
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"


def get_device():
    """
    Zwraca instancję torch.device odpowiednią dla aktualnego środowiska.
    W przypadku braku zainstalowanego pakietu PyTorch zgłasza czytelny ImportError.
    """
    try:
        import torch
        return torch.device(get_device_name())
    except ImportError as exc:
        raise ImportError(
            "Pakiet PyTorch nie jest zainstalowany. Zainstaluj zależności z pliku requirements.txt."
        ) from exc


# ==============================================================================
# 3. NIEZMIENNIKI TECHNICZNE REPREZENTACJI (Elementy Zablokowane)
# ==============================================================================

# Kształt tensora wejściowego pozycji szachowej: (Kanały, Wiersze, Kolumny)
BOARD_CHANNELS: Final[int] = 20
BOARD_ROWS: Final[int] = 8
BOARD_COLS: Final[int] = 8
BOARD_SHAPE: Final[Tuple[int, int, int]] = (BOARD_CHANNELS, BOARD_ROWS, BOARD_COLS)

# Szczegółowa specyfikacja kanałów wejściowych:
# Kanały 0-5: Białe figury (Pion, Skoczek, Goniec, Wieża, Hetman, Król)
# Kanały 6-11: Czarne figury (Pion, Skoczek, Goniec, Wieża, Hetman, Król)
# Kanał 12: Strona na posunięciu (1.0 = Białe, 0.0 = Czarne)
# Kanały 13-16: Prawa do roszady (Biała krótka, Biała długa, Czarna krótka, Czarna długa)
# Kanał 17: Pole bicia w przelocie (En Passant)
# Kanał 18: Licznik 50 ruchów (znormalizowany do [0.0, 1.0])
# Kanał 19: Licznik pełnych ruchów partii (znormalizowany do [0.0, 1.0])
CHANNEL_WHITE_PIECES_START: Final[int] = 0
CHANNEL_BLACK_PIECES_START: Final[int] = 6
CHANNEL_TURN: Final[int] = 12
CHANNEL_CASTLING_START: Final[int] = 13
CHANNEL_EN_PASSANT: Final[int] = 17
CHANNEL_HALFMOVE_CLOCK: Final[int] = 18
CHANNEL_FULLMOVE_NUMBER: Final[int] = 19

# Wymiar wyjściowej przestrzeni akcji (Standard AlphaZero: 8 x 8 x 73 płaszczyzny)
ACTION_SPACE_SIZE: Final[int] = 4672
ACTION_PLANES: Final[int] = 73

# Składowe 73 płaszczyzn ruchów:
# - 56 płaszczyzn: ruchy liniowe i ukośne królowej (8 kierunków x dystans 1..7)
# - 8 płaszczyzn: ruchy skoczka
# - 9 płaszczyzn: podpromocje pionów (3 kierunki x 3 figury: Skoczek, Goniec, Wieża)
NUM_QUEEN_MOVE_PLANES: Final[int] = 56
NUM_KNIGHT_MOVE_PLANES: Final[int] = 8
NUM_UNDERPROMOTION_PLANES: Final[int] = 9


# ==============================================================================
# 4. ARCHITEKTURA SIECI NEURONOWEJ (Dual-Head ResNet)
# ==============================================================================

# Liczba bloków rezydualnych w rdzeniu sieci (kompromis wydajność/moc na M2 i RTX 3070)
DEFAULT_NUM_RESIDUAL_BLOCKS: Final[int] = 6

# Liczba filtrów splotowych w blokach rezydualnych
DEFAULT_NUM_FILTERS: Final[int] = 128

# Liczba neuronów w warstwie ukrytej głowicy wartości (Value Head)
VALUE_HEAD_HIDDEN_DIM: Final[int] = 256

# Zakres wartości głowicy wartości: [-1.0, 1.0] (-1: przegrana, 0: remis, +1: wygrana)
VALUE_MIN: Final[float] = -1.0
VALUE_MAX: Final[float] = 1.0


def value_to_win_percentage(value: float) -> float:
    """
    Przelicza skalarny wynik głowicy wartości [-1.0, 1.0]
    na procentową szansę wygranej [0.0%, 100.0%].
    """
    clamped = max(VALUE_MIN, min(VALUE_MAX, value))
    return ((clamped + 1.0) / 2.0) * 100.0


# ==============================================================================
# 5. TRENING I PAMIĘĆ (numpy.memmap)
# ==============================================================================

# Format binarny danych dla ochrony limitu 16 GB RAM
DTYPE_STATE: Final[str] = "float32"
DTYPE_POLICY: Final[str] = "float32"
DTYPE_VALUE: Final[str] = "float32"

# Domyślny rozmiar paczki danych (batch size) dopasowany do 8 GB VRAM RTX 3070
DEFAULT_BATCH_SIZE: Final[int] = 64

# Domyślny współczynnik uczenia
DEFAULT_LEARNING_RATE: Final[float] = 1e-3
DEFAULT_WEIGHT_DECAY: Final[float] = 1e-4
