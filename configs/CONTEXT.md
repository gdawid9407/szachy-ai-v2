# PROJECT CONTEXT: Szachy AI v2 (Structure, Roadmap & State Ledger)

## 1. Rejestr Stanu Pracy (Work State Ledger)

- **Aktualna Faza:** Faza 1 – Stworzenie Silnika Gry w Szachy (Logika i Stan)
- **Aktualne Zadanie:** Zadanie 1.3 – Testy weryfikacyjne silnika – sprawdzenie poprawności zasad, roszad, bicia w przelocie, mata, pata i cofania ruchów
- **Ostatnio ukończone zadanie:** Zadanie 1.2 – Implementacja `src/engine/fen_parser.py` (wczytywanie i walidacja formatu FEN)
- **Aktywne blokery / Problemy:** Brak

### Zablokowane Niezmienniki Techniczne (Locked Invariants)

- **Silnik reguł gry:** `python-chess` (wyłączna odpowiedzialność za zasady i legalność ruchów).
- **Kształt tensora wejściowego (Plansza):** `(20, 8, 8)` typu `float32`.
- **Przestrzeń akcji (Ruchy wyjściowe):** `4672` dyskretnych indeksów (`8 x 8 x 73`).
- **Architektura sieci:** Dual-Head ResNet (Rdzeń rezydualny + Policy Head + Value Head).
- **Format danych treningowych:** Pliki binarne czytane strumieniowo przez `numpy.memmap` (ochrona limitu 16 GB RAM).
- **Obsługa sprzętu:** Dynamiczny wybór `cuda` (PC RTX 3070) -> `mps` (MacBook M2) -> `cpu`.

---

## 2. Struktura Plików Projektu (File Structure)

    szachy ai v2/
    ├── configs/
    │   ├── PROJECT_INSTRUCTIONS.md     # [Plik 1] Instrukcje dla agenta + pełny opis merytoryczny projektu
    │   └── CONTEXT.md                  # [Plik 2] Struktura plików + Plan działania (Roadmapa) + Rejestr stanu
    │
    ├── data/
    │   ├── raw/                        # Surowe pliki PGN z historycznymi partiami szachowymi
    │   └── processed/                  # Przetworzone pliki binarne pod np.memmap (states.bin, targets.bin)
    │
    ├── models/
    │   ├── checkpoints/                # Zapisane wagi sieci neuronowej (.pt) z procesów treningowych
    │   └── registry.md                 # Katalog wersji modeli (opisy wersji, parametry, dokładność)
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                   # Ścieżki (pathlib), stałe wymiarów, detekcja sprzętu (MPS/CUDA/CPU)
    │   │
    │   ├── engine/                     # Silnik reguł gry w szachy (bez udziału AI)
    │   │   ├── __init__.py
    │   │   ├── board_manager.py        # Zarządzanie planszą, legalne ruchy, cofanie (back move), historia gry
    │   │   └── fen_parser.py           # Walidacja i ładowanie pozycji z notacji FEN (np. z chess.com)
    │   │
    │   ├── ui/                         # Interfejs gry, szachownica i menu
    │   │   ├── __init__.py
    │   │   ├── board_view.py           # Wyświetlanie dużej, przejrzystej szachownicy i wskaźnika wygranej
    │   │   ├── menu.py                 # Menu: exit, back move, historia, wybór/zmiana strony, FEN, poziom AI
    │   │   └── game_controller.py      # Pętla gry i tryby: Human vs Human, Human vs AI, AI vs AI
    │   │
    │   ├── representation/             # Tłumaczenie szachownicy na matematykę dla sieci AI
    │   │   ├── __init__.py
    │   │   ├── encoder.py              # Konwersja szachownicy do tensora (20, 8, 8)
    │   │   └── move_encoder.py         # Mapowanie ruchów szachowych na indeksy 0..4671 i odwrotnie
    │   │
    │   ├── neural/                     # Sieć neuronowa, jej parametry i pełny wgląd w działanie
    │   │   ├── __init__.py
    │   │   ├── architectures.py        # Definicja sieci (ResNet + Policy Head + Value Head)
    │   │   ├── loss.py                 # Funkcja straty (błąd oceny pozycji + błąd wyboru ruchu)
    │   │   ├── inspector.py            # Wgląd w sieć: podgląd wag, aktywacji warstw, gradientów i parametrów
    │   │   └── version_manager.py      # Zapisywanie, wczytywanie i opisywanie wersji sieci neuronowej
    │   │
    │   ├── search/                     # Silnik decyzyjny AI i głębokość przewidywania
    │   │   ├── __init__.py
    │   │   ├── evaluator.py            # Obliczanie wskaźnika predykcji wygranej (%) oraz AI bez wiedzy (losowe)
    │   │   └── search_tree.py          # Algorytm przewidywania ruchów w głąb (przeszukiwanie drzewa gry)
    │   │
    │   └── training/                   # Nauka sieci neuronowej (poszczególne poziomy wytrenowania)
    │       ├── __init__.py
    │       ├── dataset.py              # Bezpieczne ładowanie danych z dysku (np.memmap)
    │       ├── supervised.py           # Nauka na podstawie historycznych gier z internetu (PGN)
    │       └── self_play.py            # Nauka na podstawie gry AI vs AI (samouczenie)
    │
    ├── pack_project.py                 # Skrypt scalający cały kod projektu do jednego pliku dla Gemini
    ├── main.py                         # Główny plik uruchomieniowy gry i systemu
    ├── requirements.txt                # Lista bibliotek (torch, chess, numpy, tqdm)
    └── .gitignore                      # Wykluczenia plików tymczasowych, dużych baz danych i modeli

---

## 3. Plan Działania – Roadmapa (Step-by-Step Implementation Plan)

Realizacja odbywa się małymi etapami. Żaden kolejny krok nie może zostać rozpoczęty przed przetestowaniem i dokładnym omówieniem kroku poprzedniego.

### Faza 0: Przygotowanie Środowiska i Narzędzi Projektowych

- [x] **0.1** Utworzenie struktury katalogów oraz plików `configs/PROJECT_INSTRUCTIONS.md` i `configs/CONTEXT.md`.
- [x] **0.2** Utworzenie `requirements.txt` oraz `.gitignore`.
- [x] **0.3** Implementacja `pack_project.py` (narzędzie zbierające wszystkie pliki z kodem w jeden plik tekstowy dla Gemini).
- [x] **0.4** Implementacja `src/config.py` (automatyczne wykrywanie MacBook M2 `mps` vs PC `cuda`, definicja ścieżek `pathlib`).

### Faza 1: Stworzenie Silnika Gry w Szachy (Logika i Stan)

- [x] **1.1** Implementacja `src/engine/board_manager.py` – obsługa planszy, sprawdzanie legalności ruchów, lista historii ruchów w aktualnej grze, funkcja cofania ruchu (`back move`).
- [x] **1.2** Implementacja `src/engine/fen_parser.py` – wczytywanie i walidacja własnych ustawień szachownicy wygenerowanych np. na chess.com (format FEN).
- [ ] **1.3** Testy weryfikacyjne silnika – sprawdzenie poprawności zasad, roszad, bicia w przelocie, mata, pata i cofania ruchów.

### Faza 2: Wygląd Szachownicy, Menu Gry i Tryby Rozgrywki (UI + Poziom 0 AI)

- [ ] **2.1** Implementacja `src/ui/board_view.py` – rysowanie dużej, przejrzystej szachownicy z oznaczeniami pól oraz miejscem na wskaźnik predykcji wygranej.
- [ ] **2.2** Implementacja `src/search/evaluator.py` (Część 1) – stworzenie **Poziomu 0 AI (AI bez wiedzy)**, które wybiera losowe, ale w 100% legalne ruchy.
- [ ] **2.3** Implementacja `src/ui/menu.py` – obsługa opcji: wyjście (`exit`), cofnięcie ruchu (`back move`), wyświetlenie historii ruchów, wybór strony przed grą, zmiana strony w trakcie gry, załadowanie pozycji FEN, wybór trybu gry i poziomu AI.
- [ ] **2.4** Implementacja `src/ui/game_controller.py` oraz `main.py` – połączenie planszy i menu w działającą grę obsługującą tryby: **Human vs Human**, **Human vs AI**, **AI vs AI**.

### Faza 3: Matematyczna Reprezentacja Szachownicy (Wejście/Wyjście dla AI)

- [ ] **3.1** Zrozumienie i implementacja `src/representation/encoder.py` – zamiana obiektu szachownicy na matematyczny tensor o wymiarach `(20, 8, 8)`.
- [ ] **3.2** Zrozumienie i implementacja `src/representation/move_encoder.py` – zamiana każdego możliwego ruchu w szachach na indeks liczbowy (od `0` do `4671`) oraz maskowanie ruchów nielegalnych.
- [ ] **3.3** Testy reprezentacji – wizualne sprawdzenie, jak komputer „widzi” poszczególne warstwy szachownicy.

### Faza 4: Budowa Sieci Neuronowej, Wgląd w Strukturę i Wersjonowanie

- [ ] **4.1** Implementacja `src/neural/architectures.py` – budowa sieci neuronowej (warstwy splotowe, bloki rezydualne, głowica oceniająca szanse na wygraną Value Head, głowica wybierająca ruchy Policy Head).
- [ ] **4.2** Implementacja `src/search/evaluator.py` (Część 2) – podłączenie **Wskaźnika predykcji wygranej** (przeliczanie wyjścia sieci `[-1, 1]` na procentową szansę zwycięstwa na żywo podczas gry).
- [ ] **4.3** Implementacja `src/neural/inspector.py` – narzędzia edukacyjne dające pełny wgląd w sieć: podgląd liczby parametrów, wartości wag, aktywacji poszczególnych warstw, gradientów oraz pewności decyzji AI.
- [ ] **4.4** Implementacja `src/neural/version_manager.py` oraz `models/registry.md` – system zapisywania, wczytywania i opisywania poszczególnych wersji sieci neuronowej.
- [ ] **4.5** Implementacja `src/neural/loss.py` – matematyczna funkcja straty mówiąca sieci, jak bardzo pomyliła się w ocenie pozycji i wyborze ruchu.

### Faza 5: Nauka Sieci Neuronowej – Poziomy Wytrenowania AI

- [ ] **5.1** Przygotowanie danych i `src/training/dataset.py` – konwersja pobranych z internetu gier PGN do formatu binarnego `np.memmap` (bezpieczne dla pamięci RAM).
- [ ] **5.2** Implementacja `src/training/supervised.py` – **Poziom 1 AI (AI wytrenowane na historycznych grach)**: proces uczenia sieci na ruchach silnych graczy, obserwacja spadku błędu (loss) i zapisanie wytrenowanej wersji modelu.
- [ ] **5.3** Implementacja `src/training/self_play.py` – **Poziom 2 AI (AI wytrenowane przez grę AI vs AI)**: system, w którym sieć gra sama ze sobą, generuje nowe doświadczenia i uczy się na własnych wygranych oraz przegranych.

### Faza 6: Głębokość Przewidywania Ruchów i Połączenie Wszystkiego

- [ ] **6.1** Implementacja `src/search/search_tree.py` – **Moc sieci i głębokość przewidywania**: algorytm przeszukiwania drzewa gry (Alpha-Beta / MCTS) z możliwością ustawienia głębokości liczenia ruchów w przód.
- [ ] **6.2** Stworzenie **Poziomu 3 AI (Połączenie wszystkiego)**: integracja wyuczonej sieci (z gier historycznych oraz AI vs AI) z algorytmem głębokiego przewidywania ruchów w przód.

### Faza 7: Zrozumienie Całości, Analiza Działania i Rozbudowa (Dodatki)

- [ ] **7.1** Eksperymenty badawcze – porównywanie różnych wersji sieci w meczach AI vs AI, analiza zachowania wag i warstw w `inspector.py`.
- [ ] **7.2** Rozbudowa projektu o nowe dodatki i ulepszenia (książki debiutowe, zaawansowana telemetria, optymalizacja szybkości działania na M2 i RTX 3070).

---

## 4. Rejestr Wersji Sieci Neuronowej (Model Version Summary)

| ID Wersji     | Poziom AI          | Opis treningu i architektury      | Liczba parametrów | Głębokość (Search) | Status             |
| :------------ | :----------------- | :-------------------------------- | :---------------- | :----------------- | :----------------- |
| `v0.0-random` | Tier 0             | Brak sieci – losowe ruchy legalne | 0                 | 0                  | Planowany (Faza 2) |
| `v1.0-init`   | Tier 1 (Untrained) | Sieć zainicjowana losowymi wagami | Do ustalenia      | 1                  | Planowany (Faza 4) |

---

## 5. Dziennik Postępów (Progress Log)

- **[2026-10-22]** – Zdefiniowano strukturę projektu, niezmienniki architektoniczne oraz plan działania (Roadmapa).
- **[2026-10-23]** – Ukończono Fazę 0: utworzono `requirements.txt`, zweryfikowano `pack_project.py` i zaimplementowano `src/config.py`.
- **[2026-10-23]** – Ukończono Zadanie 1.1: zaimplementowano i przetestowano moduł `src/engine/board_manager.py` (obsługa planszy, legalne ruchy, back move, historia SAN/UCI).
- **[2026-10-23]** – Ukończono Zadanie 1.2: zaimplementowano moduł `src/engine/fen_parser.py` z dwuetapową walidacją (syntaktyczną i semantyczną) oraz polskojęzyczną diagnostyką błędów.
