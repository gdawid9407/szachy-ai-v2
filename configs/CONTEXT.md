# PROJECT CONTEXT: Szachy AI v2 (Structure, Roadmap & State Ledger)

## 1. Rejestr Stanu Pracy (Work State Ledger)

- **Aktualna Faza:** Faza 2 – Interfejs Gry, Menu, Poziom 0 AI i Zewnętrzny Silnik UCI (Grywalne MVP)
- **Aktualne Zadanie:** Zadanie 2.5 – Implementacja `src/engine/uci_engine.py` – zewnętrzny silnik UCI (np. Stockfish) z kontrolą siły gry (`UCI_LimitStrength`, `UCI_Elo` / `depth` / `time_ms`) oraz integracja z `GameController`
- **Ostatnio ukończone zadanie:** Zadanie 2.4 – Implementacja `src/ui/game_controller.py`, `main.py` oraz `tests/test_game_controller.py` (pełna grywalna pętla CLI dla wszystkich trybów z Kreatora Wyboru, obsługa komend i eksport PGN – 68 testów przechodzi pomyślnie)
- **Aktywne blokery / Problemy:** Brak

### Zablokowane Niezmienniki Techniczne (Locked Invariants)

- **Silnik reguł gry:** `python-chess` (wyłączna odpowiedzialność za zasady i legalność ruchów).
- **Kształt tensora wejściowego (Plansza):** `(20, 8, 8)` typu `float32`.
- **Przestrzeń akcji (Ruchy wyjściowe):** `4672` dyskretnych indeksów (`8 x 8 x 73`).
- **Architektura sieci:** Dual-Head ResNet (Rdzeń rezydualny + Policy Head + Value Head).
- **Format danych treningowych:** Pliki binarne czytane strumieniowo przez `numpy.memmap` (ochrona limitu 16 GB RAM).
- **Obsługa sprzętu:** Dynamiczny wybór `cuda` (PC RTX 3070) -> `mps` (MacBook M2) -> `cpu`.
- **Model środowiskowy i konteneryzacja:** Środowisko natywne `.venv` dla ML i silnika (dostęp do akceleratorów sprzętowych) + Docker Compose dla platformy webowej (Laravel, baza danych PostgreSQL/MySQL, serwer mailingu testowego Mailpit).
- **Strategia realizacji równoległej (Parallel Track):** Platforma webowa Laravel może być rozwijana równolegle do długotrwałych procesów treningu sieci neuronowej (Fazy 4–6), z wykorzystaniem silnika szachowego i adaptera UCI (Stockfish) jako wczesnego, grywalnego MVP.

---

## 2. Struktura Plików Projektu (File Structure)

    szachy ai v2/
    ├── configs/
    │   ├── AGENTS.md                   # [Plik 0] Rola instruktora, protokół pedagogiczny, nadrzędne wytyczne systemu
    │   ├── PROJECT_INSTRUCTIONS.md     # [Plik 1] Szczegółowa specyfikacja inżynieryjna + opis modułów
    │   ├── CONTEXT.md                  # [Plik 2] Rejestr stanu + Struktura plików + Roadmapa zadań
    │   └── default_config.yaml         # [Faza 4] Deklaratywna konfiguracja parametrów sieci, treningu i przeszukiwania
    │
    ├── data/
    │   ├── raw/                        # Surowe pliki PGN z historycznymi partiami szachowymi
    │   └── processed/                  # Przetworzone pliki binarne pod np.memmap (states.bin, targets.bin)
    │
    ├── models/
    │   ├── <linia_treningowa>/         # Hierarchiczne katalogi linii (metadata.json z polem parent + checkpointy *.pt)
    │   ├── checkpoints/                # Bezstratne checkpointy robocze (.pt: wagi, optimizer, scheduler, config, lineage)
    │   └── registry.md                 # Katalog wersji modeli i rodowód eksperymentów (lineage tracking)
    │
    ├── src/
    │   ├── __init__.py
    │   ├── config.py                   # Ścieżki (pathlib), stałe wymiarów, detekcja sprzętu (MPS/CUDA/CPU)
    │   │
    │   ├── engine/                     # Silnik reguł gry w szachy (bez udziału AI) i obsługa silników zewnętrznych
    │   │   ├── __init__.py
    │   │   ├── board_manager.py        # Zarządzanie planszą, legalne ruchy, cofanie (back move), historia gry
    │   │   ├── fen_parser.py           # Walidacja i ładowanie pozycji z notacji FEN (np. z chess.com)
    │   │   └── uci_engine.py           # [MVP] Klient zewnętrznych silników szachowych UCI (Stockfish) z kontrolą siły Elo
    │   │
    │   ├── ui/                         # Interfejs gry, szachownica i menu (CLI)
    │   │   ├── __init__.py
    │   │   ├── board_view.py           # Wyświetlanie dużej, przejrzystej szachownicy i wskaźnika wygranej
    │   │   ├── menu.py                 # Kreator Wyboru (GRA / NAUKA / STAN I METRYKI), PlayerConfig/GameConfig, presety FEN, parser komend
    │   │   └── game_controller.py      # Pętla gry i tryby: Human vs Human, Human vs AI (diagnostyka), AI vs AI
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
    │   ├── search/                     # Silnik decyzyjny AI, przeszukiwanie i benchmarki
    │   │   ├── __init__.py
    │   │   ├── evaluator.py            # Obliczanie wskaźnika predykcji wygranej (%) oraz AI bez wiedzy (losowe)
    │   │   ├── search_tree.py          # Algorytm przewidywania ruchów w głąb (MCTS / Alpha-Beta) + statystyki drzewa
    │   │   └── benchmark.py            # Wielopartiowe turnieje porównawcze modeli (multi-game benchmarks)
    │   │
    │   ├── training/                   # Headless potok nauki sieci neuronowej
    │   │   ├── __init__.py
    │   │   ├── dataset.py              # Bezpieczne ładowanie danych z dysku (np.memmap)
    │   │   ├── trainer.py              # [Faza 5] Orkiestrator headless (pause, resume, checkpointing, telemetria)
    │   │   ├── supervised.py           # Nauka na podstawie historycznych gier z internetu (PGN)
    │   │   ├── self_play.py            # Nauka na podstawie gry AI vs AI (samouczenie AlphaZero)
    │   │   └── transfer.py             # Uczenie transferowe, gałęzie modeli i destylacja wiedzy (UCI)
    │   │
    │   └── api/                        # Bezstanowy mikroserwis HTTP/JSON dla platformy webowej Laravel
    │       ├── __init__.py
    │       └── app.py                  # Endpointy inferencji, walidacji stanu i ewaluacji pozycji
    │
    ├── tests/                          # Testy jednostkowe i integracyjne wszystkich modułów
    │   ├── test_board_manager.py
    │   ├── test_fen_parser.py
    │   ├── test_board_view.py
    │   ├── test_evaluator.py
    │   ├── test_menu.py
    │   └── test_game_controller.py
    │
    ├── pack_project.py                 # Skrypt scalający cały kod projektu do jednego pliku dla Gemini
    ├── main.py                         # Główny plik uruchomieniowy gry CLI
    ├── train.py                        # [Faza 5] Niezależny punkt wejścia dla headless treningu (CLI/background)
    ├── requirements.txt                # Lista bibliotek (torch, chess, numpy, tqdm)
    ├── Dockerfile                      # [Faza 8] Obraz kontenera dla bezstanowego mikroserwisu Python API
    ├── docker-compose.yml              # [Faza 8] Orkiestracja usług webowych (baza danych, Mailpit, Laravel, API)
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
- [x] **1.3** Testy weryfikacyjne silnika – sprawdzenie poprawności zasad, roszad, bicia w przelocie, mata, pata i cofania ruchów.

### Faza 2: Interfejs Gry, Menu, Poziom 0 AI i Zewnętrzny Silnik UCI (Grywalne MVP)

- [x] **2.1** Implementacja `src/ui/board_view.py` – rysowanie dużej, przejrzystej szachownicy z oznaczeniami pól oraz miejscem na wskaźnik predykcji wygranej.
- [x] **2.2** Implementacja `src/search/evaluator.py` (Część 1) – stworzenie **Poziomu 0 AI (AI bez wiedzy)**, które wybiera losowe, ale w 100% legalne ruchy.
- [x] **2.3** Implementacja `src/ui/menu.py` – Kreator Wyboru (`GRA / NAUKA / STAN I METRYKI`), wspólny wybór agenta AI (`Bot Poziom 0 / Silnik UCI / Własna Sieć`), presety FEN, podsumowanie przed startem oraz obsługa komend (`exit`, `back`, `history`, `switch`, `fen` odczyt/zapis, `eval/hint`, `save`).
- [x] **2.4** Implementacja `src/ui/game_controller.py` oraz `main.py` – połączenie planszy i menu w działającą grę obsługującą tryby: **Human vs Human**, **Human vs AI**, **AI vs AI** (w tym serie N partii i eksport PGN) oraz **Konfigurator własny**.
- [ ] **2.5** Implementacja `src/engine/uci_engine.py` – **Zewnętrzny silnik UCI (np. Stockfish)** z konfigurowalnym limitem siły gry (`UCI_LimitStrength`, `UCI_Elo` / `Skill Level`) oraz integracja z kontrolerem gry. *(Domyka grywalne MVP i odblokowuje równoległą realizację Fazy 8 – Laravel MVC).*

### Faza 3: Matematyczna Reprezentacja Szachownicy (Wejście/Wyjście dla AI)

- [ ] **3.1** Zrozumienie i implementacja `src/representation/encoder.py` – zamiana obiektu szachownicy na matematyczny tensor o wymiarach `(20, 8, 8)`.
- [ ] **3.2** Zrozumienie i implementacja `src/representation/move_encoder.py` – zamiana każdego możliwego ruchu w szachach na indeks liczbowy (od `0` do `4671`) oraz maskowanie ruchów nielegalnych.
- [ ] **3.3** Testy reprezentacji – wizualne sprawdzenie, jak komputer „widzi” poszczególne warstwy szachownicy.

### Faza 4: Konfiguracja Deklaratywna, Budowa Sieci Neuronowej, Inspekcja i Wersjonowanie

- [ ] **4.1** Deklaratywna konfiguracja (`configs/default_config.yaml` oraz rozszerzenie `src/config.py`) – wydzielenie parametrów statycznych (architektura) i dynamicznych (trening/przeszukiwanie) do plików konfiguracyjnych.
- [ ] **4.2** Implementacja `src/neural/architectures.py` – budowa sieci neuronowej (warstwy splotowe, bloki rezydualne, głowica oceniająca szanse na wygraną Value Head, głowica wybierająca ruchy Policy Head).
- [ ] **4.3** Implementacja `src/search/evaluator.py` (Część 2) – podłączenie `NeuralEvaluator` oraz **Wskaźnika predykcji wygranej** (przeliczanie wyjścia sieci `[-1, 1]` na procentową szansę zwycięstwa na żywo podczas gry).
- [ ] **4.4** Implementacja `src/neural/inspector.py` oraz **trybu diagnostycznego** w `Human vs AI` – podgląd liczby parametrów, wag, aktywacji warstw, gradientów, entropii decyzji oraz rozkładu Policy/Value na żywo.
- [ ] **4.5** Implementacja `src/neural/version_manager.py` oraz inicjalizacja `models/registry.md` – zapis/odczyt bezstratnych checkpointów (wagi, optymalizator, scheduler, `config_snapshot`, `metadata`) i śledzenie rodowodu eksperymentów (*experiment lineage tracking*).
- [ ] **4.6** Implementacja `src/neural/loss.py` – matematyczna funkcja straty mówiąca sieci, jak bardzo pomyliła się w ocenie pozycji i wyborze ruchu.

### Faza 5: Headless Potok Treningowy – Poziomy Wytrenowania AI

- [ ] **5.1** Przygotowanie danych i `src/training/dataset.py` – konwersja pobranych z internetu gier PGN do formatu binarnego `np.memmap` (bezpieczne dla pamięci RAM).
- [ ] **5.2** Implementacja `src/training/trainer.py` oraz `train.py` – **Headless Training Orchestrator & Lossless Lifecycle Control**: niezależny od interfejsu gry proces treningowy z bezstratnym pauzowaniem (`pause`), wznawianiem (`resume`, z odtworzeniem stanu optymalizatora, schedulera i epoki), automatycznym `checkpointing` oraz logowaniem telemetrii treningowej na żywo.
- [ ] **5.3** Implementacja `src/training/supervised.py` – **Poziom 1 AI (AI wytrenowane na historycznych grach)**: proces uczenia sieci na ruchach silnych graczy, obserwacja spadku błędu (loss) i zapisanie wytrenowanej wersji modelu.
- [ ] **5.4** Implementacja `src/training/self_play.py` – **Poziom 2 AI (AI wytrenowane przez grę AI vs AI)**: system, w którym sieć gra sama ze sobą, generuje nowe doświadczenia i uczy się na własnych wygranych oraz przegranych.
- [ ] **5.5** Implementacja `src/training/transfer.py` – uczenie transferowe i gałęziowe (kontynuacja treningu bez nadpisywania bazy) oraz opcjonalna destylacja wiedzy (*Knowledge Distillation*) ze Stockfisha (UCI).

### Faza 6: Głębokość Przewidywania Ruchów i Połączenie Wszystkiego

- [ ] **6.1** Implementacja `src/search/search_tree.py` – **Moc sieci i głębokość przewidywania**: algorytm przeszukiwania drzewa gry (Alpha-Beta / MCTS) z konfigurowalną głębokością oraz zbieraniem statystyk drzewa (*search tree statistics* dla trybu diagnostycznego).
- [ ] **6.2** Stworzenie **Poziomu 3 AI (Połączenie wszystkiego)**: integracja wyuczonej sieci (z gier historycznych oraz AI vs AI) z algorytmem głębokiego przewidywania ruchów w przód.

### Faza 7: Benchmarki Turniejowe, Analiza Badawcza i Optymalizacja

- [ ] **7.1** Implementacja `src/search/benchmark.py` – zautomatyzowane turnieje wielopartiowe (*multi-game benchmarks*: model vs model oraz model vs Stockfish na zadanych limitach Elo) do obiektywnego rankingu wersji.
- [ ] **7.2** Eksperymenty badawcze – porównywanie gałęzi treningowych, analiza wag i map cech w `inspector.py`, optymalizacja wydajnościowa na M2 (`mps`) i RTX 3070 (`cuda`).

### Faza 8: Integracja z Platformą Webową (Laravel MVC & Stateless API) – *Ścieżka Równoległa / Early MVP*

*Warunek startu:* Odblokowana od razu po ukończeniu **Zadania 2.5** (`BoardManager` + `uci_engine.py` ze Stockfishem). Może być rozwijana równolegle do Faz 3–7. Autorskie sieci neuronowe są podpinane do API jako kolejne poziomy AI po zakończeniu ich treningu.

- [ ] **8.1** Implementacja bezstanowego API HTTP/JSON w Pythonie (`src/api/app.py`) – walidacja ruchów, obsługa FEN oraz zapytań do silnika zewnętrznego UCI (Stockfish) na potrzeby wczesnego MVP.
- [ ] **8.2** Konteneryzacja i orkiestracja Docker (`Dockerfile` dla Python API oraz `docker-compose.yml`: PostgreSQL/MySQL, serwer testowego mailingu Mailpit, kontener aplikacji Laravel/Sail, sieć wewnętrzna).
- [ ] **8.3** Aplikacja webowa Laravel MVC – architektura, schemat relacyjnej bazy danych (konta, mecze, historia posunięć, rankingi).
- [ ] **8.4** Bezpieczeństwo i uwierzytelnianie – rejestracja, logowanie, Two-Factor Authentication (2FA) oraz transakcyjny mailing testowy przez Mailpit.
- [ ] **8.5** Klient HTTP w Laravel, webowy interfejs szachownicy oraz późniejsze podpięcie autorskich modeli sieci neuronowej (po Fazach 4–6).

---

## 4. Rejestr Wersji Sieci Neuronowej (Model Version Summary)

| ID Wersji       | Rodzic (Lineage) | Poziom AI          | Opis treningu i architektury          | Liczba parametrów | Głębokość (Search) | Status              |
| :-------------- | :--------------- | :----------------- | :------------------------------------ | :---------------- | :----------------- | :------------------ |
| `v0.0-random`   | `—`              | Tier 0             | Brak sieci – losowe ruchy legalne     | 0                 | 0                  | Ukończony (Faza 2)  |
| `uci-stockfish` | `external`       | External UCI       | Zewnętrzny silnik UCI (limit Elo)     | Zewnętrzny        | Konfigurowalna     | Planowany (Faza 2.5)|
| `v1.0-init`     | `—`              | Tier 1 (Untrained) | Sieć zainicjowana losowymi wagami     | Do ustalenia      | 1                  | Planowany (Faza 4)  |

---

## 5. Dziennik Postępów (Progress Log)

- **[2026-10-22]** – Zdefiniowano strukturę projektu, niezmienniki architektoniczne oraz plan działania (Roadmapa).
- **[2026-10-23]** – Ukończono Fazę 0: utworzono `requirements.txt`, zweryfikowano `pack_project.py` i zaimplementowano `src/config.py`.
- **[2026-10-23]** – Ukończono Zadanie 1.1: zaimplementowano i przetestowano moduł `src/engine/board_manager.py` (obsługa planszy, legalne ruchy, back move, historia SAN/UCI).
- **[2026-10-23]** – Ukończono Zadanie 1.2: zaimplementowano moduł `src/engine/fen_parser.py` z dwuetapową walidacją (syntaktyczną i semantyczną) oraz polskojęzyczną diagnostyką błędów.
- **[2026-10-28]** – Ukończono Zadanie 1.3 oraz całą Fazę 1: pełny pakiet 26 testów jednostkowych silnika (`test_board_manager.py`, `test_fen_parser.py`) zakończony sukcesem. Rozpoczęcie Fazy 2.
- **[2026-10-28]** – Ukończono Zadanie 2.1: zaimplementowano moduł `src/ui/board_view.py` (wizualizacja szachownicy w terminalu, perspektywa Białe/Czarne, wyróżnianie ruchów ANSI, boczny panel stanu gry oraz pasek predykcji wygranej) wraz z pakietem 11 testów jednostkowych w `tests/test_board_view.py`. Łącznie 37 testów przechodzi pomyślnie.
- **[2026-10-28]** – Ukończono Zadanie 2.2: zaimplementowano moduł `src/search/evaluator.py` (interfejs bazowy `BaseEvaluator`, Poziom 0 AI `RandomEvaluator`/`Tier0RandomAI` wybierający wyłącznie w 100% legalne ruchy bez sieci neuronowej) wraz z pakietem 9 testów jednostkowych w `tests/test_evaluator.py`. Łącznie 46 testów przechodzi pomyślnie.
- **[2026-10-28]** – Ukończono Zadanie 2.3: zaimplementowano moduł `src/ui/menu.py` (obsługa menu głównego, konfiguracja trybów Human vs Human, Human vs AI, AI vs AI, wybór stron i poziomów AI, parsowanie komend exit, back move, history, switch side, load FEN) wraz z pakietem 12 testów jednostkowych w `tests/test_menu.py`. Łącznie 58 testów przechodzi pomyślnie.
- **[2026-11-24]** – Zsynchronizowano dokumentację (`AGENTS.md`, `PROJECT_INSTRUCTIONS.md`, `CONTEXT.md`): dodano Docker Compose, przesunięto klienta UCI (Stockfish) do Fazy 2 (Zadanie 2.5 – grywalne MVP odblokowujące równoległą Fazę 8 Laravel), uzupełniono strukturę o `tests/`, `default_config.yaml`, `trainer.py`/`train.py`, `Dockerfile` oraz dodano kolumnę `Rodzic (Lineage)` w tabeli modeli.
- **[2026-11-25]** – Przebudowano `src/ui/menu.py` oraz `tests/test_menu.py` na docelowy Kreator Wyboru (`GRA / NAUKA / STAN I METRYKI`, wspólny kreator agenta `PlayerConfig`, uniwersalny `GameConfig`, `TrainingConfig`, presety FEN, diagnostyka sprzętu/UCI/linii modeli w `models/`, nowe komendy `fen` odczyt, `eval/hint`, `save` PGN). Zsynchronizowano `PROJECT_INSTRUCTIONS.md` oraz `CONTEXT.md`. Łącznie 63 testy jednostkowe przechodzą pomyślnie.
- **[2026-11-25]** – Ukończono Zadanie 2.4: zaimplementowano `src/ui/game_controller.py`, `main.py` oraz 5 testów integracyjnych w `tests/test_game_controller.py`. Szachy są w pełni grywalne w terminalu (`Human vs Human`, `Human vs AI`, `AI vs AI` z seriami N partii i trybem szybkim, `Konfigurator własny`, eksport PGN). Łącznie 68 testów jednostkowych przechodzi pomyślnie.
