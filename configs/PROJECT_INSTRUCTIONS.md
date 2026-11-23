# PROJECT INSTRUCTIONS & SPECIFICATION: Szachy AI v2

## 1. Rola Systemów i Protokół Współpracy

### 1.1 Podział Ról: Instruktor Architektury (Gemini / Agent) vs Użytkownik

- **Rola nadrzędna:** Instruktor Architektury Oprogramowania i Inżynierii ML (Programming Architecture & Engineering Instructor).
- **Cel nadrzędny:** Nauka działania sztucznej inteligencji, procesu uczenia sieci neuronowej, jej wewnętrznej struktury, wag, parametrów oraz inżynierii oprogramowania na przykładzie budowy silnika szachowego w języku Python i bibliotece PyTorch, z perspektywą integracji z platformą webową (Laravel MVC).
- **Zasada aktywnego uczenia (Metoda Sokratejska i Analityczna):**
  - Projekt realizowany jest etapami (krok po kroku). Każdy moduł jest najpierw omawiany koncepcyjnie, teoretycznie i architektonicznie (struktury danych, kształty tensorów, analiza zysków i strat – trade-offs).
  - Nie dostarczaj gotowych, zamkniętych rozwiązań „pod klucz” ani bezpośrednich poprawek kodu bez wyraźnego polecenia użytkownika.
  - Prowadź użytkownika do samodzielnych decyzji implementacyjnych poprzez pytania naprowadzające i analizę alternatyw.
  - Nie narzucaj przedwcześnie konkretnych bibliotek zewnętrznych ani pakietów frameworka; najpierw wyjaśnij leżący u podstaw protokół lub problem architektoniczny.
- **Krytyczna Weryfikacja (Critical Verification):**
  - Wszystkie założenia projektowe, kod i koncepcje traktuj jako hipotezy do weryfikacji.
  - Wskazuj obiektywne błędy logiczne, luki bezpieczeństwa i antywzorce architektoniczne; wyjaśniaj mechanizm ich powstawania i proponuj poprawne technicznie alternatywy.
- **Ton i komunikacja:** Język zgodny z językiem zapytania (polski lub angielski), styl formalny, akademicki, neutralny i bezpośredni. Brak zwrotów powitalnych, pożegnalnych i sztucznych uprzejmości.
- **Wiedza i niejednoznaczności:** Gdy brakuje założeń do wykonania zadania, zadawaj precyzyjne pytania uściślające.

### 1.2 Protokół Interakcji (Sekwencja Obowiązkowa)

1. **Context & Assumption Check:** Sprawdzenie aktualnej fazy w `configs/CONTEXT.md` oraz nadrzędnych wytycznych w `configs/AGENTS.md`. Weryfikacja założeń technicznych i logiki.
2. **Conceptual & Architectural Breakdown:** Wyjaśnienie mechanizmów, teorii matematycznej, struktur danych i wymiarów tensorów przed implementacją.
3. **Directional Guidance & Alternatives:** Dostarczenie modularnych schematów, pseudokodu, analizy trade-off oraz pytań diagnostycznych.
4. **Code Review & Optimization:** Analiza kodu pod kątem złożoności obliczeniowej ($O(n)$), wąskich gardeł pamięciowych, poprawności wymiarów tensorów i przypadków brzegowych.
5. **Slow Implementation & Verification:** Etapowa realizacja gwarantująca pełne zrozumienie każdego wiersza kodu i każdego parametru sieci, weryfikowana testami jednostkowymi.

---

## 2. Konfiguracja Sprzętowa i Ograniczenia Pamięciowe

### 2.1 Maszyny Robocze

- **Maszyna Główna (Development & Prototyping):** MacBook M2 (16 GB Unified Memory, macOS).
  - Akceleracja sprzętowa: Apple Silicon Metal (`torch.device("mps")`).
  - Przeznaczenie: Projektowanie architektury, interfejs gry, testy jednostkowe, inspekcja sieci, inferencja.
- **Maszyna Treningowa (Heavy Training):** PC Windows 11 (NVIDIA RTX 3070 8 GB VRAM, 32 GB RAM).
  - Akceleracja sprzętowa: NVIDIA CUDA (`torch.device("cuda")`).
  - Przeznaczenie: Długotrwałe trenowanie sieci neuronowej na dużych zbiorach partii historycznych oraz sesje Self-Play.

### 2.2 Twarde Reguły Inżynieryjne dla Agenta (Antigravity)

- **Przenośność ścieżek:** Obowiązkowe używanie `pathlib.Path` we wszystkich plikach. Zakaz stosowania sztywnych ścieżek tekstowych.
- **Przenośność sprzętowa:** Automatyczna detekcja urządzenia obliczeniowego w kolejności: `cuda` -> `mps` -> `cpu`.
- **Ochrona pamięci RAM (16 GB Limit):** Bezwzględny zakaz ładowania całych plików tekstowych PGN do pamięci RAM podczas treningu. Wymagane użycie plików binarnych i strumieniowania z dysku przez `numpy.memmap`.
- **Narzędzie scalające (`pack_project.py`):** W głównym katalogu znajduje się skrypt łączący wszystkie pliki konfiguracyjne i kod źródłowy w jeden plik tekstowy, umożliwiający szybkie przesłanie pełnego stanu projektu do analizy.

### 2.3 Środowisko Uruchomieniowe i Konteneryzacja (Docker & Hybrid Runtime)

Projekt przyjmuje model hybrydowy dopasowany do specyfiki obliczeń ML i usług sieciowych:

1. **Natywne środowisko Python (`.venv`):**
   - Obowiązkowe dla fazy badawczej, rozwoju silnika, treningu sieci oraz inspekcji wag (Fazy 0–7).
   - Zapewnia bezpośredni i bezstratny dostęp do akceleratorów sprzętowych: Apple Silicon Metal (`mps`) na macOS oraz NVIDIA CUDA (`cuda`) na PC z Windows 11 / Linuksem, bez narzutu wirtualizacji kontenerowej.
2. **Konteneryzacja Docker & Docker Compose:**
   - Dedykowana dla platformy webowej i usług pomocniczych (Faza 8).
   - Konteneryzacji podlegają:
     - Relacyjna baza danych (PostgreSQL lub MySQL).
     - Serwer testowej poczty transakcyjnej (`Mailpit` – obsługa weryfikacji e-mail i resetowania haseł w środowisku lokalnym bez zewnętrznego SMTP).
     - Aplikacja webowa Laravel (PHP-FPM, Composer, Nginx – np. przez Laravel Sail lub dedykowany `docker-compose.yml`).
     - Bezstanowy mikroserwis Python API spięty wspólną siecią mostkową (`docker network`).

---

## 3. Fundamenty Architektoniczne (Elementy Niezmienne)

Elementy ustalone na starcie, których zmiana w późniejszym etapie unieważniłaby wytrenowane modele i przetworzone bazy danych:

### 3.1 Separacja Reguł Szachowych od Sieci Neuronowej

- Biblioteka `python-chess` odpowiada w 100% za stan szachownicy, walidację zasad gry, wykrywanie szacha, mata, pata, roszad oraz bicia w przelocie.
- Sieć neuronowa **nigdy** nie generuje ruchów z poza puli ruchów legalnych – pełni wyłącznie funkcję oceny pozycji oraz nadawania priorytetów ruchom dozwolonym przez `python-chess`.

### 3.2 Matematyczna Reprezentacja Szachownicy: Tensor `(20, 8, 8)`

Szachownica jest przekształcana w trójwymiarowy tensor liczb zmiennoprzecinkowych `float32` o wymiarach `20 kanałów x 8 wierszy x 8 kolumn`:

- **Kanały 0–5:** Białe figury (Pion, Skoczek, Goniec, Wieża, Hetman, Król) – macierze binarne `0.0` / `1.0`.
- **Kanały 6–11:** Czarne figury (Pion, Skoczek, Goniec, Wieża, Hetman, Król) – macierze binarne `0.0` / `1.0`.
- **Kanał 12:** Kolej ruchu (`1.0` na całej płaszczyźnie dla białych, `0.0` dla czarnych).
- **Kanały 13–16:** Prawa do roszady (4 osobne kanały: Biała krótka, Biała długa, Czarna krótka, Czarna długa).
- **Kanał 17:** Pole bicia w przelocie (_en passant_) – oznaczone wartością `1.0` na odpowiednim polu.
- **Kanał 18:** Licznik półruchów (zasada 50 ruchów) znormalizowany do przedziału `[0.0, 1.0]`.
- **Kanał 19:** Licznik pełnych ruchów partii znormalizowany do przedziału `[0.0, 1.0]`.

### 3.3 Reprezentacja Ruchów Wyjściowych: Przestrzeń Akcji `4672`

Każdy możliwy ruch na szachownicy jest mapowany na wektor o długości `4672` (`8 x 8 x 73` płaszczyzny w standardzie AlphaZero):

- `56` płaszczyzn dla ruchów po liniach prostych i przekątnych (8 kierunków x dystans 1–7 pól).
- `8` płaszczyzn dla skoków skoczka.
- `9` płaszczyzn dla podpromocji piona (na Skoczka, Gońca lub Wieżę w 3 kierunkach: prosto, bicie w lewo, bicie w prawo; promocja na Hetmana jest traktowana jako zwykły ruch liniowy).
- **Maskowanie:** Wszystkie ruchy nielegalne w danej pozycji otrzymują przed funkcją Softmax wartość `-inf`, co daje im zerowe prawdopodobieństwo wyboru.

### 3.4 Typ Sieci Neuronowej: Dwu-głowicowa Sieć Rezydualna (Dual-Head ResNet)

Zamiast prostej sieci oceniającej tylko pojedynczą planszę, stosujemy architekturę ze wspólnym rdzeniem splotowo-rezydualnym (Convolutional & Residual Blocks) rozdzielającym się na dwie głowice:

1. **Głowica Polityki (Policy Head):** Wskazuje, które ruchy na planszy są najbardziej obiecujące (zwraca rozkład prawdopodobieństwa dla 4672 ruchów).
2. **Głowica Wartości (Value Head):** Ocenia aktualną sytuację na szachownicy, zwracając liczbę z przedziału `[-1.0, 1.0]` (`-1.0` = pewna przegrana, `0.0` = remis, `+1.0` = pewna wygrana).

### 3.5 Integracja Zewnętrznych Silników Szachowych (Protokół UCI)

System przewiduje dwukierunkową komunikację z zewnętrznymi silnikami szachowymi (np. Stockfish) poprzez uniwersalny protokół **UCI (Universal Chess Interface)**:

- **Regulacja siły gry:** Możliwość ograniczania siły silnika zewnętrznego poprzez konfigurowalny limit Elo (`UCI_LimitStrength`, `UCI_Elo`) lub poziom trudności (`Skill Level`).
- **Funkcja w systemie:**
  - Niezależny punkt odniesienia (ground-truth baseline) do obiektywnej ewaluacji własnych sieci neuronowych.
  - Opcjonalne źródło destylacji wiedzy (**Knowledge Distillation**) w procesie headless training.
  - Elastyczny przeciwnik w trybach `Human vs AI` oraz `AI vs AI`.

---

## 4. Specyfikacja Funkcjonalna Gry i Systemu AI

### 4.1 Moduł Rozgrywki i Inferencji (Gameplay & Evaluation)

Interfejs terminalowy (CLI) z czytelną separacją warstwy prezentacji, umożliwiającą późniejsze podpięcie interfejsu graficznego (GUI) oraz integrację webową:

- **Wygląd szachownicy:** Duża, przejrzysta szachownica z wyraźnymi oznaczeniami kolumn (`a-h`), wierszy (`1-8`), symbolami bierek (Unicode/ASCII) i podświetleniem ostatniego ruchu (ANSI).
- **Menu gry i komendy:**
  - `exit` – bezpieczne opuszczenie gry.
  - `back move` – cofnięcie ruchu (wraz z przywróceniem stanu sprzed ruchu).
  - `history` – wyświetlenie pełnej listy ruchów wykonanych w aktualnej grze.
  - `side select` – wybór koloru bierek przed rozpoczęciem partii.
  - `switch side` – dynamiczna zmiana strony (przejęcie bierek przeciwnika) w trakcie trwania gry.
  - `load board (FEN)` – wczytanie dowolnego ustawienia szachownicy z notacji FEN (z walidacją syntaktyczną i semantyczną).
- **Tryby rozgrywki:**
  - `Human vs Human` – klasyczna partia dwuosobowa.
  - `Human vs AI` – gra człowieka przeciwko wybranej sieci neuronowej lub zewnętrznemu silnikowi UCI. Zawiera **tryb diagnostyczno-testowy** z podglądem na żywo telemetryki modelu (wektor prawdopodobieństw ruchów, ocena pozycji Value, statystyki przeszukiwania).
  - `AI vs AI` – pojedyncze pojedynki oraz **wielopartiowe turnieje porównawcze (multi-game benchmarks)** do obiektywnego rankingu modeli i badania postępów uczenia.

### 4.2 Potok Treningowy (Headless Training Pipeline)

Proces treningu sieci neuronowej jest w pełni odseparowany (headless) od interfejsu interaktywnego i wspiera różnorodne paradygmaty uczenia:

- **Poziom 0 (AI bez wiedzy):** Losowe ruchy legalne (baza odniesienia i test integralności silnika).
- **Poziom 1 (Supervised Learning):** Uczenie nadzorowane na historycznych bazach partii mistrzowskich (PGN) konwertowanych do strumieniowanych plików binarnych (`numpy.memmap`).
- **Poziom 2 (Reinforcement Learning via Self-Play):** Uczenie ze wzmocnieniem w pętli AlphaZero – sieć gra sama ze sobą, generując dane treningowe i ewoluując bez wiedzy eksperckiej.
- **Poziom 3 (Hybrid Engine):** Połączenie wyuczonej sieci z algorytmem głębokiego przeszukiwania drzewa gry (MCTS / Alpha-Beta).
- **Uczenie transferowe i gałęziowe (Transfer & Branching Learning):** Możliwość kontynuacji treningu w nowych gałęziach badawczych bez nadpisywania wag modeli bazowych.
- **Destylacja wiedzy (Knowledge Distillation – opcjonalna):** Trenowanie głowic sieci w oparciu o ewaluacje i rozkłady generowane przez silniki UCI o wysokim Elo.

### 4.3 Zarządzanie Modelami, Obserwowalność i Wersjonowanie

- **Deklaratywna konfiguracja:** Pełna parametryzacja statyczna (liczba bloków rezydualnych, filtry, wymiary) oraz dynamiczna (learning rate, batch size, parametry MCTS/przeszukiwania) z poziomu plików konfiguracyjnych.
- **Kontrola cyklu życia (Lifecycle Control):** Mechanizmy bezpiecznego pauzowania, wznawiania i tworzenia punktów kontrolnych (checkpoints) podczas długich sesji treningowych.
- **Śledzenie rodowodu eksperymentów (Experiment Lineage Tracking):** Rejestr `models/registry.md` dokumentujący historię powstania modelu, gałęzie rodzicielskie, liczbę epok, parametry oraz uzyskany ranking turniejowy.
- **Pełny wgląd w sieć neuronową (`inspector.py`):**
  - Architektura: liczba warstw, parametry wagowe, rozkład wag i gradientów.
  - Telemetria inferencji: aktywacje warstw konwolucyjnych (mapy cech), wskaźnik entropii/pewności wyboru ruchu.
  - Wskaźnik predykcji wygranej: live evaluation bar $\text{Szansa } (\%) = \frac{\text{Value} + 1}{2} \times 100\%$.

---

## 5. Integracja z Platformą Webową (Laravel MVC & Stateless API)

Drugi etap projektu obejmuje integrację silnika szachowego i modeli AI z pełnoprawną platformą webową opartą o wzorzec MVC w ekosystemie Laravel:

### 5.1 Rola Serwisu Python (AI & Engine Microservice)
- Działa w trybie bezstanowym (stateless), wystawiając wewnętrzne API HTTP/JSON.
- Odpowiada za: walidację stanu gry, generowanie legalnych posunięć, obliczanie inferencji sieci neuronowych (predykcja ruchu, ocena pozycji Value) oraz delegowanie zapytań do silników UCI.

### 5.2 Rola Platformy Webowej (Laravel MVC)
- **Warstwa użytkownika i persystencji:** Odpowiada za autoryzację, sesje, zapis partii w relacyjnej bazie danych oraz interfejs graficzny.
- **Bezpieczeństwo i uwierzytelnianie:**
  - System rejestracji i logowania użytkowników.
  - Uwierzytelnianie dwuskładnikowe (Two-Factor Authentication – 2FA).
- **Zarządzanie kontami i rozgrywkami:**
  - Profile graczy, statystyki, historia rozegranych partii i zapis posunięć (PGN/SAN).
  - System rankingowy graczy i modeli AI.
- **Transakcyjny mailing:** Obsługa powiadomień e-mail (weryfikacja adresu, procedury resetowania hasła, powiadomienia o zdarzeniach).

### 5.3 Równoległa Ścieżka Realizacji i Grywalne MVP (Parallel Track & Early Playability)

Rozwój platformy webowej Laravel nie musi czekać na pełne zakończenie treningu sieci neuronowej (Fazy 4–6):

- **Wykorzystanie okien czasowych (Przestoje i Długi Trening):**
  Długotrwałe sesje treningowe (wielogodzinny proces uczenia nadzorowanego PGN oraz wielodniowe pętle Self-Play RL na RTX 3070) stanowią idealny moment na równoległą realizację platformy Laravel oraz orkiestracji w Dockerze.
- **Wczesne MVP z silnikiem zewnętrznym UCI:**
  Gdy tylko silnik szachowy w Pythonie osiągnie status MVP (obsługa planszy, legalność reguł, bezstanowe API oraz adapter do silnika zewnętrznego UCI, np. Stockfish z regulowanym rankingiem Elo), platforma webowa może zostać wdrożona i uruchomiona natychmiast.
- **Korzyść użytkowa:**
  Gracz otrzymuje w pełni funkcjonalną grę szachową online (konta, 2FA, zapis gier, gra przeciwko silnikowi szachowemu) na wczesnym etapie, a wytrenowane autorskie sieci neuronowe są podpinane do API jako kolejne poziomy trudności AI natychmiast po ukończeniu ich uczenia.
