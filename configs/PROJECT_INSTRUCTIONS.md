# PROJECT INSTRUCTIONS & SPECIFICATION: Szachy AI v2

## 1. Rola Systemów i Protokół Współpracy

### 1.1 Podział Ról: Architekt (Gemini) vs Wykonawca (Antigravity) i Użytkownik

- **Cel nadrzędny:** Nauka działania sztucznej inteligencji, procesu uczenia sieci neuronowej, jej wewnętrznej struktury, wag, parametrów oraz inżynierii oprogramowania na przykładzie budowy silnika szachowego w języku Python i bibliotece PyTorch.
- **Zasada aktywnego uczenia (Active Learning):** Projekt jest realizowany powoli, małymi etapami (krok po kroku). Każdy moduł jest najpierw omawiany teoretycznie i architektonicznie (struktury danych, kształty tensorów, pseudokod).
- **Weryfikacja założeń:** Wszystkie założenia projektowe podlegają krytycznej weryfikacji względem standardów inżynierii ML i silników szachowych. Błędne koncepcje są korygowane przed napisaniem kodu.
- **Ton i komunikacja:** Język polski lub angielski (poziom B2), styl formalny, akademicki, obiektywny, pozbawiony zwrotów grzecznościowych i wypełniaczy.
- **Wiedza:** Gdy brakuje informacji co do wykonania zadania dopytuj uzytkownika.

### 1.2 Protokół Interakcji (Sekwencja Obowiązkowa)

1. **Context & Assumption Check:** Sprawdzenie aktualnej fazy w `configs/CONTEXT.md`. Weryfikacja założeń technicznych i logiki.
2. **Conceptual & Architectural Breakdown:** Wyjaśnienie mechanizmów, teorii matematycznej, struktur danych i wymiarów tensorów przed implementacją.
3. **Directional Guidance:** Dostarczenie modularnych schematów, pseudokodu oraz pytań naprowadzających bez generowania gotowych, zamkniętych skryptów.
4. **Code Review & Optimization:** Analiza napisanego kodu pod kątem złożoności obliczeniowej, wąskich gardeł pamięciowych, poprawności wymiarów tensorów i przypadków brzegowych.
5. **Slow Implementation:** Realizacja powolna, etapowa, gwarantująca pełne zrozumienie każdego wiersza kodu i każdego parametru sieci.

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

---

## 4. Specyfikacja Funkcjonalna Gry i Systemu AI

### 4.1 Silnik Gry w Szachy i Interfejs (UI)

- **Wygląd szachownicy:** Duża, przejrzysta szachownica z wyraźnymi oznaczeniami kolumn (`a-h`), wierszy (`1-8`), symbolami bierek i podświetleniem ostatniego ruchu.
- **Menu gry:**
  - `exit` – bezpieczne opuszczenie gry.
  - `back move` – cofnięcie ruchu (wraz z przywróceniem stanu sprzed ruchu).
  - `history` – wyświetlenie pełnej listy ruchów wykonanych w aktualnej grze.
  - `side select` – wybór koloru bierek przed rozpoczęciem partii.
  - `switch side` – dynamiczna zmiana strony (przejęcie bierek przeciwnika) w trakcie trwania gry.
  - `load board (FEN)` – wczytanie dowolnego ustawienia szachownicy utworzonego np. w edytorze na chess.com za pomocą kodu FEN i rozpoczęcie gry od tej pozycji.
- **Tryby rozgrywki (Kto gra z kim):**
  - `Human vs Human` (Człowiek przeciwko Człowiekowi).
  - `Human vs AI` (Człowiek przeciwko Sztucznej Inteligencji).
  - `AI vs AI` (Mecz dwóch wybranych wersji/poziomów AI przeciwko sobie).

### 4.2 Poziomy Wytrenowania AI (Tiers)

Użytkownik przed grą lub w jej trakcie może wybrać poziom AI:

- **Poziom 0 (AI bez wiedzy):** Wybiera losowe posunięcia spośród ruchów dozwolonych przez przepisy.
- **Poziom 1 (AI z gier historycznych – Supervised Learning):** Sieć wytrenowana na bazie zapisów partii silnych graczy pobranych z internetu.
- **Poziom 2 (AI z gier własnych – Self-Play RL):** Sieć wytrenowana poprzez rozgrywanie tysięcy partii AI vs AI i uczenie się na własnych błędach oraz zwycięstwach.
- **Poziom 3 (Połączenie wszystkiego – Hybrid Engine):** Połączenie sieci wyuczonej na grach historycznych i Self-Play z algorytmem głębokiego przeszukiwania drzewa gry w przód.

### 4.3 Parametry Silnika AI, Inspekcja i Wersjonowanie

- **Wskaźnik predykcji wygranej:** Wyświetlany na żywo przy szachownicy pasek/procent szans na zwycięstwo każdej ze stron, obliczany bezpośrednio z wyjścia Głowicy Wartości sieci neuronowej: `Szansa (%) = ((Value + 1) / 2) * 100`.
- **Moc sieci i głębokość przewidywania:** Możliwość konfiguracji, na ile ruchów w przód (głębokość drzewa przeszukiwania) AI analizuje warianty oraz jak działa przy aktualnie ustawionych parametrach.
- **Pełny wgląd w sieć neuronową (`inspector.py`):** Narzędzie edukacyjne pozwalające w każdej chwili podejrzeć:
  - Liczbę warstw i dokładną liczbę parametrów (wag) w każdej warstwie.
  - Rozkład wartości wag oraz wartości gradientów podczas nauki (zrozumienie, jak sieć modyfikuje swoją wiedzę).
  - Aktywacje poszczególnych warstw (zobaczenie, na które pola szachownicy „zwraca uwagę” dana warstwa splotowa).
  - Poziom niepewności sieci (entropia decyzji) przy wyborze ruchu.
- **Wersjonowanie sieci neuronowej (`version_manager.py` i `models/registry.md`):** Każda zapisana wersja sieci posiada swój identyfikator, opis sposobu trenowania, liczbę rozegranych epok/gier, parametry architektury oraz zmierzoną siłę gry.
