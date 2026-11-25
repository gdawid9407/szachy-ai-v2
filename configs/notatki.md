[1] ROZGRYWKA
├── Gry z udziałem człowieka:
│ ├── [1] Osoba vs Osoba
│ ├── [2] Osoba vs Własna Sieć (z wyborem koloru i trybem testowym)
│ └── [3] Osoba vs Silnik Zewnętrzny (z wyborem koloru i Elo)
├── Gry automatyczne (Benchmark i Testy):
│ ├── [4] Własna Sieć vs Własna Sieć (porównanie wersji / parametrów)
│ ├── [5] Własna Sieć vs Silnik Zewnętrzny (kalibracja Elo)
│ ├── [6] Własna Sieć vs Bot Bazowy (test kontrolny)
│ └── [7] Konfiguracja własna (dowolna para, np. Silnik vs Silnik)

[2] NAUKA (TRENING SIECI)
├── [1] Uczenie nadzorowane (Bazy internetowe PGN)
│ └── Trening na gotowych partiach (np. Lichess) z filtrem min. Elo i podziałem Train/Val.
├── [2] Self-Play RL (Własna Sieć vs Własna Sieć — Nauka)
│ └── Generowanie nowych partii z szumem Dirichleta i temperaturą + aktualizacja wag.
├── [3] Uczenie od Nauczyciela / Destylacja (Własna Sieć + Silnik Zewnętrzny)
│ └── Trening sieci na najlepszych ruchach i ocenach dostarczanych przez silnik UCI.
├── [4] Trening Fazowy (Curriculum Learning)
│ └── Nauka wyłącznie na wyizolowanych pozycjach (np. końcówki, zagadki taktyczne EPD).
└── [5] Wznowienie treningu
└── Kontynuacja przerwanego procesu z zapisanego checkpointu i stanu optymalizatora.

menu

## Ustalona struktura menu

### Poziom 0: Menu Główne

MENU GŁÓWNE
├── [1] GRA (Bez modyfikacji wag sieci)
├── [2] NAUKA (Trening sieci i tworzenie nowych wersji)
├── [3] STAN I METRYKI SIECI (Przegląd zapisanych linii i wyników)
└── [0] WYJŚCIE

### Gałąź `[1] GRA` (Rozgrywka i porównania)

[1] GRA
├── [1] AI vs AI
│ ├── Tryb: [Pojedyncza partia] lub [Seria porównawcza N partii]
│ ├── Konfiguracja BIAŁYCH -> (wywołuje Wspólny Moduł Wyboru AI)
│ ├── Konfiguracja CZARNYCH -> (wywołuje Wspólny Moduł Wyboru AI)
│ └── Ustawienia meczu -> (jeśli seria: liczba gier, zamiana kolorów, otwarcia)
│
├── [2] Człowiek vs AI
│ ├── Wybór strony człowieka -> [Białe] / [Czarne] / [Losuj]
│ ├── Konfiguracja AI -> (wywołuje Wspólny Moduł Wyboru AI)
│ └── Tryb testowy (opcja) -> [Wyłączony] / [Włączony: podgląd Policy/Value, cofanie ruchów]
│
└── [3] Człowiek vs Człowiek
├── Przypisanie stron -> [Gracz 1 = Białe, Gracz 2 = Czarne] / [Losuj]
└── Asysta w tle -> [Brak] / [Cicha ocena pozycji przez wybraną wersję własnej sieci]

**Wspólny Moduł Wyboru AI (używany w opcjach 1 i 2):**

WYBÓR AGENTA AI:
├── [A] Silnik zewnętrzny (UCI, np. Stockfish)
│ ├── Wybór silnika z listy
│ └── Limit siły: Elo (w zakresie silnika) lub Głębokość (depth)
│
└── [B] Własna Sieć AI
├── Krok 1: Wybór linii treningowej (np. "tylko_internet", "tylko_selfplay", "internet+selfplay")
├── Krok 2: Wybór konkretnego stanu / checkpointu (np. epoka_10, epoka_50, najnowszy)
└── Krok 3: Parametry gry (Raw Policy vs MCTS, liczba symulacji, temperatura)

### Gałąź `[2] NAUKA` (Dopełnienie koncepcji treningu)

[2] NAUKA (TRENING SIECI)
├── [1] Trening na grach z internetu (Supervised Learning / PGN)
│ ├── Baza źródłowa: ścieżka do pliku PGN + filtr minimalnego Elo
│ ├── Model startowy: [Nowa sieć od zera] lub [Kontynuuj istniejący stan]
│ └── Zapis jako linia: np. "internet_v1"
│
├── [2] Trening AI vs AI (Self-Play Reinforcement Learning)
│ ├── Model startowy: [Nowa sieć od zera] lub [Wznowienie istniejącego stanu]
│ ├── Parametry eksploracji: szum Dirichleta, temperatura, liczba symulacji MCTS
│ └── Zapis jako linia: np. "selfplay_v1"
│
├── [3] Trening hybrydowy / Dotrenowanie (Transfer Learning: np. Internet -> Self-Play)
│ ├── Wybór bazowego checkpointu (np. "internet_v1 / epoka_20")
│ ├── Wybór kolejnej fazy nauki: [Self-Play RL] / [Inna baza PGN] / [Destylacja z silnika UCI]
│ └── Zapis jako nowa linia potomna (zachowuje model bazowy bez nadpisywania)
│
└── [4] Trening od silnika zewnętrznego (Destylacja wiedzy — opcjonalnie)
└── Silnik UCI ocenia pozycje i wskazuje najlepsze ruchy jako cel dla własnej sieci

Aby w menu `AI vs AI` można było łatwo wybierać między liniami i stanami, katalog modeli powinien mieć stałą strukturę hierarchiczną:

models/
├── internet_only_v1/
│ ├── metadata.json # źródło danych, architektura, historia straty (Loss)
│ ├── epoch_005.pt
│ └── epoch_020.pt
├── selfplay_only_v1/
│ ├── metadata.json
│ ├── iter_010.pt
│ └── iter_050.pt
└── hybrid_pgn_then_selfplay_v1/
├── metadata.json # zawiera pole: "parent": "internet_only_v1/epoch_020.pt"
├── iter_010.pt
└── iter_030.pt

wersja ostateczna

Poziom 0: Menu Główne
[1] GRA – rozgrywka i testy (bez zmiany wag sieci)
[2] NAUKA – trening i tworzenie nowych wersji sieci
[3] STAN I METRYKI – podgląd modeli, linii treningowych i diagnostyka sprzętu
[0] WYJŚCIE
Gałąź [1] GRA (Kreator rozgrywki)
Krok 1 – Wybór trybu:
[1] Człowiek vs AI
[2] Człowiek vs Człowiek
[3] AI vs AI (Pojedyncza partia lub Seria N partii)
[4] Konfigurator własny (dowolna para + własny FEN)
Krok 2 – Wybór graczy:
Tam, gdzie gra komputer, uruchamia się Wspólny Kreator Agenta AI:
[1] Bot bazowy (Poziom 0 – losowe ruchy)
[2] Silnik zewnętrzny (UCI, np. Stockfish) → wybór silnika → tryb limitu (Elo / Głębokość / Czas na ruch w ms)
[3] Własna Sieć AI → linia z models/ → checkpoint (najnowszy lub konkretna epoka) → tryb (Raw Policy / MCTS + liczba symulacji + temperatura)
Krok 3 – Ustawienia dodatkowe (zależne od trybu):
Dla Człowiek vs AI: kolor (Białe / Czarne / Losowo) + Tryb testowy [T/N].
Dla Człowiek vs Człowiek: Asysta AI w tle [T/N] (jeśli T → uruchamia Wspólny Kreator Agenta AI).
Dla AI vs AI: liczba partii (domyślnie 1), zamiana kolorów (T/N), tryb wyświetlania (Na żywo z pauzą / Krok po kroku [Enter] / Szybki benchmark bez rysowania planszy).
Krok 4 – Pozycja początkowa (opcjonalnie):
[Enter] Pozycja startowa | [1] Wklej własny FEN | [2] Gotowy zestaw testowy (np. końcówka, debiut).
