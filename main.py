"""
main.py

Główny punkt uruchomieniowy aplikacji terminalowej Szachy AI v2.
Łączy Menu Główne (Kreator Wyboru) z kontrolerem rozgrywki (GameController),
podglądem stanu/metryk środowiska oraz szkieletem gałęzi NAUKA.
"""

from __future__ import annotations

import sys
from typing import Callable

from src.config import ensure_directories
from src.ui.board_view import BoardView
from src.ui.game_controller import GameController
from src.ui.menu import MainMenuAction, Menu


def run_cli(input_func: Callable[[str], str] = input) -> None:
    """Uruchamia główną pętlę aplikacji CLI (Menu Główne -> akcje)."""
    ensure_directories()

    menu = Menu(use_colors=True)
    board_view = BoardView(use_colors=True, use_unicode_pieces=True)
    controller = GameController(board_view=board_view, menu=menu)

    while True:
        action = menu.prompt_main_menu(input_func=input_func)

        if action == MainMenuAction.EXIT:
            print("Zamykanie Szachy AI v2. Do zobaczenia!")
            break

        if action == MainMenuAction.PLAY:
            game_config = menu.prompt_game_wizard(input_func=input_func)
            if game_config is not None:
                controller.run_match(
                    game_config,
                    input_func=input_func,
                    ask_save_at_end=True,
                )
            continue

        if action == MainMenuAction.TRAIN:
            training_config = menu.prompt_training_wizard(input_func=input_func)
            if training_config is not None:
                print("\n--- KONFIGURACJA TRENINGU (SZKIELET) ---")
                print(f"  Metoda nauki   : {training_config.method.value}")
                print(f"  Punkt startowy : {training_config.start_mode.value}")
                print(f"  Linia docelowa : {training_config.target_line_name}")
                print(
                    "  [Info] Potok treningowy (trainer.py / train.py) zostanie uruchomiony w Fazie 5."
                )
            continue

        if action == MainMenuAction.STATUS:
            print(menu.format_status_and_metrics())
            input_func("Naciśnij [Enter], aby wrócić do Menu Głównego...")
            continue


def main() -> int:
    """Punkt wejścia z bezpieczną obsługą przerwania Ctrl+C."""
    try:
        run_cli()
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nPrzerwano działanie programu. Do zobaczenia!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
