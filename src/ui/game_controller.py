"""
src/ui/game_controller.py

Kontroler rozgrywki łączący silnik (BoardManager), widok (BoardView),
Kreator Wyboru (Menu) oraz ewaluatory AI (BaseEvaluator / RandomEvaluator).

Obsługuje:
- Tryby: Człowiek vs Człowiek (z opcjonalną asystą w tle), Człowiek vs AI
  (z trybem testowym), AI vs AI (pojedyncza partia lub seria N partii
  z zamianą kolorów i wyborem trybu wyświetlania) oraz Konfigurator własny.
- Wszystkie komendy w trakcie partii: ruch (UCI/SAN), back/undo, history,
  switch, fen (odczyt i wczytanie), eval/hint, save (eksport PGN), restart,
  help, exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from typing import Callable, List, Optional
import chess
import chess.pgn

from src.config import DATA_DIR
from src.engine.board_manager import BoardManager
from src.engine.fen_parser import FenParser
from src.search.evaluator import BaseEvaluator, RandomEvaluator
from src.ui.board_view import BoardView
from src.ui.menu import (
    AgentType,
    AiDisplayMode,
    GameConfig,
    InGameAction,
    Menu,
    PlayerConfig,
)


@dataclass(frozen=True)
class GameResult:
    """Podsumowanie pojedynczej rozegranej partii."""
    result: str
    reason: str
    moves_count: int
    white_name: str
    black_name: str
    final_fen: str
    pgn_path: Optional[Path] = None


@dataclass
class SeriesSummary:
    """Zbiorcze podsumowanie meczu (1 partia lub seria N partii)."""
    games_played: int
    player_a_name: str
    player_b_name: str
    player_a_wins: int = 0
    player_b_wins: int = 0
    draws: int = 0
    results: List[GameResult] = field(default_factory=list)

    def format_report(self) -> str:
        """Zwraca czytelną tabelę wyników serii meczowej."""
        lines = [
            "============================================================",
            "                 PODSUMOWANIE SERII MECZOWEJ                ",
            "============================================================",
            f"  Rozegrane partie : {self.games_played}",
            f"  {self.player_a_name:<30} : {self.player_a_wins} wygranych",
            f"  {self.player_b_name:<30} : {self.player_b_wins} wygranych",
            f"  Remisy                         : {self.draws}",
            "============================================================",
        ]
        return "\n".join(lines)


class GameController:
    """
    Zarządza cyklem życia rozgrywki CLI dla pojedynczych partii i serii AI vs AI.
    """

    def __init__(
        self,
        board_view: Optional[BoardView] = None,
        menu: Optional[Menu] = None,
        pgn_output_dir: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.board_view: BoardView = board_view if board_view is not None else BoardView()
        self.menu: Menu = menu if menu is not None else Menu()
        self.pgn_output_dir: Path = (
            pgn_output_dir if pgn_output_dir is not None else DATA_DIR / "pgn_exports"
        )
        self._seed: Optional[int] = seed

    # --------------------------------------------------------------------------
    # Fabryka agentów (PlayerConfig -> BaseEvaluator | None)
    # --------------------------------------------------------------------------

    def build_evaluator(self, player_config: PlayerConfig) -> Optional[BaseEvaluator]:
        """
        Tworzy instancję ewaluatora dla zadanego PlayerConfig.
        Dla człowieka zwraca None. Dla jeszcze niezaimplementowanych modułów
        (UCI przed Zadaniem 2.5, Własna Sieć przed Fazą 4) informuje użytkownika
        i przechodzi bezpiecznie na Poziom 0 (RandomEvaluator).
        """
        if player_config.agent_type == AgentType.HUMAN:
            return None

        if player_config.agent_type == AgentType.RANDOM_BOT:
            return RandomEvaluator(seed=self._seed)

        if player_config.agent_type == AgentType.UCI_ENGINE:
            print(
                f"[Info] Adapter silnika zewnętrznego UCI ({player_config.uci_engine_name}) "
                "zostanie podpięty w Zadaniu 2.5. Tymczasowo używam Bota bazowego (Poziom 0)."
            )
            return RandomEvaluator(seed=self._seed)

        if player_config.agent_type == AgentType.NEURAL_NET:
            print(
                "[Info] Ewaluator Własnej Sieci AI zostanie podpięty w Fazie 4. "
                "Tymczasowo używam Bota bazowego (Poziom 0)."
            )
            return RandomEvaluator(seed=self._seed)

        return RandomEvaluator(seed=self._seed)

    # --------------------------------------------------------------------------
    # Eksport partii do formatu PGN
    # --------------------------------------------------------------------------

    def export_pgn(
        self,
        board_manager: BoardManager,
        white_name: str,
        black_name: str,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Zapisuje bieżący stan i historię partii do standardowego pliku .pgn."""
        game = chess.pgn.Game.from_board(board_manager.board)
        game.headers["Event"] = "Szachy AI v2 CLI"
        game.headers["Site"] = "Local"
        game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = white_name
        game.headers["Black"] = black_name
        game.headers["Result"] = board_manager.get_game_result()

        if output_path is None:
            self.pgn_output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = self.pgn_output_dir / f"game_{timestamp}.pgn"
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        output_path.write_text(str(game) + "\n", encoding="utf-8")
        return output_path

    # --------------------------------------------------------------------------
    # Uruchamianie meczu (1 partia lub seria N partii)
    # --------------------------------------------------------------------------

    def run_match(
        self,
        config: GameConfig,
        input_func: Callable[[str], str] = input,
        sleep_func: Callable[[float], None] = time.sleep,
        max_plies: Optional[int] = None,
    ) -> SeriesSummary:
        """
        Przeprowadza pojedynczą partię lub serię N partii zgodnie z GameConfig.
        Obsługuje opcjonalną zamianę kolorów co partię.
        """
        summary = SeriesSummary(
            games_played=0,
            player_a_name=config.white.describe(),
            player_b_name=config.black.describe(),
        )

        for game_idx in range(config.num_games):
            swapped = config.swap_colors and (game_idx % 2 == 1)
            white_cfg = config.black if swapped else config.white
            black_cfg = config.white if swapped else config.black

            if config.num_games > 1:
                print(
                    f"\n--- Partia {game_idx + 1}/{config.num_games}: "
                    f"Białe = {white_cfg.describe()} vs Czarne = {black_cfg.describe()} ---"
                )

            game_res = self.play_single_game(
                white_cfg=white_cfg,
                black_cfg=black_cfg,
                initial_fen=config.initial_fen,
                display_mode=config.display_mode,
                move_delay_sec=config.move_delay_sec,
                test_mode=config.test_mode,
                assist_cfg=config.assist_agent,
                save_pgn=config.save_pgn,
                input_func=input_func,
                sleep_func=sleep_func,
                max_plies=max_plies,
            )

            summary.games_played += 1
            summary.results.append(game_res)

            if game_res.result == "1-0":
                if swapped:
                    summary.player_b_wins += 1
                else:
                    summary.player_a_wins += 1
            elif game_res.result == "0-1":
                if swapped:
                    summary.player_a_wins += 1
                else:
                    summary.player_b_wins += 1
            elif game_res.result == "1/2-1/2":
                summary.draws += 1

            # Jeśli użytkownik przerwał partię (exit), przerwij również serię
            if game_res.result == "*" and game_res.reason == "Przerwano przez użytkownika.":
                break

        if config.num_games > 1:
            print(summary.format_report())

        return summary

    # --------------------------------------------------------------------------
    # Pętla pojedynczej partii
    # --------------------------------------------------------------------------

    def play_single_game(
        self,
        white_cfg: PlayerConfig,
        black_cfg: PlayerConfig,
        initial_fen: Optional[str] = None,
        display_mode: AiDisplayMode = AiDisplayMode.LIVE_DELAY,
        move_delay_sec: float = 0.5,
        test_mode: bool = False,
        assist_cfg: Optional[PlayerConfig] = None,
        save_pgn: bool = False,
        input_func: Callable[[str], str] = input,
        sleep_func: Callable[[float], None] = time.sleep,
        max_plies: Optional[int] = None,
    ) -> GameResult:
        """
        Główna pętla rozgrywki pojedynczej partii szachów.
        """
        bm = BoardManager(fen=initial_fen)

        white_eval = self.build_evaluator(white_cfg)
        black_eval = self.build_evaluator(black_cfg)
        assist_eval = self.build_evaluator(assist_cfg) if assist_cfg is not None else None

        is_ai_vs_ai = (white_eval is not None) and (black_eval is not None)
        fast_mode = is_ai_vs_ai and (display_mode == AiDisplayMode.FAST_BENCHMARK)

        # Domyślna perspektywa: jeśli tylko Czarne są człowiekiem -> odwróć planszę
        perspective: chess.Color = (
            chess.BLACK
            if (black_eval is None and white_eval is not None)
            else chess.WHITE
        )
        last_move: Optional[chess.Move] = None
        aborted = False

        while not bm.is_game_over():
            if max_plies is not None and len(bm) >= max_plies:
                break

            active_eval = white_eval if bm.turn == chess.WHITE else black_eval
            eval_source = assist_eval or active_eval or white_eval or black_eval
            current_eval: Optional[float] = (
                eval_source.evaluate(bm.board) if eval_source is not None else None
            )

            if not fast_mode:
                self.board_view.display(
                    board_manager=bm,
                    perspective=perspective,
                    eval_value=current_eval,
                    last_move=last_move,
                    clear_screen=False,
                )

            # ---- TURA AI ----
            if active_eval is not None:
                if is_ai_vs_ai and display_mode == AiDisplayMode.STEP_BY_STEP:
                    step_cmd = input_func(
                        "[Enter] Następny ruch AI | [q] Przerwij partię: "
                    ).strip().lower()
                    if step_cmd in ("q", "exit", "quit"):
                        aborted = True
                        break

                ai_move = active_eval.select_move(bm)
                if ai_move is None:
                    break

                bm.push_move(ai_move)
                last_move = ai_move
                if not fast_mode:
                    last_san = bm.get_san_history()[-1]
                    side_label = "Białe" if bm.turn == chess.BLACK else "Czarne"
                    print(f"Ruch AI ({side_label}): {last_san} ({ai_move.uci()})")
                    if is_ai_vs_ai and display_mode == AiDisplayMode.LIVE_DELAY and move_delay_sec > 0:
                        sleep_func(move_delay_sec)
                continue

            # ---- TURA CZŁOWIEKA ----
            if test_mode and eval_source is not None:
                suggested = eval_source.select_move(bm)
                sug_str = suggested.uci() if suggested is not None else "brak"
                val_str = f"{current_eval:+.2f}" if current_eval is not None else "n/d"
                print(f"[Tryb testowy] Ocena: {val_str} | Proponowany ruch: {sug_str}")

            prompt_label = f"Ruch ({bm.turn_name}) [np. e2e4, 'help']: "
            raw_input = input_func(prompt_label)
            cmd = self.menu.parse_in_game_input(raw_input)

            if cmd.action == InGameAction.EXIT:
                aborted = True
                break

            if cmd.action == InGameAction.HELP:
                print(self.menu.format_help())
                continue

            if cmd.action == InGameAction.HISTORY:
                print(self.menu.format_history(bm))
                continue

            if cmd.action == InGameAction.SHOW_FEN:
                print(f"Aktualny FEN: {bm.get_fen()}")
                continue

            if cmd.action == InGameAction.LOAD_FEN:
                fen_payload = cmd.payload or ""
                res = FenParser.validate(fen_payload)
                if res.is_valid and res.cleaned_fen is not None:
                    bm.set_fen(res.cleaned_fen)
                    last_move = None
                    print(f"Załadowano pozycję FEN: {res.cleaned_fen}")
                else:
                    print(f"Błąd FEN: {res.error_message}")
                continue

            if cmd.action == InGameAction.BACK:
                # W grze z AI cofamy 2 półruchy (ruch AI + ruch człowieka), aby znów był ruch człowieka
                is_human_vs_ai = (white_eval is None) != (black_eval is None)
                steps_to_pop = 2 if (is_human_vs_ai and len(bm) >= 2) else 1
                popped_any = False
                for _ in range(steps_to_pop):
                    if bm.pop_move() is not None:
                        popped_any = True
                if popped_any:
                    last_move = bm.board.peek() if len(bm) > 0 else None
                    print("Cofnięto ruch.")
                else:
                    print("Brak ruchów do cofnięcia.")
                continue

            if cmd.action == InGameAction.SWITCH_SIDE:
                white_cfg, black_cfg = black_cfg, white_cfg
                white_eval, black_eval = black_eval, white_eval
                perspective = chess.BLACK if perspective == chess.WHITE else chess.WHITE
                print(
                    f"Zamieniono strony! Białe: {white_cfg.describe()} | Czarne: {black_cfg.describe()}"
                )
                continue

            if cmd.action == InGameAction.EVAL:
                evaluator_for_hint = eval_source if eval_source is not None else RandomEvaluator(seed=self._seed)
                val = evaluator_for_hint.evaluate(bm.board)
                hint_move = evaluator_for_hint.select_move(bm)
                hint_str = hint_move.uci() if hint_move is not None else "brak"
                print(
                    f"[Ocena pozycji] Value: {val:+.2f} "
                    f"({evaluator_for_hint.get_win_percentage(bm.board):.1f}% Białe) | "
                    f"Podpowiedź: {hint_str}"
                )
                continue

            if cmd.action == InGameAction.SAVE_PGN:
                saved_path = self.export_pgn(bm, white_cfg.describe(), black_cfg.describe())
                print(f"Zapisano partię do pliku PGN: {saved_path}")
                continue

            if cmd.action == InGameAction.RESTART:
                bm = BoardManager(fen=initial_fen)
                last_move = None
                print("Zrestartowano planszę.")
                continue

            if cmd.action == InGameAction.MOVE:
                move_str = cmd.payload or ""
                if self._try_apply_human_move(bm, move_str):
                    last_move = bm.board.peek()
                else:
                    print(
                        f"Nielegalny lub niepoprawny ruch: '{move_str}'. "
                        "Wpisz ruch w formacie UCI (np. e2e4) lub 'help'."
                    )
                continue

        # ---- ZAKOŃCZENIE PARTII ----
        if aborted:
            result_str = "*"
            reason_str = "Przerwano przez użytkownika."
        elif bm.is_game_over():
            result_str = bm.get_game_result()
            reason_str = bm.get_game_over_reason() or "Partia zakończona."
            if not fast_mode:
                final_eval_src = assist_eval or white_eval or black_eval
                final_eval = final_eval_src.evaluate(bm.board) if final_eval_src else None
                self.board_view.display(
                    board_manager=bm,
                    perspective=perspective,
                    eval_value=final_eval,
                    last_move=last_move,
                    clear_screen=False,
                )
        else:
            result_str = "*"
            reason_str = f"Osiągnięto limit półruchów ({max_plies})."

        pgn_path: Optional[Path] = None
        if save_pgn:
            pgn_path = self.export_pgn(bm, white_cfg.describe(), black_cfg.describe())

        print(f"\nKoniec partii: {result_str} — {reason_str} (Półruchów: {len(bm)})")
        if pgn_path is not None:
            print(f"Zapisano PGN: {pgn_path}")

        return GameResult(
            result=result_str,
            reason=reason_str,
            moves_count=len(bm),
            white_name=white_cfg.describe(),
            black_name=black_cfg.describe(),
            final_fen=bm.get_fen(),
            pgn_path=pgn_path,
        )

    @staticmethod
    def _try_apply_human_move(board_manager: BoardManager, move_str: str) -> bool:
        """
        Próbuje wykonać ruch podany przez człowieka – najpierw jako UCI (np. 'e2e4'),
        a w razie niepowodzenia również jako notację SAN (np. 'e4', 'Nf3').
        """
        if board_manager.push_uci(move_str):
            return True
        try:
            san_move = board_manager.board.parse_san(move_str)
            return board_manager.push_move(san_move)
        except ValueError:
            return False
