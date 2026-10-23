#!/usr/bin/env python3
"""
pack_project.py

Skrypt narzędziowy scalający pliki źródłowe, konfiguracyjne i dokumentację projektu
do pojedynczego pliku tekstowego (np. na potrzeby analizy przez LLM / Gemini).

Wymagania:
- Zgodność z wytycznymi projektu (użycie pathlib.Path, brak sztywnych ścieżek).
- Pomijanie plików binarnych, wag modeli, baz danych PGN oraz plików tymczasowych systemu (w tym plików '._*' na macOS).
- Generowanie czytelnego drzewa struktury katalogów na początku wygenerowanego pliku.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


# Domyślne katalogi wykluczane z procesu scalania
DEFAULT_IGNORED_DIRS: Set[str] = {
    ".git",
    ".vscode",
    ".idea",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "venv",
    ".venv",
    "env",
    "raw",          # data/raw (surowe pliki PGN)
    "processed",    # data/processed (binarne pliki memmap)
    "checkpoints",  # models/checkpoints (wagi sieci neuronowej)
}

# Domyślne rozszerzenia wykluczane (pliki binarne, archiwa, wagi modeli)
DEFAULT_IGNORED_EXTENSIONS: Set[str] = {
    ".pt",
    ".pth",
    ".bin",
    ".npy",
    ".npz",
    ".pgn",
    ".pyc",
    ".pyo",
    ".pyd",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
    ".png",
    ".jpg",
    ".jpeg",
    ".ico",
    ".mp4",
}

# Dozwolone rozszerzenia plików tekstowych w projekcie
DEFAULT_ALLOWED_EXTENSIONS: Set[str] = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".gitignore",
}

# Nazwy pojedynczych plików do wykluczenia
DEFAULT_IGNORED_FILENAMES: Set[str] = {
    ".DS_Store",
    "project_packed.txt",
}


def should_ignore_dir(dir_name: str) -> bool:
    """Sprawdza, czy katalog powinien zostać pominięty podczas indeksowania."""
    if dir_name.startswith("._"):
        return True
    return dir_name in DEFAULT_IGNORED_DIRS


def should_include_file(file_path: Path, output_file: Path) -> bool:
    """
    Weryfikuje, czy plik kwalifikuje się do dołączenia do paczki projektu.
    
    Kryteria:
    - Plik nie jest plikiem tymczasowym AppleDouble (prefix '._')
    - Plik nie jest plikiem wyjściowym skryptu
    - Nazwa pliku nie znajduje się na liście wykluczonych
    - Rozszerzenie pliku nie znajduje się na liście wykluczonych rozszerzeń
    - Rozszerzenie znajduje się na liście dozwolonych (lub jest to .gitignore)
    """
    name = file_path.name

    if name.startswith("._"):
        return False

    if name in DEFAULT_IGNORED_FILENAMES:
        return False

    if file_path.resolve() == output_file.resolve():
        return False

    suffix = file_path.suffix.lower()

    if suffix in DEFAULT_IGNORED_EXTENSIONS:
        return False

    if suffix in DEFAULT_ALLOWED_EXTENSIONS or name == ".gitignore":
        return True

    return False


def collect_project_files(root_dir: Path, output_file: Path) -> List[Path]:
    """Rekurencyjnie zbiera ścieżki wszystkich dozwolonych plików w projekcie."""
    collected: List[Path] = []

    for item in root_dir.iterdir():
        if item.is_dir():
            if not should_ignore_dir(item.name):
                collected.extend(_collect_dir(item, output_file))
        elif item.is_file():
            if should_include_file(item, output_file):
                collected.append(item)

    return sorted(collected, key=lambda p: str(p.relative_to(root_dir)))


def _collect_dir(current_dir: Path, output_file: Path) -> Iterable[Path]:
    """Funkcja pomocnicza do rekurencyjnego przeszukiwania podkatalogów."""
    for item in current_dir.iterdir():
        if item.is_dir():
            if not should_ignore_dir(item.name):
                yield from _collect_dir(item, output_file)
        elif item.is_file():
            if should_include_file(item, output_file):
                yield item


def build_ascii_tree(root_dir: Path, files: List[Path]) -> str:
    """
    Generuje eleganckie, hierarchiczne drzewo ASCII dla zebranych plików.
    """
    tree: Dict[str, Any] = {}
    for f in files:
        rel = f.relative_to(root_dir)
        curr = tree
        for part in rel.parts[:-1]:
            curr = curr.setdefault(part, {})
        curr[rel.parts[-1]] = None  # None oznacza plik liściowy

    lines: List[str] = [f"{root_dir.name}/"]

    def _render(node: Dict[str, Any], prefix: str = "") -> None:
        # Sortowanie: katalogi alfabetycznie, pliki alfabetycznie
        sorted_keys = sorted(
            node.keys(),
            key=lambda k: (0 if node[k] is not None else 1, k.lower())
        )
        total = len(sorted_keys)
        for i, key in enumerate(sorted_keys):
            is_last = (i == total - 1)
            connector = "└── " if is_last else "├── "
            child_prefix = "    " if is_last else "│   "
            val = node[key]

            if val is None:
                lines.append(f"{prefix}{connector}{key}")
            else:
                lines.append(f"{prefix}{connector}{key}/")
                _render(val, prefix + child_prefix)

    _render(tree)
    return "\n".join(lines)


def pack_project(root_dir: Path, output_path: Path) -> None:
    """
    Scala pliki projektu w jeden plik tekstowy z nagłówkiem i metadanymi.
    """
    resolved_root = root_dir.resolve()
    resolved_output = output_path.resolve()

    files = collect_project_files(resolved_root, resolved_output)

    header_border = "=" * 80
    section_border = "-" * 80

    total_bytes = 0
    total_lines = 0

    with resolved_output.open("w", encoding="utf-8") as out:
        out.write(f"{header_border}\n")
        out.write("SZACHY AI V2 - PROJECT CODEBASE DUMP\n")
        out.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        out.write(f"Root Directory: {resolved_root}\n")
        out.write(f"Total Files Included: {len(files)}\n")
        out.write(f"{header_border}\n\n")

        out.write("DIRECTORY STRUCTURE OVERVIEW:\n")
        out.write(f"{section_border}\n")
        out.write(build_ascii_tree(resolved_root, files))
        out.write(f"\n{section_border}\n\n")

        for file_path in files:
            rel_path = file_path.relative_to(resolved_root)
            out.write(f"{header_border}\n")
            out.write(f"FILE: {rel_path.as_posix()}\n")
            out.write(f"{header_border}\n")

            try:
                content = file_path.read_text(encoding="utf-8")
                lines_count = len(content.splitlines())
                file_size = len(content.encode("utf-8"))
                total_lines += lines_count
                total_bytes += file_size
                out.write(content)
                if not content.endswith("\n"):
                    out.write("\n")
            except Exception as exc:
                out.write(f"[BŁĄD ODCZYTU PLIKU: {exc}]\n")

            out.write("\n")

    print(f"Pomyślnie scalono projekt do: {resolved_output}")
    print(f"Liczba plików: {len(files)}")
    print(f"Łączna liczba linii kodu: {total_lines}")
    print(f"Rozmiar pliku wyjściowego: {resolved_output.stat().st_size:,} bajtów")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skrypt łączący pliki projektu Szachy AI v2 w jeden plik tekstowy."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Katalog główny projektu (domyślnie: katalog skryptu).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "project_packed.txt",
        help="Ścieżka pliku wyjściowego (domyślnie: project_packed.txt w katalogu projektu).",
    )

    args = parser.parse_args()
    pack_project(root_dir=args.root, output_path=args.output)


if __name__ == "__main__":
    main()
