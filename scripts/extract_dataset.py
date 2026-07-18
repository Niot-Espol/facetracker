"""Extrae un dataset ZIP sin permitir rutas fuera del destino indicado."""

from __future__ import annotations

import argparse
import stat
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extrae dataset_modelo.zip de forma segura.")
    parser.add_argument("--zip", required=True, type=Path, dest="zip_path")
    parser.add_argument("--destination", required=True, type=Path)
    return parser.parse_args()


def safe_extract(zip_path: Path, destination: Path) -> None:
    if not zip_path.is_file():
        raise FileNotFoundError(f"No existe el archivo ZIP: {zip_path}")

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if target != destination and destination not in target.parents:
                raise ValueError(f"Ruta no segura dentro del ZIP: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"No se permiten enlaces simbólicos: {member.filename}")
        archive.extractall(destination)


def main() -> None:
    args = parse_args()
    safe_extract(args.zip_path, args.destination)
    print(f"Dataset extraído en: {args.destination.resolve()}")


if __name__ == "__main__":
    main()
