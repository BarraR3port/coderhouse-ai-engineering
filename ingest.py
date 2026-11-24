import argparse
import logging
import sys

from pydantic import ValidationError

from rag import IngestError, RagSettings, ingest, local_embeddings


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fragmenta los documentos de data/ y los guarda en la base vectorial local."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="reindexa aunque los documentos y la configuración no hayan cambiado",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        settings = RagSettings()
    except ValidationError as error:
        print(f"Configuración inválida en el .env:\n{error}")
        return 2
    try:
        report = ingest(settings, local_embeddings(settings), rebuild=args.rebuild)
    except IngestError as error:
        print(error)
        return 1
    if report.reindexed:
        print(
            f"Listo: {report.chunks} fragmentos de {report.documents} documentos "
            f"guardados en {settings.vectorstore_dir}"
        )
    else:
        print(f"La base vectorial ya estaba al día ({report.chunks} fragmentos): no se reindexó")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main(sys.argv[1:]))
