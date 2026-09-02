"""SberDS source-нода исполнимого ядра проверки качества телеметрии."""

from __future__ import annotations

from pathlib import Path

from _package import build_archive, encode_archive


VALIDATION_ROOT = Path(__file__).resolve().parent / "data" / "laim-ars-validation"


def main() -> dict[str, str]:
    codebase, checksum = encode_archive(build_archive(VALIDATION_ROOT))
    print({"worker": "validation", "checksum": checksum})
    return {
        "codebase_validation": codebase,
        "checksum_validation": checksum,
    }


if __name__ == "__main__":
    main()
