"""Check that an sdist contains source and release metadata only."""

from __future__ import annotations

import argparse
import sys
import tarfile
from pathlib import Path, PurePosixPath

REQUIRED_PATHS = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "src/openclip_gemma/__init__.py",
    "src/openclip_gemma/constants.py",
    "src/openclip_gemma/factory.py",
    "src/openclip_gemma/model.py",
    "src/openclip_gemma/tokenizer.py",
    "src/openclip_gemma/transforms.py",
}
EXCLUDED_TOP_LEVEL = {
    "benchmarks",
    "docs",
    "examples",
    "experiments",
    "tests",
}
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
MAX_ARCHIVE_FILE_SIZE = 50_000_000


def verify_sdist(archive_path: Path) -> None:
    if not archive_path.is_file() or not archive_path.name.endswith(".tar.gz"):
        raise ValueError(f"Not a source distribution archive: {archive_path}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        paths = {PurePosixPath(member.name).parts[1:] for member in members}
        normalized_paths = {"/".join(parts) for parts in paths}
        missing = REQUIRED_PATHS.difference(normalized_paths)
        if missing:
            raise ValueError(f"Source distribution is missing files: {sorted(missing)}")

        excluded = sorted(
            path
            for path in normalized_paths
            if PurePosixPath(path).parts[0] in EXCLUDED_TOP_LEVEL
        )
        if excluded:
            raise ValueError(f"Source distribution contains development files: {excluded}")

        weight_files = [
            path
            for path in normalized_paths
            if PurePosixPath(path).suffix.lower() in WEIGHT_SUFFIXES
        ]
        if weight_files:
            raise ValueError(
                f"Source distribution must not contain model or checkpoint files: {weight_files}"
            )

        oversized = [
            (member.name, member.size)
            for member in members
            if member.size > MAX_ARCHIVE_FILE_SIZE
        ]
        if oversized:
            raise ValueError(
                f"Source distribution contains unexpectedly large files: {oversized}"
            )

    print(f"Verified source distribution contents: {archive_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    try:
        verify_sdist(args.archive)
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"Source distribution verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
