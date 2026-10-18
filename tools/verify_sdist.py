"""Check that an sdist contains source and release metadata only."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
from pathlib import Path, PurePosixPath

from readme_validation import (
    validate_distribution_metadata,
    validate_readme,
)

REQUIRED_PATHS = {
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "src/openclip_gemma/__init__.py",
    "src/openclip_gemma/batching.py",
    "src/openclip_gemma/constants.py",
    "src/openclip_gemma/factory.py",
    "src/openclip_gemma/image_search.py",
    "src/openclip_gemma/index_storage.py",
    "src/openclip_gemma/model.py",
    "src/openclip_gemma/tokenizer.py",
    "src/openclip_gemma/transforms.py",
    "src/openclip_gemma/video_search.py",
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


def verify_sdist(archive_path: Path, expected_version: str = "0.2.0") -> None:
    if not archive_path.is_file() or not archive_path.name.endswith(".tar.gz"):
        raise ValueError(f"Not a source distribution archive: {archive_path}")

    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        paths = {PurePosixPath(member.name).parts[1:] for member in members}
        normalized_paths = {"/".join(parts) for parts in paths}
        missing = REQUIRED_PATHS.difference(normalized_paths)
        if missing:
            raise ValueError(f"Source distribution is missing files: {sorted(missing)}")

        project_root = Path(__file__).resolve().parents[1]
        readme_path = project_root / "README.md"
        readme_members = [
            member for member in members if PurePosixPath(member.name).name == "README.md"
        ]
        if len(readme_members) != 1:
            raise ValueError("Source distribution must contain one README.md file.")
        readme = (
            archive.extractfile(readme_members[0])
            .read()
            .decode("utf-8")
            .replace("\r\n", "\n")
        )
        source_readme = readme_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if readme != source_readme:
            raise ValueError("Source distribution README does not match the project README.")
        validate_readme(readme, readme_path, "sdist README", expected_version)

        pyproject_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "pyproject.toml"
        ]
        if len(pyproject_members) != 1:
            raise ValueError("Source distribution must contain one pyproject.toml file.")
        project = tomllib.loads(
            archive.extractfile(pyproject_members[0])
            .read()
            .decode("utf-8")
        )["project"]

        metadata_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "PKG-INFO"
        ]
        if len(metadata_members) != 1:
            raise ValueError("Source distribution must contain one PKG-INFO file.")
        metadata = (
            archive.extractfile(metadata_members[0])
            .read()
            .decode("utf-8")
            .replace("\r\n", "\n")
        )
        validate_distribution_metadata(
            metadata,
            expected_version,
            project,
            readme,
            "sdist",
            readme_path,
        )
        if f"Version: {expected_version}\n" not in metadata:
            raise ValueError("Source distribution metadata has an unexpected version.")
        if "Name: openclip-gemma\n" not in metadata:
            raise ValueError("Source distribution metadata has an unexpected name.")
        if "Author: kotadymo\n" not in metadata:
            raise ValueError("Source distribution metadata has an unexpected author.")
        if any(line.startswith("Author-email:") for line in metadata.splitlines()):
            raise ValueError("Source distribution metadata must not contain an author email.")
        if "License-Expression: Apache-2.0\n" not in metadata:
            raise ValueError("Source distribution metadata has an unexpected license.")

        license_members = [
            member
            for member in members
            if PurePosixPath(member.name).name == "LICENSE"
        ]
        if len(license_members) != 1:
            raise ValueError("Source distribution must contain one LICENSE file.")
        expected_license = (
            Path(__file__).resolve().parents[1] / "LICENSE"
        ).read_bytes().replace(b"\r\n", b"\n")
        packaged_license = archive.extractfile(license_members[0]).read().replace(
            b"\r\n",
            b"\n",
        )
        if packaged_license != expected_license:
            raise ValueError("Source distribution LICENSE does not match the project LICENSE.")

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
    parser.add_argument("--expected-version", default="0.2.0")
    args = parser.parse_args()
    try:
        verify_sdist(args.archive, expected_version=args.expected_version)
    except (OSError, ValueError, tarfile.TarError) as error:
        print(f"Source distribution verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
