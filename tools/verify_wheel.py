"""Check that a built wheel contains only the intended Python package."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REQUIRED_MODULES = {
    "openclip_gemma/__init__.py",
    "openclip_gemma/batching.py",
    "openclip_gemma/constants.py",
    "openclip_gemma/factory.py",
    "openclip_gemma/image_search.py",
    "openclip_gemma/index_storage.py",
    "openclip_gemma/model.py",
    "openclip_gemma/tokenizer.py",
    "openclip_gemma/transforms.py",
    "openclip_gemma/video_search.py",
}
WEIGHT_SUFFIXES = {".bin", ".ckpt", ".onnx", ".pt", ".pth", ".safetensors"}
MAX_WHEEL_FILE_SIZE = 50_000_000


def verify_wheel(wheel_path: Path, expected_version: str = "0.2.0rc1") -> None:
    if not wheel_path.is_file() or wheel_path.suffix != ".whl":
        raise ValueError(f"Not a wheel file: {wheel_path}")

    with zipfile.ZipFile(wheel_path) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        missing_modules = REQUIRED_MODULES.difference(names)
        if missing_modules:
            raise ValueError(f"Wheel is missing package modules: {sorted(missing_modules)}")

        package_entries = {
            name
            for name in names
            if name.startswith("openclip_gemma/")
            or ".dist-info/" in name
        }
        unexpected = names.difference(package_entries)
        if unexpected:
            raise ValueError(f"Wheel contains unexpected paths: {sorted(unexpected)}")

        weight_files = [
            name for name in names if Path(name).suffix.lower() in WEIGHT_SUFFIXES
        ]
        if weight_files:
            raise ValueError(f"Wheel must not contain model or checkpoint files: {weight_files}")

        oversized = [
            (entry.filename, entry.file_size)
            for entry in entries
            if entry.file_size > MAX_WHEEL_FILE_SIZE
        ]
        if oversized:
            raise ValueError(f"Wheel contains unexpectedly large files: {oversized}")

        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        wheel_files = [name for name in names if name.endswith(".dist-info/WHEEL")]
        if len(metadata_files) != 1 or len(wheel_files) != 1:
            raise ValueError("Wheel must contain one METADATA and one WHEEL file.")
        license_files = [
            name
            for name in names
            if name.endswith(".dist-info/licenses/LICENSE")
        ]
        if len(license_files) != 1:
            raise ValueError("Wheel must contain the project LICENSE file.")

        metadata = archive.read(metadata_files[0]).decode("utf-8").replace(
            "\r\n",
            "\n",
        )
        if "Name: openclip-gemma\n" not in metadata:
            raise ValueError("Wheel metadata has an unexpected distribution name.")
        if f"Version: {expected_version}\n" not in metadata:
            raise ValueError("Wheel metadata has an unexpected version.")
        if "Author: kotadymo\n" not in metadata:
            raise ValueError("Wheel metadata has an unexpected author.")
        if any(line.startswith("Author-email:") for line in metadata.splitlines()):
            raise ValueError("Wheel metadata must not contain an author email.")
        if "License-Expression: Apache-2.0\n" not in metadata:
            raise ValueError("Wheel metadata has an unexpected license expression.")
        expected_license = (
            Path(__file__).resolve().parents[1] / "LICENSE"
        ).read_bytes().replace(b"\r\n", b"\n")
        packaged_license = archive.read(license_files[0]).replace(b"\r\n", b"\n")
        if packaged_license != expected_license:
            raise ValueError("Wheel LICENSE does not match the project LICENSE.")

    print(f"Verified wheel contents: {wheel_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected-version", default="0.2.0rc1")
    args = parser.parse_args()
    try:
        verify_wheel(args.wheel, expected_version=args.expected_version)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(f"Wheel verification failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
