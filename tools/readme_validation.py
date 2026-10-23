"""Shared checks for release-stable README and distribution metadata."""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from email.parser import Parser
from pathlib import Path
from urllib.parse import unquote, urlsplit

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))")
REGISTERED_MODEL_IDS = (
    "embeddinggemma-2",
    "dinov2-dinotxt",
    "siglip2-base",
    "siglip2-naflex-base",
    "qwen3-vl-embedding-2b",
)
STALE_REGISTRY_WORDING = (
    re.compile(r"\bdevelopment registry\b", re.IGNORECASE),
    re.compile(
        r"\bSigLIP\s+2\s+and\s+Qwen3-VL\b[^\n.]{0,100}\bnot\s+in\b"
        r"[^\n.]{0,100}\bpublished\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bQwen3-VL\b[^\n.]{0,60}\bDraft\s+PR\s*#44\b",
        re.IGNORECASE,
    ),
)


def validate_release_wording(readme: str, source: str, version: str) -> None:
    tagged_version = rf"v?{re.escape(version)}"
    patterns = (
        re.compile(
            rf"\b{tagged_version}\b.{{0,100}}\b(?:not yet|has not been|is not|isn't)\b"
            r".{0,50}\b(?:published|released|available)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{tagged_version}\b.{{0,60}}\b(?:publication|release)\s+pending\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:current|latest)\s+(?:(?:production|PyPI)\s+)?"
            r"(?:release|version)\b.{0,60}\bv?\d+\.\d+\.\d+(?:[a-z]+\d+)?\b"
            r"|\bv?\d+\.\d+\.\d+(?:[a-z]+\d+)?\b.{0,60}"
            r"\b(?:current|latest)\s+(?:(?:production|PyPI)\s+)?release\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{tagged_version}\b.{{0,60}}\b(?:publication|release)\s+"
            r"(?:is\s+)?(?:pending|planned|scheduled)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\bafter\s+(?:publishing|publication of)\s+{tagged_version}\b"
            rf"|\bafter\s+{tagged_version}\s+(?:is\s+)?(?:published|released)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:planned\s+for|planned\s+release\s+of|coming\s+in|upcoming\s+in)"
            rf"\s+{tagged_version}\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{tagged_version}\b.{{0,60}}\b(?:planned|upcoming|coming soon)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"\b{tagged_version}\b.{{0,50}}\bwill\s+(?:add|include|provide)\b",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{tagged_version}[^\n。]{{0,80}}"
            r"(?:未公開|公開待ち|公開予定|公開前|まだ公開されていない|"
            r"未リリース|リリース予定|追加予定)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?:現行|現在)(?:の)?(?:正式版|最新(?:版)?|PyPI(?:正式)?版)"
            r"[^\n。]{0,50}v?\d+\.\d+\.\d+|v?\d+\.\d+\.\d+"
            r"[^\n。]{0,50}(?:現行|最新)",
            re.IGNORECASE,
        ),
        re.compile(r"(?:公開後|リリース後)[^\n。]{0,80}(?:README|更新|修正)"),
    )
    for pattern in patterns:
        match = pattern.search(readme)
        if match:
            raise ValueError(
                f"{source} contains release-state wording that can become stale: "
                f"{match.group(0)!r}"
            )


def validate_local_links(readme: str, readme_path: Path) -> None:
    for match in MARKDOWN_LINK.finditer(readme):
        target = match.group(1) or match.group(2)
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        target_path = (readme_path.parent / unquote(parsed.path)).resolve()
        if not target_path.exists():
            raise ValueError(
                f"README contains a broken local link {target!r} "
                f"(resolved to {target_path})."
            )


def validate_registry_status(readme: str, source: str) -> None:
    missing_models = [
        model_id for model_id in REGISTERED_MODEL_IDS if f"`{model_id}`" not in readme
    ]
    if missing_models:
        raise ValueError(
            f"{source} does not document all five v0.7.0 registry models: "
            f"{missing_models}."
        )
    if not re.search(
        r"\bQwen\s+8B\b.{0,100}\b(?:deferred|not\s+available|not\s+supported|"
        r"not\s+registered)\b",
        readme,
        re.IGNORECASE,
    ):
        raise ValueError(
            f"{source} must state that Qwen 8B is not registered or supported."
        )
    for pattern in STALE_REGISTRY_WORDING:
        match = pattern.search(readme)
        if match:
            raise ValueError(
                f"{source} contains stale model-registry wording: "
                f"{match.group(0)!r}"
            )


def validate_readme(
    readme: str,
    readme_path: Path,
    source: str,
    version: str,
) -> None:
    validate_release_wording(readme, source, version)
    validate_registry_status(readme, source)
    validate_local_links(readme, readme_path)


def read_project_config(path: Path) -> dict[str, object]:
    with path.open("rb") as file:
        return tomllib.load(file)["project"]


def _requirement_key(requirement: Requirement) -> tuple[object, ...]:
    return (
        canonicalize_name(requirement.name),
        frozenset(canonicalize_name(extra) for extra in requirement.extras),
        tuple(sorted(str(specifier) for specifier in requirement.specifier)),
        str(requirement.marker) if requirement.marker else "",
    )


def validate_distribution_metadata(
    metadata_text: str,
    expected_version: str,
    project: dict[str, object],
    readme: str,
    source: str,
    readme_path: Path,
) -> None:
    metadata = Parser().parsestr(metadata_text)
    if canonicalize_name(metadata.get("Name", "")) != canonicalize_name(
        str(project["name"])
    ):
        raise ValueError(f"{source} metadata has an unexpected distribution name.")
    if metadata.get("Version") != expected_version:
        raise ValueError(f"{source} metadata has an unexpected version.")

    expected_python = SpecifierSet(str(project["requires-python"]))
    actual_python = SpecifierSet(metadata.get("Requires-Python", ""))
    if actual_python != expected_python:
        raise ValueError(f"{source} metadata has an unexpected Python requirement.")

    expected_requirements: list[Requirement] = [
        Requirement(dependency) for dependency in project["dependencies"]
    ]
    for extra, dependencies in project.get("optional-dependencies", {}).items():
        for dependency in dependencies:
            core, separator, existing_marker = dependency.partition(";")
            marker = f"({existing_marker.strip()}) and extra == '{extra}'"
            if not separator:
                marker = f"extra == '{extra}'"
            expected_requirements.append(Requirement(f"{core.strip()}; {marker}"))

    actual_requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist", [])
    ]
    if Counter(map(_requirement_key, actual_requirements)) != Counter(
        map(_requirement_key, expected_requirements)
    ):
        raise ValueError(f"{source} metadata has unexpected Requires-Dist entries.")

    expected_extras = set(project.get("optional-dependencies", {}))
    actual_extras = set(metadata.get_all("Provides-Extra", []))
    if actual_extras != expected_extras:
        raise ValueError(f"{source} metadata has unexpected optional extras.")

    normalized_description = metadata_text.replace("\r\n", "\n").partition("\n\n")[2]
    expected_description = readme.replace("\r\n", "\n").rstrip("\n")
    if normalized_description.rstrip("\n") != expected_description:
        raise ValueError(
            f"{source} metadata Description does not match the packaged README."
        )
    validate_release_wording(
        normalized_description,
        f"{source} metadata Description",
        expected_version,
    )
    validate_local_links(normalized_description, readme_path)
