import importlib.util
from pathlib import Path

import pytest

_VALIDATION_PATH = Path(__file__).resolve().parents[1] / "tools" / "readme_validation.py"
_VALIDATION_SPEC = importlib.util.spec_from_file_location(
    "openvlclip_readme_validation",
    _VALIDATION_PATH,
)
if _VALIDATION_SPEC is None or _VALIDATION_SPEC.loader is None:
    raise RuntimeError(f"Could not load release validator from {_VALIDATION_PATH}.")
_VALIDATION_MODULE = importlib.util.module_from_spec(_VALIDATION_SPEC)
_VALIDATION_SPEC.loader.exec_module(_VALIDATION_MODULE)
validate_distribution_metadata = _VALIDATION_MODULE.validate_distribution_metadata
validate_local_links = _VALIDATION_MODULE.validate_local_links
validate_release_wording = _VALIDATION_MODULE.validate_release_wording


@pytest.mark.parametrize(
    "text",
    [
        "v0.2.0 has not been published to production PyPI.",
        "v0.2.0 publication pending.",
        "After publishing v0.2.0, update README.",
        "The v0.2.0 release is planned for the next month.",
        "The current production release is v0.1.1.",
        "v0.1.1 remains the latest production release.",
        "v0.2.0の機能は公開予定です。",
        "v0.2.0は公開前です。",
        "公開後にREADMEを更新する。",
    ],
)
def test_rejects_time_sensitive_release_wording(text: str) -> None:
    with pytest.raises(ValueError, match="can become stale"):
        validate_release_wording(text, "README", "0.2.0")


def test_allows_historical_rc_publication_statement() -> None:
    validate_release_wording(
        "The v0.2.0rc1 release candidate was published to TestPyPI. "
        "At that time v0.1.1 was the production release.",
        "README",
        "0.2.0",
    )


def test_validates_local_markdown_links(tmp_path: Path) -> None:
    readme_path = tmp_path / "README.md"
    target = tmp_path / "guide.md"
    target.touch()

    validate_local_links("[Guide](guide.md)", readme_path)
    with pytest.raises(ValueError, match="broken local link"):
        validate_local_links("[Missing](missing.md)", readme_path)


def test_checks_metadata_against_project_and_readme(tmp_path: Path) -> None:
    readme_path = tmp_path / "README.md"
    readme = "# OpenVLCLIP v0.5.0"
    readme_path.write_text(readme, encoding="utf-8")
    project: dict[str, object] = {
        "name": "openvlclip",
        "requires-python": ">=3.12,<3.15,!=3.14.1",
        "dependencies": ["numpy>=1.26,<3"],
        "optional-dependencies": {"video": ["av>=19.0.1,<20"]},
    }
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: openvlclip\n"
        "Version: 0.5.0\n"
        "Requires-Python: >=3.12, !=3.14.1, <3.15\n"
        "Requires-Dist: numpy>=1.26,<3\n"
        "Provides-Extra: video\n"
        "Requires-Dist: av>=19.0.1,<20 ; extra == 'video'\n\n"
        f"{readme}\n"
    )

    validate_distribution_metadata(
        metadata,
        "0.5.0",
        project,
        readme,
        "wheel",
        readme_path,
    )

    with pytest.raises(ValueError, match="Requires-Dist"):
        validate_distribution_metadata(
            metadata.replace("av>=19.0.1", "av>=18.0.0"),
            "0.5.0",
            project,
            readme,
            "wheel",
            readme_path,
        )
