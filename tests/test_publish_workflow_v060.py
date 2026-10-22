from __future__ import annotations

import re
from pathlib import Path

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "publish-pypi-v0.6.0.yml"
)


def test_v060_workflow_is_tag_only_and_verifies_main_ci() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert re.search(
        r"(?m)^on:\n  push:\n    tags:\n      - v0\.6\.0$",
        workflow,
    )
    assert "EXPECTED_TAG: v0.6.0" in workflow
    assert "EXPECTED_VERSION: 0.6.0" in workflow
    assert "git merge-base --is-ancestor $env:GITHUB_SHA FETCH_HEAD" in workflow
    assert "actions/workflows/ci.yml/runs?head_sha=$env:GITHUB_SHA" in workflow
    assert "$_.event -ceq \"push\"" in workflow
    assert "$_.head_branch -ceq \"main\"" in workflow
    assert "uv lock --check" in workflow
    assert "tools\\verify_wheel.py" in workflow
    assert "tools\\verify_sdist.py" in workflow


def test_v060_workflow_uses_trusted_publishing_and_fail_closed_license_gate() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow.count("id-token: write") == 1
    publish_job = workflow.split("\n  publish:\n", maxsplit=1)[1]
    assert "environment:\n      name: pypi" in publish_job
    assert "pypa/gh-action-pypi-publish@" in publish_job
    assert "OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED" in workflow
    assert '$env:DINOTXT_LICENSE_CONFIRMED -cne "true"' in workflow
    assert "testpypi" not in workflow.lower()
