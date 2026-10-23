from __future__ import annotations

import re
from pathlib import Path

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[1]
    / ".github"
    / "workflows"
    / "publish-pypi.yml"
)


def test_common_workflow_is_tag_only_and_checks_release_identity() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert re.search(
        r'(?m)^on:\n  push:\n    tags:\n      - "v\*"$',
        workflow,
    )
    assert "workflow_dispatch" not in workflow
    assert "$env:GITHUB_EVENT_NAME -cne \"push\"" in workflow
    assert "$env:GITHUB_REF_TYPE -cne \"tag\"" in workflow
    assert "$env:GITHUB_REPOSITORY -cne \"kotadymo/openvlclip\"" in workflow
    assert (
        r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$" in workflow
    )
    assert '[version]$version -lt [version]"0.7.0"' in workflow
    assert 'refs/tags/$tag^{commit}' in workflow
    assert "git merge-base --is-ancestor $env:GITHUB_SHA FETCH_HEAD" in workflow


def test_common_workflow_requires_same_sha_main_ci_and_absent_pypi_version() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert (
        "actions/workflows/ci.yml/runs?head_sha=$env:GITHUB_SHA"
        "&status=completed&per_page=100"
    ) in workflow
    assert '$_.head_sha -ceq $env:GITHUB_SHA' in workflow
    assert '$_.event -ceq "push"' in workflow
    assert '$_.head_branch -ceq "main"' in workflow
    assert '$_.conclusion -ceq "success"' in workflow
    assert "https://pypi.org/pypi/openvlclip/$version/json" in workflow
    assert "$pypiResponse.StatusCode -eq 200" in workflow
    assert "$pypiResponse.StatusCode -ne 404" in workflow
    assert "projectName -cne \"openvlclip\"" in workflow
    assert "projectVersion -cne $version" in workflow
    assert "lockVersion -cne $version" in workflow
    assert "uv lock --check" in workflow


def test_common_workflow_keeps_license_gates_and_isolates_oidc_publish_job() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "OPENVLCLIP_DINOTXT_LICENSE_CONFIRMED" in workflow
    assert "OPENVLCLIP_V070_MODELS_LICENSE_CONFIRMED" in workflow
    assert '$env:DINOTXT_LICENSE_CONFIRMED -cne "true"' in workflow
    assert '$env:V070_MODELS_LICENSE_CONFIRMED -cne "true"' in workflow
    assert "permissions: {}" in workflow
    assert workflow.count("id-token: write") == 1

    build_job, publish_job = workflow.split("\n  publish:\n", maxsplit=1)
    assert "id-token: write" not in build_job
    assert "environment:\n      name: pypi" in publish_job
    assert "pypa/gh-action-pypi-publish@" in publish_job
    assert "pypi-openvlclip-${{ needs.build-and-verify.outputs.version }}-distributions" in workflow
    assert "uv run --locked python tools\\verify_wheel.py" in workflow
    assert "uv run --locked python tools\\verify_sdist.py" in workflow
    assert "testpypi" not in workflow.lower()


def test_common_workflow_uses_commit_pinned_actions_and_removes_legacy_publishers() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    action_refs = re.findall(r"(?m)^\s*uses:\s*[^@\n]+@([^\s]+)$", workflow)

    assert action_refs
    assert all(re.fullmatch(r"[0-9a-f]{40}", reference) for reference in action_refs)
    assert not (
        WORKFLOW_PATH.parent / "publish-pypi-v0.6.0.yml"
    ).exists()
