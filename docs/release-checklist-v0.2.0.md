# v0.2.0 Production Release Checklist

## Current status

- [x] v0.2.0rc1 published to TestPyPI; clean CPU and CUDA package checks passed.
- [x] `pyproject.toml` is set to `0.2.0` and `uv.lock` is synchronized.
- [x] Final README and changelog reviewed and validated.
- [x] `uv lock --check`, pytest, Ruff, wheel/sdist verification, and actionlint
  pass on the release branch.
- [ ] Merge the release changes to `main` and confirm CI succeeds on the exact
  release commit.
- [ ] Confirm PyPI Trusted Publishing is configured for the workflow below.
- [ ] After separate approval, create and push the `v0.2.0` tag.
- [ ] Confirm the production publish workflow succeeds and verify PyPI files.
- [ ] Create the GitHub Release only after publication verification.

No tag, package publication, or GitHub Release is created by this checklist.

## Trusted Publishing

The production workflow is
`.github/workflows/publish-pypi-v0.2.0.yml` and uses the existing GitHub
Environment `pypi`. Register the following publisher in the PyPI project
settings:

| Setting | Value |
| --- | --- |
| Owner | `kotadymo` |
| Repository | `openclip-gemma` |
| Workflow filename | `publish-pypi-v0.2.0.yml` |
| GitHub Environment | `pypi` |

PyPI's account-side publisher settings are not exposed by the repository;
verify the registration in the project settings before tagging.

## README and package metadata

`README.md` describes the v0.2.0 feature set, installation commands, verified
environments, and limitations without stating whether or when the package is
available from a package index. Its content is therefore suitable both before
and after the tag is published; no publication-triggered README edit is
required.

The wheel and sdist verifiers compare the packaged README description with the
repository README and reject release-state wording that can become stale.
They also check the project name, version, Python requirement, and dependency
metadata. Keep the changelog's dated RC entry as a historical TestPyPI record.
