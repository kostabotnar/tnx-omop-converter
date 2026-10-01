# Releasing

## One-time setup on PyPI

The release workflow publishes with PyPI trusted publishing, so no API token is stored in GitHub.

1. Sign in to pypi.org and open Your projects, Publishing, then add a pending publisher (the project does not exist on PyPI before the first release).
2. Enter these values:
   - PyPI project name: `tnx-omop-converter`
   - Owner: `kostabotnar`
   - Repository name: `tnx-omop-converter`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. In the GitHub repository, open Settings, Environments and create an environment named `pypi`. Optionally require a reviewer so each publish needs approval.

## Release steps

1. Change `version` in `pyproject.toml` (for example to `0.2.0`).
2. Move the Unreleased entries in `CHANGELOG.md` under the new version heading.
3. Merge the change to `main`.
4. Tag the merge commit and push the tag:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The workflow runs the tests, builds the package, checks that the tag matches the version in `pyproject.toml`, publishes to PyPI and creates a GitHub release with the built files. If the tag and the version differ, delete the tag, fix the version and tag again.
