* Add `RELEASE_BOT` access token to actions. Should be a personal access token with "repo" permissions.
* Update `pyproject.toml` with
  * semantic_release section
  * Match version in .toml with latest release on github. If no release on github, either 0.0.0, or create a tagged release.
* GA files
  * Add `main_test_and_release`
  * Add `test_prs`
  * Add `automerge`
  * Add `dependabot`
  * Remove unused CI
* Update the PR template
* Get the first build done
