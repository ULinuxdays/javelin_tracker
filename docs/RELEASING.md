## Release Checklist

Use this runbook when publishing to package registries or shipping Docker images.

### 1. Prep

1. Update `pyproject.toml` version and ensure `javelin_tracker/__init__.py` (and export metadata) report the same value.
2. Add a new section to `CHANGELOG.md` summarising features/fixes.
3. Sync the version string in `CITATION.cff` and, if needed, refresh `docs/methods.md` with algorithm notes.
4. Regenerate demo data if schemas changed: `python scripts/generate_demo_data.py`.

### 2. Quality gates

```bash
ruff check .
pytest -q
pytest tests/test_smoke_cli.py -q  # optional focused smoke run
```

GitHub Actions (`.github/workflows/ci.yml`) mirrors these checks on Python 3.9/3.11/3.12.

### 3. Build artifacts

```bash
rm -rf dist build
python -m build
```

The command produces both `wheel` and `sdist` artifacts under `dist/`.

### 4. Publish to PyPI

```bash
python -m twine upload dist/*
```

Use `--repository testpypi` for staging uploads. Confirm installation with `pip install --index-url ... javelin-tracker`.

### 5. Docker image

```bash
docker build -t ghcr.io/uday/javelin-tracker:<version> .
docker run --rm -e JAVELIN_TRACKER_BOOTSTRAP_DEMO=1 ghcr.io/uday/javelin-tracker:<version> summary
docker push ghcr.io/uday/javelin-tracker:<version>
```

- `/data` is declared as a volume; publish usage instructions so operators bind-mount persistent storage.
- The entrypoint seeds demo data automatically when `JAVELIN_TRACKER_BOOTSTRAP_DEMO=1`, which is useful for smoke validation in container registries.

### 6. Post-release

- Tag the commit (`git tag vX.Y.Z && git push origin --tags`).
- Create a GitHub release referencing the changelog and uploaded artifacts.
- Update documentation links if URLs changed.

### 7. Zenodo DOI workflow

1. Enable the repository inside https://zenodo.org/account/settings/github/ (one-time).
2. Publish a GitHub release (with the vX.Y.Z tag); Zenodo will automatically archive the snapshot and return a DOI.
3. Edit `CITATION.cff` and `README.md` with the latest DOI badge/link.
4. Mention the DOI in any manuscripts, posters, or academic talks to ensure the software snapshot is citable.
