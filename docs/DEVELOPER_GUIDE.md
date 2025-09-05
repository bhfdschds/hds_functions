# Developer Guide

## Overview

The `hds-functions` project contains shared PySpark utilities developed by the BHF-DSC Health Data Science team for NHS England's Secure Data Environment (SDE).

This guide covers:
- Project structure  
- Branching strategy  
- Commit message conventions  
- Code formatting and linting  
- Running tests  
- Continuous integration and release workflow  
- VSCode configuration  

---

## Project Structure

```
hds-functions/
├── src/                  # Python package modules
│   └── hds_functions/    # Importable package
├── tests/                # Unit tests
├── docs/                 # Project documentation
├── .vscode/              # VSCode configuration
├── .github/workflows/    # GitHub Actions workflows
├── .gitignore            # Git ignore rules
├── CHANGELOG.md          # Release notes and version history
├── LICENSE               # Project license
├── README.md             # Project overview
└── pyproject.toml        # Poetry and project configuration
```

---

## Branches

- **main**: Stable production-ready code.  
- **dev**: Active development occurs here. All features and fixes are integrated before merging into `main`.  

**Workflow:**  
A typical development flow:

1. Create a feature or fix branch from `dev` using a descriptive name, e.g.:  
   - `feat/add-csv-loader`  
   - `fix/date-parsing-bug`  
2. Work on the branch and open a pull request (PR) into `dev` for review (squash merge).  
3. After checks pass, the feature/fix merges into `dev`.  
4. When `dev` contains a set of stable changes, open a PR into `main`. After review and checks pass, merge into `main` (standard merge commit).  
5. The `main` branch is automatically synced back to `dev` via CI to keep development up to date (fast-forward merge).  


---

## Commit Message Rules

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/).  

Format:

```
<type>(<scope>): <description>
```

**Examples:**
```
feat(csv): add CSV loader
fix(date): correct date parsing logic
docs(project): update README
```

- **Scope tags** are recommended but not enforced.
- **Allowed types:** `build`, `chore`, `ci`, `docs`, `feat`, `fix`, `perf`, `style`, `refactor`, `test`.
- Commit messages following this convention are automatically used to generate the **CHANGELOG.md**.

---

## Code Formatting & Linting

[Ruff](https://docs.astral.sh/ruff/) is used for linting and formatting.  

### Rules:
- `E`, `F`: PEP8 style and Pyflakes checks  
- `B`: flake8-bugbear for common Python issues  
- `I`: Import sorting (isort-style)  
- `D`: Google-style docstring checks  

### Formatting:
- Line length: 88 characters  
- Quotes: double (`"`)  
- Python version target: 3.10  

**Run locally:**
```bash
poetry run ruff check .
poetry run ruff format --check .
```

Linting and formatting are checked on all pull requests into `dev` and `main` via GitHub Actions.

---

## Running Tests

[pytest](https://docs.pytest.org/) with coverage is used for testing.

**Run tests locally:**
```bash
poetry run pytest
```

Test configuration:
- Coverage is measured for the `hds_functions` package.
- Tests are located in the `tests/` directory.
- Pytest automatically adds `src` to the Python path.

---

## Continuous Integration & Release

CI/CD is handled via GitHub Actions. Key workflows:

### 1. Lint & Test on Pull Requests
Runs on PRs to `dev` and `main`. The workflow:
- Checkout the repository
- Set up Python 3.10
- Install Poetry dependencies
- Install Spark for PySpark tests
- Run Ruff for linting and formatting
- Run pytest for unit tests and coverage

> To merge, all linting, formatting, and tests must pass. Any issues should be fixed before merging.

### 2. Release & Sync
- Triggered on pushes to `main`.
- Steps performed in the workflow:
  - Checkout `main` with full history
  - Run [python-semantic-release](https://python-semantic-release.readthedocs.io/) to determine version and generate release notes
  - Publish release assets (tags, notes)
  - Fast-forward merge `main` into `dev`
  
---

## VSCode Setup

Recommended settings (via `.vscode/settings.json`):

- **Python files**: Ruff for formatting and linting; auto-format on save.  
- **JSON files**: Format on save, 4-space indentation.  

The [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) can be installed for seamless editor integration.

---

## Contact

For help or questions, contact the BHF-DSC Health Data Science team:  
**bhfdsc_hds [at] hdruk [dot] ac [dot] uk**

