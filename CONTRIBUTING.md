# Contributing to Maic

Thank you for your interest in contributing to Maic! We welcome contributions of all kinds, from bug reports and documentation improvements to new features and optimizations.

## Code of Conduct

This project adheres to the Contributor Covenant Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to the maintainers.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/maic.git
   cd maic
   ```
3. **Install everything** (Python venv + MLX + frontend) in one step:
   ```bash
   just setup
   ```
   Requires Python 3.10+, Node.js 22+, and [`just`](https://github.com/casey/just) (`brew install just`).

## Development Workflow

### Backend Development
- Make changes in `maic/` or `main.py`
- Run backend: `just dev`
- Default server URL: `http://localhost:8000`
- Run with a specific model:
  ```bash
  just dev --model mlx-community/Llama-3.2-1B-Instruct-4bit
  ```

### Frontend Development
- Make changes in `frontend/src/`
- Run frontend dev server:
  ```bash
  just dev-frontend
  ```
- Build frontend for production:
  ```bash
  just build
  ```
- Run backend + built frontend together:
  ```bash
  just run
  ```

### Testing
- Run backend tests:
  ```bash
  just test
  ```
- Run all lint checks:
  ```bash
  just lint
  ```
- Run full local CI equivalent:
  ```bash
  just ci
  ```
- For frontend-only work, also run:
  ```bash
  cd frontend && npm run lint && npm run build
  ```

## Submitting Changes

### Before You Commit
1. Ensure your code follows the existing style
2. Test your changes thoroughly
3. Add comments for complex logic
4. Update documentation if behavior changes (`README.md`, `CHANGELOG.md`, and/or `docs/CURRENT/*`)

### Commit Messages
Write clear, concise commit messages:
```
[type]: Brief description

Longer explanation if needed.
- Mention related issues: fixes #123
- Explain reasoning and impact
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Examples:
- `feat: add model deletion from storage`
- `fix: correct TPM chart label formatting`
- `docs: update API usage examples`
- `perf: optimize model loading time`

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```

2. **Push to your fork:**
   ```bash
   git push origin feat/your-feature-name
   ```

3. **Open a Pull Request** on GitHub with:
   - Clear title and description
   - Reference to any related issues
   - Summary of changes
   - Testing notes

4. **Code Review:** Maintainers will review and request changes if needed.
5. Ensure required checks pass in GitHub Actions (CI gate).

### Release Notes

- Maic is currently on the `1.0.x` line.
- Production release publishing is automated:
  - Merge to `main` triggers the release workflow.
  - Version is read from `pyproject.toml`.
  - A matching `vX.Y.Z` tag and GitHub Release are created.
  - Python artifacts are published to PyPI.
- If your PR changes user-facing behavior, include a `CHANGELOG.md` update in the same PR.

## Coding Standards

### Python
- Use type hints (project runs mypy in strict mode).
- Keep route handlers thin; business logic belongs in `maic/core`.
- Keep OpenAI response shaping in `maic/adapters`.
- Use docstrings for modules, classes, and non-trivial functions.

### TypeScript/JavaScript
- Run frontend lint/build before PR:
  ```bash
  cd frontend && npm run lint && npm run build
  ```
- Keep API calls in `frontend/src/api/client.ts`.
- Keep shared API types in `frontend/src/api/types.ts`.
- Keep business/state logic in hooks; keep components mostly presentational.

### General
- Write readable code, not clever code
- Add comments for "why", not "what"
- Keep files organized and modular
- Avoid large functions and files

## Reporting Issues

Before creating an issue, check if it already exists. When reporting:

1. **Use a clear, descriptive title**
2. **Describe the problem** with steps to reproduce
3. **Expected vs actual behavior**
4. **Environment:** macOS version, Python version, hardware
5. **Error messages or logs** (if applicable)
6. **Screenshots** (if relevant)

## Feature Requests

1. **Check existing issues** first
2. **Describe the use case** and why it would be useful
3. **Provide examples** of how it would work
4. **Consider implementation** if possible

## Documentation

- Update README.md for user-facing changes
- Update CHANGELOG.md following the format
- Add docstrings to new functions/classes
- Keep `docs/CURRENT/*` aligned with the current implementation reality

## Questions?

- Open a GitHub Discussion
- Check existing issues and PRs
- Review documentation in the `/docs` folder (when available)

---

Thank you for contributing to make Maic better! 🪄
