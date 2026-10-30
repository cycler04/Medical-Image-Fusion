# Python Environment Guidelines

- **Primary Interpreter**: Always use the project's dedicated virtual environment:
  `.venv_main/bin/python`
- **No Global System Scans**: Never run system-wide searches across `/` (e.g., `find / -name python` or `find / -name torch`) or assume global `conda` is in PATH.
- **Dependency Checks**: Verify imports using `.venv_main/bin/python -c "import <package>"`.
