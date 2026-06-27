# Building the docs locally

If `make html` fails with "sphinx-build: command not found", install Sphinx and dependencies first.

## Quick setup (macOS/Linux)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install sphinx sphinx-rtd-theme myst-parser
```

Then build:

```bash
cd docs
make html
open _build/html/index.html  # macOS
# or xdg-open _build/html/index.html  # Linux
```

## Alternative using project extras

If the project has dev extras configured:

```bash
pip install -e .[dev]
cd docs
make html
```

## Troubleshooting

- **Missing extensions**: Install missing packages with pip
- **Permission issues**: Use `pip install --user` or ensure virtualenv is activated
- **Math not rendering**: Ensure internet connection for MathJax CDN, or configure local MathJax
