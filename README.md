# Introduction
The `mutools` package is a consolidated collection of often-reused functions that I have developed over time for various projects. It serves as a personal utility library to enhance organization and efficiency in coding tasks.

# Installation

## Requirements

- Python >= 3.9

## Using `uv` (recommended)

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd mutools
uv sync
```

To include development dependencies (e.g. `ruff`):

```bash
uv sync --group dev
```

## Using `pip`

Clone the repository and install in editable mode:

```bash
git clone <repo-url>
cd mutools
pip install -e .
```