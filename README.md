# Movie Sharjeel Final

movie-sharjeel-final is a movie-review sentiment classification application.

![Flask](https://img.shields.io/badge/Flask-2563EB?style=flat-square)

## Overview

The Flask implementation is organized around the entry points and modules below.

## Implemented workflows

The source defines these handlers:

- **Index** — `index`

## Getting started

### 1. Get the project

```bash
git clone https://github.com/nasirabbas827/movie-sharjeel-final.git
cd movie-sharjeel-final
```

### 2. Prepare Python

Create an isolated environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows Command Prompt, or `source .venv/bin/activate` on macOS/Linux.

No requirements file is included. The inspected Python imports indicate these dependencies as a starting point:

```bash
python -m pip install Flask joblib numpy tensorflow
```

Dependency versions are not pinned in this repository.

### 3. Configure and run

Run from the repository root so relative file paths resolve correctly:

```bash
python app.py
```

Open http://127.0.0.1:5000/, or use the address printed by the application.

### Model files

The source references these artifact paths whose filenames were not found in the repository: `tfidf_logreg_model.pkl`, `tfidf_vectorizer.pkl`, `lstm_model.h5`, `lstm_tokenizer.pkl`. Check the loading/training code to determine whether these files are generated automatically or must be supplied separately.

Keep models, preprocessing objects, and label mappings together in the locations expected by the source.

## Repository guide

| Path | Purpose |
| --- | --- |
| [`app.py`](app.py) | Application entry point |
| [`templates`](templates) | HTML page templates |

## Development notes

These setup instructions are based on the repository source. The application has not been executed as part of this documentation update; verify dependencies and configuration in your local environment.
