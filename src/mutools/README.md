# mutools

Personal utility library providing shared plotting and I/O helpers across projects.

## Structure

```
mutools/
├── io/          # Data loading and parsing utilities
└── plotting/    # Matplotlib-based plotting utilities
```

## Submodules

### [`io/`](io/)

Utilities for loading and pre-processing data from various sources. Currently covers SPINE training log files.

See [`io/README.md`](io/README.md) for full documentation.

### [`plotting/`](plotting/)

Utilities for creating publication-quality figures, including PROfit-style stacked histograms, systematic uncertainty plots, style sheet management, and a TOML-driven plot dispatcher.

See [`plotting/README.md`](plotting/README.md) for full documentation.
