# mutools

Personal utility library providing shared plotting and I/O helpers across projects.

## Structure

```
mutools/
├── exposure/    # Quality-cut definition, composition, and analysis
├── io/          # Data loading and parsing utilities
└── plotting/    # Matplotlib-based plotting utilities
```

## Submodules

### [`exposure/`](exposure/)

Infrastructure for defining, composing, and applying quality cuts to beam spill DataFrames. Includes a cut registry, expression-tree-backed `QualityCut` objects, hierarchical pass-rate summaries, and pairwise correlation analysis with built-in visualisations.

See [`exposure/README.md`](exposure/README.md) for full documentation.

### [`io/`](io/)

Utilities for loading and pre-processing data from various sources. Currently covers SPINE training log files.

See [`io/README.md`](io/README.md) for full documentation.

### [`plotting/`](plotting/)

Utilities for creating publication-quality figures, including PROfit-style stacked histograms, systematic uncertainty plots, SPINE training performance plots, style sheet management, and a TOML-driven plot dispatcher.

See [`plotting/README.md`](plotting/README.md) for full documentation.
