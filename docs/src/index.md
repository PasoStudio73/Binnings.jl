```@meta
CurrentModule = Binnings
```

# Binnings

`Binnings.jl` is a Julia package for **discretizing numeric signals into bins**
(a.k.a. quantization).

It provides a common interface for converting continuous values into integer bin
indices using three different strategies:

- **Uniform**: equally spaced bins between the minimum and maximum values
- **Quantile**: bins based on empirical quantiles
- **Jenks**: iterative natural-breaks binning that minimizes within-bin deviation

The package is designed to work with several input layouts, including:

- scalar vectors
- tabular data (`n_samples × n_features` matrices)
- time series stored as cells of vectors
- images stored as cells of matrices
- arbitrary tensors stored as cells of N-D arrays

## Quick start

```julia
using Binnings

X = rand(Float32, 100, 4)

config = Uniform(; nbins=16)
X_bin, edges = bin(config, X)

config = Quantile(; nbins=16, type=:linear)
X_bin, edges = bin(config, X)

config = Jenks(; nbins=16, errornorm=:l1)
X_bin, edges = bin(config, X)
```

## Available algorithms

| Config | Strategy | Main parameters |
|---|---|---|
| `Uniform` | Linearly spaced edges between min and max | `nbins` |
| `Quantile` | Edges at empirical quantiles | `nbins`, `type` |
| `Jenks` | Iterative optimization of within-bin deviation | `nbins`, `maxiter`, `flux`, `errornorm` |

All configurations share a common interface through `nbins`, `max_nobs`, and `rng`.

## Output format

`bin(config, X)` returns:

- `X_bin`: binned indices as `UInt8`
- `edges`: one edge vector per feature

The original shape is preserved for multidimensional inputs.

## Documentation

- See [Algorithms](@ref algorithms) for method details.
- See the API documentation below for full type and function references.

```@index
```

```@autodocs
Modules = [Binnings]
```

## License

MIT License

## About

Developed by the [ACLAI Lab](https://aclai.unife.it/en/) at the University of Ferrara.