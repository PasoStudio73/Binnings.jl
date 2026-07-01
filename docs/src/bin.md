```@meta
CurrentModule = Binnings
```

# Binning functions

This page documents the `bin` methods provided by `Binnings.jl`.

`bin` converts numeric data into integer bin indices and returns a tuple of:

- `X_bin`: binned values as `UInt8`
- `edges`: learned bin boundaries

The exact return shape depends on the input layout:

- `AbstractVector{<:Real}`: a binned vector and one edge vector
- `AbstractArray{<:Real}`: one binned vector per feature/column
- `Matrix{<:AbstractArray}`: preserves the original cell layout for time series,
  images, and tensors

## Common behavior

All binning methods follow the same general pattern:

1. select a subset of observations when edge estimation would be too expensive,
2. learn bin boundaries from the selected values,
3. map each observation to its bin index.

The `max_nobs` and `rng` fields of the configuration control subsampling.

## Vector binning

### Uniform binning

Uniform binning uses linearly spaced edges between the minimum and maximum value.

```@docs
bin(::Uniform, ::AbstractVector{<:Real})
```

### Quantile binning

Quantile binning places edges at empirical quantile positions.

```@docs
bin(::Quantile, ::AbstractVector{<:Real})
```

### Jenks binning

Jenks binning iteratively adjusts class breaks to reduce within-bin deviation.

```@docs
bin(::Jenks, ::AbstractVector{<:Real})
```

## Tabular data

For matrices of numeric values, each column is binned independently.

```@docs
bin(::BinningConfig, ::AbstractArray{<:Real})
```

get_idxs
```