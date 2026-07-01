```@meta
CurrentModule = Binnings
```

# [Algorithms](@id algorithms)

`Binnings.jl` provides three discretization algorithms, each implementing a different
strategy for mapping continuous numeric values into a fixed number of integer bins.
All algorithms share a common interface: they accept a configuration struct and return
`(X_bin, edges)`, where `X_bin` contains `UInt8` bin indices and `edges` contains the
bin boundaries per feature.

---

## Uniform

**Config:** `Uniform(; nbins=64, max_nobs=1000, rng=Xoshiro(42))`

The uniform algorithm places bin edges at **linearly spaced intervals** between the
minimum and maximum observed values of each feature. It is the simplest and fastest
method.

### Steps

1. **Subsampling** (optional): if the number of observations exceeds
   `max_nobs × nbins`, a random subsample is drawn using `rng` to estimate the range.
2. **Edge computation**: `nbins - 1` edges are placed uniformly between `min` and `max`.
3. **Binning**: each value is mapped to a bin index via `searchsortedfirst` on the
   edge vector.

### Properties

- **O(n)** complexity per feature.
- Sensitive to outliers, since the range `[min, max]` drives edge placement.
- Best suited for approximately uniformly distributed features.

---

## Quantile

**Config:** `Quantile(; type=:linear, nbins=64, max_nobs=1000, rng=Xoshiro(42))`

The quantile algorithm places bin edges at **empirical quantile positions**, so that
each bin contains approximately the same number of observations.

### Steps

1. **Subsampling** (optional): as in `Uniform`.
2. **Quantile estimation**: `nbins - 1` evenly spaced quantile levels in `(0, 1)` are
   computed using `Statistics.quantile` with interpolation parameters `(alpha, beta)`
   derived from the chosen `type`.
3. **Deduplication**: duplicate edges (arising from discrete or heavily tied data) are
   removed, so the actual number of bins may be less than `nbins`.
4. **Binning**: same as `Uniform`.

### Interpolation types

| `type`      | `(alpha, beta)` | Notes                        |
|-------------|-----------------|------------------------------|
| `:linear`   | `(1.0, 1.0)`    | Default, R type 7            |
| `:inverted` | `(0.0, 0.0)`    | R type 1                     |
| `:average`  | `(0.0, 1.0)`    | Averaged                     |
| `:median`   | `(1/3, 1/3)`    | Median-unbiased, R type 8    |
| `:normal`   | `(3/8, 3/8)`    | Normal-unbiased, R type 9    |
| `:matlab`   | `(0.5, 0.5)`    | MATLAB / SciPy default       |

### Properties

- Robust to outliers and skewed distributions.
- Guarantees approximately equal bin population (equi-frequency binning).
- Slightly more expensive than `Uniform` due to sorting.

---

## Jenks

**Config:** `Jenks(; nbins=64, maxiter=200, flux=0.1, fluxadjust=1.03, fluxadjust_bothways=true, errornorm=:l1, max_nobs=1000, rng=Xoshiro(42))`

The Jenks algorithm is an **iterative optimization** method inspired by Jenks natural
breaks. It minimizes the total within-bin deviation by repeatedly shifting bin
boundaries.

### Steps

1. **Initialization**: bin edges are initialized (e.g., using quantile positions).
2. **Iterative optimization**: at each iteration, each internal boundary is shifted
   by a fraction `flux` of the local inter-edge spacing. The shift is accepted if it
   reduces the total within-bin error according to the chosen `errornorm`.
3. **Flux adaptation**: after each iteration, `flux` is multiplied by `fluxadjust`
   if the error improved, or divided by `fluxadjust` if
   `fluxadjust_bothways = true`, allowing the step size to grow or shrink adaptively.
4. **Termination**: the loop stops after `maxiter` iterations or when no boundary
   move reduces the error.
5. **Binning**: same as `Uniform`.

### Error norms

| `errornorm` | Function           | Formula                              |
|-------------|--------------------|--------------------------------------|
| `:l1`       | `lin_deviation`    | ``\sum_i |x_i - \bar{x}|``          |
| `:l2`       | `sq_deviation`     | ``\sum_i (x_i - \bar{x})^2``        |

### Properties

- Produces **natural breaks**: edges align with gaps in the data distribution.
- More expensive than `Uniform` and `Quantile` — scales with `maxiter`.
- `:l1` norm is more robust to outliers; `:l2` penalizes large deviations more.
- Flux adaptation allows fine-grained convergence without hand-tuning a fixed step.

---

## Comparison summary

| Property              | `Uniform`   | `Quantile`      | `Jenks`              |
|-----------------------|-------------|-----------------|----------------------|
| Edge placement        | Equi-width  | Equi-frequency  | Data-adaptive        |
| Outlier sensitivity   | High        | Low             | Low–Medium           |
| Computational cost    | O(n)        | O(n log n)      | O(n · maxiter)       |
| Handles skewed data   | Poor        | Good            | Good                 |
| Aligns with data gaps | No          | No              | Yes                  |