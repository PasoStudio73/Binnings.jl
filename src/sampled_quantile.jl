# Binarization: histogram-based gradient boosting
# This is the technique used by LightGBM and EvoTrees
# Key benefits:
# 1 - Comparing UInt8 is faster than floats during split search
# 2 - Instead of searching over all unique float values,
#     only nbins thresholds need to be evaluated

# Reference:
# https://apxml.com/courses/julia-for-machine-learning/chapter-2-julia-data-manipulation-preparation/julia-data-transformation
# https://github.com/milankl/Jenks.jl
# https://medium.com/@adnan.mazraeh1993/comprehensive-guide-to-binning-discretization-in-data-science-from-basics-to-super-advanced-006c2e215a9f

# https://github.com/sisl/Discretizers.jl
# https://github.com/myersm0/SymbolicApproximators.jl
# https://bkamins.github.io/julialang/2020/12/11/binning.html
# https://github.com/carstenbauer/BinningAnalysis.jl
# https://github.com/kirklong/BinnedStatistics.jl

"""
    sampled_quantile(X::AbstractMatrix{T}; feature_names, nbins, rng=Random.TaskLocalRNG()) where {T}
    sampled_quantile(df; feature_names, nbins, rng=Random.TaskLocalRNG())

Get the histogram breaking points of the feature data and
Transform feature data into a UInt8 sampled_quantiled matrix.
"""
function sampled_quantile(X::Matrix{T}; nbins::Int, rng::AbstractRNG) where {T<:Real}
    nrows, nfeats = size(X)
    nobs = min(nrows, 1000 * nbins)
    idx = sample(rng, 1:nrows, nobs, replace=false, ordered=true)

    # edges = Vector{Vector{T}}(undef, nfeats)
    edges = Vector{Vector{T}}(undef, nfeats)
    # featbins = Vector{UInt8}(undef, nfeats)
    # feattypes = trues(nfeats) # forse non serve, sono tutti uni
    x_bin = Matrix{UInt8}(undef, nrows, nfeats)

    Threads.@threads for j in 1:nfeats
        edges[j] = quantile(view(X, idx, j), (1:nbins-1) / nbins)
        length(edges[j]) == 1 && (edges[j] = [minimum(view(X, idx, j))])
        # featbins[j] = length(edges[j]) + 1
        x_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
    end

    # return edges, featbins, feattypes
    return edges, x_bin
end

# Here's a comprehensive overview of discretization/binning methods for float datasets and their Julia implementations:

# ## 1. Equal‑Width Binning
# Divides the data range into intervals of equal size.
# - **Use when:** Data is roughly uniformly distributed; simple baseline.
# - **Julia packages:**
#   - `Discretizers.jl` — `KBinsDiscretizer` with `strategy=:uniform`
#   - Base Julia: `cut(X, nbins)` (not built‑in; you'd roll your own)

# ## 2. Equal‑Frequency (Quantile) Binning
# Each bin contains approximately the same number of observations.
# - **Use when:** Data is skewed; you want balanced histograms.
# - **Julia packages:**
#   - `Discretizers.jl` — `KBinsDiscretizer` with `strategy=:quantile`
#   - `BinningAnalysis.jl` — for time‑series binning
#   - `BinnedStatistics.jl` — binned stats with user‑defined edges

# ## 3. Fisher‑Jenks Natural Breaks
# Minimizes within‑class variance and maximizes between‑class variance (1D k‑means).
# - **Use when:** You need meaningful, data‑driven class breaks (e.g., choropleth maps).
# - **Julia packages:**
#   - `Jenks.jl` — dedicated, high‑performance Fisher‑Jenks implementation

# ## 4. K‑Means Clustering
# Applies k‑means to 1D data to find natural cluster boundaries.
# - **Use when:** Data has clear modes/clusters.
# - **Julia packages:**
#   - `Clustering.jl` — `kmeans(data', k)` and extract sorted centroids as bin edges
#   - `Discretizers.jl` — `strategy=:kmeans`

# ## 5. Entropy‑Based / MDLP (Supervised)
# Uses the target variable to find splits that minimize class entropy (Minimum Description Length Principle).
# - **Use when:** You have a labelled target and want predictive bins (classification).
# - **Julia packages:**
#   - `Discretizers.jl` — `MDLPDiscRes` and `CAIMDiscRes`
#   - DecisionTree.jl — internal supervised splitting logic (usable for pre‑binning with adapters)

# ## 6. Custom / Domain‑Driven
# User‑supplied breakpoints based on domain knowledge (e.g., tax brackets, medical thresholds).
# - **Julia packages:** Base Julia with `searchsortedfirst` / `searchsortedlast` + a vector of edges.

# ## 7. Symbolic Aggregate approXimation (SAX)
# Binning for time series: PAA (Piecewise Aggregate Approximation) + equal‑frequency discretisation assuming a Gaussian distribution.
# - **Julia packages:**
#   - `SymbolicApproximators.jl`
#   - `TimeseriesTools.jl` (optional binning utilities)

# ## 8. Adaptive / Recursive Partitioning
# Recursively split regions with high variance (regression trees without labels).
# - **Julia packages:** No dedicated package; easily built with `Clustering.jl` or tree‑recursion.

# ---

# ### Quick Summary Table

# | Method | Julia Package | Key Function |
# |--------|---------------|--------------|
# | Equal‑Width | `Discretizers.jl` | `KBinsDiscretizer(n_bins, :uniform)` |
# | Equal‑Frequency | `Discretizers.jl` | `KBinsDiscretizer(n_bins, :quantile)` |
# | Fisher‑Jenks | `Jenks.jl` | `classify_jenks(data, k)` |
# | K‑Means | `Clustering.jl` | `kmeans(reshape(data,1,:), k)` |
# | Entropy / MDLP | `Discretizers.jl` | `un_supervised_discretization` or `mdlp_disc` |
# | SAX | `SymbolicApproximators.jl` | `sax(...)` |

# For your histogram‑based gradient boosting use case, **equal‑frequency (quantile)** is the standard choice because it balances bin cardinality, which is critical for stable split gain computations. `Discretizers.jl` provides a clean, tested implementation.