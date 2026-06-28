# fitted attributes
bin_edges_::Vector{Vector{Float64}}
n_bins_::Vector{Int}
n_features_in_::Int
feature_names_in_::Union{Vector{String}, Nothing}
_encoder::Union{Nothing, OneHotEncoderHelper}  # defined below

# inner constructor with defaults
function KBinsDiscretizer(;
    n_bins::Union{Int, Vector{Int}} = 5,
    encode::String = "onehot",
    strategy::String = "quantile",
    quantile_method::String = "linear",
    dtype::Union{Type{<:AbstractFloat}, Nothing} = nothing,
    subsample::Union{Int, Nothing} = 200_000,
    random_state::Union{Int, AbstractRNG, Nothing} = nothing,
)
    # validate basic parameters
    @assert n_bins isa Int || n_bins isa Vector{Int} "n_bins must be Int or Vector{Int}"
    @assert encode in ("onehot", "onehot-dense", "ordinal")
    @assert strategy in ("uniform", "quantile", "kmeans")
    # dtype can be Float64, Float32, or nothing
    if dtype !== nothing
        @assert dtype <: AbstractFloat "$dtype must be a subtype of AbstractFloat"
    end
    new(
        n_bins, encode, strategy, quantile_method, dtype, subsample, random_state,
        Vector{Vector{Float64}}(),  # bin_edges_ (empty initially)
        Int[],
        0,
        nothing,
        nothing,
    )
end

# Helper struct for one-hot encoding (simplified)
mutable struct OneHotEncoderHelper
    categories::Vector{Vector{Int}}
    sparse::Bool
    dtype::DataType
    function OneHotEncoderHelper(categories, sparse::Bool, dtype::DataType)
        new(categories, sparse, dtype)
    end
end

# Parameter validation
function _validate_n_bins(n_bins::Union{Int,Vector{Int}}, n_features::Int)
    if n_bins isa Int
        if n_bins < 2
            throw(ArgumentError("n_bins must be at least 2, got $n_bins"))
        end
        return fill(n_bins, n_features)
    end
    # vector case
    if length(n_bins) != n_features
        throw(ArgumentError(
            "n_bins must be a scalar or a vector of length n_features ($n_features)"
        ))
    end
    for (i, b) in enumerate(n_bins)
        if b < 2 || !(b isa Integer)
            throw(ArgumentError(
                "Invalid number of bins at index $i: $b (must be integer ≥2)"
            ))
        end
    end
    return Int.(n_bins)
end

# Resampling helper
function _resample_with_replacement(X::AbstractMatrix, n_samples::Int, rng::AbstractRNG;
    sample_weight=nothing)
    n, p = size(X)
    if sample_weight === nothing
        idx = rand(rng, 1:n, n_samples)
    else
        # weighted sampling: use weights as probabilities
        w = sample_weight ./ sum(sample_weight)
        idx = sample(rng, 1:n, StatsBase.Weights(w), n_samples; replace=true)
    end
    return X[idx, :], nothing # return weights as nothing (weights already consumed)
end

# Weighted percentile
function _weighted_percentile(x::AbstractVector, w::AbstractVector, probs::AbstractVector;
    average::Bool=false)
    # Sort by x
    ord = sortperm(x)
    xs = x[ord]
    ws = w[ord]
    cs = cumsum(ws) / sum(ws) # cumulative distribution
    # For each probability, find value
    result = similar(probs, Float64)
    for (i, p) in enumerate(probs)
        if p <= cs[1]
            result[i] = xs[1]
        elseif p >= cs[end]
            result[i] = xs[end]
        else
            # find smallest index where cs >= p
            idx = searchsortedfirst(cs, p)
            if idx == 1
                result[i] = xs[1]
            else
                if average && (cs[idx-1] < p < cs[idx])
                    # linear interpolation between xs[idx-1] and xs[idx]
                    t = (p - cs[idx-1]) / (cs[idx] - cs[idx-1])
                    result[i] = (1 - t) * xs[idx-1] + t * xs[idx]
                else
                    # closest observation or inverted_cdf
                    result[i] = xs[idx]
                end
            end
        end
    end
    return result
end

# Simple 1D k-means (Lloyd)
function _kmeans_1d(column::AbstractVector, k::Int; sample_weight=nothing, max_iter=100)
    # deterministic init: uniform spacing
    col_min, col_max = minimum(column), maximum(column)
    init_centers = range(col_min, col_max, length=k)
    centers = sort(init_centers)
    if sample_weight === nothing
        sample_weight = ones(length(column))
    end
    weights = Float64.(sample_weight)
    # Lloyd iteration
    for iter in 1:max_iter
        # assign points to nearest center
        distances = abs.(column .- centers')
        labels = argmin(distances, dims=2)[:, 1]
        # recompute centers as weighted means
        new_centers = zeros(k)
        for c in 1:k
            mask = labels .== c
            if sum(mask) == 0
                new_centers[c] = centers[c] # keep unchanged
            else
                new_centers[c] = sum(weights[mask] .* column[mask]) / sum(weights[mask])
            end
        end
        if maximum(abs, new_centers - centers) < 1e-8
            centers = new_centers
            break
        end
        centers = new_centers
    end
    sort!(centers)
    return centers
end

# Fit method
"""
fit!(model::KBinsDiscretizer, X; y=nothing, sample_weight=nothing)

Fit the discretizer to the data X.
"""
function fit!(model::KBinsDiscretizer, X::AbstractMatrix{<:Real};
    y=nothing, sample_weight=nothing)
    # basic checks
    n_samples, n_features = size(X)
    if model.dtype === nothing
        output_dtype = eltype(X)
    else
        output_dtype = model.dtype
    end

    # sample_weight validation
    if sample_weight !== nothing
        sample_weight = Float64.(vec(sample_weight))
        @assert length(sample_weight) == n_samples
    end

    # subsampling
    rng = (model.random_state isa AbstractRNG) ? model.random_state :
          (model.random_state isa Integer) ? Random.MersenneTwister(model.random_state) :
          Random.default_rng()
    if model.subsample !== nothing && n_samples > model.subsample
        X_sub, _ = _resample_with_replacement(
            X, model.subsample, rng; sample_weight=sample_weight
        )
        n_samples = model.subsample
        sample_weight = nothing
        X = X_sub
    end

    n_bins = _validate_n_bins(model.n_bins, n_features)
    bin_edges = Vector{Vector{Float64}}(undef, n_features)

    # mask for zero-weight samples (for uniform/kmeans)
    if model.strategy != "quantile" && sample_weight !== nothing
        nnz_mask = sample_weight .!= 0
    else
        nnz_mask = trues(n_samples)
    end

    for j in 1:n_features
        column = X[:, j][nnz_mask]
        col_min = minimum(column)
        col_max = maximum(column)

        if col_min ≈ col_max
            @warn "Feature $j is constant and will be replaced with 0."
            n_bins[j] = 1
            bin_edges[j] = [-Inf, Inf]
            continue
        end

        if model.strategy == "uniform"
            bin_edges[j] = collect(range(col_min, col_max, length=n_bins[j] + 1))

        elseif model.strategy == "quantile"
            percentile_levels = range(0, 100, length=n_bins[j] + 1)
            if sample_weight === nothing
                # use Julia's quantile with method mapping
                edges = _quantile(column, percentile_levels ./ 100, model.quantile_method)
                bin_edges[j] = Float64.(edges)
            else
                average = (model.quantile_method == "averaged_inverted_cdf")
                bin_edges[j] = _weighted_percentile(
                    column, sample_weight, percentile_levels ./ 100; average=average
                )
            end

        elseif model.strategy == "kmeans"
            centers = _kmeans_1d(column, n_bins[j]; sample_weight=sample_weight)
            # bin edges are midpoints between centers
            edges = (centers[1:(end-1)] .+ centers[2:end]) ./ 2
            bin_edges[j] = [col_min; edges; col_max]
        end

        # Remove too-narrow bins
        if model.strategy in ("quantile", "kmeans")
            diffs = diff(bin_edges[j])
            mask = diffs .> 1e-8
            # always keep first element
            bin_edges[j] = bin_edges[j][[true; mask]]
            if length(bin_edges[j]) - 1 != n_bins[j]
                @warn "Bins whose width are too small in feature $j removed."
                n_bins[j] = length(bin_edges[j]) - 1
            end
        end
    end

    model.bin_edges_ = bin_edges
    model.n_bins_ = n_bins
    model.n_features_in_ = n_features

    # Prepare one-hot encoder if needed
    if occursin("onehot", model.encode)
        categories = [collect(0:(b-1)) for b in n_bins]
        sparse = (model.encode == "onehot")
        model._encoder = OneHotEncoderHelper(categories, sparse, output_dtype)
    end

    return model
end

# helper: map quantile_method to Julia's quantile args
function _quantile(x, probs, method::String)
    # mapping of common methods to (alpha, beta) for Julia's quantile
    # Reference: Hyndman & Fan (1996)
    mapping = Dict(
        "linear" => (1, 1),
        "inverted_cdf" => (0, 0),
        "averaged_inverted_cdf" => (0, 1), # approx? Actually this isn't exact
        "median_unbiased" => (1//3, 1//3),
        "normal_unbiased" => (3//8, 3//8),
    )
    if haskey(mapping, method)
        alpha, beta = mapping[method]
        # Julia's quantile expects alpha, beta as Real
        return Statistics.quantile(x, probs; alpha=alpha, beta=beta)
    else
        # fallback to linear
        @warn "Quantile method '$method' not explicitly supported; using linear."
        return Statistics.quantile(x, probs) # default alpha=1, beta=1
    end
end

# Transform
"""
transform(model::KBinsDiscretizer, X)

Discretize the data X using the fitted model.
"""
function transform(model::KBinsDiscretizer, X::AbstractMatrix{<:Real})
    @assert model.n_features_in_ > 0 "Model not fitted."
    dtype = model.dtype === nothing ? eltype(X) : model.dtype
    Xt = Matrix{dtype}(X) # copy
    n_samples, n_features = size(Xt)
    @assert n_features == model.n_features_in_

    bin_edges = model.bin_edges_
    for j in 1:n_features
        # internal edges (excluding -Inf and Inf)
        edges = bin_edges[j][2:(end-1)]
        Xt[:, j] = searchsortedlast.(Ref(edges), X[:, j])  # returns 0 .. length(edges)
    end

    if model.encode == "ordinal"
        return Xt
    end

    # One-hot encoding
    return _onehotencode(Int.(Xt), model.n_bins_; sparse=model.encode=="onehot", dtype=dtype)
end

# One-hot encoding helper
function _onehotencode(X::Matrix{Int}, n_bins_per_feat::Vector{Int};
    sparse::Bool=false, dtype::DataType=Float64)
    n, p = size(X)
    total_bins = sum(n_bins_per_feat)
    if sparse
        # build sparse matrix
        I = Int[]
        J = Int[]
        V = dtype[]
        offset = 0
        for j in 1:p
            nb = n_bins_per_feat[j]
            # Code 0..(nb-1) maps to column offset+code+1? Actually code are 0..nb-1
            for i in 1:n
                code = X[i, j]
                if 0 <= code <= nb-1 # sanity
                    push!(I, i)
                    push!(J, offset + code + 1)
                    push!(V, one(dtype))
                end
            end
            offset += nb
        end
        return SparseArrays.sparse(I, J, V, n, total_bins)
    else
        # dense output
        Xout = zeros(dtype, n, total_bins)
        offset = 0
        for j in 1:p
            nb = n_bins_per_feat[j]
            for i in 1:n
                code = X[i, j]
                if 0 <= code <= nb-1
                    Xout[i, offset+code+1] = one(dtype)
                end
            end
            offset += nb
        end
        return Xout
    end
end

# Inverse transform
"""
inverse_transform(model::KBinsDiscretizer, X)

Transform discretized data back to original feature space.
"""
function inverse_transform(model::KBinsDiscretizer, X::AbstractMatrix)
    @assert model.n_features_in_ > 0 "Model not fitted."
    if model.encode in ("onehot", "onehot-dense")
        X = onehotdecode(X, model.n_bins)
    end
    Xinv = Matrix{Float64}(X) # ensure float
    n_features = model.n_features_in_
    @assert size(Xinv, 2) == n_features "Feature count mismatch."
    for j in 1:n_features
        edges = model.bin_edges_[j]
        centers = (edges[1:(end-1)] .+ edges[2:end]) ./ 2
        # Xinv codes are 0..(length(centers)-1)
        codes = round.(Int, Xinv[:, j])
        Xinv[:, j] = centers[codes .+ 1]
    end
    return Xinv
end

function _onehotdecode(X::AbstractMatrix, n_bins_per_feat::Vector{Int})
    n, total_bins = size(X)
    p = length(n_bins_per_feat)
    Xdec = zeros(Int, n, p)
    offset = 0
    for j in 1:p
        nb = n_bins_per_feat[j]
        for i in 1:n
            # find which bin column is active
            for k in 0:(nb-1)
                if X[i, offset+k+1] != 0
                    Xdec[i, j] = k
                    break
                end
            end
        end
        offset += nb
    end
    return Xdec
end

# Feature names out
function get_feature_names_out(model::KBinsDiscretizer, input_features::Union{Vector{String},Nothing}=nothing)
    @assert model.n_features_in_ > 0 "Model not fitted."
    if input_features === nothing
        feature_names = model.feature_names_in_
        if feature_names === nothing
            # generate generic names
            feature_names = ["x$i" for i in 1:model.n_features_in_]
        end
    else
        feature_names = input_features
        @assert length(feature_names) == model.n_features_in_
    end
    if model.encode in ("onehot", "onehot-dense")
        out_names = String[]
        for (j, fname) in enumerate(feature_names)
            for b in 0:(model.n_bins_[j]-1)
                push!(out_names, "$(fname)_$b")
            end
        end
        return out_names
    end
    return feature_names  # ordinal
end