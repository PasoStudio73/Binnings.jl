# ---------------------------------------------------------------------------- #
#                                    utils                                     #
# ---------------------------------------------------------------------------- #
"""
    get_idxs(x, max_nobs, nbins, rng)

Return observation indices used to estimate bin edges.

If `length(x) > max_nobs * nbins`, a reproducible, ordered sample without
replacement is drawn using `rng`. Otherwise, all indices are returned.

This limits edge-estimation cost on large datasets while preserving input-order
indexing.
"""
function get_idxs(
    x::AbstractVector{T},
    max_nobs::Int,
    nbins::UInt8,
    rng::AbstractRNG
) where {T<:Real}
    nsamples = length(x)
    nobs = min(nsamples, max_nobs * nbins)
    return nobs < nsamples ?
        sample(rng, 1:nsamples, nobs, replace=false, ordered=true) :
        collect(1:nsamples)
end

# ---------------------------------------------------------------------------- #
#                                     bin                                      #
# ---------------------------------------------------------------------------- #
"""
    bin(config::Uniform, x)

Discretize a numeric vector `x` into uniformly spaced bins.

Edges are linearly spaced between `minimum(x)` and `maximum(x)`.
Returns:

- `x_bin::Vector{UInt8}`: 1-based bin index for each value in `x`
- `edges::Vector`: bin edge values used for discretization
"""
function bin(config::Uniform, x::AbstractVector{T}) where {T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)

    idxs = get_idxs(x, max_nobs, nbins, rng)

    edges = collect(range(minimum(x), maximum(x); length=nbins))
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = UInt8.(searchsortedfirst.(Ref(edges), x))

    return x_bin, edges
end

"""
    bin(config::Quantile, x)

Discretize a numeric vector `x` using quantile-based bins.

Internal edges are computed from quantiles of sampled observations (`get_idxs`),
with interpolation controlled by `alpha` and `beta` from `config`.
Returns `(x_bin, edges)` where `x_bin` contains 1-based bin indices.
"""
function bin(config::Quantile, x::AbstractVector{T}) where {T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)
    alpha, beta = get_alpha(config), get_beta(config)

    idxs = get_idxs(x, max_nobs, nbins, rng)

    edges = T.(quantile(view(x, idxs), (1:nbins-1) / nbins; alpha, beta))
    edges = vcat(minimum(x), edges)
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = UInt8.(searchsortedfirst.(Ref(edges), x))

    return x_bin, edges
end

"""
    bin(config::Jenks, x)

Discretize a numeric vector `x` with an iterative Jenks-style optimization.

The algorithm adjusts class breaks to reduce within-bin deviation using the
configured deviation function and flux parameters. Returns `(x_bin, edges)`,
where `x_bin` are 1-based bin indices and `edges` are learned break values.
"""
function bin(config::Jenks, x::AbstractVector{T}) where {T<:Real}
    nbins, maxiter = get_nbins(config), get_maxiter(config)
    fluxadjust_bothways = get_fluxadjust_bothways(config)
    fluxadjust = get_fluxadjust(config)
    deviation = get_deviation(config)

    _x = sort(x)
    ndata = length(_x)

    np = ndata ÷ nbins
    breaks = [1:np:np*nbins; ndata+1]
    
    fluxes = fill(get_flux(config),nbins-1)
    devs = Vector(undef,nbins)
    devs_pre = Vector(undef,nbins)

    for iter in 1:maxiter
        devs_pre .= devs

        for iclass in 1:nbins
            devs[iclass] =
                deviation(
                    @view(_x[breaks[iclass]:breaks[iclass+1]-1])
                )
        end

        for iclass in 1:nbins-1
            a, b, c = breaks[iclass:iclass+2]
            if devs[iclass] < devs[iclass+1]
                if iter > 1
                    if devs_pre[iclass] >= devs_pre[iclass+1]
                        fluxes[iclass] /= fluxadjust
                    elseif fluxadjust_bothways
                        fluxes[iclass] *= fluxadjust
                    end
                end

                cs_factor = Int(round((b-a)*fluxes[iclass]))
                newbreak = breaks[iclass+1]+cs_factor
                breaks[iclass+1] = min(newbreak,breaks[iclass+2]-2)
            else

                if iter > 1
                    if devs_pre[iclass] <= devs_pre[iclass+1]
                        fluxes[iclass] /= fluxadjust
                    elseif fluxadjust_bothways
                        fluxes[iclass] *= fluxadjust
                    end
                end

                cs_factor = Int(round((c-b)*fluxes[iclass]))
                newbreak = breaks[iclass+1]-cs_factor
                breaks[iclass+1] = max(newbreak,breaks[iclass]+2)
            end
        end
    end

    breaks[end] = breaks[end] - 1
    edges = _x[breaks][1:end-1]
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = UInt8.(searchsortedlast.(Ref(edges), x))

    return x_bin, edges
end

"""
    bin(config, X::AbstractArray{T})

Feature-wise binning for tabular numeric data (`n_samples × n_features`).

Each column is binned independently (threaded), returning:

- `X_bin::Vector{Vector{UInt8}}`: one binned vector per feature
- `edges::Vector{Vector}`: one edge vector per feature
"""
function bin(
    config::BinningConfig,
    X::AbstractArray{T}
) where {T<:Real}
    nfeats = size(X, 2)
    edges = Vector{Vector}(undef, nfeats)
    X_bin = Vector{Vector{UInt8}}(undef, nfeats)

    Threads.@threads for j in 1:nfeats
        X_bin[j], edges[j] = bin(config, view(X, :, j))
    end

    return X_bin, edges
end

"""
    bin(config, X::Matrix{<:AbstractArray{T}})

Binning for datasets where each cell is a multidimensional item
(e.g., time series vectors, images, or tensors).

For each column/feature:
1. all per-row items are flattened and concatenated,
2. bin edges are learned once on the flattened values,
3. binned values are reshaped back to each original item shape.

Returns:

- binned data with original `(nrows, ncols)` structure and per-item shapes
  preserved.
- `edges::Vector{Vector}` with one edge vector per column.
"""
function bin(
    config::BinningConfig,
    X::Matrix{<:AbstractArray{T}}
) where {T<:Real}
    nrows, ncols = size(X)
    el_shape = size(first(X))
    el_len = prod(el_shape)

    bins = Vector{Tuple{Vector{UInt8}, Vector}}(undef, ncols)
    Threads.@threads for j in 1:ncols
        flat = vec(stack(vec, view(X, :, j)))
        bins[j] = bin(config, flat)
    end

    edges = last.(bins)

    X_bin = map(1:ncols) do j
        flat_bin = first(bins[j])
        arr = reshape(flat_bin, el_shape..., nrows)
        [copy(selectdim(arr, ndims(arr), i)) for i in 1:nrows]
    end

    return reshape(reduce(hcat, X_bin), nrows, ncols), edges
end
