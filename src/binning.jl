# function bin(config::Uniform, X::Matrix{T}) where {T<:Real}
#     nbins, max_nobs =
#         get_nbins(config), get_max_nobs(config)

#     nsamples, nfeats = size(X)
#     nobs = min(nsamples, max_nobs * nbins)
#     idxs = nobs < nsamples ?
#         sample(rng, 1:nsamples, nobs, replace=false, ordered=true) :
#         collect(1:nsamples)

#     # edges = Vector{Vector{T}}(undef, nfeats)
#     edges = Vector{Vector{T}}(undef, nfeats)
#     # featbins = Vector{UInt8}(undef, nfeats)
#     # feattypes = trues(nfeats) # forse non serve, sono tutti uni
#     X_bin = Matrix{UInt8}(undef, nsamples, nfeats)

#     Threads.@threads for j in 1:nfeats
#         col_min = minimum(X[:, j])
#         col_max = maximum(X[:, j])
#         edges[j] = collect(range(col_min, col_max, length=nbins))
#         length(edges[j]) == 1 && (edges[j] = [minimum(view(X, idxs, j))])
#         # featbins[j] = length(edges[j]) + 1
#         X_bin[:, j] .= searchsortedfirst.(Ref(edges[j]), view(X, :, j))
#     end

#     # return edges, featbins, feattypes
#     return X_bin, edges
# end

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

function bin(config::Uniform{S}, x::AbstractVector{T}) where {S<:Float,T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)

    idxs = get_idxs(x, max_nobs, nbins, rng)

    edges = collect(range(minimum(x), maximum(x); length=nbins))
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = searchsortedfirst.(Ref(edges), x)

    return S.(edges), x_bin
end

function bin(config::Quantile{S}, x::AbstractVector{T}) where {S<:Float,T<:Real}
    nbins, max_nobs, rng =
        get_nbins(config), get_max_nobs(config), get_rng(config)
    alpha, beta = get_alpha(config), get_beta(config)

    idxs = get_idxs(x, max_nobs, nbins, rng)

    edges = quantile(view(x, idxs), (1:nbins-1) / nbins; alpha, beta)
    length(edges) == 1 && (edges = [minimum(view(x, idxs))])
    x_bin = searchsortedfirst.(Ref(edges), x)

    return S.(edges), x_bin
end

function bin(
    config::BinningConfig{S},
    X::AbstractArray{T}
) where {S<:Float,T<:Real}
    nfeats = size(X, 2)
    edges = Vector{Vector{S}}(undef, nfeats)
    X_bin = Vector{Vector{UInt8}}(undef, nfeats)

    Threads.@threads for j in 1:nfeats
        edges[j], X_bin[j] = bin(config, view(X, :, j))
    end

    return X_bin, edges
end

function bin(
    config::BinningConfig{S},
    X::Matrix{<:AbstractArray{T}}
) where {S<:Float,T<:Real}

end