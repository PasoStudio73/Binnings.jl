const Float = Union{Float16,Float32,Float64}

const AlphaBetaQuantiles = Dict(
    :linear => (1.0, 1.0),
    :inverted => (0.0, 0.0),
    :average => (0.0, 1.0),
    :median => (1//3, 1//3),
    :normal => (3//8, 3//8),
    :matlab => (0.5, 0.5)
)

"""
max_nobs::Int determine how many rows to sample for the quantile estimation.
The heuristic 1000 * nbins means “allow roughly 1000 observations per bin”.
Sampling more than that rarely improves the quality of the bin edges,
so it caps the cost.
The min ensures that if the dataset is smaller than that cap,
all rows are used (no sampling losses).
"""
struct Uniform{S<:Float} <: AbstractBinningConfig
    nbins::UInt8
    max_nobs::Int
    rng::AbstractRNG

    function Uniform(;
        nbins::Int=64,
        max_nobs::Int=1000,
        rng::AbstractRNG=Xoshiro(42),
        float_type::Type{S}=Float32
    ) where {S<:Float}
        check_parameters(nbins, max_nobs)
        new{float_type}(nbins, max_nobs, rng)
    end
end

struct Quantile{S<:Float} <: AbstractBinningConfig
    nbins::UInt8
    alpha::Float16
    beta::Float16
    max_nobs::Int
    rng::AbstractRNG

    function Quantile(;
        type::Symbol=:linear,
        nbins::Int=64,
        max_nobs::Int=1000,
        rng::AbstractRNG=Xoshiro(42),
        float_type::Type{S}=Float32
    ) where {S<:Float}
        check_parameters(type)
        check_parameters(nbins, max_nobs)

        new{float_type}(nbins, AlphaBetaQuantiles[type]..., max_nobs, rng)
    end
end

function check_parameters(type::Symbol)
    @assert haskey(AlphaBetaQuantiles, type)
        "type must be one of $(keys(AlphaBetaQuantiles)), got :$type"
end

function check_parameters(nbins::Int, max_nobs::Int)
    @assert 2 ≤ nbins ≤ 255 "nbins must be in [2, 255], got $nbins"
    @assert max_nobs ≥ 1 "max_nobs must be ≥ 1, got $max_nobs"
end

get_nbins(b::AbstractBinningConfig) = b.nbins
get_max_nobs(b::AbstractBinningConfig) = b.max_nobs
get_rng(b::AbstractBinningConfig) = b.rng

get_alpha(b::Quantile) = b.alpha
get_beta(b::Quantile) = b.beta

const BinningConfig{S} = Union{Uniform{S}, Quantile{S}}


