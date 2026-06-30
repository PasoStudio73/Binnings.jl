const Float = Union{Float16,Float32,Float64}

const AlphaBetaQuantiles = Dict(
    :linear => (1.0, 1.0),
    :inverted => (0.0, 0.0),
    :average => (0.0, 1.0),
    :median => (1//3, 1//3),
    :normal => (3//8, 3//8),
    :matlab => (0.5, 0.5)
)

const JenksErrNorm = Dict(
    :l1 => lin_deviation,
    :l2 => sq_deviation
)

"""
max_nobs::Int determine how many rows to sample for the quantile estimation.
The heuristic 1000 * nbins means “allow roughly 1000 observations per bin”.
Sampling more than that rarely improves the quality of the bin edges,
so it caps the cost.
The min ensures that if the dataset is smaller than that cap,
all rows are used (no sampling losses).
"""
struct Uniform <: AbstractBinningConfig
    nbins::UInt8
    max_nobs::Int
    rng::AbstractRNG

    function Uniform(;
        nbins::Int=64,
        max_nobs::Int=1000,
        rng::AbstractRNG=Xoshiro(42),
    )
        check_parameters(nbins, max_nobs)
        new(nbins, max_nobs, rng)
    end
end

struct Quantile <: AbstractBinningConfig
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
    )
        check_parameters(nbins, max_nobs)
        check_quantiles(type)

        new(nbins, AlphaBetaQuantiles[type]..., max_nobs, rng)
    end
end

struct Jenks <: AbstractBinningConfig
    nbins::UInt8
    maxiter::Int
    flux::Real
    fluxadjust::Real
    fluxadjust_bothways::Bool
    errornorm::Base.Callable
    max_nobs::Int
    rng::AbstractRNG

    function Jenks(;
        nbins::Int=64,
        maxiter::Int=200,
        flux::Real=0.1,
        fluxadjust::Real=1.03,
        fluxadjust_bothways::Bool=true,
        errornorm::Symbol=:l1,
        max_nobs::Int=1000,
        rng::AbstractRNG=Xoshiro(42),
    )
        check_parameters(nbins, max_nobs)
        check_errnorm(errornorm)

        new(
            nbins,
            maxiter,
            flux,
            fluxadjust,
            fluxadjust_bothways,
            JenksErrNorm[errornorm],
            max_nobs,
            rng
        )
    end
end

function check_parameters(nbins::Int, max_nobs::Int)
    @assert 2 ≤ nbins ≤ 255 "nbins must be in [2, 255], got $nbins"
    @assert max_nobs ≥ 1 "max_nobs must be ≥ 1, got $max_nobs"
end

function check_quantiles(type::Symbol)
    @assert haskey(AlphaBetaQuantiles, type)
        "type must be one of $(keys(AlphaBetaQuantiles)), got :$type"
end

function check_errnorm(errornorm::Symbol)
    @assert haskey(JenksErrNorm, errornorm)
        "errornorm must be one of $(keys(JenksErrNorm)), got :$errornorm"
end

get_nbins(b::AbstractBinningConfig) = b.nbins
get_max_nobs(b::AbstractBinningConfig) = b.max_nobs
get_rng(b::AbstractBinningConfig) = b.rng

get_alpha(b::Quantile) = b.alpha
get_beta(b::Quantile) = b.beta

get_maxiter(b::Jenks) = b.maxiter
get_flux(b::Jenks) = b.flux
get_fluxadjust(b::Jenks) = b.fluxadjust
get_fluxadjust_bothways(b::Jenks) = b.fluxadjust_bothways
get_errornorm(b::Jenks) = b.errornorm
get_initmode(b::Jenks) = b.initmode
get_deviation(b::Jenks) = b.errornorm

const BinningConfig = Union{Uniform, Quantile, Jenks}


