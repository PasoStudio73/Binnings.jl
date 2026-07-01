# ---------------------------------------------------------------------------- #
#                                    types                                     #
# ---------------------------------------------------------------------------- #
const Float = Union{Float16,Float32,Float64}

"""
Mapping of quantile interpolation `type` to `(alpha, beta)` used by
`Statistics.quantile`.

Supported types:
- `:linear`   => `(1.0, 1.0)` (default)
- `:inverted` => `(0.0, 0.0)`
- `:average`  => `(0.0, 1.0)`
- `:median`   => `(1//3, 1//3)`
- `:normal`   => `(3//8, 3//8)`
- `:matlab`   => `(0.5, 0.5)`
"""
const AlphaBetaQuantiles = Dict(
    :linear => (1.0, 1.0),
    :inverted => (0.0, 0.0),
    :average => (0.0, 1.0),
    :median => (1//3, 1//3),
    :normal => (3//8, 3//8),
    :matlab => (0.5, 0.5)
)

"""
Mapping of Jenks error norms to deviation functions.

- `:l1` uses `lin_deviation` (sum of absolute deviations)
- `:l2` uses `sq_deviation` (sum of squared deviations)
"""
const JenksErrNorm = Dict(
    :l1 => lin_deviation,
    :l2 => sq_deviation
)

"""
    Uniform(; nbins=64, max_nobs=1000, rng=Xoshiro(42))
    Uniform <: AbstractBinningConfig

Uniform-width discretization config.

Fields:
- `nbins::UInt8`: number of bins (`2 ≤ nbins ≤ 255`).
- `max_nobs::Int`: sampling budget per bin used by edge estimation
  (effective cap: `max_nobs * nbins`).
- `rng::AbstractRNG`: RNG for reproducible subsampling.

Constructor keywords:
- `nbins`: number of bins.
- `max_nobs`: observations-per-bin cap factor.
- `rng`: random generator used when sampling is required.
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

"""
    Quantile(; type=:linear, nbins=64, max_nobs=1000, rng=Xoshiro(42))
    Quantile <: AbstractBinningConfig

Quantile-based discretization config.

Fields:
- `nbins::UInt8`: number of bins (`2 ≤ nbins ≤ 255`).
- `alpha::Float16`, `beta::Float16`: quantile interpolation parameters.
- `max_nobs::Int`: sampling budget per bin for quantile estimation.
- `rng::AbstractRNG`: RNG for reproducible subsampling.

Constructor keywords:
- `type`: interpolation mode key in `AlphaBetaQuantiles`.
- `nbins`: number of bins.
- `max_nobs`: observations-per-bin cap factor.
- `rng`: random generator used when sampling is required.
"""
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

"""
    Jenks(; nbins=64, maxiter=200, flux=0.1, fluxadjust=1.03,
            fluxadjust_bothways=true, errornorm=:l1, max_nobs=1000,
            rng=Xoshiro(42))
    Jenks <: AbstractBinningConfig

Jenks-style iterative discretization config.

Fields:
- `nbins::UInt8`: number of bins (`2 ≤ nbins ≤ 255`).
- `maxiter::Int`: maximum optimization iterations.
- `flux::Real`: initial boundary-shift ratio.
- `fluxadjust::Real`: multiplicative flux adaptation factor.
- `fluxadjust_bothways::Bool`: allow both increase/decrease of flux.
- `errornorm::Base.Callable`: deviation function (`lin_deviation` or `sq_deviation`).
- `max_nobs::Int`: shared-interface sampling parameter.
- `rng::AbstractRNG`: shared-interface RNG parameter.

Constructor keywords:
- `nbins`, `maxiter`, `flux`, `fluxadjust`, `fluxadjust_bothways`.
- `errornorm`: `:l1` or `:l2`.
- `max_nobs`, `rng`.
"""
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

"""
    check_parameters(nbins, max_nobs)

Validate generic binning parameters.

- `nbins` must be in `[2, 255]` (stored as `UInt8`)
- `max_nobs` must be `≥ 1`
"""
function check_parameters(nbins::Int, max_nobs::Int)
    @assert 2 ≤ nbins ≤ 255 "nbins must be in [2, 255], got $nbins"
    @assert max_nobs ≥ 1 "max_nobs must be ≥ 1, got $max_nobs"
end

"""
    check_quantiles(type)

Validate quantile interpolation type key for `Quantile`.
"""
function check_quantiles(type::Symbol)
    @assert haskey(AlphaBetaQuantiles, type)
        "type must be one of $(keys(AlphaBetaQuantiles)), got :$type"
end

"""
    check_errnorm(errornorm)

Validate Jenks error norm key (`:l1` or `:l2`).
"""
function check_errnorm(errornorm::Symbol)
    @assert haskey(JenksErrNorm, errornorm)
        "errornorm must be one of $(keys(JenksErrNorm)), got :$errornorm"
end

"Return number of bins for any binning config."
get_nbins(b::AbstractBinningConfig) = b.nbins

"Return `max_nobs` sampling budget factor."
get_max_nobs(b::AbstractBinningConfig) = b.max_nobs

"Return RNG associated with config."
get_rng(b::AbstractBinningConfig) = b.rng

"Return quantile interpolation alpha parameter."
get_alpha(b::Quantile) = b.alpha

"Return quantile interpolation beta parameter."
get_beta(b::Quantile) = b.beta

"Return maximum number of Jenks iterations."
get_maxiter(b::Jenks) = b.maxiter

"Return current Jenks flux value."
get_flux(b::Jenks) = b.flux

"Return Jenks flux adaptation factor."
get_fluxadjust(b::Jenks) = b.fluxadjust

"Return whether Jenks flux adapts in both directions."
get_fluxadjust_bothways(b::Jenks) = b.fluxadjust_bothways

"Return Jenks error function."
get_errornorm(b::Jenks) = b.errornorm

"Return Jenks initialization mode."
get_initmode(b::Jenks) = b.initmode

"Return the deviation callable used by Jenks optimization."
get_deviation(b::Jenks) = b.errornorm

"Union of all supported binning configuration types."
const BinningConfig = Union{Uniform, Quantile, Jenks}


