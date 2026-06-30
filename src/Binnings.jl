module Binnings

using Random: AbstractRNG, Xoshiro
using StatsBase: mean, sample
using Statistics: quantile

abstract type AbstractBinningConfig end

function lin_deviation(x::AbstractVector{T}) where {T<:Real}
    μ = mean(x)
    return sum(xi -> abs(xi - μ), x)
end

function sq_deviation(x::AbstractVector{T}) where {T<:Real}
    μ = mean(x)
    return sum(xi -> (xi - μ)^2, x) 
end

export Uniform, Quantile, Jenks
include("structs.jl")

# export bin
export bin
include("binning.jl")

end
