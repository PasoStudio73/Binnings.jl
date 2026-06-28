module Binnings

using Random: AbstractRNG, Xoshiro
using StatsBase: sample
using Statistics: quantile

# abstract type AbstractBinning end
abstract type AbstractBinningConfig end

# abstract type AbstractAlphaBetaConfig <: AbstractBinningConfig end

# export Uniform, SampledQuantile, LinearQuantile, InvertedQuantile,
#     AvgInvertedQuantile, MedianUnbiasedQuantile, NormalUnbiasedQuantile
export Uniform, Quantile
include("structs.jl")

# export bin
export bin
include("binning.jl")

end
