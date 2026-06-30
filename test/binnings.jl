using Test
using Binnings

using DataFrames, Random
using RDatasets

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
iris = dataset("datasets", "iris")
Xc = Matrix(select(iris, Not(:Species)))

config = Binnings.Uniform(;nbins=16)
X_bin, edges = bin(config, Xc)

config = Binnings.Quantile(;nbins=16)
X_bin, edges = bin(config, Xc)

config = Binnings.Jenks(;nbins=16)
X_bin, edges = bin(config, Xc)
