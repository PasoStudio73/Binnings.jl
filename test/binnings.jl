using Test
using Binnings

using SoleData
using DataFrames, StableRNGs, MLJ

# ---------------------------------------------------------------------------- #
#                                   tabular                                    #
# ---------------------------------------------------------------------------- #
Xc, yc = @load_iris
Xc = Matrix(DataFrame(Xc))

config = Binnings.Uniform(;nbins=16)
X_bin, edges = bin(config, Xc)

config = Binnings.Quantile(;nbins=16)
X_bin, edges = bin(config, Xc)

config = Binnings.Jenks(;nbins=16)
X_bin, edges = bin(config, Xc)

# ---------------------------------------------------------------------------- #
#                                time series                                   #
# ---------------------------------------------------------------------------- #
natopsloader = SoleData.Artifacts.NatopsLoader()
Xts, yts = SoleData.Artifacts.load(natopsloader)
Xts = Matrix(Xts)

config = Binnings.Uniform(;nbins=32)
X_bin, edges = bin(config, Xts)

config = Binnings.Quantile(;nbins=32)
X_bin, edges = bin(config, Xts)

config = Binnings.Jenks(;nbins=32)
X_bin, edges = bin(config, Xts)

# ---------------------------------------------------------------------------- #
#                                   images                                     #
# ---------------------------------------------------------------------------- #
rng = StableRNG(42)
X = [round.(rand(rng, Float32, 2, 2); digits=2) for i in 1:3, j in 1:2]

config = Binnings.Uniform(;nbins=3)
X_bin, edges = bin(config, X)

config = Binnings.Quantile(;nbins=3)
X_bin, edges = bin(config, X)

config = Binnings.Jenks(;nbins=3)
X_bin, edges = bin(config, X)

# ---------------------------------------------------------------------------- #
#                             multi dimensional                                #
# ---------------------------------------------------------------------------- #
rng = StableRNG(42)
X4 = [round.(rand(rng, Float32, 3, 2, 2); digits=2) for i in 1:3, j in 1:2]

config = Binnings.Uniform(;nbins=3)
X_bin, edges = bin(config, X4)

config = Binnings.Quantile(;nbins=3)
X_bin, edges = bin(config, X4)

config = Binnings.Jenks(;nbins=3)
X_bin, edges = bin(config, X4)
