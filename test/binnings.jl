using Test
using Binnings

using DataFrames, MLJ, Random

# ---------------------------------------------------------------------------- #
#                                load dataset                                  #
# ---------------------------------------------------------------------------- #
Xc, yc = MLJ.@load_iris
Xc = Matrix(DataFrame(Xc))

config = Binnings.Uniform()
X_bin, edges = bin(config, Matrix(Xc))

config = Binnings.Quantile()
X_bin, edges = bin(config, Matrix(Xc))

# using Discretizers

# a=DiscretizeUniformWidth(16)
# b=DiscretizeUniformCount(16)
# c=DiscretizeQuantile(16)
# d=DiscretizeBayesianBlocks()

# da = LinearDiscretizer(binedges(a, Xc[:,1]))
# db = LinearDiscretizer(binedges(b, Xc[:,1]))
# dc = LinearDiscretizer(binedges(c, Xc[:,1]))
# dd = LinearDiscretizer(binedges(d, Xc[:,1]))



expected_uniform = [
    12 10 5 1;
    9 3 5 1;
    6 6 1 1;
    4 4 9 1;
    10 12 5 1;
    16 16 16 16;
    4 9 5 9;
    10 9 9 1;
    1 1 5 1
]

config = Binnings.Uniform(; nbins)
X_bin, edges = bin(config, X)
@test Int.(X_bin) == expected_uniform





# KBinsDiscretizer(encode='ordinal', n_bins=16, strategy='quantile')
# [[5. 5. 1. 0.]
#  [3. 1. 1. 0.]
#  [2. 3. 0. 0.]
#  [1. 2. 2. 0.]
#  [4. 6. 1. 0.]
#  [5. 6. 2. 1.]
#  [1. 4. 1. 1.]
#  [4. 4. 2. 0.]
#  [0. 0. 1. 0.]]

config = Binnings.LinearQuantile(; nbins)
X_bin, edges = bin(config, X)
@show Int.(X_bin)

config = Binnings.InvertedQuantile(; nbins)
X_bin, edges = bin(config, X)
@show Int.(X_bin)

config = Binnings.AvgInvertedQuantile(; nbins)
X_bin, edges = bin(config, X)
@show Int.(X_bin)

config = Binnings.MedianUnbiasedQuantile(; nbins)
X_bin, edges = bin(config, X)
@show Int.(X_bin)

config = Binnings.NormalUnbiasedQuantile(; nbins)
X_bin, edges = bin(config, X)
@show Int.(X_bin)



