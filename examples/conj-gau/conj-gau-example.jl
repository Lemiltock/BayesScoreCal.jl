using Random
using Turing
using Turing: Variational
import DataFrames: transform as dtransform, transform! as dtransform!, DataFrame, nrow
import LinearAlgebra: PosDefException
using Optim
using BayesScoreCal

# Helpers
vrand(dist,N) = [rand(dist) for i in 1:N]

# Data setup
n = 10
trueμ = 1
σ² = 1
y = vrand(Normal(trueμ, σ²), n)


# Datagen
postμ(μ₀, σ₀, y) = (1/(σ₀^(-2) + n * (σ²^(-2))))*((μ₀/(σ₀^2))+ (sum(y)/σ²))   # Cant get sub and  super right?
