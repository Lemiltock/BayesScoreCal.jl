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

# Posterior sample size
N_samples = 2000 
N_importance = 1000

# Optimisation settings
N_energy = 1000
energyβ = 1.0
vmultiplier = 1.0
alphalevels = [0.0, 0.25, 0.5, 0.9, 1.0]

# Datagen
# Initial values
μ₀ = 0
σ²₀ = 4^2

# True model
post_μ = (1/(σ²₀^(-1) + n * (σ²^(-1))))*((μ₀/(σ²₀))+ (sum(y)/σ²))
post_σ² = (1/(σ²₀^(-1) + n * σ²^(-1)))
true_samples = vrand(Normal(post_μ, post_σ²), N_samples )

@model function truemodel(y)
    μ ~ Normal(μ₀, σ²₀)
    for i in eachindex(y)
        y[i] ~ Normal(((μ/16)+(sum(y))/(4^(-2)+10)), 1/(4^(-2)+10))
    end
end


# Approx model
err_μ = vrand(Normal(0.5, 0.025^2), N_samples )
err_σ² = abs.(vrand(Normal(1.5, 0.025^2), N_samples ))
app_μ = [(post_μ - err_μ[i])/(err_σ²[i]) for i in 1:N_samples ]
app_σ² = [post_σ²/err_σ²[i] for i in 1:N_samples ]
app_samples = [rand(Normal(app_μ[i], app_σ²[i])) for i in 1:N_samples ]

# Model corrections
caldist = sample(1:length(app_samples), N_importance)
calpoints = app_samples[caldist]
