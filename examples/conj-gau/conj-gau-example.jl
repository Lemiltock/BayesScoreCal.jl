using Random
using Turing
using Turing: Variational
import DataFrames: transform as dtransform, transform! as dtransform!, DataFrame, nrow
import LinearAlgebra: PosDefException
using Distributions
using Optim
using BayesScoreCal

# Helpers
vrand(dist,N) = [rand(dist) for i in 1:N]

function rescale(x, s::Real)
    mx = mean(x)
    (s .* (x .- [mx])) .+ [mx]
end

function rescaler(s::Real)
    function(x) 
        mx = mean(x)
        (s .* (x .- [mx])) .+ [mx]
    end
end

# Data setup
n = 10 
trueμ = 1
σ² = 1
y = vrand(Normal(trueμ, σ²), n)

# Posterior sample size
N_samples = 2000 
N_adapt = 10 # M, number of calibration samples

# Optimisation settings
N_energy = 1000
energyβ = 1.0
vmultiplier = 1.0
alphalevels = [0.0, 0.25, 0.5, 0.9, 1.0]
options = Optim.Options(f_tol = 0.00001)

# Datagen
# Initial values
μ₀ = 0
σ²₀ = 4^2

# True model
post_μ = (1/(σ²₀^(-1) + n * (σ²^(-1))))*((μ₀/(σ²₀))+ (sum(y)/σ²))
post_σ² = (1/(σ²₀^(-1) + n * σ²^(-1)))
true_model = Normal(post_μ, post_σ²)
true_samples = vrand(true_model, N_samples)

# Approx model
err_μ = vrand(Normal(0.5, 0.025^2), N_samples)
err_σ² = abs.(vrand(Normal(1.5, 0.025^2), N_samples))
app_μ = [(post_μ - err_μ[i])/(err_σ²[i]) for i in 1:N_samples]
app_σ² = [post_σ²/err_σ²[i] for i in 1:N_samples]
app_samples = [rand(Normal(app_μ[i], app_σ²[i])) for i in 1:N_samples]

# Model corrections
# Get M calibration samples
caldist = sample(1:length(app_samples), N_adapt)
calpoints = app_samples[caldist]

# I think rescaler not needed as v(y) = 1?

# Find importance weights
ℓprior = [logpdf(true_model, cp) for cp in calpoints]
ℓapp = [logpdf(Normal(app_μ[i], app_σ²[i]), app_samples[i]) for i in caldist]
is_weights = ℓprior - ℓapp

# Generate new data
newy = rand.(Normal.(calpoints, [σ²]), [n])

# New approx models: pre-allocate
tr_app_samples = Matrix{Float64}(undef, N_energy, N_adapt)

# Generate samples for each calibration dataset
for t in eachindex(newy)
    # Not sure if need to resample from true model
    t_post_μ = (1/(σ²₀^(-1) + n * (σ²^(-1))))*((μ₀/(σ²₀))+ (sum(newy[t])/σ²))
    t_true_model = Normal(t_post_μ, post_σ²)
    t_true_samples = vrand(t_true_model, N_samples)
    t_err_μ = vrand(Normal(0.5, 0.025^2), N_samples)
    t_err_σ² = abs.(vrand(Normal(1.5, 0.025^2), N_samples))
    t_app_μ = [(t_post_μ - t_err_μ[i])/(t_err_σ²[i]) for i in 1:N_samples]
    t_app_σ² = [post_σ²/t_err_σ²[i] for i in 1:N_samples]
    t_app_samples = [rand(Normal(t_app_μ[i], t_app_σ²[i])) for i in 1:N_samples]

    # Find energy score
    downsampleid = sample(1:N_samples, N_energy)
    tr_app_samples[:, t] = [t_app_samples[rw] for rw in downsampleid]
end

# Temp set iter
iter = 1
dfsamples  = DataFrame[]

# Update with weights, for appropriate alpha level
for alpha in alphalevels
    trimval = quantile(is_weights, 1 - alpha)
    w = [is_weights[i] > trimval ? trimval : is_weights[i] for i in eachindex(is_weights)]
    w = exp.(w .- maximum(w))
    cal = Calibration(calpoints, tr_app_samples)
    tf = UnivariateAffine() # Energy score calibration updates this.
    res = energyscorecalibrate!(tf, cal, w; β = energyβ, options = options)
    samplecomp = DataFrame(
        samples = tf.(map(x-> x, app_samples), [mean(app_samples)]),
        method = "Adjust-post",
        iter = 1,
        alpha = alpha,
    )

    push!(dfsamples, samplecomp)
end

