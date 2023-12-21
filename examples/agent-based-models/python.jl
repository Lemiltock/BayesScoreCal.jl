using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using Turing
using DataFrames
using StatsPlots
using CSV
using Agents
using InteractiveDynamics
#using CairoMakie
using Optim
using BayesScoreCal


N_ind = 1000
N_adapt = 1000
N_samples = 100


is_weights = ones(N_samples)
energyβ = 1.0
vmultiplier = 1.0
# Setup rescale helper
function multiplyscale(x::Matrix{Vector{Float64}}, scale::Float64) 
    μ = mean(x)
    scale .* (x .- [μ]) .+ [μ]
end

# Setup logit /inv logit helper
function logit(p::Float64)
    return log(p/(1-p))
end

function invlogit(p::Float64)
    return 1/(1+exp(-p))
end


options = Optim.Options(f_tol = 0.00001)
tspan = (0,60)
u0 = [N_ind*0.99, N_ind*0.01, 0, 0]

# Setup SIR ODE
function sir_ode!(du,u,p,t)
    (S,I,R,C) = u
    (β,c,γ) = p
    N = S+I+R
    infection = β*c*I/N*S
    recovery = γ*I
    @inbounds begin
        du[1] = -infection
        du[2] = infection - recovery
        du[3] = recovery
        du[4] = infection
    end
nothing
end;

ODEprob = ODEProblem(sir_ode!,
              u0,
              tspan,
              [0.25, 1.0, 0.35])


@model bayes_sir(y) = begin
  # Calculate number of timepoints
  l = length(y)
  γ ~ Uniform(0.0,1.0)
  β ~ Uniform(0.0,1.0)
  I = 0.01*Float64(N_ind)
  u0=[Float64(N_ind)-I,I,0.0,0.0]
  #p=[β,10.0,0.25]
  p=[β,1.0,γ]
  #tspan = (0.0,float(l))
  prob = ODEProblem(sir_ode!,
              u0,
              tspan,
              p) #[0.05, 10.0, 0.25])
#remake(ODEprob, u=u0, p=p)
  sol = solve(prob,
              Tsit5(),
              saveat = 1.0)
  sol_X = Array(sol)[2,:] # Infections
  #sol_C = Array(sol)[4,:] # Cumulative cases
  #sol_X = sol_C[2:end] - sol_C[1:(end-1)]
  l = length(y)
  for i in 1:l
    y[i] ~ Poisson(sol_X[i])
  end
end;

y = CSV.read("traindata.csv", DataFrame, header=false)
Z = y[:,1]

sir_bayes_model = bayes_sir(Z)
next = false
tmp_ode_nuts = Chains{}
while next == false
    try
        tmp_ode_nuts = sample(sir_bayes_model,NUTS(),N_samples);
        next = true
    catch e
        next = false
    end
end

app_samples_γ = tmp_ode_nuts[:γ]
app_samples_β = tmp_ode_nuts[:β]

CSV.write("gamma.csv", Tables.table(app_samples_γ), writeheader=false)
CSV.write("beta.csv", Tables.table(app_samples_β), writeheader=false)


# Read in post samples and new data make sure to transpose testvals first
post_samp = CSV.read("testdata.csv", DataFrame, header=false)
post_γ = post_samp[:, 1]
post_β = post_samp[:, 2]

test_data = CSV.read("testvals.csv", DataFrame, header=false)


tr_app_samples_γ = Matrix{Float64}(undef, N_adapt, N_samples)
tr_app_samples_β = Matrix{Float64}(undef, N_adapt, N_samples)
for t in 1:N_samples
    Z = test_data[:, t]
    sir_bayes_model = bayes_sir(Z)
    next = false
    tmp_ode_nuts = Chains{}
    while next == false
        try
            tmp_ode_nuts = sample(sir_bayes_model,NUTS(),N_adapt);
            tr_app_samples_γ[:, t] = tmp_ode_nuts[:γ]
            tr_app_samples_β[:, t] = tmp_ode_nuts[:β]
            next = true
        catch e
            next = false
        end
    end
end

# Xform data and setup for score-cal
# Transform and calibration
bij_all = bijector(sir_bayes_model)
bij = Bijectors.Stacked(bij_all.bs[1:2]...)

# Get joint samples
tr_app_samples_joint = Matrix{Vector{Float64}}(undef, N_adapt, N_samples)
for i in 1:N_samples
    tr_app_samples_joint[:,i] = [[a,b] for (a,b) in 
                                 zip(tr_app_samples_β[:,i], 
                                     tr_app_samples_γ[:,i])]
end
# prexfer_joint = deepcopy(tr_app_samples_joint)
# prexfer_β = deepcopy(tr_app_samples_β)
# prexfer_i₀ = deepcopy(tr_app_samples_i₀)

# X-form them TODO approx post for tdgp - rename
tr_app_samples_joint = 
        inverse(bij).(multiplyscale(bij.(tr_app_samples_joint),
                                    vmultiplier))

# Grab univariate out again and logit xform
for j in 1:N_samples
    for i in 1:N_adapt
        tr_app_samples_β[i, j] = logit(tr_app_samples_joint[i, j][1])
        tr_app_samples_γ[i, j] = logit(tr_app_samples_joint[i, j][2])
        tr_app_samples_joint[i, j] = [tr_app_samples_β[i, j], 
                                      tr_app_samples_γ[i, j]]
    end
    post_β[j] = logit(post_β[j])
    post_γ[j] = logit(post_γ[j])
end
post_joint = [[a, b] for (a,b) in zip(post_β, post_γ)]


# Calibrate samples
# Univariate
# β
cal = Calibration(post_β, tr_app_samples_β)
tf = UnivariateAffine() # Energy score calibration updates this.
res = energyscorecalibrate!(tf, cal, is_weights; β = energyβ, options = options)
samples_β = tf.(map(x-> x, tr_app_samples_β), [mean(tr_app_samples_β)])
# i₀
cal = Calibration(post_γ, tr_app_samples_γ)
tf = UnivariateAffine() # Energy score calibration updates this.
res = energyscorecalibrate!(tf, cal, is_weights; β = energyβ, options = options)
samples_γ = tf.(map(x-> x, tr_app_samples_γ), [mean(tr_app_samples_γ)])
# Multivariate
cal = Calibration(post_joint, tr_app_samples_joint)
d = BayesScoreCal.dimension(cal)[1]
tf = CholeskyAffine(d)
res = energyscorecalibrate!(tf, cal, is_weights)
samples_joint = tf.(map(x-> x, tr_app_samples_joint), 
                    [mean(tr_app_samples_joint)])

# Inv-logit transform B and i
# Save data
samples_joint_β = deepcopy(samples_β)
samples_joint_γ = deepcopy(samples_γ)
for j in 1:N_samples
    for i in 1:N_adapt
        samples_joint_β[i, j] = invlogit(samples_joint[i, j][1])
        samples_joint_γ[i, j] = invlogit(samples_joint[i, j][2])
        samples_β[i, j] = invlogit(samples_β[i, j])
        samples_γ[i, j] = invlogit(samples_γ[i, j])
        tr_app_samples_β[i, j] = invlogit(tr_app_samples_β[i, j])
        tr_app_samples_γ[i, j] = invlogit(tr_app_samples_γ[i, j])
    end
    post_β[j] = invlogit(post_β[j])
    post_γ[j] = invlogit(post_γ[j])
end
post_joint = [[a, b] for (a,b) in zip(post_β, post_γ)]

# Use for save purposes
for i in 1:N_adapt
    for j in 1:N_samples
        samples_joint_β[i, j] = samples_joint[i, j][1]
        samples_joint_γ[i, j] = samples_joint[i, j][2]
    end
end

CSV.write("preB-100.csv", Tables.table(tr_app_samples_β), writeheader=false)
CSV.write("preG-100.csv", Tables.table(tr_app_samples_γ), writeheader=false)
CSV.write("uniB-100.csv", Tables.table(samples_β), writeheader=false)
CSV.write("uniG-100.csv", Tables.table(samples_γ), writeheader=false)
CSV.write("trueB-100.csv", Tables.table(post_β), writeheader=false)
CSV.write("trueG-100.csv", Tables.table(post_γ), writeheader=false)
CSV.write("joinB-100.csv", Tables.table(samples_joint_β), writeheader=false)
CSV.write("joinG-100.csv", Tables.table(samples_joint_γ), writeheader=false)


