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
using CairoMakie
using Optim
using BayesScoreCal

# Approximate/true model settings
N_samples = 1000
N_adapt = 1000

# Optimisation settings
N_importance = 200
N_energy = 1000
energyβ = 1.0
vmultiplier = 1.0
options = Optim.Options(f_tol = 0.00001)

# Generate data with ABM
const steps_per_day = 24

@agent PoorSoul ContinuousAgent{2} begin
    mass::Float64
    days_infected::Int # number of days since infection
    status::Symbol #  :S, :I, or :R
    β::Float64 # Transmision prob
end

using DrWatson: @dict
function sir_initiation(; 
        infection_period = 4 * steps_per_day, # 30
        detection_time = 1 * steps_per_day, # 14 
        reinfection_probability = 0.0, # 0.05, 
        isolated = 0.0, # Percentage 
        interaction_radius = 0.005, # 0.012 
        dt = 1.0, 
        speed = 0.001,
        death_rate = 0, # 0.044
        N = 1000,
        initial_infected = 10, # 5
        seed = 42,
        βmin = 0.05, # 0.4
        βmax = 0.05, # 0.8
    )
    properties = (;
                  infection_period,
                  reinfection_probability,
                  detection_time,
                  death_rate,
                  interaction_radius,
                  dt,
                 )
    space = ContinuousSpace((1, 1); spacing = 0.02)
    model = ABM(PoorSoul,
                space,
                properties = properties,
                rng = MersenneTwister(seed))
    
    # Add 1000 agents to the model
    for ind in 1:N
        pos = Tuple(rand(model.rng, 2))
        status = ind ≤ N - initial_infected ? :S : :I
        isisolated = ind ≤ isolated * N
        mass = isisolated ? Inf : 1.0
        vel = isisolated ? (0.0, 0.0) : sincos(2π * rand(model.rng)) .* speed
        β = (βmax - βmin) * rand(model.rng) + βmin
        add_agent!(pos, model, vel, mass, 0, status, β)
    end
    return model
end

function transmit!(a1, a2, rp)
    # Need only 1 infected to transmit
    count(a.status == :I for a in (a1, a2)) ≠ 1  && return
    infected, healthy = a1.status == :I ? (a1, a2) : (a2, a1)

    rand(sir_model.rng) > infected.β && return

    if healthy.status == :R
        return
        rand(sir_model.rng) > rp && return
    end
    healthy.status = :I
end

function sir_model_step!(model)
    r = model.interaction_radius
    for (a1, a2) in interacting_pairs(model, r, :nearest)
        transmit!(a1, a2, model.reinfection_probability)
        elastic_collision!(a1, a2, :mass)
    end
end

function sir_agent_step!(agent, model)
    move_agent!(agent, model, model.dt)
    update!(agent)
    recover_or_die!(agent, model)
end

update!(agent) = agent.status == :I && (agent.days_infected += 1)

function recover_or_die!(agent, model)
    if agent.days_infected ≥ model.infection_period
        if rand(model.rng) ≤ model.death_rate
            kill_agent!(agent, model)
        else
            agent.status = :R
            agent.days_infected = 0
        end
    end
end

sir_model = sir_initiation()

infected(x) = count(i == :I for i in x)
recovered(x) = count(i == :R for i in x)
adata = [(:status, infected), (:status, recovered)]

data1, _ = run!(sir_model, sir_agent_step!,  sir_model_step!, 40*24; adata)

# Generate data for ODE
tmp = data1[1:24:end, 2:3] # Grab one obs per day
infect = tmp[2:end,1] - tmp[1:(end-1),1]
recov = tmp[2:end,2] - tmp[1:(end-1),2]
Y = infect + recov
 
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

@model bayes_sir(y) = begin
  # Calculate number of timepoints
  l = length(y)
  i₀ ~ Uniform(0.0,1.0)
  β ~ Uniform(0.0,1.0)
  I = i₀*1000.0
  u0=[1000.0-I,I,0.0,0.0]
  p=[β,10.0,0.25]
  tspan = (0.0,float(l))
  prob = ODEProblem(sir_ode!,
          u0,
          tspan,
          p)
  sol = solve(prob,
              Tsit5(),
              saveat = 1.0)
  sol_C = Array(sol)[4,:] # Cumulative cases
  sol_X = sol_C[2:end] - sol_C[1:(end-1)]
  l = length(y)
  for i in 1:l
    y[i] ~ Poisson(sol_X[i])
  end
end;

sir_bayes_model = bayes_sir(Y)
ode_nuts = sample(sir_bayes_model,NUTS(1.0),1000);

#describe(ode_nuts)
#StatsPlots.plot(ode_nuts)
# Generate weights for each theta
#
# logjoint(bayes_sir(Y), need named fileds to pass here)
is_weights = ones(1000)

# Get posterior samples (TODO: update to joint samples)
post_i₀ = vec(ode_nuts[:i₀].data)
post_β = vec(ode_nuts[:β].data)

tr_app_samples_i₀ = Matrix{Float64}(undef, N_energy, N_adapt)
tr_app_samples_β = Matrix{Float64}(undef, N_energy, N_adapt)

for t in 1:1000
    next = false # This allows for ode solver error catching
    reps = 0
    while next == false
        try
            # Setup for approximate distribution (for first dist)
            tmax = 40.0
            tspan = (0.0,tmax)
            obstimes = 1.0:1.0:tmax
            I = post_i₀[t]*1000.0
            u0 = [1000.0 - I,I,0.0,0.0] # S,I.R,C
            p = [post_β[t],10.0,0.25]; # β,c,γ

            prob_ode = ODEProblem(sir_ode!,u0,tspan,p); sol_ode = solve(prob_ode,
                            Tsit5(),
                            saveat = 1.0);
            sol_ode = solve(prob_ode,
                            Tsit5(),
                            saveat = 1.0);

            # Generate new data from ODE above
            C = Array(sol_ode)[4,:] # Cumulative cases
            X = C[2:end] - C[1:(end-1)];
            Z = rand.(Poisson.(X))
            tmp_ode_nuts = sample(bayes_sir(Z), NUTS(1.0), 1000); 

            tr_app_samples_i₀[:, t] = tmp_ode_nuts[:i₀]
            tr_app_samples_β[:, t] = tmp_ode_nuts[:β]
            next = true

        catch e
            next = false
            reps += 1
            if(reps == 10) # Avoid infinite loops adjust as req
                next = true
            end
        end
    end
end

# Store un-xform samples
CSV.write("examples/agent-based-models/preB.csv", Tables.table(tr_app_samples_β), writeheader=false)
CSV.write("examples/agent-based-models/prei.csv", Tables.table(tr_app_samples_i₀), writeheader=false)

# Transform and calibration
function multiplyscale(x::Matrix{Vector{Float64}}, scale::Float64) 
    μ = mean(x)
    scale .* (x .- [μ]) .+ [μ]
end

bij_all = bijector(sir_bayes_model)
bij = Bijectors.Stacked(bij_all.bs[1:2]...)

# Get joint samples
post_joint = [[a, b] for (a,b) in zip(post_β, post_i₀)]
tr_app_samples_joint = Matrix{Vector{Float64}}(undef, 1000, 1000)
for i in 1:1000
    tr_app_samples_joint[:,i] = [[a,b] for (a,b) in 
                                 zip(tr_app_samples_β[:,i], 
                                     tr_app_samples_i₀[:,i])]
end

# X-form them
tr_app_samples_joint = inverse(bij).(multiplyscale(bij.(tr_app_samples_joint),
                                                   1.0))

# Grab univariate out again
for i in 1:N_adapt
    for j in 1:N_energy
        tr_app_samples_β[i, j] = tr_app_samples_joint[i, j][1]
        tr_app_samples_i₀[i, j] = tr_app_samples_joint[i, j][2]
    end
end

# Univariate
# β
cal = Calibration(post_β, tr_app_samples_β)
tf = UnivariateAffine() # Energy score calibration updates this.
res = energyscorecalibrate!(tf, cal, is_weights; β = energyβ, options = options)
samples_β = tf.(map(x-> x, tr_app_samples_β), [mean(tr_app_samples_β)])
# i₀
cal = Calibration(post_i₀, tr_app_samples_i₀)
tf = UnivariateAffine() # Energy score calibration updates this.
res = energyscorecalibrate!(tf, cal, is_weights; β = energyβ, options = options)
samples_i₀ = tf.(map(x-> x, tr_app_samples_i₀), [mean(tr_app_samples_i₀)])
# Multivariate

cal = Calibration(post_joint, tr_app_samples_joint)
d = BayesScoreCal.dimension(cal)[1]
tf = CholeskyAffine(d)
res = energyscorecalibrate!(tf, cal, is_weights)
samples_joint = tf.(map(x-> x, tr_app_samples_joint), 
                    [mean(tr_app_samples_joint)])

# Use for save purposes
# for i in 1:N_adapt
#     for j in 1:N_energy
#         samples_β[i, j] = samples_joint[i, j][1]
#         samples_i₀[i, j] = samples_joint[i, j][2]
#     end
# end
CSV.write("examples/agent-based-models/uniB.csv", Tables.table(tr_app_samples_β), writeheader=false)
CSV.write("examples/agent-based-models/unii.csv", Tables.table(tr_app_samples_i₀), writeheader=false)
CSV.write("examples/agent-based-models/trueB.csv", Tables.table(post_β), writeheader=false)
CSV.write("examples/agent-based-models/truei.csv", Tables.table(post_i₀), writeheader=false)
CSV.write("examples/agent-based-models/joinB.csv", Tables.table(samples_β), writeheader=false)
CSV.write("examples/agent-based-models/joini.csv", Tables.table(samples_i₀), writeheader=false)

#samplecomp = DataFrame(
#    samples = tf.(map(x-> x, tr_app_samples_β), [mean(tr_app_samples_β)]),
#    method = "Adjust-post",
#    iter = 1,
#    alpha = alpha,
#)


# data_cache = load_object("examples/agent-based-models/sir-cache.jld2")
# post_β = data_cache.post_β
# post_i₀ = data_cache.post_i₀
# tr_app_samples_β = data_cache.tr_app_samples_β
# tr_app_samples_i₀ = data_cache.tr_app_samples_i₀
