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

using DrWatson: @dict

# Approximate/true model settings
N_samples = 100
N_adapt = 1000
N_ind = 1000

# Optimisation settings
N_runs = 100
energyβ = 1.0
vmultiplier = 1.0
alphalevels = [0.0, 0.25, 0.5, 0.9, 1.0]
abmpars = [10, 0.05, 4, 1, 0.001, 0.005, N_ind] 
#[initial infected (int), beta value, infect period (days), 
# detection (days), speed, interaction radius, total individuals]
options = Optim.Options(f_tol = 0.00001)
tspan = (0,40)
u0 = [N_ind*0.9, N_ind*0.1, 0, 0]

# Setup ABM for data generation
const steps_per_day = 24

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
              [0.05, 1.0, 0.25])
              #[0.05, 10.0, 0.25])


@model bayes_sir(y) = begin
  # Calculate number of timepoints
  l = length(y)
  i₀ ~ Uniform(0.0,1.0)
  β ~ Uniform(0.0,1.0)
  I = i₀*Float64(N_ind)
  u0=[Float64(N_ind)-I,I,0.0,0.0]
  #p=[β,10.0,0.25]
  p=[β,1.0,0.25]
  #tspan = (0.0,float(l))
  prob = ODEProblem(sir_ode!,
              u0,
              tspan,
              p) #[0.05, 10.0, 0.25])
#remake(ODEprob, u=u0, p=p)
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

# Need function wrapper here
function testfun(N_adapt::Int64, N_samples::Int64, vmultiplier::Float64,
        iter::Int64, abmpars::Vector{Float64},
        options::Optim.Options = Optim.Options()) 

    # Setup dataframe for results
    dfsamples = DataFrame[]
    # Setup ABM
    @agent PoorSoul ContinuousAgent{2} begin
        mass::Float64
        days_infected::Int # number of days since infection
        status::Symbol #  :S, :I, or :R
        β::Float64 # Transmision prob
    end

    function sir_initiation(; 
            β = abmpars[2],
            initial_infected = abmpars[1],
            infection_period = Int(abmpars[3]) * steps_per_day,
            detection_time = Int(abmpars[4]) * steps_per_day,
            reinfection_probability = 0.0,
            isolated = 0.0,
            interaction_radius = abmpars[6],
            dt = 1.0, 
            speed = abmpars[5],
            death_rate = 0,
            N = abmpars[7],
            #seed = 42,
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
                    rng = MersenneTwister())
                    #rng = MersenneTwister(seed))
        
        # Add 1000 agents to the model
        for ind in 1:N
            pos = Tuple(rand(model.rng, 2))
            status = ind ≤ N - initial_infected ? :S : :I
            isisolated = ind ≤ isolated * N
            mass = isisolated ? Inf : 1.0
            vel = isisolated ? (0.0, 0.0) : sincos(2π * rand(model.rng)) .* speed
            β = β
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
            #elastic_collision!(a1, a2, :mass)
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
                remove_agent!(agent, model)
            else
                agent.status = :R
                agent.days_infected = 0
            end
        end
    end
    # Generate ABM data
    sir_model = sir_initiation()

    infected(x) = count(i == :I for i in x)
    recovered(x) = count(i == :R for i in x)
    adata = [(:status, infected), (:status, recovered)]
    # Need to add variable for time length here
    data1, _ = run!(sir_model, sir_agent_step!,  sir_model_step!, 40*24; adata)

    # Generate data for ODE
    tmp = data1[1:steps_per_day:end, 2:3] # Grab one obs per day
    infect = tmp[2:end,1] - tmp[1:(end-1),1]
    recov = tmp[2:end,2] - tmp[1:(end-1),2]
    Y = infect + recov
    sir_bayes_model = bayes_sir(Y)
    next = false
    ode_nuts = Chains{}
    while next == false
        try
            # Use 4 times thinning
            ode_nuts = sample(sir_bayes_model,NUTS(),N_samples*4);
            next = true;
            break;
        catch e
            next = false
        end
    end

    is_weights = ones(N_samples)

    # Get posterior samples (TODO: update to joint samples) thin here
    # app_post_tdgp
    post_i₀ = vec(ode_nuts[:i₀].data[1:4:end])
    post_β = vec(ode_nuts[:β].data[1:4:end])
    
    samplecomp = DataFrame(
        samples = post_β,
        method = "True-post",
        iter = iter,
        index = 1,
        param = "β",
        trueval = post_β[1]
       )
    push!(dfsamples, samplecomp)
    samplecomp = DataFrame(
        samples = post_i₀,
        method = "True-post",
        iter = iter, index = 1, param = "i₀",
        trueval = post_i₀[1]
       )
    push!(dfsamples, samplecomp)

    tr_app_samples_i₀ = Matrix{Float64}(undef, N_adapt, N_samples)
    tr_app_samples_β = Matrix{Float64}(undef, N_adapt, N_samples)

    for t in 1:N_samples
        next = false # This allows for ode solver error catching
        reps = 0
        while next == false
            try
                # Setup for approximate distribution (for first dist)
                #tmax = 40.0 # Need to set variable
                #tspan = (0.0,tmax)
                #obstimes = 1.0:1.0:tmax
                #I = post_i₀[t]*N_samples
                #u0 = [float(N_samples) - I,I,0.0,0.0] # S,I.R,C
                #p = [post_β[t],10.0,0.25]; # β,c,γ

                #prob_ode = ODEProblem(sir_ode!,u0,tspan,p); 
                #                sol_ode = solve(prob_ode,
                #                                Tsit5(),
                #                                saveat = 1.0);
                #sol_ode = solve(prob_ode,
                #                Tsit5(),
                #                saveat = 1.0);

                ## Generate new data from ODE above
                #C = Array(sol_ode)[4,:] # Cumulative cases
                #X = C[2:end] - C[1:(end-1)];
                #Z = rand.(Poisson.(X))
                #tmp_ode_nuts = sample(bayes_sir(Z), NUTS(1.0), N_samples); 
                
                sir_model = sir_initiation(β = post_β[t], 
                                           initial_infected = 
                                                post_i₀[t]*Float64(N_ind))

                infected(x) = count(i == :I for i in x)
                recovered(x) = count(i == :R for i in x)
                adata = [(:status, infected), (:status, recovered)]
                # Need to add variable for time length here
                data1, _ = run!(sir_model, sir_agent_step!,  sir_model_step!, 40*24; adata)

                # Generate data for ODE
                tmp = data1[1:steps_per_day:end, 2:3] # Grab one obs per day
                infect = tmp[2:end,1] - tmp[1:(end-1),1]
                recov = tmp[2:end,2] - tmp[1:(end-1),2]
                Z = infect + recov
                sir_bayes_model = bayes_sir(Z)
                next = false
                tmp_ode_nuts = Chains{}
                while next == false
                    try
                        tmp_ode_nuts = sample(sir_bayes_model,NUTS(),N_adapt);
                        next = true
                    catch e
                        next = false
                    end
                end

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

    # Transform and calibration
    bij_all = bijector(sir_bayes_model)
    bij = Bijectors.Stacked(bij_all.bs[1:2]...)

    # Get joint samples
    tr_app_samples_joint = Matrix{Vector{Float64}}(undef, N_adapt, N_samples)
    for i in 1:N_samples
        tr_app_samples_joint[:,i] = [[a,b] for (a,b) in 
                                     zip(tr_app_samples_β[:,i], 
                                         tr_app_samples_i₀[:,i])]
    end
    prexfer_joint = deepcopy(tr_app_samples_joint)
    prexfer_β = deepcopy(tr_app_samples_β)
    prexfer_i₀ = deepcopy(tr_app_samples_i₀)

    # X-form them TODO approx post for tdgp - rename
    tr_app_samples_joint = 
            inverse(bij).(multiplyscale(bij.(tr_app_samples_joint),
                                        vmultiplier))

    # Grab univariate out again and logit xform
    for j in 1:N_samples
        for i in 1:N_adapt
            tr_app_samples_β[i, j] = logit(tr_app_samples_joint[i, j][1])
            tr_app_samples_i₀[i, j] = logit(tr_app_samples_joint[i, j][2])
            tr_app_samples_joint[i, j] = [tr_app_samples_β[i, j], 
                                          tr_app_samples_i₀[i, j]]
        end
        post_β[j] = logit(post_β[j])
        post_i₀[j] = logit(post_i₀[j])
    end
    post_joint = [[a, b] for (a,b) in zip(post_β, post_i₀)]
    
    # Calibrate samples
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
    # Inv-logit transform B and i
    # Save data
    samples_joint_β = deepcopy(samples_β)
    samples_joint_i₀ = deepcopy(samples_i₀)
    for j in 1:N_samples
        for i in 1:N_adapt
            samples_joint_β[i, j] = invlogit(samples_joint[i, j][1])
            samples_joint_i₀[i, j] = invlogit(samples_joint[i, j][2])
            samples_β[i, j] = invlogit(samples_β[i, j])
            samples_i₀[i, j] = invlogit(samples_i₀[i, j])
            tr_app_samples_β[i, j] = invlogit(tr_app_samples_β[i, j])
            tr_app_samples_i₀[i, j] = invlogit(tr_app_samples_i₀[i, j])
        end
        post_β[j] = invlogit(post_β[j])
        post_i₀[j] = invlogit(post_i₀[j])
    end
    post_joint = [[a, b] for (a,b) in zip(post_β, post_i₀)]
    for k in 1:N_samples
        samplecomp = DataFrame(
            samples = samples_β[:,k],
            method = "Uni-post",
            iter = iter,
            index = k,
            param = "β",
            trueval = post_β[k]
           )
        push!(dfsamples, samplecomp)
        samplecomp = DataFrame(
            samples = samples_i₀[:,k],
            method = "Uni-post",
            iter = iter,
            index = k,
            param = "i₀",
            trueval = post_i₀[k]
           )
        push!(dfsamples, samplecomp)
        samplecomp = DataFrame(
            samples = samples_joint_β[:,k],
            method = "Joint-post",
            iter = iter,
            index = k,
            param = "β",
            trueval = post_β[k]
           )
        push!(dfsamples, samplecomp)
        samplecomp = DataFrame(
            samples = samples_joint_i₀[:,k],
            method = "Joint-post",
            iter = iter,
            index = k,
            param = "i₀",
            trueval = post_i₀[k]
           )
        push!(dfsamples, samplecomp)
    end

    return dfsamples
end

# Single run
res = testfun(N_adapt, N_samples, vmultiplier, 1, abmpars, options)
# Join results
allres = reduce(vcat, res)

# Multiple run
res = testfun.([N_adapt], 
               [N_samples], 
               [vmultiplier], 
               1:10, 
               [abmpars], 
               [options])
# Join results
allres = vcat(reduce(vcat, res)...)

#preB = tr_app_samples_β
#prei = tr_app_samples_i₀
# Use for save purposes
# for i in 1:N_adapt
#     for j in 1:N_energy
#         samples_β[i, j] = samples_joint[i, j][1]
#         samples_i₀[i, j] = samples_joint[i, j][2]
#     end
# end
CSV.write("preB-100.csv", Tables.table(tr_app_samples_β), writeheader=false)
CSV.write("prei-100.csv", Tables.table(tr_app_samples_i₀), writeheader=false)
CSV.write("uniB-100.csv", Tables.table(samples_β), writeheader=false)
CSV.write("unii-100.csv", Tables.table(samples_i₀), writeheader=false)
CSV.write("trueB-100.csv", Tables.table(post_β), writeheader=false)
CSV.write("truei-100.csv", Tables.table(post_i₀), writeheader=false)
CSV.write("joinB-100.csv", Tables.table(samples_joint_β), writeheader=false)
CSV.write("joini-100.csv", Tables.table(samples_joint_i₀), writeheader=false)

# Test
mean(([mean(tr_app_samples_β[:,col]) for col=1:size(tr_app_samples_β)[2]] - post_β))
mean(([mean(samples_β[:,col]) for col=1:size(samples_β)[2]] - post_β))
mean(([mean(samples_joint_β[:,col]) for col=1:size(samples_joint_β)[2]] - post_β))

mean(([mean(tr_app_samples_i₀[:,col]) for col=1:size(tr_app_samples_i₀)[2]] - post_i₀))
mean(([mean(samples_i₀[:,col]) for col=1:size(samples_i₀)[2]] - post_i₀))
mean(([mean(samples_joint_i₀[:,col]) for col=1:size(samples_joint_i₀)[2]] - post_i₀))

#CSV.write("../examples/agent-based-models/multi.csv", Tables.table(res), writeheader=true, bufsize=263015500)
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
