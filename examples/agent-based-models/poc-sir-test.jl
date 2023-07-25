using DifferentialEquations
using DiffEqSensitivity
using Random
using Distributions
using Turing
using DataFrames
using StatsPlots

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
tmax = 40.0
tspan = (0.0,tmax)
obstimes = 1.0:1.0:tmax
u0 = [990.0,10.0,0.0,0.0] # S,I.R,C
p = [0.05,10.0,0.25]; # β,c,γ

prob_ode = ODEProblem(sir_ode!,u0,tspan,p);
sol_ode = solve(prob_ode,
                Tsit5(),
                saveat = 1.0);
StatsPlots.plot(sol_ode)

# Generate data from ODE above.
C = Array(sol_ode)[4,:] # Cumulative cases
X = C[2:end] - C[1:(end-1)];
Random.seed!(1234)
Y = rand.(Poisson.(X));
StatsPlots.bar(obstimes,Y,legend=false)
StatsPlots.plot!(obstimes,X,legend=false)

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

ode_nuts = sample(bayes_sir(Y),NUTS(1.0),10000);

describe(ode_nuts)
StatsPlots.plot(ode_nuts)

# Agent based model version (Continuous for now, use pred/prey for discreet)
using Agents, Random
using InteractiveDynamics
using CairoMakie

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
        interaction_radius = 0.01, # 0.012 
        dt = 1.0, 
        speed = 0.002,
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

# Makes a picture of initial state
#sir_model = sir_initiation()

#sir_colours(a) = a.status == :S ? "#2b2b33" : a.status == :I ? "#bf2642" : "#338c54"

#fig, ax, abmp = abmplot(sir_model; ac = sir_colours)
#fig

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
# Make data Z

tmp = data1[1:24:end, 2:3] # Grab one obs per day
infect = tmp[2:end,1] - tmp[1:(end-1),1]
recov = tmp[2:end,2] - tmp[1:(end-1),2]
Z = infect + recov

figure = Figure()
ax = figure[1, 1] = Axis(figure; ylabel = "Individuals")
l1 = lines!(ax, data1[:, dataname((:status, infected))], colour = "#bf2642")
l2 = lines!(ax, data1[:, dataname((:status, recovered))], colour = "#338c54")
figure[1, 2][1,1] =
    Legend(figure, [l1, l2], ["Infected", "Recovered"])
figure

# Creates animation of ABM
#sir_model = sir_initiation()
#
#abmvideo(
#         "socialdist2.mp4",
#         sir_model,
#         sir_agent_step!,
#         sir_model_step!;
#         title = "Move",
#         frames = 500,
#         spf = 2,
#         framerate = 25,
#        )
