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
plot(sol_ode)

# Agent based model version (Continuous for now, use pred/prey for discreet)
using Agents, Random
using InteractiveDynamics
using CairoMakie

@agent SocialAgent ContinuousAgent{2} begin
    mass::Float64
end

function ball_model(; speed = 0.002)
    space2d = ContinuousSpace((2, 2); spacing = 0.02)
    model = ABM(SocialAgent,
                space2d,
                properties = Dict(:dt => 1.0),
                rng = MersenneTwister(42))
    
    # Add 1000 agents to the model
    for ind in 1:1000
        pos = Tuple(rand(model.rng, 2)*2)
        vel = sincos(2π * rand(model.rng)) .* speed
        add_agent!(pos, model, vel, 1.0)
    end
    return model
end

model = ball_model()

agent_step!(agent, model) = move_agent!(agent, model, model.dt)

function model_step!(model)
    for (a1, a2) in interacting_pairs(model, 0.012, :nearest)
        elastic_collision!(a1, a2, :mass)
    end
end

abmvideo(
         "socialdist2.mp4",
         model,
         agent_step!,
         model_step!;
         title = "Move",
         frames = 500,
         spf = 2,
         framerate = 25,
        )
