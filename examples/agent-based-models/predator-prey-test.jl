# This is to map LV to ABM

using Turing
using DifferentialEquations
using OrdinaryDiffEq

# Load StatsPlots for visualizations and diagnostics.
using StatsPlots

using LinearAlgebra

# Set a seed for reproducibility.
using Random
Random.seed!(71758); # Match seed for ABM

# Define Lotka-Volterra model.
function lotka_volterra(du, u, p, t)
    # Model parameters.
    α, β, a, δ, γ, b, c, d, e = p
    # Current state.
    x, y, z = u

    # Evaluate differential equations.
    du[1] = (α - β * y - a * z) * x # Grass
    du[2] = (δ * x - γ - b * z) * y # Sheep
    du[3] = (c * x + d * y - e) * z # Wolf

    return nothing
end

# Define initial-value problem.
u0 = [24.0, 140.0, 20.0] # Half grass, 140 sheep, 20 wolves
p = [0.212672, 1/900, 0.0, 0.31/900, 0.1, 1/900, 0.0, 0.06/900, 1/60]
tspan = (0.0, 2000.0)
prob = ODEProblem(lotka_volterra, u0, tspan, p)

# Plot simulation.
plot(solve(prob, Tsit5()))
