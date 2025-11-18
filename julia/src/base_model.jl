module BaseDengueModel

using DifferentialEquations

"""
Parameter definitions for the module
"""

Base.@kwdef mutable struct BaseParams
    # Human demography
    Π_H::Float64    = 4.4e-5      # 1/day, crude birth rate per capita
    μ_H::Float64    = 5.0e-5      # 1/day, natural mortality per capita

    # Human dengue progression (fixed from table at representative temperature)
    σ_H::Float64    = 0.15        # 1/day, exposed -> infectious
    γ_H::Float64    = 0.1428      # 1/day, recovery
    δ_H::Float64    = 1e-4        # 1/day, dengue-induced mortality (small)

    # Forces of infection (to be calibrated)
    λ_VH::Float64   = 1e-8        # 1/day, vector -> human FOI
    λ_HV::Float64   = 1e-8        # 1/day, human -> vector FOI

    # Mosquito demography (no temperature dependence in this base model)
    Π_V::Float64    = 0.5         # 1/day, per capita recruitment of immatures from adults (placeholder)
    μ_VI::Float64   = 0.1         # 1/day, immature natural mortality (placeholder)
    μ_VA::Float64   = 0.1         # 1/day, adult natural mortality (placeholder)

    # Mosquito progression and disease-related mortality (fixed from table)
    σ_V::Float64    = 0.1         # 1/day, maturation immature -> adult
    δ_VI::Float64   = 0.01        # 1/day, extra mortality infected immature
    δ_VA::Float64   = 0.01        # 1/day, extra mortality infected adult
end

# State indices to keep everything readable
const IDX_SH  = 1
const IDX_EH  = 2
const IDX_IH  = 3
const IDX_RH  = 4
const IDX_SVI = 5
const IDX_IVI = 6
const IDX_SVA = 7
const IDX_IVA = 8

"""
Right-hand side of the ODE system (no temperature dependence).

State vector u = [
    S_H, E_H, I_H, R_H,
    S_VI, I_VI, S_VA, I_VA
]
"""

function dengue_rhs!(du, u, p::BaseParams, t)
    # unpack state
    S_H  = u[IDX_SH]
    E_H  = u[IDX_EH]
    I_H  = u[IDX_IH]
    R_H  = u[IDX_RH]
    S_VI = u[IDX_SVI]
    I_VI = u[IDX_IVI]
    S_VA = u[IDX_SVA]
    I_VA = u[IDX_IVA]

    # unpack parameters
    Π_H  = p.Π_H
    μ_H  = p.μ_H
    σ_H  = p.σ_H
    γ_H  = p.γ_H
    δ_H  = p.δ_H

    λ_VH = p.λ_VH
    λ_HV = p.λ_HV

    Π_V  = p.Π_V
    μ_VI = p.μ_VI
    μ_VA = p.μ_VA
    σ_V  = p.σ_V
    δ_VI = p.δ_VI
    δ_VA = p.δ_VA

    # ------------------
    # Human SEIR dynamics (no T(t))
    # ------------------
    # dS_H/dt = (Π_H - μ_H) S_H - λ_VH S_H
    du[IDX_SH] = (Π_H - μ_H) * S_H - λ_VH * S_H

    # dE_H/dt = λ_VH S_H - σ_H E_H - μ_H E_H
    du[IDX_EH] = λ_VH * S_H - σ_H * E_H - μ_H * E_H

    # dI_H/dt = σ_H E_H - γ_H I_H - δ_H I_H - μ_H I_H
    du[IDX_IH] = σ_H * E_H - (γ_H + δ_H + μ_H) * I_H

    # dR_H/dt = γ_H I_H - μ_H R_H
    du[IDX_RH] = γ_H * I_H - μ_H * R_H

    # ------------------
    # Mosquito immature (aquatic) dynamics, with vertical transmission
    # ------------------
    # dS_VI/dt = Π_V S_VA - μ_VI S_VI - σ_V S_VI
    du[IDX_SVI] = Π_V * S_VA - (μ_VI + σ_V) * S_VI

    # dI_VI/dt = Π_V I_VA - (μ_VI + δ_VI) I_VI - σ_V I_VI
    du[IDX_IVI] = Π_V * I_VA - (μ_VI + δ_VI + σ_V) * I_VI

    # ------------------
    # Mosquito adult dynamics, with vertical transmission and FOI from humans
    # ------------------
    # dS_VA/dt = σ_V S_VI  - μ_VA S_VA - λ_HV S_VA
    du[IDX_SVA] = σ_V * S_VI - (μ_VA + λ_HV) * S_VA

    # dI_VA/dt = σ_V I_VI  - (μ_VA + δ_VA) I_VA + λ_HV S_VA
    du[IDX_IVA] = σ_V * I_VI - (μ_VA + δ_VA) * I_VA + λ_HV * S_VA

    return nothing
end

"""
Construct a reasonable initial condition vector.

Arguments:
  NH0   - initial total human population (approx)
  NMI0  - initial total immature mosquito population
  NMA0  - initial total adult mosquito population
  ih_frac - initial infectious human fraction of NH0
"""
function initial_conditions(NH0::Float64,
                            NMI0::Float64,
                            NMA0::Float64;
                            ih_frac::Float64 = 1e-5)

    S_H0  = NH0 * (1.0 - ih_frac)
    E_H0  = 0.0
    I_H0  = NH0 * ih_frac
    R_H0  = 0.0

    S_VI0 = NMI0
    I_VI0 = 0.0

    S_VA0 = NMA0
    I_VA0 = 0.0

    return [S_H0, E_H0, I_H0, R_H0, S_VI0, I_VI0, S_VA0, I_VA0]
end

"""
Solve the base model over a given time span.

Returns the DifferentialEquations.jl solution object.
"""
function solve_base_model(u0::AbstractVector, p::BaseParams, tspan::Tuple{Real,Real})
    prob = ODEProblem(dengue_rhs!, u0, (float(tspan[1]), float(tspan[2])), p)
    return solve(prob)
end

end # module