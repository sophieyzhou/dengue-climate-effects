module BaseDengueModel

using DifferentialEquations

"""
Parameter definitions for the module
"""

Base.@kwdef mutable struct BaseParams
    # Human demography [Peru 2015]
    Π_H::Float64    = 10.54      # 1000s/week crude birth rate DONE
    μ_H::Float64    = 0.000000252      # /1000people/week, natural mortality per capita Peru DONE

    # Human dengue progression (fixed)
    σ_H::Float64    = 1.05        # 1/week, exposeure incubation -> infectious DONE
    γ_H::Float64    = 0.9996      # 1/week, recovery DONE

    # dengue induced mortality TO BE LEARNED
    δ_H::Float64    = 1e-7        # 1000s/week, dengue-induced mortality (small) DONE

    # Forces of infection (TO BE LEARNED)
    Β_V::Float64   = 0.5         # bite infection rate of vector from biting Human DONE
    Β_H::Float64   = 0.4         # bite infection rate of human from getting bit by vector DONE

    # biting rate of F mosquitoes on all humans (fixed)
    α::Float64     = 0.84         # bites/mosquito/week DONE

    # Mosquito demography (no temperature dependence in this base model)
    Π_V::Float64    = 75.04         # 1/week,, per capita birth rate DONE
    μ_VI::Float64   = 0.3261         # 1/week, immature mortality DONE
    μ_VA::Float64   = 0.2495         # 1/week, adult mortality DONE

    # Mosquito progression and disease-related mortality (fixed from table)
    σ_V::Float64    = 0.5502         #1/week, maturation immature -> adult DONE
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

    # totals
    N_H  = S_H + E_H + I_H + R_H
    # N_VA = S_VA + I_VA

    # unpack parameters
    Π_H  = p.Π_H
    μ_H  = p.μ_H
    σ_H  = p.σ_H
    γ_H  = p.γ_H
    δ_H  = p.δ_H

    Β_V = p.Β_V
    Β_H = p.Β_H
    α   = p.α

    Π_V  = p.Π_V
    μ_VI = p.μ_VI
    μ_VA = p.μ_VA
    σ_V  = p.σ_V
    # δ_VI = p.δ_VI
    # δ_VA = p.δ_VA

    # ------------------
    # Vector Transmission Forces Equations
    # ------------------
    if N_H > 0
        λ_HV = α * Β_V * (I_H / N_H)  # humans -> vectors
        λ_VH = α * Β_H * (I_H / N_H)  # vectors -> humans
    else
        λ_HV = 0.0
        λ_VH = 0.0
    end
    
    # ------------------
    # Human SEIR dynamics (no T(t))
    # ------------------
    # dS_H/dt = Π_H - μ_H S_H - λ_VH S_H
    du[IDX_SH] = Π_H - μ_H * S_H - λ_VH * S_H

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
    du[IDX_IVI] = Π_V * I_VA - (μ_VI + σ_V) * I_VI

    # ------------------
    # Mosquito adult dynamics, with vertical transmission and FOI from humans
    # ------------------
    # dS_VA/dt = σ_V S_VI  - μ_VA S_VA - λ_HV S_VA
    du[IDX_SVA] = σ_V * S_VI - (μ_VA + λ_HV) * S_VA

    # dI_VA/dt = σ_V I_VI  - (μ_VA + δ_VA) I_VA + λ_HV S_VA
    du[IDX_IVA] = σ_V * I_VI - (μ_VA) * I_VA + λ_HV * S_VA

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