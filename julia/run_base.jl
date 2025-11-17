#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations
using Plots
using .BaseDengueModel


# --- Extract solution ---
t   = sol.t
S_H = getindex.(sol.u, 1)
E_H = getindex.(sol.u, 2)
I_H = getindex.(sol.u, 3)
R_H = getindex.(sol.u, 4)
S_M = getindex.(sol.u, 5)
I_M = getindex.(sol.u, 6)

# --- Plot humans ---

# --- Plot mosquitoes ---