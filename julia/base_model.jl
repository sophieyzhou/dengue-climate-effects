module BaseDengueModel

using DifferentialEquations

"""
Parameter definitions for the module
"""

Base.@kwdef mutable struct BaseDengueParams
    # fixed human demography related parameters
    \Pi_H::Float64 = 1       # TODO: find human recruitment rate
    \mu_H::Float64 = 1   # TODO: find human natural death rate

    # human dengue related parameters 
    \gamma_H::Float64 = 1        # TODO: find human recovery rate from dengue
end

"""