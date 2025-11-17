#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)

using DifferentialEquations
using Plots
using .BaseDengueModel