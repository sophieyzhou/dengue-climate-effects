#!/usr/bin/env julia

using Pkg
# Activate the Julia project in the julia/ folder
Pkg.activate(@__DIR__)

# Bring in your modules
include(joinpath(@__DIR__, "src", "read_data.jl"))
include(joinpath(@__DIR__, "src", "base_model.jl"))

using .IDNData
using .BaseDengueModel
using Dates
using Printf
using Plots
using DifferentialEquations
using Optimization, OptimizationOptimJL
using Optimization.AutoZygote

# ---------------------------
# 1. Load Peru monthly dengue incidence
# ---------------------------

# repo root is one level up from julia/
const REPO_ROOT = dirname(@__DIR__)

idn = IDNData.load_idn_monthly_cases(REPO_ROOT)
records = idn.records

println("Loaded $(length(records)) monthly records for $(idn.country) ($(idn.iso_code)).")

# Take the first record's start date as t = 0
start_date = first(records).start_date
end_date   = last(records).end_date
total_days = Dates.value(end_date - start_date) + 1

# We'll run the ODE in **weeks**, so convert days -> weeks
tspan_weeks = (0.0, total_days / 7.0)

# Build time points (in weeks) for each record (use midpoints of months)
t_data = Float64[]
cases_data = Float64[]
years = Int[]
months = Int[]

for rec in records
    Δ_start   = rec.start_date - start_date
    Δ_end     = rec.end_date   - start_date
    tmid_days = (Dates.value(Δ_start) + Dates.value(Δ_end)) / 2
    tmid_weeks = tmid_days / 7.0

    push!(t_data, tmid_weeks)
    push!(cases_data, Float64(rec.dengue_total))
    push!(years, rec.year)
    push!(months, month(rec.start_date))
end

println("Time span: $(tspan_weeks[1]) to $(tspan_weeks[2]) weeks.")
println("Number of data points: $(length(t_data))")

# ---------------------------
# 2. Set parameters and initial conditions
# ---------------------------

# Rough scales – you can refine these later
NH0  = 30457.6    # ~ total population of Peru in thousands in 2015
NMA0 = 30457000*2     # adult mosquitoes (rough guess)
NMI0 = NMA0 * 0.5     # aquatic mosquitoes (rough guess)

u0 = BaseDengueModel.initial_conditions(NH0, NMI0, NMA0; ih_frac = 1e-5)

# Baseline parameter template (everything fixed except the three we fit)
p0 = BaseDengueModel.BaseParams()

# Free parameters: [Β_V, Β_H, δ_H]
p_free0 = [p0.Β_V, p0.Β_H, p0.δ_H]

# Helper to rebuild full BaseParams from free params
function make_params(p_free, p_template::BaseDengueModel.BaseParams)
    Β_V_fit, Β_H_fit, δ_H_fit = p_free

    return BaseDengueModel.BaseParams(
        Π_H  = p_template.Π_H,
        μ_H  = p_template.μ_H,
        σ_H  = p_template.σ_H,
        γ_H  = p_template.γ_H,
        δ_H  = δ_H_fit,
        Β_V  = Β_V_fit,
        Β_H  = Β_H_fit,
        α    = p_template.α,
        Π_V  = p_template.Π_V,
        μ_VI = p_template.μ_VI,
        μ_VA = p_template.μ_VA,
        σ_V  = p_template.σ_V
    )
end

const IDX_IH = BaseDengueModel.IDX_IH

# ---------------------------
# 3. Simulation + loss for optimization
# ---------------------------

function simulate_IH(p_free)
    p_all = make_params(p_free, p0)
    prob  = ODEProblem(BaseDengueModel.dengue_rhs!, u0, tspan_weeks, p_all)

    sol = solve(prob; saveat = t_data)

    # I_H at each data time point
    return [sol[i][IDX_IH] for i in eachindex(t_data)]
end

# Simple sum-of-squares loss between model I_H and reported monthly cases
# (we're treating I_H as a proxy for incidence here)
function loss(p_free, _)
    I_model = simulate_IH(p_free)
    return sum((I_model .- cases_data).^2)
end

# ---------------------------
# 4. Run optimization
# ---------------------------

println("\nRunning parameter optimization for [Β_V, Β_H, δ_H]...")

lower = [0.1, 0.05, 0.0]      # Β_V, Β_H, δ_H ≥ 0
upper = [0.85, 0.85, 1e-3]    # biologically reasonable bounds

opt_fun  = OptimizationFunction(loss, AutoZygote())
opt_prob = OptimizationProblem(opt_fun, p_free0; lb = lower, ub = upper)

res = Optimization.solve(opt_prob, Fminbox(BFGS()))

p_free_est = res.u
println("\n=== Estimated free parameters ===")
println("Β_V  (human -> vector bite infection): ", p_free_est[1])
println("Β_H  (vector -> human bite infection): ", p_free_est[2])
println("δ_H  (dengue-induced human mortality): ", p_free_est[3])
println("Final loss: ", res.minimum)

p_est = make_params(p_free_est, p0)
println("\nFull fitted BaseParams: ")
println(p_est)

# ---------------------------
# 5. Compare model vs data and plot
# ---------------------------

# Re-simulate with fitted parameters
prob_est = ODEProblem(BaseDengueModel.dengue_rhs!, u0, tspan_weeks, p_est)
sol_est  = solve(prob_est; saveat = t_data)
I_model_est = [sol_est[i][IDX_IH] for i in eachindex(t_data)]

println("\nFirst few months: model infectious humans vs reported cases")
println("Year  Month  t(weeks)   I_H(model)      cases(data)")
println("----  -----  ---------  -------------  -----------")

for i in 1:min(6, length(t_data))
    println(@sprintf("%4d  %5d  %9.2f  %13.3e  %11.0f",
        years[i], months[i], t_data[i], I_model_est[i], cases_data[i]))
end

# Time-series plot
plt = plot(
    t_data, I_model_est,
    label = "Model I_H(t)",
    xlabel = "Time (weeks since first record)",
    ylabel = "Count (arbitrary units)",
    lw = 2,
)

plot!(t_data, cases_data,
      label = "Reported cases",
      lw = 2,
      seriestype = :scatter)

display(plt)
