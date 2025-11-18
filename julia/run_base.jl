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

# ---------------------------
# 1. Load Indonesia monthly dengue incidence
# ---------------------------

# repo root is one level up from julia/
const REPO_ROOT = dirname(@__DIR__)

idn = IDNData.load_idn_monthly_cases(REPO_ROOT)
records = idn.records

println("Loaded $(length(records)) monthly records for $(idn.country) ($(idn.iso_code)).")

# Time span in days relative to the first record start date
start_date = first(records).start_date
end_date   = last(records).end_date
total_days = Dates.value(end_date - start_date) + 1
tspan = (0.0, float(total_days))

# ---------------------------
# 2. Set parameters and initial conditions
# ---------------------------

# Rough scales – you can refine these later
NH0  = 270e6     # ~ total population of Indonesia (order of magnitude)
NMI0 = 1e8       # aquatic mosquitoes (arbitrary but large)
NMA0 = 1e8       # adult mosquitoes (arbitrary but large)

# Parameter set: fixed parameters from table, λ_VH and λ_HV as initial guesses
p = BaseDengueModel.BaseParams(
    λ_VH = 1e-8,   # initial guess – to be calibrated
    λ_HV = 1e-8    # initial guess – to be calibrated
    # everything else uses defaults from BaseParams
)

u0 = BaseDengueModel.initial_conditions(NH0, NMI0, NMA0; ih_frac = 1e-5)

println("Solving base ODE model over $(total_days) days...")
sol = BaseDengueModel.solve_base_model(u0, p, tspan)

# ---------------------------
# 3. Sample model at monthly points and compare with data
# ---------------------------

# Use the end of each calendar interval as the sampling time
ts_monthly = Float64[]
for rec in records
    t = Dates.value(rec.end_date - start_date) |> float
    push!(ts_monthly, t)
end

# Model "incidence proxy": infectious humans at that time (I_H)
I_H_model = [sol(t)[BaseDengueModel.IDX_IH] for t in ts_monthly]
cases_data = [rec.dengue_total for rec in records]

println("\nFirst few months: model infectious humans vs reported cases")
println("Year  Month  t(days)  I_H(model)      cases(data)")
println("----  -----  -------  ------------   -----------")

for i in 1:min(6, length(records))
    rec = records[i]
    t   = ts_monthly[i]
    println(@sprintf("%4d  %5d  %7.1f  %12.3e   %11d",
        rec.year, month(rec.start_date), t, I_H_model[i], cases_data[i]))
end

println("\n=== Calibrated parameter values ===")

# Replace these with your actual field names
println("λ_VH (vector → human FOI): ", p.λ_VH)
println("λ_HV (human → vector FOI): ", p.λ_HV)

println("σ_V  (maturation rate VI → VA): ", p.σ_V)
println("δ_VI (extra immature mosquito mortality): ", p.δ_VI)
println("δ_VA (extra adult mosquito mortality): ", p.δ_VA)

println("μ_H  (fixed human natural mortality): ", p.μ_H)
println("Π_H  (fixed human recruitment): ", p.Π_H)

ts  = IDNData.load_idn_monthly_cases()

records = ts.records

# Take the first record's start date as t = 0
t0 = records[1].start_date

t_days   = Float64[]
I_model  = Float64[]
cases    = Float64[]
years    = Int[]
months   = Int[]

for rec in records
    # center of the month in "days since t0"
    Δ_start = rec.start_date - t0   # Day
    Δ_end   = rec.end_date   - t0   # Day
    tmid = (Dates.value(Δ_start) + Dates.value(Δ_end)) / 2  # Float64 days

    push!(t_days, tmid)
    push!(years, rec.year)
    push!(months, month(rec.start_date))   # uses Dates.month(...)
    
    # Evaluate solution at tmid (assuming u = [S_H, E_H, I_H, ...])
    u_t = sol(tmid)
    I_H_t = u_t[3]   # Infectious humans
    push!(I_model, I_H_t)

    push!(cases, rec.dengue_total)
end

# Simple time-series plot
plt = plot(
    t_days, I_model,
    label = "Model I_H(t)",
    xlabel = "Days since first record",
    ylabel = "Count",
    lw = 2,
)

plot!(t_days, cases,
      label = "Reported cases",
      lw = 2,
      seriestype = :scatter,
)

display(plt)

# (Optional) you can add plotting if Plots.jl is in your environment:
# using Plots
# plot(ts_monthly ./ 30.0, I_H_model, label="I_H model (scaled)",
#      xlabel="time (months since start)", ylabel="infectious humans")
# scatter!(ts_monthly ./ 30.0, cases_data, label="reported cases")
