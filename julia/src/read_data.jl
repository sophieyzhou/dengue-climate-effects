module IDNData

using CSV
using DataFrames
using Dates

"One monthly dengue record for a country."
struct DengueMonthlyRecord
    year::Int
    start_date::Date
    end_date::Date
    dengue_total::Int
end

"Time series of monthly dengue totals for a single country."
struct DengueCountryMonthly
    country::String
    iso_code::String
    records::Vector{DengueMonthlyRecord}
end

function load_idn_monthly_cases(root::AbstractString = pwd())
    path = joinpath(root, "data", "idn_caseload.csv")

    df = CSV.read(path, DataFrame)

    # 1) Filter for monthly resolution rows
    monthly_df = df[df.T_res .== "Month", :]

    if nrow(monthly_df) == 0
        @warn "No rows with T_res == \"Month\" found in idn_cases.csv"
    end

    # 2) Pull unique country + ISO code from the filtered data
    unique_countries = unique(monthly_df.adm_0_name)
    unique_iso       = unique(monthly_df.ISO_A0)

    if length(unique_countries) != 1 || length(unique_iso) != 1
        @warn "More than one country/ISO in monthly data; using the first."
    end

    country = String(unique_countries[1])
    iso_code = String(unique_iso[1])

    # 3) Build DengueMonthlyRecord vector
    records = DengueMonthlyRecord[]
    for row in eachrow(monthly_df)
        year         = Int(row.Year)
        start_date   = row.calendar_start_date
        end_date     = row.calendar_end_date
        dengue_total = Int(row.dengue_total)

        push!(records, DengueMonthlyRecord(year, start_date, end_date, dengue_total))
    end

    # 4) Sort chronologically
    sort!(records, by = r -> r.start_date)

    return DengueCountryMonthly(country, iso_code, records)
end

end # module