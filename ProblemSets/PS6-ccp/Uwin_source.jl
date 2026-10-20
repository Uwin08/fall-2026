# Problem Set 6 | ECON 6343 | Uwin Ariyarathna, Division of Finance
# Function definitions only. Load packages in the script or tests before including.
# Y = 1 denotes continuing; Y = 0 denotes replacement. CCP p0 is the replacement probability.

function _getcol(df::DataFrame, candidates::Vector{String}; required=true)
    lookup = Dict(lowercase(replace(String(n), r"[^a-zA-Z0-9]" => "")) => n for n in names(df))
    for c in candidates
        key = lowercase(replace(c, r"[^a-zA-Z0-9]" => ""))
        haskey(lookup, key) && return lookup[key]
    end
    required && error("Missing column. Looked for $(join(candidates, ", ")). Available: $(join(names(df), ", "))")
    return nothing
end

function _period_columns(df::DataFrame, stems::Vector{String})
    hits = Tuple{Int,String}[]
    for n in names(df)
        name = lowercase(replace(String(n), "_" => ""))
        for stem in stems
            m = match(Regex("^" * lowercase(stem) * raw"(\d+)$"), name)
            if m !== nothing
                push!(hits, (parse(Int, m.captures[1]), n))
                break
            end
        end
    end
    sort!(hits, by=first)
    return hits
end

"""Load the second PS5 CSV and create bus-major, time-minor panel observations.

Supported wide columns: Xst1..XstT, Zst or Zstate, Branded, and
Odometer1..T / X1..T, Y1..T / Decision1..T. RouteUsage is read
from RouteUsage / Z (or PS5's route grid from Zst).
Uses the PS5 busdataBeta0.csv column names, including Odo1..Odo20.
"""
function load_and_reshape_data(url::String)
    raw = if startswith(lowercase(url), "http://") || startswith(lowercase(url), "https://")
        response = HTTP.get(url; status_exception=true)
        CSV.read(IOBuffer(response.body), DataFrame)
    else
        CSV.read(url, DataFrame)
    end
    n = nrow(raw)
    n > 0 || error("CSV contains no rows")
    xst_cols = _period_columns(raw, ["xst", "xstate"])
    isempty(xst_cols) && error("Expected PS5 discretized mileage columns Xst1, Xst2, ...")
    periods = first.(xst_cols)
    T = length(periods)
    all(diff(periods) .== 1) || error("Xst periods must be consecutive")
    Xstate = Matrix{Int}(undef, n, T)
    for (j, (_, c)) in enumerate(xst_cols)
        Xstate[:, j] = Int.(raw[!, c])
    end
    zcol = _getcol(raw, ["Zst", "Zstate", "RouteUsageState"])
    Zstate = Int.(raw[!, zcol])
    bcol = _getcol(raw, ["Branded", "Brand", "B"])
    Branded = Int.(raw[!, bcol])
    all(in((0,1)), Branded) || error("Branded indicators must be 0 or 1")

    ycols = _period_columns(raw, ["y", "replace", "replacement", "decision", "d"])
    ocols = _period_columns(raw, ["odo", "odometer", "mileage", "x"])
    ymap = Dict(ycols)
    omap = Dict(ocols)
    all(haskey(ymap, t) for t in periods) || error("Missing binary choice columns such as Y1...YT; review PS5 naming")
    all(haskey(omap, t) for t in periods) || error("Missing Odometer1...T or X1...T columns; review PS5 naming")
    rcol = _getcol(raw, ["RouteUsage", "Z", "Route", "Usage"]; required=false)
    # Default route grid matches the supplied create_grids() implementation.
    route = rcol === nothing ? (0.25 .+ 0.01 .* (Zstate .- (minimum(Zstate) == 0 ? 0 : 1))) : Float64.(raw[!, rcol])
    idcol = _getcol(raw, ["bus_id", "busid", "id"]; required=false)
    ids = idcol === nothing ? collect(1:n) : collect(raw[!, idcol])
    length(unique(ids)) == n || error("Bus identifiers must be unique in the wide data")

    # Match FVT1'[:] below: all T periods of bus 1, then bus 2, etc.
    long = DataFrame(bus_id=repeat(ids, inner=T), time=repeat(periods, outer=n),
                     Odometer=Vector{Float64}(undef,n*T),
                     RouteUsage=repeat(route, inner=T),
                     Branded=repeat(Branded, inner=T),
                     Y=Vector{Int}(undef,n*T))
    for i in 1:n, j in 1:T
        k = (i-1)*T+j
        long.Odometer[k] = Float64(raw[i, omap[periods[j]]])
        long.Y[k] = Int(raw[i, ymap[periods[j]]])
    end
    all(in((0,1)), long.Y) || error("Replacement decisions Y must be 0 or 1")
    return long, Xstate, Zstate, Branded
end

"""Add the three squared predictors in both training and prediction data."""
function add_squared_terms!(df::DataFrame)
    df[!, :Odometer_sq] = Float64.(df.Odometer).^2
    df[!, :RouteUsage_sq] = Float64.(df.RouteUsage).^2
    df[!, :time_sq] = Float64.(df.time).^2
    return df
end

"""Estimate the complete seven-factor interaction logit (2^7 = 128 candidate terms)."""
function estimate_flexible_logit(df::DataFrame)
    training = add_squared_terms!(copy(df))
    formula = @formula(Y ~ Odometer * Odometer_sq * RouteUsage * RouteUsage_sq * Branded * time * time_sq)
    return glm(formula, training, Binomial(), LogitLink(); dropcollinear=true)
end

"""Return states in precisely the same ordering as create_grids() rows."""
function construct_state_space(xbin::Int, zbin::Int, xval::Vector, zval::Vector)
    length(xval) == xbin || throw(ArgumentError("xbin does not match xval"))
    length(zval) == zbin || throw(ArgumentError("zbin does not match zval"))
    return DataFrame(Odometer=kron(ones(zbin), xval),
                     RouteUsage=kron(zval, ones(xbin)),
                     Branded=zeros(Int, xbin*zbin), time=zeros(Int, xbin*zbin))
end

"""Compute -β log(p0) for each state and choice-specific brand and period.
At time T+1 the future-value term is zero (terminal period).
"""
function compute_future_values(state_df::DataFrame, flex_logit, xtran::Matrix,
                               xbin::Int, T::Int, β::Float64)
    size(state_df,1) == size(xtran,1) || error("State data and transition rows differ")
    FV = zeros(Float64, size(xtran,1), 2, T+1)
    newdata = copy(state_df)
    for t in 2:T, b in 0:1
        newdata[!, :time] .= t
        newdata[!, :Branded] .= b
        add_squared_terms!(newdata)
        p0 = 1.0 .- predict(flex_logit, newdata)
        # Avoid log(0) due to floating-point saturation from the rich logit.
        FV[:,b+1,t] .= -β .* log.(clamp.(p0, eps(Float64), 1.0))
    end
    return FV
end

"""Map state-grid value differences back to the bus-major long panel."""
function compute_fvt1(df_long::DataFrame, FV::Array{Float64,3}, xtran::Matrix,
                      Xstate::Matrix, Zstate::Vector, xbin::Int, B::Vector)
    N,T = size(Xstate)
    nrow(df_long) == N*T || error("Long panel size does not match N*T")
    length(Zstate) == N && length(B) == N || error("Bus-level state vectors mismatch")
    size(FV,3) >= T+1 || error("Future value array needs T+1 slices")
    xzero = minimum(Xstate) == 0 ? 0 : 1
    zzero = minimum(Zstate) == 0 ? 0 : 1
    FVT1 = zeros(Float64,N,T)
    for i in 1:N, t in 1:T
        x = Xstate[i,t] - xzero + 1
        z = Zstate[i] - zzero + 1
        1 <= x <= xbin || error("Mileage state outside grid at bus $i period $t")
        row0 = (z-1)*xbin + 1
        row1 = row0 + x-1
        row1 <= size(xtran,1) || error("Route usage state outside grid at bus $i")
        brand = Int(B[i]) + 1
        1 <= brand <= 2 || error("Invalid Branded state")
        FVT1[i,t] = dot(xtran[row1,:] .- xtran[row0,:],
                        FV[row0:row0+xbin-1,brand,t+1])
    end
    return vec(permutedims(FVT1))
end

"""Structural logit with future-value term offset constrained to coefficient one."""
function estimate_structural_params(df_long::DataFrame, fvt1::Vector)
    length(fvt1) == nrow(df_long) || error("Future value vector has wrong length")
    df = copy(df_long)
    df[!, :fv] = fvt1
    return glm(@formula(Y ~ Odometer + Branded), df,
               Binomial(), LogitLink(); offset=df.fv)
end


#=
PS6 WRITEUP
Uwin Ariyarathna | Division of Finance | University of Oklahoma
ECON 6343: Econometrics III | Problem Set 6

Q1. Data loading and long-panel preparation
The second PS5 CSV file, busdataBeta0.csv, was loaded and reshaped from
wide format to a bus-by-period panel. The resulting dataset has 1,000 buses
observed for 20 periods, yielding 20,000 observations. The corresponding
mileage and route-usage state indices were retained for Q3.

Q2. Flexible logit estimation
I estimated a binomial logit for the decision Y, where Y=1 means continuing
to operate the bus and Y=0 means replacing its engine. The specification
uses Odometer, Odometer squared, RouteUsage, RouteUsage squared, Branded,
time, and time squared, including all interactions among these seven
factors (128 model-matrix terms counting the intercept). Some terms are
collinear and are dropped by the estimator, as allowed by the assignment.
The fitted model provides the conditional choice probabilities for Q3.

Q3(a). State transitions
I reused the PS5 create_grids() helper, which returns 201 mileage bins
and 101 route-usage bins, together with the transition probabilities.

Q3(b). CCP future-value terms
With the discount factor set to beta=0.9, I evaluated the replacement
probability p0 = Pr(Y=0 | state) at each grid state and relevant future
time and brand state. I then computed -beta*log(p0), and used differences
between transition probabilities after continuing versus replacing to
map the future-value contribution to each observed bus-period row.

Q3(c). Structural estimation
I estimated a binomial logit of Y on Odometer and Branded with the CCP
future-value contribution entered as an offset, fixing its coefficient
to one. The run reported the following estimates (standard errors):
   Intercept:  1.88501   (0.03383)
   Odometer:  -0.06674   (0.00402)
   Branded:    1.39266   (0.04445)
All three reported coefficients were statistically significant at 1%.
Higher mileage is associated with lower conditional utility of continuing
to run the bus; a branded bus has higher conditional utility of continuing,
holding the other included model components fixed. These are model-based
conditional associations, not separate causal estimates.

Q3(d). The assignment's custom binary-logit estimator is optional and
was not implemented. Q3(e). The main() wrapper executes Q1-Q3; running
@time main() in Julia reports total execution time.

Q4. Unit testing
The separate Uwin_tests.jl file contains independently written unit
tests covering long-panel reshaping, model inputs, state-space construction,
CCP future-value calculations, and structural estimation. The recorded
Julia execution reported 22 tests passed out of 22 (14.3 seconds).

Submission notes
Uwin_script.jl writes PS6_Ariyarathna_structural_estimates.csv and
PS6_Ariyarathna_panel_with_fv.csv in the working assignment folder.
This commentary serves as the written explanation permitted in the PDF.
=#
