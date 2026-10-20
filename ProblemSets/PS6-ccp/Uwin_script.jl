# Problem Set 6 | ECON 6343 | Uwin Ariyarathna, Division of Finance
# Run from VS Code using: julia "Uwin_script.jl" "path/to/PS5_second.csv"
# Alternatively set ENV["PS6_DATA_URL"] to the second CSV URL used in PS5.

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV

include(joinpath(@__DIR__, "Uwin_source.jl"))
# The professor's supplied helper was named create_grids(1).jl.
include(joinpath(@__DIR__, "create_grids(1).jl"))

function main(; data_url::String=get(ENV, "PS6_DATA_URL", joinpath(@__DIR__, "busdataBeta0.csv")))
    println(repeat("=",60))
    println("Rust (1987) Bus Engine Replacement Model - CCP Estimation")
    println("Uwin Ariyarathna | ECON 6343 | Problem Set 6")
    println(repeat("=",60))
    isfile(data_url) || startswith(data_url, "https://") || error("Cannot find data: $data_url")
    β = 0.9

    println("\nQ1: Reading and reshaping PS5 data")
    df_long, Xstate, Zstate, Branded = load_and_reshape_data(data_url)
    println("Long panel: ", size(df_long), " | bus × periods: ", size(Xstate))

    println("\nQ2: Estimating fully interacted flexible logit")
    flex_logit = estimate_flexible_logit(df_long)
    println("Flexible logit estimated with full 7-factor formula")

    println("\nQ3(a): Constructing the PS5 transition grids")
    zval,zbin,xval,xbin,xtran = create_grids()
    println("Mileage bins = $xbin, route bins = $zbin")

    println("\nQ3(b): Evaluating CCP future values on the state grid")
    state_df = construct_state_space(xbin,zbin,xval,zval)
    T = size(Xstate,2)
    FV = compute_future_values(state_df,flex_logit,xtran,xbin,T,β)

    println("\nQ3(b): Mapping state-grid terms to observed bus periods")
    fvt1 = compute_fvt1(df_long,FV,xtran,Xstate,Zstate,xbin,Branded)

    println("\nQ3(c): Estimating structural parameters")
    theta_hat_ccp_glm = estimate_structural_params(df_long,fvt1)
    println("\nStructural model (offset coefficient fixed at 1):")
    show(stdout, MIME("text/plain"), coeftable(theta_hat_ccp_glm))
    println("\n\nCoefficients = ", coef(theta_hat_ccp_glm))
    println("Standard errors = ", stderror(theta_hat_ccp_glm))

    # Required output files. A separate writeup may be included as these code comments.
    output = DataFrame(parameter=String.(coefnames(theta_hat_ccp_glm)),
                       estimate=coef(theta_hat_ccp_glm),
                       std_error=stderror(theta_hat_ccp_glm))
    CSV.write(joinpath(@__DIR__, "PS6_Ariyarathna_structural_estimates.csv"), output)
    panel = copy(df_long)
    panel[!, :fv] = fvt1
    CSV.write(joinpath(@__DIR__, "PS6_Ariyarathna_panel_with_fv.csv"), panel)
    return (flex_logit=flex_logit, structural=theta_hat_ccp_glm, data=panel)
end

if abspath(PROGRAM_FILE) == @__FILE__
    url = !isempty(ARGS) ? ARGS[1] : get(ENV,"PS6_DATA_URL", joinpath(@__DIR__, "busdataBeta0.csv"))
    @time main(data_url=url)
end
