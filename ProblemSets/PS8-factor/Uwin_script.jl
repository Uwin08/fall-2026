# ============================================================================
# Uwin Ariyarathna | ECON 6343 Econometrics III | Problem Set 8
# Run from VS Code: julia Uwin_script.jl
# The source file deliberately contains no imports or package management.
import Pkg
for package in ("Optim", "DataFrames", "CSV", "HTTP", "GLM",
                "Distributions", "MultivariateStats", "ForwardDiff")
    if Base.find_package(package) === nothing
        println("Installing missing PS8 package: ", package)
        Pkg.add(package)
    end
end

using Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, CSV, HTTP, GLM
using MultivariateStats, ForwardDiff

include(joinpath(@__DIR__, "Uwin_source.jl"))

function main()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS8-factor/nlsy.csv"
    location = isempty(ARGS) ? url : ARGS[1]
    println("="^72, "\nECON 6343 | Problem Set 8 | Uwin Ariyarathna\n", "="^72)
    df_raw = load_data(location)
    df = clean_data(df_raw)
    println("NLSY: $(nrow(df)) complete observations, $(nrow(df_raw)-nrow(df)) excluded")
    outdir = joinpath(@__DIR__, "Uwin_PS8_outputs")
    mkpath(outdir)

    all_regs = DataFrame[]
    println("\nQ1. BASELINE LOG-WAGE REGRESSION")
    base = estimate_base_regression(df)
    println(coeftable(base))
    push!(all_regs, regression_table(base, "Q1_base"))

    println("\nQ2. ASVAB CORRELATION MATRIX")
    correlations = compute_asvab_correlations(df)
    display(correlations)
    CSV.write(joinpath(outdir, "Uwin_PS8_Q2_correlations.csv"), correlations)
    C = Matrix(correlations)
    upper = [abs(C[i,j]) for i in 1:5 for j in i+1:6]
    println("Maximum absolute pairwise ASVAB correlation: ", round(maximum(upper), digits=4))
    println("Interpretation: Strong correlation can inflate standard errors of individual ASVAB coefficients, although OLS is still defined if there is no exact collinearity.")

    println("\nQ3. REGRESSION WITH SIX ASVAB SCORES")
    full = estimate_full_regression(df)
    println(coeftable(full))
    push!(all_regs, regression_table(full, "Q3_six_ASVAB"))

    println("\nQ4. PCA: FIRST PRINCIPAL COMPONENT")
    pca_scores, pca_fit = first_asvab_pca(df)
    pca_reg = estimate_pca_regression(df)
    println(coeftable(pca_reg))
    println("PC1 correlation with grad4yr: ", cor(pca_scores, df.grad4yr))
    println("PC1 sign is arbitrary; reversing score signs reverses its regression coefficient.")
    push!(all_regs, regression_table(pca_reg, "Q4_PCA"))

    println("\nQ5. FACTOR ANALYSIS: FIRST FACTOR")
    factor_scores, factor_fit = first_asvab_factor(df)
    factor_reg = estimate_factor_regression(df)
    println(coeftable(factor_reg))
    println("Factor 1 correlation with grad4yr: ", cor(factor_scores, df.grad4yr))
    println("Factor orientation is arbitrary, as with PCA.")
    push!(all_regs, regression_table(factor_reg, "Q5_factor_analysis"))
    CSV.write(joinpath(outdir, "Uwin_PS8_Q1_Q3_Q4_Q5_regressions.csv"), vcat(all_regs...))
    CSV.write(joinpath(outdir, "Uwin_PS8_component_scores.csv"),
              DataFrame(pca_score=pca_scores, factor_score=factor_scores))

    println("\nQ6. FULL MEASUREMENT SYSTEM: MLE")
    println("Latent factor N(0,1), 31-point Gauss-Legendre quadrature on [-8,8].")
    start = factor_starting_values(df)
    X, y, Xfac, M = prepare_factor_matrices(df)
    println("Parameter count: ", length(start), "; initial log likelihood: ",
            -factor_model(start, X, Xfac, M, y, 31))
    theta, se, loglike, converged = run_estimation(df, start; R=31, max_iterations=350)
    result_table = format_results(theta, se, loglike, M)
    println("Converged: ", converged, "; maximized log likelihood: ", loglike)
    display(result_table)
    CSV.write(joinpath(outdir, "Uwin_PS8_Q6_factor_MLE.csv"), result_table)
    CSV.write(joinpath(outdir, "Uwin_PS8_Q6_summary.csv"),
              DataFrame(observations=[length(y)], parameters=[length(theta)],
                        converged=[converged], log_likelihood=[loglike], quadrature_points=[31]))

    open(joinpath(outdir, "Uwin_PS8_writeup.txt"), "w") do io
        println(io, "ECON 6343 - PS8 - Uwin Ariyarathna")
        println(io, "Common complete-case sample: N = $(nrow(df))")
        println(io, "Q2: Max absolute ASVAB pairwise correlation = $(maximum(upper)).")
        println(io, "Q3: Correlated ASVAB variables may lead to multicollinearity, which inflates")
        println(io, "coefficient uncertainty and makes individual estimates sensitive to small changes.")
        println(io, "Q4: PCA summarizes the maximum observed common variance in a single component.")
        println(io, "Q5: Factor analysis models shared variation separately from unique measurement variation.")
        println(io, "Q4-Q5: The signs of principal component and factor scores are arbitrary.")
        println(io, "Q6: Joint MLE integrates N(0,1) latent ability using Gauss-Legendre quadrature.")
        println(io, "Q6: Measurement equation covariates are intercept, black, hispanic, female.")
        println(io, "Q6: Wage equation adds schoolt, gradHS, and grad4yr.")
        println(io, "Q6: Error SDs are estimated as exp(log_sigma) for positivity.")
        println(io, "Q6: Converged = $converged; log likelihood = $loglike.")
        println(io, "Q6: Check convergence and standard errors before substantive interpretation.")
    end
    println("\nCompleted Q1-Q6. Saved CSV and TXT files to $outdir")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
