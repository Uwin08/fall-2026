# ============================================================================
# Uwin Ariyarathna | ECON 6343 Econometrics III | Problem Set 8
# ============================================================================

import Pkg
for package in ("Optim", "DataFrames", "CSV", "HTTP", "GLM",
                "Distributions", "MultivariateStats", "ForwardDiff")
    if Base.find_package(package) === nothing
        println("Installing missing PS8 package: ", package)
        Pkg.add(package)
    end
end

using Test, Random, LinearAlgebra, Statistics, Distributions
using Optim, DataFrames, CSV, HTTP, GLM
using MultivariateStats, ForwardDiff

include(joinpath(@__DIR__, "Uwin_source.jl"))

"""Deterministic synthetic NLSY-shaped dataset, requiring no network access."""
function mock_nlsy(n::Int=160; seed::Int=6343)
    rng = MersenneTwister(seed)
    black = Float64.(rand(rng, n) .< 0.30)
    hispanic = Float64.(rand(rng, n) .< 0.20)
    female = Float64.(rand(rng, n) .< 0.50)
    schoolt = Float64.(rand(rng, 10:18, n))
    gradHS = Float64.(schoolt .>= 12)
    grad4yr = Float64.(schoolt .>= 16)
    xi = randn(rng, n)
    tests = [0.15 .* black .- 0.10 .* female .+ (0.6+0.1j).*xi .+
             (0.45+0.03j).*randn(rng,n) for j in 1:6]
    logwage = 1.5 .+ 0.06 .* schoolt .+ 0.10 .* grad4yr .+ 0.2 .* xi .+
              0.3 .* randn(rng,n)
    df = DataFrame(logwage=logwage, black=black, hispanic=hispanic, female=female,
                   schoolt=schoolt, gradHS=gradHS, grad4yr=grad4yr)
    for j in 1:6
        df[!, ASVAB_COLS[j]] = tests[j]
    end
    return df
end

@testset "PS8 independent unit tests" begin
    df = mock_nlsy()

    @testset "Data validation and file loading" begin
        @test size(clean_data(df), 1) == 160
        bad = copy(df)
        allowmissing!(bad, :logwage)
        bad[1, :logwage] = missing
        @test nrow(clean_data(bad)) == 159
        @test_throws Exception clean_data(select(df, Not(:asvabAR)))
        mktempdir() do d
            path = joinpath(d, "sample.csv")
            CSV.write(path, df)
            @test size(load_data(path)) == size(df)
        end
    end

    @testset "Q1 Q2 Q3 ordinary regression and correlations" begin
        base = estimate_base_regression(df)
        full = estimate_full_regression(df)
        @test length(coef(base)) == 7
        @test length(coef(full)) == 13
        @test all(isfinite, coef(base))
        @test all(isfinite, coef(full))
        @test deviance(full) <= deviance(base) + 1e-7
        C = Matrix(compute_asvab_correlations(df))
        @test size(C) == (6,6)
        @test C ≈ C' atol=1e-12
        @test diag(C) ≈ ones(6) atol=1e-12
        @test all(abs.(C) .<= 1+1e-10)
        @test nrow(regression_table(base, "base")) == 7
    end

    @testset "Q4 PCA and Q5 one-factor scores" begin
        pc, _ = first_asvab_pca(df)
        fac, _ = first_asvab_factor(df)
        @test length(pc) == nrow(df)
        @test length(fac) == nrow(df)
        @test abs(mean(pc)) < 1e-8
        @test std(pc) > 0
        @test std(fac) > 0
        @test length(coef(estimate_pca_regression(df))) == 8
        @test length(coef(estimate_factor_regression(df))) == 8
        @test :asvabPCA ∉ Symbol.(names(df))
        @test :asvabFactor ∉ Symbol.(names(df))
        # Extra columns must not change the selected six named ASVAB measures.
        augmented = copy(df)
        augmented[!, :irrelevant] = 100 .* randn(MersenneTwister(1), nrow(df))
        pc2, _ = first_asvab_pca(augmented)
        @test abs(cor(pc, pc2)) ≈ 1 atol=1e-8
    end

    @testset "Optimizer compatibility with current Optim.jl" begin
        # Catch changes to Optim's automatic differentiation keyword API.
        objective = x -> sum(abs2, x)
        result = Optim.optimize(objective, [1.0, -2.0], Optim.LBFGS(),
                                Optim.Options(iterations=100);
                                autodiff=Optim.ADTypes.AutoForwardDiff())
        @test Optim.converged(result)
        @test norm(Optim.minimizer(result)) < 1e-5
    end

    @testset "Q6 matrices, quadrature, parameter ordering" begin
        X, y, Xfac, M = prepare_factor_matrices(df)
        @test size(X) == (160,7)
        @test size(Xfac) == (160,4)
        @test size(M) == (160,6)
        @test all(X[:,1] .== 1)
        @test all(Xfac[:,1] .== 1)
        @test y == df.logwage
        nodes, weights = gauss_legendre(41)
        @test length(nodes) == 41 && length(weights) == 41
        @test sum(weights) ≈ 16.0 atol=1e-11
        @test sum(nodes .* weights) ≈ 0 atol=1e-11
        @test sum((nodes.^2) .* weights) ≈ 2*8^3/3 atol=1e-9
        theta = factor_starting_values(df)
        @test length(theta) == 45
        alpha, beta, gamma, delta, sigma = unpack_factor_parameters(theta, 7,4,6)
        @test size(alpha) == (4,6)
        @test length(beta) == 7
        @test length(gamma) == 6
        @test delta isa Real
        @test length(sigma) == 7
        @test all(sigma .> 0)
        @test_throws DimensionMismatch unpack_factor_parameters(theta[1:end-1], 7,4,6)
    end

    @testset "Q6 likelihood: numerical, sign, derivative and invariance checks" begin
        tiny = mock_nlsy(18)
        X, y, Xfac, M = prepare_factor_matrices(tiny)
        theta = factor_starting_values(tiny)
        val = factor_model(theta, X, Xfac, M, y, 41)
        @test isfinite(val)
        @test factor_model(theta, X, Xfac, M, y, 41) ≈ val
        @test all(isfinite, ForwardDiff.gradient(t -> factor_model(t, X, Xfac, M, y, 21), theta))
        # Gaussian latent factor is symmetric: negating all loadings changes nothing.
        flipped = copy(theta)
        flipped[32:38] .*= -1  # 24 alpha + 7 beta + 6 gamma + 1 delta
        @test factor_model(flipped, X, Xfac, M, y, 41) ≈ val atol=1e-7
        # When every loading is zero, integral of prior is ~1, so likelihood
        # should match the product of independent normal measurement/wage pdfs.
        independent = copy(theta)
        independent[32:38] .= 0.0
        alpha,beta,gamma,delta,sigma = unpack_factor_parameters(independent,7,4,6)
        reference = sum(logpdf.(Normal(0,1), (y .- X*beta)./sigma[end]) .- log(sigma[end]))
        for j in 1:6
            reference += sum(logpdf.(Normal(0,1), (M[:,j] .- Xfac*alpha[:,j])./sigma[j]) .- log(sigma[j]))
        end
        @test factor_model(independent, X, Xfac, M, y, 41) ≈ -reference atol=1e-6
        table = format_results(theta, ones(length(theta)), -val, M)
        @test nrow(table) == 45
        @test all(table.std_error .> 0)
        @test table.variable[end] == "sigma_error_sd"
    end
end
println("PS8 independent unit tests completed.")
