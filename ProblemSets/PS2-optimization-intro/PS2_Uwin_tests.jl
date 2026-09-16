# ==============================================================================
# ECON 6343: Econometrics III
# Problem Set 2
# Name: Uwin Ariyarathna
# Division of Finance
# University of Oklahoma
#
# File: PS2_Uwin_tests.jl
# Purpose: Unit tests for functions/components used.
# ==============================================================================

using Test
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables

cd(@__DIR__)

include("PS2_Uwin_source.jl")

Random.seed!(1234)

@testset "ECON 6343 Problem Set 2 - Uwin Ariyarathna" begin

    # ==========================================================================
    # Question 1 component test: basic optimization
    # ==========================================================================
    @testset "Question 1: Basic optimization" begin
        f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
        negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2

        result = optimize(negf, [-7.0], LBFGS())

        @test Optim.converged(result)
        @test isapprox(result.minimizer[1], -7.3782434055; atol=1e-5)
        @test isapprox(-result.minimum, 964.3133837824; atol=1e-4)
        @test isapprox(f(result.minimizer), -result.minimum; atol=1e-8)
    end


    # ==========================================================================
    # Question 2 tests: OLS objective function
    # ==========================================================================
    @testset "Question 2: OLS" begin
        X = [
            1.0 0.0
            1.0 1.0
            1.0 2.0
            1.0 3.0
        ]
        beta_true = [1.0, 2.0]
        y = X * beta_true

        # Perfect fit should have SSR = 0.
        @test isapprox(ols(beta_true, X, y), 0.0; atol=1e-12)

        result = optimize(
            b -> ols(b, X, y),
            zeros(2),
            LBFGS(),
            Optim.Options(g_tol=1e-8, iterations=100_000)
        )

        bols = inv(X' * X) * X' * y

        @test Optim.converged(result)
        @test isapprox(result.minimizer, beta_true; atol=1e-6)
        @test isapprox(result.minimizer, bols; atol=1e-6)
    end


    # ==========================================================================
    # Question 3 tests: binary logit likelihood
    # ==========================================================================
    @testset "Question 3: Binary logit likelihood" begin
        X = ones(4, 1)
        y = [0.0, 1.0, 0.0, 1.0]

        # alpha = 0 gives fitted probability 0.5 for every observation.
        # Negative log-likelihood therefore equals 4*log(2).
        @test isapprox(logit([0.0], X, y), 4 * log(2); atol=1e-10)
    end


    # ==========================================================================
    # Question 4 component test: Optim logit versus GLM
    # ==========================================================================
    @testset "Question 4: Optim versus GLM logit" begin
        x = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        y = [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        X = [ones(length(x)) x]

        result = optimize(
            alpha -> logit(alpha, X, y),
            zeros(2),
            LBFGS(),
            Optim.Options(g_tol=1e-8, iterations=100_000)
        )

        df_test = DataFrame(y=y, x=x)
        model_glm = glm(@formula(y ~ x), df_test, Binomial(), LogitLink())

        @test Optim.converged(result)
        @test isapprox(result.minimizer, coef(model_glm); atol=1e-5)
    end


    # ==========================================================================
    # Question 5 tests: multinomial logit likelihood
    # ==========================================================================
    @testset "Question 5: Multinomial logit" begin
        X = [
            1.0 0.0
            1.0 1.0
            1.0 2.0
            1.0 0.5
            1.0 1.5
            1.0 2.5
        ]

        y = [1, 2, 3, 1, 2, 3]

        K = size(X, 2)
        J = length(unique(y))
        alpha_zero = zeros(K * (J - 1))

        # With all coefficients equal to zero, each alternative has probability 1/J.
        expected_nll = length(y) * log(J)

        @test isapprox(mlogit(alpha_zero, X, y), expected_nll; atol=1e-10)
        @test isfinite(mlogit(alpha_zero, X, y))
    end


    # ==========================================================================
    # Question 6 test: wrapper exists
    # ==========================================================================
    @testset "Question 6: Wrapper" begin
        @test hasmethod(PS2, Tuple{})
    end

end

println("\nAll Problem Set 2 unit tests finished.")
