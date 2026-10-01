# Uwin Ariyarathna
# Division of Finance
# ECON 6343: Econometrics III
# Problem Set 4 Unit Tests

using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions, ForwardDiff
using ADTypes: AutoForwardDiff

cd(@__DIR__)
Random.seed!(1234)

include("PS4_Uwin_source.jl")

@testset "PS4 Unit Tests" begin

    @testset "Question 1: Multinomial logit" begin
        Random.seed!(11)
        N, K, J = 40, 3, 4
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)

        theta = [0.1 .* randn(K*(J-1)); 0.25]
        nll = mlogit_with_Z(theta, X_test, Z_test, y_test)

        @test isa(nll, Real)
        @test isfinite(nll)
        @test nll > 0
        @test length(theta) == K*(J-1) + 1

        # If all coefficients are zero, all J choices have probability 1/J.
        theta_zero = zeros(K*(J-1) + 1)
        nll_zero = mlogit_with_Z(theta_zero, X_test, Z_test, y_test)
        @test nll_zero ≈ N*log(J) atol=1e-10
    end

    @testset "Question 3a: Quadrature" begin
        nodes, weights = lgwt(7, -4, 4)
        d = Normal(0, 1)

        integral_density = sum(weights .* pdf.(d, nodes))
        expectation = sum(weights .* nodes .* pdf.(d, nodes))

        @test length(nodes) == 7
        @test length(weights) == 7
        @test all(weights .> 0)
        @test integral_density ≈ 1.0 atol=0.01
        @test abs(expectation) < 1e-10
    end

    @testset "Question 3b: Variance quadrature" begin
        variance_7pts, variance_10pts = variance_quadrature()

        @test variance_7pts > 0
        @test variance_10pts > 0
        @test variance_10pts ≈ 4.0 atol=0.1
        @test abs(variance_10pts - 4.0) < abs(variance_7pts - 4.0)
    end

    @testset "Question 3c: Monte Carlo formula" begin
        Random.seed!(22)

        mc_integrate_test = function(f, a, b, D)
            draws = rand(D) * (b-a) .+ a
            return (b-a) * mean(f.(draws))
        end

        result = mc_integrate_test(x -> x^2, 0.0, 1.0, 100_000)
        @test result ≈ 1/3 atol=0.01
    end

    @testset "Question 4: Mixed logit quadrature" begin
        Random.seed!(33)
        N, K, J = 20, 2, 3
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        theta = [0.1 .* randn(K*(J-1)); 0.0; 0.5]

        nodes, weights = lgwt(7, -4, 4)
        nll = mixed_logit_quad(theta, X_test, Z_test, y_test, nodes, weights)

        @test isa(nll, Real)
        @test isfinite(nll)
        @test nll > 0
    end

    @testset "Question 5: Mixed logit Monte Carlo" begin
        Random.seed!(44)
        N, K, J = 20, 2, 3
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)
        theta = [0.1 .* randn(K*(J-1)); 0.0; 0.5]

        nll = mixed_logit_mc(theta, X_test, Z_test, y_test, 1_000)

        @test isa(nll, Real)
        @test isfinite(nll)
        @test nll > 0
    end

    @testset "Optimization setup functions" begin
        Random.seed!(55)
        N, K, J = 20, 2, 3
        X_test = randn(N, K)
        Z_test = randn(N, J)
        y_test = rand(1:J, N)

        q4_start = optimize_mixed_logit_quad(X_test, Z_test, y_test)
        q5_start = optimize_mixed_logit_mc(X_test, Z_test, y_test)

        @test length(q4_start) == K*(J-1) + 2
        @test length(q5_start) == K*(J-1) + 2
    end
end

println("All PS4 unit tests completed successfully!")
