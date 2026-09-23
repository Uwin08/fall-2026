# Uwin Ariyarathna
# Division of Finance
# ECON 6343: Econometrics III
# Problem Set 3

using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

# Problem Set 3 - Unit Tests


cd(@__DIR__)
include("PS3_Uwin_source.jl")

@testset "PS3 Uwin Ariyarathna Unit Tests" begin

    @testset "Multinomial Logit" begin
        Random.seed!(42)
        N, K, J = 40, 3, 8
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        theta = [randn(K * (J-1)); 0.1]

        value = mlogit_with_Z(theta, X, Z, y)
        @test isa(value, Real)
        @test isfinite(value)
        @test value >= 0

        # At zero parameters every alternative has probability 1/J.
        theta0 = zeros(K * (J-1) + 1)
        @test isapprox(mlogit_with_Z(theta0, X, Z, y), N * log(J); atol=1e-10)
    end

    @testset "MNL Probability Components" begin
        Random.seed!(123)
        N, K, J = 20, 3, 8
        X = randn(N, K)
        Z = randn(N, J)
        theta = [randn(K * (J-1)); 0.2]

        alpha = theta[1:end-1]
        gamma = theta[end]
        bigAlpha = [reshape(alpha, K, J-1) zeros(K)]
        num = zeros(N, J)
        for j = 1:J
            num[:,j] = exp.(X * bigAlpha[:,j] .+ gamma .* (Z[:,j] .- Z[:,J]))
        end
        P = num ./ sum(num, dims=2)

        @test all(P .>= 0)
        @test all(P .<= 1)
        @test all(abs.(sum(P, dims=2) .- 1) .< 1e-10)
    end

    @testset "Nested Logit" begin
        Random.seed!(456)
        N, K, J = 40, 3, 8
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        nesting_structure = [[1,2,3], [4,5,6,7]]
        theta = [randn(2*K); 0.8; 0.9; 0.1]

        value = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
        @test isa(value, Real)
        @test isfinite(value)
        @test value >= 0
    end

    @testset "Nested Logit at lambda = 1" begin
        Random.seed!(789)
        N, K, J = 30, 3, 8
        X = randn(N, K)
        Z = randn(N, J)
        y = rand(1:J, N)
        nesting_structure = [[1,2,3], [4,5,6,7]]
        theta = [randn(2*K); 1.0; 1.0; 0.1]

        value = nested_logit_with_Z(theta, X, Z, y, nesting_structure)
        @test isfinite(value)
        @test value > 0
    end

    @testset "Coefficient Matrix Structure" begin
        K, J = 3, 8
        nesting_structure = [[1,2,3], [4,5,6,7]]
        alpha = collect(1.0:6.0)
        bigAlpha = zeros(K, J)
        bigAlpha[:, nesting_structure[1]] .= repeat(alpha[1:K], 1, 3)
        bigAlpha[:, nesting_structure[2]] .= repeat(alpha[K+1:2K], 1, 4)

        @test bigAlpha[:,1] == bigAlpha[:,2] == bigAlpha[:,3]
        @test bigAlpha[:,4] == bigAlpha[:,5] == bigAlpha[:,6] == bigAlpha[:,7]
        @test bigAlpha[:,8] == zeros(K)
    end

    @testset "Optimization Functions on Small Data" begin
        Random.seed!(100)
        N, K, J = 50, 2, 4
        X = randn(N, K)
        Z = 0.1 .* randn(N, J)
        y = repeat(1:J, inner=ceil(Int, N/J))[1:N]

        mhat = optimize_mlogit(X, Z, y)
        @test length(mhat) == K * (J-1) + 1
        @test all(isfinite.(mhat))

        nesting_structure = [[1,2], [3]]
        nhat = optimize_nested_logit(X, Z, y, nesting_structure)
        @test length(nhat) == 2*K + 3
        @test all(isfinite.(nhat))
    end
end
