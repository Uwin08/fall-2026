# PS7 | Uwin Ariyarathna | Division of Finance


using Test, Random, LinearAlgebra, Statistics, Distributions, Optim
using DataFrames, DataFramesMeta, CSV, HTTP, GLM, FreqTables
include(joinpath(@__DIR__, "Uwin_source.jl"))

function test_ps7()
    @testset "PS7 - Independent function tests" begin
        @testset "Data transformation and design matrices" begin
            sample = DataFrame(occupation=[1,8,13,missing], race=[1,2,1,1],
                               age=[25,30,35,40], collgrad=[1,0,1,0])
            clean, X, y = prepare_occupation_data(sample)
            @test size(X) == (3,4)
            @test collect(y) == [1,7,7]
            @test X[:,1] == ones(3)
            @test collect(clean.white) == [true,false,true]
        end

        @testset "Q1: OLS GMM identity objective" begin
            X = [1.0 0; 1 1; 1 2; 1 3]
            y = [2.0, 3.0, 4.0, 5.0]
            b = X \ y
            @test isapprox(ols_gmm(b,X,y),0.0; atol=1e-10)
            @test ols_gmm([0.,0.],X,y) > ols_gmm(b,X,y)
            fit = optimize(a -> ols_gmm(a,X,y), [0.,0.], LBFGS())
            @test isapprox(Optim.minimizer(fit),b; atol=1e-4)
        end

        @testset "Q2: probability, MLE, and GMM moment shapes" begin
            X = [1. 0.; 1. 1.; 1. -1.; 1. 2.; 1. -2.]
            y = [1,2,3,1,2]
            a = [0.2,-0.1, -0.3,0.4]
            P = choice_probabilities(a,X,3)
            @test size(P) == (5,3)
            @test all(P .> 0)
            @test all(isapprox.(vec(sum(P,dims=2)),ones(5);atol=1e-12))
            @test isfinite(mlogit_mle(a,X,y))
            @test isfinite(mlogit_gmm(a,X,y))
            @test isfinite(mlogit_gmm_overid(a,X,y))
            D = Float64.(reshape(y,:,1) .== reshape(1:3,1,:))
            @test isapprox(mlogit_gmm_overid(a,X,y), sum(abs2,D-P);atol=1e-10)
            g = vec(X'*(D[:,1:2]-P[:,1:2])/length(y))
            @test isapprox(mlogit_gmm(a,X,y),length(y)*sum(abs2,g);atol=1e-10)
            @test mlogit_mle(a,X,y) != mlogit_gmm_overid(a,X,y)
            # Independent gradient checks against symmetric finite differences.
            for (f, gradfun) in ((mlogit_mle, mlogit_mle_gradient!),
                                 (mlogit_gmm_overid, mlogit_gmm_overid_gradient!))
                gg = zeros(length(a))
                gradfun(gg,a,X,y)
                h = 1e-6
                for k in eachindex(a)
                    plus, minus = copy(a), copy(a)
                    plus[k] += h
                    minus[k] -= h
                    approximate = (f(plus,X,y)-f(minus,X,y))/(2h)
                    @test isapprox(gg[k],approximate; atol=2e-5, rtol=1e-4)
                end
            end
        end

        @testset "Q3: inverse CDF and Gumbel simulation" begin
            Random.seed!(2026)
            for j in (3,4,5)
                y1, X1 = sim_logit(300,j)
                y2, X2 = sim_logit_w_gumbel(300,j)
                @test size(X1) == size(X2) == (300,3)
                @test length(y1) == length(y2) == 300
                @test all((1 .<= y1) .& (y1 .<= j))
                @test all((1 .<= y2) .& (y2 .<= j))
                @test simulation_coefficients(j)[:,end] == zeros(3)
            end
            @test_throws ArgumentError sim_logit(0,4)
            @test_throws ArgumentError sim_logit(20,2)
        end

        @testset "Q2: separate starting points and convergence" begin
            X = [ones(60) collect(range(-1.5,1.5,length=60))]
            rng = MersenneTwister(701)
            y = [rand(rng, 1:3) for _ in 1:60]
            f = a -> mlogit_gmm_overid(a, X, y)
            fit1 = optimize(f, zeros(4), LBFGS(), Optim.Options(iterations=500))
            fit2 = optimize(f, fill(0.1,4), LBFGS(), Optim.Options(iterations=500))
            @test isfinite(Optim.minimum(fit1))
            @test isfinite(Optim.minimum(fit2))
            @test Optim.minimum(fit1) <= f(zeros(4)) + 1e-8
            @test Optim.minimum(fit2) <= f(fill(0.1,4)) + 1e-8
        end

        @testset "Q5: simulated moments and reproducibility" begin
            Random.seed!(42)
            y, X = sim_logit(90,4)
            a = vec(simulation_coefficients(4)[:,1:3])
            v1 = mlogit_smm_overid(a,X,y,8;seed=777)
            v2 = mlogit_smm_overid(a,X,y,8;seed=777)
            @test v1 == v2
            @test isfinite(v1) && v1 >= 0
            @test_throws ArgumentError mlogit_smm_overid(a,X,y,0)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_ps7()
end
