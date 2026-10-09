# Uwin Ariyarathna | Division of Finance | University of Oklahoma
# Q4: Separate unit tests
using Test, Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, GLM

cd(@__DIR__)

include("PS5_Uwin_source.jl")

@testset "PS5 Uwin source tests" begin
	# Test create_grids (structure and basic properties)
	@testset "create_grids" begin
		zval, zbin, xval, xbin, xtran = create_grids()
		@test isa(zval, AbstractVector)
		@test zbin == length(zval)
		@test xbin == length(xval)
		@test size(xtran) == (zbin * xbin, xbin)
		# probabilities should be in [0,1] (allow tiny numerical slack)
		@test all(xtran .>= -1e-12)
		@test all(xtran .<= 1.0 + 1e-8)
		# x grid should be strictly non-decreasing
		@test all(diff(xval) .>= 0.0)
	end

	# Test load_static_data (quick structural checks)
	@testset "load_static_data" begin
		df_long = load_static_data()
		@test isa(df_long, DataFrame)
		# expected columns
		for col in [:bus_id, :Y, :time, :Odometer, :RouteUsage, :Branded]
			@test col in propertynames(df_long)
		end
		# number of rows should be a multiple of 20 (20 periods per bus)
		@test nrow(df_long) % 20 == 0
		# Y should be binary (0/1)
		uniqY = unique(df_long.Y)
		@test all(x -> x in (0,1), uniqY)
	end

	# Test static GLM estimation and coefficient extraction
	@testset "estimate_static_model" begin
		df_long = load_static_data()
		res = estimate_static_model(df_long)
		@test length(coef(res)) == 3 && all(isfinite, coef(res))
	end

	# Test load_dynamic_data returns expected named tuple and dimensions
	@testset "load_dynamic_data" begin
		d = load_dynamic_data()
		# required fields
		for field in (:Y, :X, :B, :Xstate, :Zstate, :N, :T, :xval, :xbin, :zbin, :xtran, :β)
			@test haskey(pairs(d) |> Dict, field) || hasproperty(d, field)
		end
		@test size(d.Y) == (d.N, d.T)
		@test size(d.xtran, 2) == d.xbin
		@test size(d.xtran, 1) == d.xbin * d.zbin
	end

	# Build a small synthetic dataset to exercise compute_future_value! and log_likelihood_dynamic quickly
	@testset "dynamic computations (small synthetic)" begin
		# small state space: 2 mileage bins, 2 route usage bins
		xval = [0.0, 1.0]
		xbin = length(xval)
		zbin = 2
		β = 0.9

		# simple transition: if continue, 70% stay same, 30% move to next; if at top, stay
		# Build xtran of size (zbin*xbin, xbin)
		xtran = zeros(zbin * xbin, xbin)
		for z in 1:zbin
			for x in 1:xbin
				row = x + (z-1)*xbin
				if x == 1
					xtran[row, :] = [0.7, 0.3]
				else
					xtran[row, :] = [0.0, 1.0]
				end
			end
		end

		# Create tiny panel: N=2, T=3
		N = 2; T = 3
		# Observed odometer values
		X = [0.0 0.5 1.0; 1.0 1.2 1.5]
		# Observable discretized states (1..xbin)
		Xstate = [1 1 2; 2 2 2]
		# Route usage state per individual (1..zbin)
		Zstate = [1, 2]
		# Brand indicator 0/1
		B = [0, 1]
		# Decisions Y (binary)
		Y = [1 0 1; 0 1 0]

		d_small = (
			Y = Y,
			X = X,
			B = B,
			Xstate = Xstate,
			Zstate = Zstate,
			N = N,
			T = T,
			xval = xval,
			xbin = xbin,
			zbin = zbin,
			xtran = xtran,
			β = β
		)

		# FV tensor
		FV = zeros(d_small.zbin * d_small.xbin, 2, d_small.T + 1)
		θ_zero = zeros(3)
		compute_future_value!(FV, θ_zero, d_small)
		# Terminal period remains zero
		@test all(FV[:, :, end] .== 0.0)
		# FV should have been updated for t <= T (not all zeros)
		@test any(abs.(FV[:, :, 1]) .> 0.0)
		@test all(isfinite, FV)

		# Test log likelihood is finite and changes with θ
		ll0 = log_likelihood_dynamic(θ_zero, d_small)
		ll1 = log_likelihood_dynamic([1.0, -0.5, 0.3], d_small)
		@test isfinite(ll0)
		@test isfinite(ll1)
		@test ll0 != ll1
	end
end

# Q3: At the terminal date, choice differences equal flow utility.
@testset "Terminal period and parameter sensitivity" begin
    tiny = (N=1, T=1, Y=reshape([1],1,1), X=reshape([0.0],1,1),
            B=[0], Xstate=reshape([1],1,1), Zstate=[1],
            xval=[0.0,1.0], xbin=2, zbin=1,
            xtran=[0.8 0.2; 0.0 1.0], β=0.9)
    θ = [1.2, -0.3, 0.4]
    FV = zeros(2,2,2)
    compute_future_value!(FV, θ, tiny)
    @test FV[:,:,2] == zeros(2,2)
    @test isapprox(FV[1,1,1], 0.9*log(1+exp(1.2)); atol=1e-12)
    # One observed choice of continuing: negative log likelihood = log(1+exp(-θ₀)).
    @test isapprox(log_likelihood_dynamic(θ,tiny), log1p(exp(-1.2)); atol=1e-12)
    @test log_likelihood_dynamic(θ,tiny) >= 0
end
