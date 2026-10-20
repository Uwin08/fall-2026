# Problem Set 6 | ECON 6343 | Uwin Ariyarathna, Division of Finance
# Run: julia "Uwin_tests.jl"
using Test, Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, DataFramesMeta, CSV
include(joinpath(@__DIR__, "Uwin_source.jl"))
include(joinpath(@__DIR__, "create_grids(1).jl"))

@testset "PS6 Ariyarathna" begin
    @testset "Q1: read and long reshape" begin
        mktemp() do path, io
            write(io, "bus_id,Zst,Branded,RouteUsage,Xst1,Xst2,Odometer1,Odometer2,Y1,Y2\n")
            write(io, "11,1,0,0.25,1,2,0,0.125,0,1\n")
            write(io, "22,2,1,0.26,2,1,0.125,0,1,0\n")
            flush(io)
            df,X,Z,B = load_and_reshape_data(path)
            @test size(df) == (4,6)
            @test size(X) == (2,2)
            @test Z == [1,2]
            @test B == [0,1]
            @test df.bus_id == [11,11,22,22]
            @test df.time == [1,2,1,2]
            @test df.Y == [0,1,1,0]
        end
    end

    @testset "Q2: square terms and interaction formula" begin
        d = DataFrame(Odometer=[1.0,2.0],RouteUsage=[0.5,0.6],time=[1,2])
        add_squared_terms!(d)
        @test d.Odometer_sq == [1.0,4.0]
        @test d.RouteUsage_sq ≈ [0.25,0.36]
        @test d.time_sq == [1.0,4.0]
    end

    @testset "Q3(a): transition grid ordering" begin
        zval,zbin,xval,xbin,xtran = create_grids()
        s = construct_state_space(xbin,zbin,xval,zval)
        @test nrow(s) == size(xtran,1) == xbin*zbin
        @test s.Odometer[1] == xval[1]
        @test s.Odometer[xbin+1] == xval[1]
        @test s.RouteUsage[xbin+1] == zval[2]
        @test all(isapprox.(vec(sum(xtran,dims=2)),1.0; atol=1e-8))
    end

    @testset "Q3(b): future-value mapping and order" begin
        # Two mileage bins, one route bin; row0=1, row1=2.
        xtran = [1.0 0.0; 0.3 0.7]
        FV = zeros(2,2,4)
        FV[:,1,2] = [2.0,4.0]
        FV[:,2,2] = [1.0,3.0]
        FV[:,1,3] = [5.0,7.0]
        FV[:,2,3] = [3.0,9.0]
        X = [1 2; 2 1]
        Z = [1,1]
        B = [0,1]
        df = DataFrame(bus_id=[1,1,2,2],time=[1,2,1,2],
                       Odometer=[0.0,1.0,1.0,0.0],Branded=[0,0,1,1],
                       RouteUsage=fill(0.25,4),Y=[0,1,0,1])
        v = compute_fvt1(df,FV,xtran,X,Z,2,B)
        @test length(v) == 4
        @test v[1] ≈ 0.0
        @test v[2] ≈ 1.4
        @test v[3] ≈ 1.4
        @test v[4] ≈ 0.0
    end

    @testset "Q3(c): structural GLM offset" begin
        Random.seed!(6343)
        n = 200
        x = rand(n).*4
        b = rand(0:1,n)
        fv = 0.15 .* x
        p = 1.0 ./ (1.0 .+ exp.(-(-1 .+ 0.3.*x .+ 0.2.*b .+ fv)))
        y = Int.(rand(n) .< p)
        d = DataFrame(Y=y,Odometer=x,Branded=b)
        fit = estimate_structural_params(d,fv)
        @test length(coef(fit)) == 3
        @test all(isfinite,coef(fit))
    end
end

println("Tests completed. Run Uwin_script.jl separately for full estimation.")
