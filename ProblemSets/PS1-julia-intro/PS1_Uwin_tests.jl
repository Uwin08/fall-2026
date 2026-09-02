# ==============================================================================
# ECON 6343: Econometrics III
# Problem Set 1
# Name: Uwin Ariyarathna
# Division of Finance
# University of Oklahoma
#
# File: PS1_Uwin_tests.jl
# ==============================================================================

using Test
using JLD
using CSV
using DataFrames

include("PS1_Uwin_source.jl")

@testset "ECON 6343 Problem Set 1 - Uwin Ariyarathna" begin

    @testset "q1()" begin
        A, B, C, D = q1()

        # Required dimensions
        @test size(A) == (10, 7)
        @test size(B) == (10, 7)
        @test size(C) == (5, 7)
        @test size(D) == (10, 7)

        # A follows the required support
        @test all((-5 .<= A) .& (A .<= 10))

        # C is constructed from the required pieces of A and B
        @test C[:, 1:5] == A[1:5, 1:5]
        @test C[:, 6:7] == B[1:5, 6:7]

        # D keeps nonpositive A values and replaces positive values with zero
        @test D == A .* (A .<= 0)
        @test all(D .<= 0)

        # Required files are produced
        @test isfile("matrixpractice.jld")
        @test isfile("firstmatrix.jld")
        @test isfile("Cmatrix.csv")
        @test isfile("Dmatrix.dat")

        # Check saved dimensions in firstmatrix.jld
        saved = load("firstmatrix.jld")
        @test size(saved["A"]) == (10, 7)
        @test size(saved["B"]) == (10, 7)
        @test size(saved["C"]) == (5, 7)
        @test size(saved["D"]) == (10, 7)
    end


    @testset "q2(A, B, C)" begin
        A, B, C, D = q1()

        # The assignment specifies that q2 returns nothing.
        @test q2(A, B, C) === nothing
    end


    @testset "q3()" begin
        # nlsw88.csv must be in the same working folder as the three Julia files.
        @test isfile("nlsw88.csv")

        @test q3() === nothing
        @test isfile("nlsw88_processed.csv")

        processed = CSV.read("nlsw88_processed.csv", DataFrame)

        # Core columns needed by Questions 3 and 4 should be present
        @test :never_married in propertynames(processed)
        @test :grade in propertynames(processed)
        @test :race in propertynames(processed)
        @test :industry in propertynames(processed)
        @test :occupation in propertynames(processed)
        @test :wage in propertynames(processed)
        @test :ttl_exp in propertynames(processed)
    end


    @testset "matrixops(A, B)" begin
        A = [1.0 2.0; 3.0 4.0]
        B = [5.0 6.0; 7.0 8.0]

        element_product, transpose_product, total_sum = matrixops(A, B)

        @test element_product == [5.0 12.0; 21.0 32.0]
        @test transpose_product == [26.0 30.0; 38.0 44.0]
        @test total_sum == 36.0

        # Vector inputs should also work, as required for ttl_exp and wage
        x = [1.0, 2.0, 3.0]
        y = [4.0, 5.0, 6.0]

        xy, xTy, xplusy = matrixops(x, y)

        @test xy == [4.0, 10.0, 18.0]
        @test xTy == 32.0
        @test xplusy == 21.0

        # Unequal dimensions must produce the required error
        err = try
            matrixops([1.0 2.0], [1.0 2.0; 3.0 4.0])
            nothing
        catch e
            e
        end

        @test err isa ErrorException
        @test err.msg == "inputs must have the same size."
    end


    @testset "q4()" begin
        # q4 requires outputs created by q1 and q3.
        q1()
        q3()

        @test isfile("firstmatrix.jld")
        @test isfile("nlsw88_processed.csv")
        @test q4() === nothing
    end
end

println("\nAll requested unit tests have finished.")
