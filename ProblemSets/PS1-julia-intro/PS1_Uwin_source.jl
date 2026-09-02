# ==============================================================================
# ECON 6343: Econometrics III
# Problem Set 1
# Name: Uwin Ariyarathna
# Division of Finance
# University of Oklahoma
#
# File: PS1_Uwin_source.jl
# Purpose: Function definitions only. This file is read by the script and tests.
# ==============================================================================

using JLD
using Random
using LinearAlgebra
using Statistics
using CSV
using DataFrames
using FreqTables
using Distributions

# ------------------------------------------------------------------------------
# Question 1: Initializing variables and basic matrix operations
# ------------------------------------------------------------------------------

function q1()
    # 1(a): Set seed and create A, B, C, and D
    Random.seed!(1234)

    # A: 10 x 7, Uniform[-5, 10]
    A = rand(Uniform(-5, 10), 10, 7)

    # B: 10 x 7, Normal(mean = -2, standard deviation = 15)
    B = rand(Normal(-2, 15), 10, 7)

    # C: first 5 rows and first 5 columns of A,
    #    plus the last 2 columns and first 5 rows of B
    C = [A[1:5, 1:5] B[1:5, 6:7]]

    # D[i,j] = A[i,j] if A[i,j] <= 0, and 0 otherwise
    D = A .* (A .<= 0)

    # 1(b): Number of elements of A
    println("Q1(b) Number of elements in A: ", length(A))

    # 1(c): Number of unique elements of D
    println("Q1(c) Number of unique elements in D: ", length(unique(D)))

    # 1(d): Vectorize B and store as a 70 x 1 matrix
    E = reshape(B, length(B), 1)

    # An easier vector form would be:
    # E_vector = vec(B)

    # 1(e): A in the first slice of dimension 3 and B in the second
    F = cat(A, B; dims=3)

    # 1(f): Permute F from 10 x 7 x 2 to 2 x 10 x 7
    F = permutedims(F, (3, 1, 2))

    # 1(g): Kronecker product B ⊗ C
    G = kron(B, C)

    # kron(C, F) is not valid because F is three-dimensional, while kron here
    # is being applied to two-dimensional matrices.

    # 1(h): Save A, B, C, D, E, F, and G
    save(
        "matrixpractice.jld",
        "A", A,
        "B", B,
        "C", C,
        "D", D,
        "E", E,
        "F", F,
        "G", G
    )

    # 1(i): Save only A, B, C, and D
    save(
        "firstmatrix.jld",
        "A", A,
        "B", B,
        "C", C,
        "D", D
    )

    # 1(j): Export C to CSV after converting it to a DataFrame
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    # 1(k): Export D as a tab-delimited .dat file
    CSV.write("Dmatrix.dat", DataFrame(D, :auto); delim='\t')

    # 1(l): q1 has zero inputs and returns A, B, C, and D
    return A, B, C, D
end


# ------------------------------------------------------------------------------
# Question 2: Loops and comprehensions
# ------------------------------------------------------------------------------

function q2(A, B, C)
    # 2(a): Element-by-element product using loops
    AB = zeros(Float64, size(A))

    for i in axes(A, 1)
        for j in axes(A, 2)
            AB[i, j] = A[i, j] * B[i, j]
        end
    end

    # Same calculation without a loop
    AB2 = A .* B

    # 2(b): Elements of C between -5 and 5 inclusive, using loops
    Cprime = Float64[]

    for j in axes(C, 2)
        for i in axes(C, 1)
            if -5 <= C[i, j] <= 5
                push!(Cprime, C[i, j])
            end
        end
    end

    # Same calculation without a loop
    Cprime2 = C[(C .>= -5) .& (C .<= 5)]

    # Check that the two approaches agree
    @assert AB ≈ AB2
    @assert Cprime == Cprime2

    # 2(c): Build X with dimensions N x K x T
    N = 15_169
    K = 6
    T = 5

    X = zeros(Float64, N, K, T)

    # Columns 1, 5, and 6 remain stationary over time.
    X[:, 1, :] .= 1.0

    x5 = rand(Binomial(20, 0.6), N)
    x6 = rand(Binomial(20, 0.5), N)

    for t in 1:T
        # Dummy variable with probability 0.75 * (6 - t) / 5
        X[:, 2, t] = rand(Bernoulli(0.75 * (6 - t) / 5), N)

        # Normal with mean 15 + t - 1 and sd 5(t - 1)
        # At t = 1 the standard deviation is 0, so the column is constant.
        if t == 1
            X[:, 3, t] .= 15.0
        else
            X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
        end

        # Normal with mean pi(6 - t)/3 and sd 1/e
        X[:, 4, t] = rand(Normal(pi * (6 - t) / 3, 1 / exp(1)), N)

        # Stationary columns
        X[:, 5, t] = x5
        X[:, 6, t] = x6
    end

    # 2(d): K x T beta matrix using comprehensions
    beta = zeros(Float64, K, T)

    beta[1, :] = [1 + 0.25 * (t - 1) for t in 1:T]
    beta[2, :] = [log(t) for t in 1:T]
    beta[3, :] = [-sqrt(t) for t in 1:T]
    beta[4, :] = [exp(t) - exp(t + 1) for t in 1:T]
    beta[5, :] = [t for t in 1:T]
    beta[6, :] = [t / 3 for t in 1:T]

    # 2(e): Y_t = X_t beta_t + epsilon_t, epsilon ~ N(0, 0.36)
    Y = zeros(Float64, N, T)

    for t in 1:T
        epsilon = rand(Normal(0, 0.36), N)
        Y[:, t] = X[:, :, t] * beta[:, t] + epsilon
    end

    # The assignment specifies that q2 returns nothing.
    return nothing
end


# ------------------------------------------------------------------------------
# Question 3: Read data and calculate summary statistics
# ------------------------------------------------------------------------------

function q3()
    # 3(a): Read nlsw88.csv as a DataFrame.
    # normalizenames=true makes column names valid Julia identifiers.
    # Common text representations of missing values are treated as missing.
    nlsw88 = CSV.read(
        "nlsw88.csv",
        DataFrame;
        normalizenames=true,
        missingstring=["", ".", "NA"]
    )

    CSV.write("nlsw88_processed.csv", nlsw88)

    # 3(b): Percentage never married
    never_married_pct = 100 * mean(skipmissing(nlsw88.never_married))

    # Percentage college graduates.
    # The standard nlsw88 data include collgrad. The fallback uses grade >= 16.
    college_grad_pct = if :collgrad in propertynames(nlsw88)
        100 * mean(skipmissing(nlsw88.collgrad))
    else
        grades = collect(skipmissing(nlsw88.grade))
        100 * mean(grades .>= 16)
    end

    println("Q3(b) Percentage never married: ",
            round(never_married_pct; digits=2), "%")
    println("Q3(b) Percentage college graduates: ",
            round(college_grad_pct; digits=2), "%")

    # 3(c): Race distribution as percentages
    race_counts = freqtable(nlsw88.race)
    race_percent = 100 .* race_counts ./ sum(race_counts)

    println("Q3(c) Race distribution (%):")
    println(race_percent)

    # 3(d): Requested summary statistics
    summarystats = describe(
        nlsw88,
        :mean,
        :median,
        :std,
        :min,
        :max,
        :nunique,
        :nmissing
    )

    println("Q3(d) Summary statistics:")
    println(summarystats)

    missing_grade = count(ismissing, nlsw88.grade)
    println("Q3(d) Number of missing grade observations: ", missing_grade)

    # 3(e): Joint distribution of industry and occupation
    joint_distribution = freqtable(nlsw88.industry, nlsw88.occupation)

    println("Q3(e) Joint distribution of industry and occupation:")
    println(joint_distribution)

    # 3(f): Mean wage by industry and occupation
    wage_data = select(nlsw88, :industry, :occupation, :wage)

    mean_wage = combine(
        groupby(wage_data, [:industry, :occupation]),
        :wage => (x -> mean(skipmissing(x))) => :mean_wage
    )

    println("Q3(f) Mean wage by industry and occupation:")
    println(mean_wage)

    # The assignment specifies that q3 has no outputs.
    return nothing
end


# ------------------------------------------------------------------------------
# Question 4: Practice with functions
# ------------------------------------------------------------------------------

"""
    matrixops(A, B)

Perform three operations on arrays `A` and `B` of the same size:

1. Return the element-by-element product `A .* B`.
2. Return the product `A' * B`.
3. Return the sum of all elements of `A + B`.

An error with the message `"inputs must have the same size."` is thrown when
`A` and `B` do not have identical dimensions.
"""
function matrixops(A, B)
    # 4(e): Check that the inputs have the same size
    if size(A) != size(B)
        error("inputs must have the same size.")
    end

    # 4(b)(i): Element-by-element product
    element_product = A .* B

    # 4(b)(ii): A'B
    transpose_product = A' * B

    # 4(b)(iii): Sum of all elements of A + B
    total_sum = sum(A + B)

    return element_product, transpose_product, total_sum
end


function q4()
    # 4(a): Load firstmatrix.jld
    data = load("firstmatrix.jld")

    A = data["A"]
    B = data["B"]
    C = data["C"]
    D = data["D"]

    # 4(d): Evaluate matrixops(A, B)
    AB_results = matrixops(A, B)

    println("Q4(d) matrixops(A, B):")
    println("  Size of elementwise product: ", size(AB_results[1]))
    println("  Size of A'B: ", size(AB_results[2]))
    println("  Sum of all elements of A+B: ", AB_results[3])

    # 4(f): Evaluate matrixops(C, D). Their sizes differ, so an error is expected.
    println("Q4(f) matrixops(C, D):")

    try
        matrixops(C, D)
    catch err
        println("  ", err)
    end

    # 4(g): Evaluate matrixops using ttl_exp and wage
    nlsw88 = CSV.read("nlsw88_processed.csv", DataFrame)

    ttl_exp = collect(skipmissing(nlsw88.ttl_exp))
    wage = collect(skipmissing(nlsw88.wage))

    # The standard nlsw88 data have nonmissing ttl_exp and wage. If a modified
    # file has missing values in only one of these columns, use complete cases.
    if length(ttl_exp) != length(wage)
        complete = dropmissing(select(nlsw88, :ttl_exp, :wage))
        ttl_exp = Array(complete.ttl_exp)
        wage = Array(complete.wage)
    end

    ttl_wage_results = matrixops(ttl_exp, wage)

    println("Q4(g) matrixops(ttl_exp, wage):")
    println("  Length of elementwise product: ", length(ttl_wage_results[1]))
    println("  A'wage result: ", ttl_wage_results[2])
    println("  Sum of ttl_exp + wage: ", ttl_wage_results[3])

    # The assignment specifies that q4 has no outputs.
    return nothing
end
