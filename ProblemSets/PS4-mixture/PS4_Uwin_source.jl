# Uwin Ariyarathna
# Division of Finance
# ECON 6343: Econometrics III
# Problem Set 4
# Source file: function definitions only

include("lgwt.jl")

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS4-mixture/nlsw88t.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [df.age df.white df.collgrad]

    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)

    y = df.occ_code

    return df, X, Z, y
end


#---------------------------------------------------
# Question 1:
# Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------
function mlogit_with_Z(theta, X, Z, y)

    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    alpha = theta[1:end-1]
    gamma = theta[end]

    bigY = zeros(N, J)

    for j = 1:J
        bigY[:, j] = y .== j
    end

    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)

    for j = 1:J
        num[:, j] = exp.(X * bigAlpha[:, j] .+
                         gamma .* (Z[:, j] .- Z[:, J]))
    end

    dem = sum(num, dims = 2)

    P = num ./ dem

    loglike = -sum(bigY .* log.(P))

    return loglike
end


#---------------------------------------------------
# Question 3a:
# Quadrature Practice
#---------------------------------------------------
function practice_quadrature()

    println("=== Question 3a: Quadrature Practice ===")

    d = Normal(0, 1)

    nodes, weights = lgwt(7, -4, 4)

    integral_density = sum(weights .* pdf.(d, nodes))

    expectation = sum(weights .* nodes .* pdf.(d, nodes))

    println("∫φ(x)dx = ", integral_density, " (should be ≈ 1)")
    println("∫xφ(x)dx = ", expectation, " (should be ≈ 0)")

    return integral_density, expectation
end


#---------------------------------------------------
# Question 3b:
# More Quadrature Practice
#---------------------------------------------------
function variance_quadrature()

    println("\n=== Question 3b: Variance using Quadrature ===")

    σ = 2
    d = Normal(0, σ)

    nodes7, weights7 = lgwt(7, -5*σ, 5*σ)

    variance_7pts = sum(weights7 .* (nodes7.^2) .* pdf.(d, nodes7))

    nodes10, weights10 = lgwt(10, -5*σ, 5*σ)

    variance_10pts = sum(weights10 .* (nodes10.^2) .* pdf.(d, nodes10))

    println("Variance with 7 quadrature points: ", variance_7pts)
    println("Variance with 10 quadrature points: ", variance_10pts)
    println("True variance: ", σ^2)

    println("Comment: The 10-point quadrature approximation should generally be closer to the true variance.")

    return variance_7pts, variance_10pts
end


#---------------------------------------------------
# Question 3c:
# Monte Carlo Practice
#---------------------------------------------------
function practice_monte_carlo()

    println("\n=== Question 3c: Monte Carlo Integration ===")

    σ = 2
    d = Normal(0, σ)

    a, b = -5*σ, 5*σ

    function mc_integrate(f, a, b, D)
        draws = rand(D) * (b - a) .+ a
        return (b - a) * mean(f.(draws))
    end

    results = Dict()

    for D in [1000, 1000000]

        println("\nWith D = $D draws:")

        variance_mc = mc_integrate(x -> x^2 * pdf(d, x), a, b, D)

        mean_mc = mc_integrate(x -> x * pdf(d, x), a, b, D)

        density_mc = mc_integrate(x -> pdf(d, x), a, b, D)

        println("MC Variance: ", variance_mc, " (true: ", σ^2, ")")
        println("MC Mean: ", mean_mc, " (true: 0)")
        println("MC Density integral: ", density_mc, " (true: 1)")

        results[D] = (variance_mc, mean_mc, density_mc)
    end

    println("Comment: D = 1,000,000 should generally produce a more accurate approximation than D = 1,000.")

    return results
end


#---------------------------------------------------
# Question 4:
# Mixed Logit with Quadrature
# DO NOT RUN OPTIMIZATION
#---------------------------------------------------
function mixed_logit_quad(theta, X, Z, y, nodes, weights)

    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]

    bigY = zeros(N, J)

    for j = 1:J
        bigY[:, j] = y .== j
    end

    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(theta))

    P_integrated = zeros(T, N, J)

    for r in eachindex(nodes)

        gamma_r = mu_gamma + sigma_gamma * nodes[r]

        num_r = zeros(T, N, J)

        for j = 1:J
            num_r[:, j] = exp.(X * bigAlpha[:, j] .+
                               gamma_r .* (Z[:, j] .- Z[:, J]))
        end

        dem_r = sum(num_r, dims = 2)

        P_r = num_r ./ dem_r

        density_weight = weights[r] * pdf(Normal(0, 1), nodes[r])

        P_integrated .+= P_r * density_weight
    end

    loglike = -sum(bigY .* log.(P_integrated))

    return loglike
end


#---------------------------------------------------
# Question 5:
# Mixed Logit with Monte Carlo
# DO NOT RUN OPTIMIZATION
#---------------------------------------------------
function mixed_logit_mc(theta, X, Z, y, D)

    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    alpha = theta[1:(K*(J-1))]
    mu_gamma = theta[end-1]
    sigma_gamma = theta[end]

    bigY = zeros(N, J)

    for j = 1:J
        bigY[:, j] = y .== j
    end

    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(theta))

    P_integrated = zeros(T, N, J)

    gamma_dist = Normal(mu_gamma, sigma_gamma)

    for d = 1:D

        gamma_d = rand(gamma_dist)

        num_d = zeros(T, N, J)

        for j = 1:J
            num_d[:, j] = exp.(X * bigAlpha[:, j] .+
                               gamma_d .* (Z[:, j] .- Z[:, J]))
        end

        dem_d = sum(num_d, dims = 2)

        P_d = num_d ./ dem_d

        P_integrated .+= P_d / D
    end

    loglike = -sum(bigY .* log.(P_integrated))

    return loglike
end


#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------
function optimize_mlogit(X, Z, y)

    K = size(X, 2)
    J = length(unique(y))

    startvals = [2 * rand(K*(J-1)) .- 1; 0.1]

    result = optimize(
        theta -> mlogit_with_Z(theta, X, Z, y),
        startvals,
        LBFGS(),
        Optim.Options(
            g_tol = 1e-5,
            iterations = 100_000,
            show_trace = true
        );
        autodiff = AutoForwardDiff()
    )

    theta_hat = Optim.minimizer(result)

    return theta_hat
end


#---------------------------------------------------
# Question 4 Optimization Setup
# DO NOT RUN
#---------------------------------------------------
function optimize_mixed_logit_quad(X, Z, y)

    K = size(X, 2)
    J = length(unique(y))

    nodes, weights = lgwt(7, -4, 4)

    startvals = [2 * rand(K*(J-1)) .- 1; 0.1; 1.0]

    # DO NOT RUN
    #
    # result = optimize(
    #     theta -> mixed_logit_quad(theta, X, Z, y, nodes, weights),
    #     startvals,
    #     LBFGS(),
    #     Optim.Options(
    #         g_tol = 1e-5,
    #         iterations = 100_000,
    #         show_trace = true
    #     );
    #     autodiff = AutoForwardDiff()
    # )

    println("Mixed logit quadrature optimization setup complete (not executed)")

    return startvals
end


#---------------------------------------------------
# Question 5 Optimization Setup
# DO NOT RUN
#---------------------------------------------------
function optimize_mixed_logit_mc(X, Z, y)

    K = size(X, 2)
    J = length(unique(y))

    D = 1000

    startvals = [2 * rand(K*(J-1)) .- 1; 0.1; 1.0]

    # DO NOT RUN
    #
    # result = optimize(
    #     theta -> mixed_logit_mc(theta, X, Z, y, D),
    #     startvals,
    #     LBFGS(),
    #     Optim.Options(
    #         g_tol = 1e-5,
    #         iterations = 100_000,
    #         show_trace = true
    #     );
    #     autodiff = AutoForwardDiff()
    # )

    println("Mixed logit Monte Carlo optimization setup complete (not executed)")

    return startvals
end


#---------------------------------------------------
# Question 6:
# Main Function
#---------------------------------------------------
function allwrap()

    println("=== Problem Set 4: Multinomial and Mixed Logit ===")
    println("Uwin Ariyarathna | Division of Finance")

    df, X, Z, y = load_data()

    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", length(unique(y)))

    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")

    theta_hat_mle = optimize_mlogit(X, Z, y)

    println("Estimates: ", theta_hat_mle)

    gamma_hat = theta_hat_mle[end]

    println("γ̂ = ", gamma_hat)

    println("\n=== QUESTION 2: INTERPRETATION ===")

    println(
        "Yes. The estimated γ makes more sense here because " *
        "Z contains occupation-specific expected log wages. " *
        "A positive γ means that, holding X fixed, an occupation " *
        "is more likely to be chosen when its expected log wage " *
        "is higher relative to the normalized occupation."
    )

    practice_quadrature()

    variance_quadrature()

    practice_monte_carlo()

    println("\n=== QUESTION 4: MIXED LOGIT QUADRATURE (SETUP ONLY) ===")

    optimize_mixed_logit_quad(X, Z, y)

    println("\n=== QUESTION 5: MIXED LOGIT MONTE CARLO (SETUP ONLY) ===")

    optimize_mixed_logit_mc(X, Z, y)

    println("\n=== ALL ANALYSES COMPLETE ===")
end
