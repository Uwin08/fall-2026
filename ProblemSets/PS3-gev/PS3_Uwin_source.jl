# Uwin Ariyarathna
# Division of Finance
# ECON 6343: Econometrics III
# Problem Set 3

#---------------------------------------------------
# Data Loading Function
#---------------------------------------------------
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    X = [df.age df.white df.collgrad]
    Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
             df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
    y = df.occupation
    return df, X, Z, y
end

#---------------------------------------------------
# Question 1: Multinomial Logit with Alternative-Specific Covariates
#---------------------------------------------------
function mlogit_with_Z(theta, X, Z, y)
    alpha = theta[1:end-1]
    gamma = theta[end]

    K = size(X, 2)
    J = size(Z, 2)
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    # beta_J = 0 normalization
    bigAlpha = [reshape(alpha, K, J-1) zeros(K)]

    T = promote_type(eltype(X), eltype(theta))
    num = zeros(T, N, J)

    for j = 1:J
        num[:, j] = exp.(X * bigAlpha[:, j] .+ gamma .* (Z[:, j] .- Z[:, J]))
    end

    dem = sum(num, dims=2)
    P = num ./ dem

    loglike = -sum(bigY .* log.(P))
    return loglike
end

#---------------------------------------------------
# Question 3: Nested Logit
#---------------------------------------------------
function nested_logit_with_Z(theta, X, Z, y, nesting_structure)
    alpha = theta[1:end-3]
    lambda = theta[end-2:end-1]
    gamma = theta[end]

    K = size(X, 2)
    J = size(Z, 2)
    N = length(y)

    bigY = zeros(N, J)
    for j = 1:J
        bigY[:, j] = y .== j
    end

    # Common coefficients within each nest. Other is normalized to zero.
    bigAlpha = zeros(K, J)
    bigAlpha[:, nesting_structure[1]] .= repeat(alpha[1:K], 1, length(nesting_structure[1]))
    bigAlpha[:, nesting_structure[2]] .= repeat(alpha[K+1:2K], 1, length(nesting_structure[2]))

    T = promote_type(eltype(X), eltype(theta))
    lidx = zeros(T, N, J)
    num = zeros(T, N, J)

    for j = 1:J
        if j in nesting_structure[1]
            lidx[:, j] = exp.((X * bigAlpha[:, j] .+ gamma .* (Z[:, j] .- Z[:, J])) ./ lambda[1])
        elseif j in nesting_structure[2]
            lidx[:, j] = exp.((X * bigAlpha[:, j] .+ gamma .* (Z[:, j] .- Z[:, J])) ./ lambda[2])
        else
            lidx[:, j] .= 1.0
        end
    end

    wc_sum = sum(lidx[:, nesting_structure[1]], dims=2)
    bc_sum = sum(lidx[:, nesting_structure[2]], dims=2)

    for j = 1:J
        if j in nesting_structure[1]
            num[:, j] = lidx[:, j] .* wc_sum .^ (lambda[1] - 1)
        elseif j in nesting_structure[2]
            num[:, j] = lidx[:, j] .* bc_sum .^ (lambda[2] - 1)
        else
            num[:, j] = lidx[:, j]
        end
    end

    dem = sum(num, dims=2)
    P = num ./ dem

    loglike = -sum(bigY .* log.(P))
    return loglike
end

#---------------------------------------------------
# Optimization Functions
#---------------------------------------------------
function optimize_mlogit(X, Z, y)
    J = size(Z, 2)
    startvals = [2 .* rand((J-1) * size(X,2)) .- 1; 0.1]

    result = optimize(theta -> mlogit_with_Z(theta, X, Z, y),
                      startvals, LBFGS(),
                      Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    return result.minimizer
end

function optimize_nested_logit(X, Z, y, nesting_structure)
    startvals = [2 .* rand(2 * size(X,2)) .- 1; 0.5; 0.5; 0.1]

    result = optimize(theta -> nested_logit_with_Z(theta, X, Z, y, nesting_structure),
                      startvals, LBFGS(),
                      Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=true))
    return result.minimizer
end

#---------------------------------------------------
# Question 4: Wrap Everything into One Function
#---------------------------------------------------
function allwrap()
    Random.seed!(1234)

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS3-gev/nlsw88w.csv"
    df, X, Z, y = load_data(url)

    println("Uwin Ariyarathna - Division of Finance")
    println("ECON 6343 - Problem Set 3")
    println("Data loaded successfully!")
    println("Sample size: ", size(X, 1))
    println("Number of covariates in X: ", size(X, 2))
    println("Number of alternatives: ", size(Z, 2))

    println("\n=== QUESTION 1: MULTINOMIAL LOGIT RESULTS ===")
    theta_hat_mle = optimize_mlogit(X, Z, y)
    println("Estimates: ", theta_hat_mle)

    println("\n=== QUESTION 2: INTERPRETATION OF GAMMA ===")
    println("Estimated gamma: ", theta_hat_mle[end])
    println("Gamma measures the change in latent utility associated with a one-unit increase")
    println("in occupation j's expected log wage relative to the normalized Other occupation,")
    println("holding age, white, and college-graduate status fixed.")

    println("\n=== QUESTION 3: NESTED LOGIT RESULTS ===")
    nesting_structure = [[1, 2, 3], [4, 5, 6, 7]]
    nlogit_theta_hat = optimize_nested_logit(X, Z, y, nesting_structure)
    println("Estimates: ", nlogit_theta_hat)
    println("beta_WC: ", nlogit_theta_hat[1:3])
    println("beta_BC: ", nlogit_theta_hat[4:6])
    println("lambda_WC: ", nlogit_theta_hat[7])
    println("lambda_BC: ", nlogit_theta_hat[8])
    println("gamma: ", nlogit_theta_hat[9])

    return theta_hat_mle, nlogit_theta_hat
end
