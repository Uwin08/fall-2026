# ==============================================================================
# ECON 6343: Econometrics III
# Problem Set 2
# Name: Uwin Ariyarathna
# Division of Finance
# University of Oklahoma
#
# File: PS2_Uwin_source.jl
# Purpose: Source code for Problem Set 2.
# ==============================================================================


# ==============================================================================
# Question 2: OLS objective function
# Professor's form from the PS2 PDF
# ==============================================================================
function ols(beta, X, y)
    ssr = (y .- X * beta)' * (y .- X * beta)
    return ssr
end


# ==============================================================================
# Question 3: Binary logit negative log-likelihood
# Optim minimizes, so return the negative log-likelihood.
# ==============================================================================
function logit(alpha, X, y)
    loglike = sum(y .* (X * alpha) - log.(1 .+ exp.(X * alpha)))
    return -loglike
end


# ==============================================================================
# Question 5: Multinomial logit negative log-likelihood
# Occupation 7 is the reference category.
# With J = 7 alternatives, estimate K*(J-1) = K*6 coefficients.
# ==============================================================================
function mlogit(alpha, X, y)
    K = size(X, 2)
    J = length(unique(y))
    N = length(y)

    # Choice indicators
    d = zeros(N, J)
    for j = 1:J
        d[:, j] = y .== j
    end

    # Coefficients for alternatives 1,...,J-1.
    # Alternative J is the reference category, normalized to zero.
    alpha_matrix = [reshape(alpha, K, J - 1) zeros(K, 1)]

    # Multinomial logit choice probabilities
    num = zeros(N, J)
    den = zeros(N)

    for j = 1:J
        num[:, j] = exp.(X * alpha_matrix[:, j])
        den .+= num[:, j]
    end

    p = num ./ repeat(den, 1, J)

    loglike = sum(d .* log.(p))
    return -loglike
end


# ==============================================================================
# Question 6: Wrap Questions 1-5 into one function
# ==============================================================================
function PS2()

    println("==============================================================")
    println("ECON 6343: Econometrics III - Problem Set 2")
    println("Uwin Ariyarathna | Division of Finance")
    println("==============================================================")


    # ==========================================================================
    # Question 1: Basic optimization in Julia
    # Code follows the PS2 PDF.
    # ==========================================================================
    println("\nQUESTION 1: Basic optimization")

    f(x) = -x[1]^4 - 10x[1]^3 - 2x[1]^2 - 3x[1] - 2
    negf(x) = x[1]^4 + 10x[1]^3 + 2x[1]^2 + 3x[1] + 2
    startval = rand(1) # random number as starting value
    result = optimize(negf, startval, LBFGS())

    println(result)
    println("Argmax: ", result.minimizer)
    println("Maximum value of f(x): ", -result.minimum)


    # ==========================================================================
    # Question 2: OLS using Optim
    # Code follows the PS2 PDF.
    # ==========================================================================
    println("\nQUESTION 2: OLS using Optim")

    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)

    X = [ones(size(df, 1), 1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.married .== 1

    beta_hat_ols = optimize(
        b -> ols(b, X, y),
        rand(size(X, 2)),
        LBFGS(),
        Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true)
    )

    println("OLS estimates from Optim:")
    println(beta_hat_ols.minimizer)

    # Checks supplied in the PS2 PDF
    bols = inv(X' * X) * X' * y
    df.white = df.race .== 1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)

    println("OLS closed-form estimates:")
    println(bols)

    println("OLS estimates from lm():")
    println(coef(bols_lm))


    # ==========================================================================
    # Question 3: Logit using Optim
    # ==========================================================================
    println("\nQUESTION 3: Logit using Optim")

    alpha_hat_logit = optimize(
        alpha -> logit(alpha, X, y),
        rand(size(X, 2)),
        LBFGS(),
        Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true)
    )

    println("Logit estimates from Optim:")
    println(alpha_hat_logit.minimizer)


    # ==========================================================================
    # Question 4: Check logit answer with glm()
    # ==========================================================================
    println("\nQUESTION 4: Check logit estimates using glm()")

    alpha_hat_glm = glm(
        @formula(married ~ age + white + collgrad),
        df,
        Binomial(),
        LogitLink()
    )

    println("Logit estimates from glm():")
    println(coef(alpha_hat_glm))

    println("Difference: Optim minus glm() coefficients:")
    println(alpha_hat_logit.minimizer .- coef(alpha_hat_glm))


    # ==========================================================================
    # Question 5: Multinomial logit for occupation
    # Cleaning code follows the PS2 PDF line by line.
    # ==========================================================================
    println("\nQUESTION 5: Multinomial logit for occupation")

    println("Occupation frequencies before cleaning:")
    println(freqtable(df, :occupation)) # note small number of obs in some occupations

    df = dropmissing(df, :occupation)
    df[df.occupation .== 8,  :occupation] .= 7
    df[df.occupation .== 9,  :occupation] .= 7
    df[df.occupation .== 10, :occupation] .= 7
    df[df.occupation .== 11, :occupation] .= 7
    df[df.occupation .== 12, :occupation] .= 7
    df[df.occupation .== 13, :occupation] .= 7

    println("Occupation frequencies after cleaning and aggregation:")
    println(freqtable(df, :occupation)) # problem solved

    # Re-define X and y because the number of rows changed.
    X = [ones(size(df, 1), 1) df.age df.race .== 1 df.collgrad .== 1]
    y = df.occupation

    K = size(X, 2)
    J = length(unique(y))
    n_params = K * (J - 1)

    # The PDF suggests trying several different sets of starting values.
    start_zero = zeros(n_params)
    start_u01 = rand(n_params)
    start_um11 = 2 .* rand(n_params) .- 1

    result_zero = optimize(
        alpha -> mlogit(alpha, X, y),
        start_zero,
        LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false)
    )

    result_u01 = optimize(
        alpha -> mlogit(alpha, X, y),
        start_u01,
        LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false)
    )

    result_um11 = optimize(
        alpha -> mlogit(alpha, X, y),
        start_um11,
        LBFGS(),
        Optim.Options(g_tol=1e-5, iterations=100_000, show_trace=false)
    )

    results = [result_zero, result_u01, result_um11]
    objective_values = [r.minimum for r in results]
    best_index = argmin(objective_values)
    best_result = results[best_index]

    println("Starting-value objective values:")
    println("Zeros:          ", result_zero.minimum)
    println("Uniform [0,1]:  ", result_u01.minimum)
    println("Uniform [-1,1]: ", result_um11.minimum)

    println("Best multinomial-logit result:")
    println("Converged: ", Optim.converged(best_result))
    println("Negative log-likelihood: ", best_result.minimum)

    alpha_matrix = reshape(best_result.minimizer, K, J - 1)

    println("Multinomial-logit coefficient matrix:")
    println("Rows: Intercept, Age, Race==1, College graduate")
    println("Columns: Occupations 1-6 relative to occupation 7")
    println(alpha_matrix)
    println("Occupation 7 is the reference category.")


    println("\n==============================================================")
    println("Problem Set 2 completed.")
    println("==============================================================")

    return nothing
end
