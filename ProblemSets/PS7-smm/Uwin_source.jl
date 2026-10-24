# PS7 | Uwin Ariyarathna | Division of Finance
# ECON 6343: Econometrics III
# Function definitions only: packages are imported by the executable script.

################################################################################
# Data Loading and Preparation Functions
################################################################################

"""
    load_data(url)

Load wage data and create design matrix for OLS regression.
Returns: DataFrame, X matrix, log wage vector
"""
function load_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :log_wage = log.(:wage))
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.log_wage
    # could also do:
    # y = log.(df.wage)
    return df, X, y
end


"""
    prepare_occupation_data(df)

Prepare occupation data for multinomial logit.
Collapse occupation categories and create covariates.
"""
function prepare_occupation_data(df)
    df = dropmissing(df, :occupation)
    df[df.occupation.==8 ,:occupation] .= 7
    df[df.occupation.==9 ,:occupation] .= 7
    df[df.occupation.==10,:occupation] .= 7
    df[df.occupation.==11,:occupation] .= 7
    df[df.occupation.==12,:occupation] .= 7
    df[df.occupation.==13,:occupation] .= 7
    df.white = df.race .== 1
    X = [ones(size(df,1),1) df.age df.white df.collgrad]
    y = df.occupation
    return df, X, y
end


# Numerically stable multinomial-logit choice probabilities.
function choice_probabilities(α, X, J)
    K = size(X, 2)
    B = hcat(reshape(α, K, J-1), zeros(K))
    utility = X * B
    utility .-= maximum(utility, dims=2)
    ex = exp.(utility)
    return ex ./ sum(ex, dims=2)
end

# Fixed data-generating parameter matrix, with final alternative normalized to zero.
function simulation_coefficients(J)
    J >= 3 || throw(ArgumentError("J must be greater than 2"))
    if J == 4
        return hcat([1.0,-1.0,0.5], [-2.0,0.5,0.3],
                    [0.0,-0.5,2.0], zeros(3))
    end
    # Deterministic non-base coefficients for arbitrary J
    return hcat([ [0.5*sin(j), -0.75*cos(j), 0.4*sin(2j)] for j in 1:J-1 ]..., zeros(3))
end

################################################################################
# Question 1: OLS via GMM
################################################################################

"""
    ols_gmm(β, X, y)

GMM objective function for OLS regression.
Uses identity weighting matrix.

Mathematical form:
    J(β) = g'Wg where W = I and g = (y - Xβ)

Arguments:
- β: coefficient vector
- X: N×K design matrix
- y: N×1 outcome vector

Returns: scalar objective function value
"""
function ols_gmm(β, X, y)
    # Compute predicted values ŷ = Xβ
    ŷ = X*β
    
    # Compute residuals g = y - ŷ
    g = y - ŷ
    
    # Compute objective function J = g'Ig
    # Hint: Use the identity matrix or just g'*g
    J = g⋅g # g \cdot g
    # I prefer \cdot because it will return a scalar;
    # g'*g will return a 1x1 matrix, which I then have to return J[1]
    
    return J
end

################################################################################
# Question 2: Multinomial Logit via MLE and GMM
################################################################################

"""
    mlogit_mle(alpha, X, y)

Maximum likelihood objective function for multinomial logit.

Model:
    P(y_i = j) = exp(X_i'β_j) / Σ_k exp(X_i'β_k)
    where β_J = 0 (normalization)

Arguments:
- alpha: vectorized coefficients of dimension K*(J-1)
- X: N×K covariate matrix
- y: N×1 choice vector (integer values 1,2,...,J)

Returns: negative log-likelihood value
"""
function mlogit_mle(alpha, X, y)
    N, K = size(X)
    J = length(alpha) ÷ K + 1
    P = choice_probabilities(alpha, X, J)
    return -sum(log(max(P[i, Int(y[i])], eps(Float64))) for i in 1:N)
end

"""
    mlogit_gmm(α, X, y)

GMM objective function for multinomial logit using just-identified moments.

Moment conditions:
    E[X_i(d_ij - P_ij(α))] = 0 for j = 1,...,J-1

This gives K*(J-1) moments for K*(J-1) parameters.
"""
function mlogit_gmm(α, X, y)
    N, K = size(X)
    J = length(α) ÷ K + 1
    P = choice_probabilities(α, X, J)
    D = Float64.(reshape(y, :, 1) .== reshape(1:J, 1, :))
    g = vec(X' * (D[:, 1:J-1] - P[:, 1:J-1]) / N)
    return N * dot(g,g)
end

"""
    mlogit_gmm_overid(α, X, y)

Over-identified GMM for multinomial logit.

Uses N*J moments: d_ij - P_ij(α) for all i,j
This is over-identified since we only have K*(J-1) parameters.
"""
function mlogit_gmm_overid(α, X, y)
    N, K = size(X)
    J = length(α) ÷ K + 1
    P = choice_probabilities(α, X, J)
    D = Float64.(reshape(y, :, 1) .== reshape(1:J, 1, :))
    g = vec(D - P)  # exactly N*J stacked individual-alternative moments
    return dot(g, g)  # identity weighting matrix
end

# Analytic derivatives for accurate convergence checks in Question 2.
function mlogit_mle_gradient!(gradient, alpha, X, y)
    N, K = size(X)
    J = length(alpha) ÷ K + 1
    P = choice_probabilities(alpha, X, J)
    D = Float64.(reshape(y, :, 1) .== reshape(1:J, 1, :))
    gradient .= vec(X' * (P[:, 1:J-1] - D[:, 1:J-1]))
    return nothing
end

function mlogit_gmm_overid_gradient!(gradient, alpha, X, y)
    N, K = size(X)
    J = length(alpha) ÷ K + 1
    P = choice_probabilities(alpha, X, J)
    D = Float64.(reshape(y, :, 1) .== reshape(1:J, 1, :))
    E = D - P
    weighted_error = vec(sum(E .* P, dims=2))
    grad_mat = zeros(K, J-1)
    for j in 1:J-1
        grad_mat[:,j] = -2 .* (X' * (P[:,j] .* (E[:,j] .- weighted_error)))
    end
    gradient .= vec(grad_mat)
    return nothing
end

################################################################################
# Question 3: Simulate Data from Multinomial Logit
################################################################################

"""
    sim_logit(N, J)

Simulate multinomial logit data using inverse CDF method.

Steps:
1. Generate X ~ N(μ, Σ)
2. Set coefficient matrix β (K×J)
3. Compute choice probabilities P_ij
4. Draw uniform random variable ε_i
5. Assign y_i based on which cumulative probability bracket ε_i falls in

Arguments:
- N: number of observations (default: 100,000)
- J: number of choice alternatives (default: 4)

Returns: (Y, X) where Y is N×1 choice vector, X is N×K covariate matrix
"""
function sim_logit(N=100_000, J=4)
    N > 0 || throw(ArgumentError("N must be positive"))
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    β = simulation_coefficients(J)
    P = choice_probabilities(vec(β[:, 1:J-1]), X, J)
    draw = rand(N)
    Y = ones(Int, N)
    cumulative = zeros(N)
    for j in 1:J-1
        cumulative .+= P[:, j]
        Y .+= draw .>= cumulative
    end
    return Y, X
end

"""
    sim_logit_w_gumbel(N, J)

Simulate multinomial logit data using Gumbel shocks (alternative method).

This method directly uses the fact that if ε ~ Gumbel(0,1), then
    y_i = argmax_j (X_i'β_j + ε_ij)
generates choices from a multinomial logit model.

This is often simpler and more numerically stable than the inverse CDF method.
"""
function sim_logit_w_gumbel(N=100_000, J=4)
    N > 0 || throw(ArgumentError("N must be positive"))
    X = hcat(ones(N), randn(N), 2 .+ 2 .* randn(N))
    β = simulation_coefficients(J)
    ε = rand(Gumbel(0,1), N, J)
    Y = [argmax(row) for row in eachrow(X * β + ε)]
    return Y, X
end

################################################################################
# Question 5: Multinomial Logit via SMM
################################################################################

"""
    mlogit_smm_overid(α, X, y, D)

Simulated Method of Moments for multinomial logit.

Algorithm:
1. For given parameters α, compute model-implied probabilities
2. Simulate D datasets using the Gumbel method
3. Compute average simulated choice frequencies
4. Match simulated frequencies to actual frequencies

Arguments:
- α: parameter vector
- X: covariate matrix
- y: actual choices
- D: number of simulation draws

Returns: SMM objective function value
"""
function mlogit_smm_overid(α, X, y, D; seed=1234)
    D >= 1 || throw(ArgumentError("D must be positive"))
    N, K = size(X)
    J = length(α) ÷ K + 1
    B = hcat(reshape(α, K, J-1), zeros(K))
    bigY = Float64.(reshape(y, :, 1) .== reshape(1:J, 1, :))
    bigYsim = zeros(N,J)
    # Common random numbers ensure a deterministic objective for optimization.
    rng = MersenneTwister(seed)
    utilities = X * B
    for d in 1:D
        ε = rand(rng, Gumbel(0,1), N, J)
        sim_y = [argmax(row) for row in eachrow(utilities + ε)]
        for j in 1:J
            bigYsim[:, j] .+= (sim_y .== j) ./ D
        end
    end
    g = vec(bigY - bigYsim)
    return dot(g, g)
end

################################################################################
# Main Function - Question 6: Wrap everything in a function
################################################################################

"""
    main()

Main function that runs all estimation procedures.
Keeps everything out of global scope for better performance.
"""
function main(; sim_n=10_000, smm_n=1_200, smm_draws=100, run_smm=true)
    println("Uwin Ariyarathna | Division of Finance | PS7")
    println("Q1: Linear regression by GMM with identity weighting")
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2026/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df, Xw, yw = load_data(url)
    df, X, y = prepare_occupation_data(df)
    rng = MersenneTwister(1234)
    ols_fit = optimize(b -> ols_gmm(b, Xw, yw), rand(rng, size(Xw,2)),
                       LBFGS(), Optim.Options(iterations=2000))
    ols_closed = Xw \ yw
    println("GMM beta: ", Optim.minimizer(ols_fit))
    println("OLS beta: ", ols_closed)
    println("Maximum difference: ", maximum(abs.(Optim.minimizer(ols_fit) - ols_closed)))

    println("\nQ2: Multinomial logit MLE and stacked N x J GMM")
    J = 7
    K = size(X,2)
    # Normalize covariates for numerical optimization (not model identification).
    # Recover coefficients in the original variable units for output and Q5.
    scales = [std(X[:,k]) > 1e-10 ? std(X[:,k]) : 1.0 for k in 1:K]
    Xopt = X ./ reshape(scales, 1, :)
    indicators = Float64.(reshape(y, :, 1) .== reshape(1:J, 1, :))
    lpm = Xopt \ indicators
    initial = vec(lpm[:, 1:J-1] .- lpm[:, J])
    f_mle = a -> mlogit_mle(a, Xopt, y)
    grad_mle! = (g,a) -> mlogit_mle_gradient!(g,a,Xopt,y)
    f_gmm = a -> mlogit_gmm_overid(a, Xopt, y)
    grad_gmm! = (g,a) -> mlogit_gmm_overid_gradient!(g,a,Xopt,y)
    opts = Optim.Options(iterations=8_000, g_tol=1e-5, f_reltol=1e-10)
    mle_fit = optimize(f_mle, grad_mle!, initial, LBFGS(), opts)
    mle_scaled = Optim.minimizer(mle_fit)
    gmm1 = optimize(f_gmm, grad_gmm!, mle_scaled, LBFGS(), opts)
    gmm2 = optimize(f_gmm, grad_gmm!, 0.2 .* randn(rng,length(mle_scaled)), LBFGS(), opts)
    original_units = a -> vec(reshape(a, K, J-1) ./ reshape(scales, K, 1))
    mle = original_units(mle_scaled)
    gmm_mle = original_units(Optim.minimizer(gmm1))
    gmm_random = original_units(Optim.minimizer(gmm2))
    println("MLE: ", mle)
    println("GMM (MLE start): ", gmm_mle)
    println("GMM (random start): ", gmm_random)
    println("MLE converged: ", Optim.converged(mle_fit))
    println("GMM (MLE start) converged: ", Optim.converged(gmm1))
    println("GMM (random start) converged: ", Optim.converged(gmm2))
    println("GMM objective at MLE start: ", Optim.minimum(gmm1))
    println("GMM objective at random start: ", Optim.minimum(gmm2))
    println("Max coefficient gap: ", maximum(abs.(gmm_mle-gmm_random)))
    if !(Optim.converged(mle_fit) && Optim.converged(gmm1) && Optim.converged(gmm2))
        println("WARNING: At least one Q2 optimizer did not meet convergence tolerance.")
        println("Inspect iteration limits and solver diagnostics before submission.")
    end
    println("The stacked-moment GMM objective is not generally globally convex.")
    println("Different starts alone do not prove a global mathematical property.")
    println("Compare convergence status and objective values before drawing conclusions.")

    println("\nQ3: Simulate multinomial logit and recover parameters")
    Random.seed!(1234)
    ys, Xs = sim_logit(sim_n,4)
    yg, Xg = sim_logit_w_gumbel(sim_n,4)
    true_beta = vec(simulation_coefficients(4)[:,1:3])
    fit_sim = optimize(a -> mlogit_mle(a,Xs,ys), true_beta .+ 0.1,
                       LBFGS(), Optim.Options(iterations=1500))
    fit_gumbel = optimize(a -> mlogit_mle(a,Xg,yg), true_beta .+ 0.1,
                          LBFGS(), Optim.Options(iterations=1500))
    println("True beta: ", true_beta)
    println("Inverse-CDF MLE: ", Optim.minimizer(fit_sim))
    println("Gumbel MLE: ", Optim.minimizer(fit_gumbel))
    println("Max inverse-CDF estimation error: ", maximum(abs.(Optim.minimizer(fit_sim)-true_beta)))
    println("Max Gumbel estimation error: ", maximum(abs.(Optim.minimizer(fit_gumbel)-true_beta)))

    smm_est = nothing
    if run_smm
        println("\nQ5: Multinomial logit by simulated method of moments")
        # A subsample keeps repeated simulated optimization practical.
        n = min(smm_n, length(y))
        Xsmall, ysmall = X[1:n, :], y[1:n]
        f = a -> mlogit_smm_overid(a, Xsmall, ysmall, smm_draws)
        smm_fit = optimize(f, mle, NelderMead(), Optim.Options(iterations=700, f_reltol=1e-8))
        smm_est = Optim.minimizer(smm_fit)
        println("MLE full-sample estimates: ", mle)
        println("GMM full-sample estimates: ", gmm_mle)
        println("SMM subsample estimates: ", smm_est)
        println("SMM converged: ", Optim.converged(smm_fit))
        println("SMM objective: ", Optim.minimum(smm_fit))
        println("SMM based on n=",n," and D=",smm_draws,"; estimates are approximate.")
    end
    # Save a compact output CSV and a separate written interpretation for GitHub.
    output = DataFrame(
        quantity=["Q1 max absolute OLS-GMM difference",
                  "Q2 GMM (MLE start) objective",
                  "Q2 GMM (random start) objective",
                  "Q2 largest coefficient difference",
                  "Q3 inverse-CDF max recovery error",
                  "Q3 Gumbel max recovery error",
                  "Q5 SMM objective (if estimated)"],
        value=[maximum(abs.(Optim.minimizer(ols_fit)-ols_closed)),
               Optim.minimum(gmm1), Optim.minimum(gmm2),
               maximum(abs.(gmm_mle-gmm_random)),
               maximum(abs.(Optim.minimizer(fit_sim)-true_beta)),
               maximum(abs.(Optim.minimizer(fit_gumbel)-true_beta)),
               run_smm ? Optim.minimum(smm_fit) : NaN])
    CSV.write(joinpath(@__DIR__, "Uwin_PS7_results.csv"), output)

    open(joinpath(@__DIR__, "Uwin_PS7_writeup.txt"), "w") do io
        println(io, "PS7 | Uwin Ariyarathna | Division of Finance")
        println(io, "Q1: Minimize squared OLS residuals with the identity weighting matrix.")
        println(io, "The numerical GMM solution matches the matrix least-squares estimator.")
        println(io, "Q2: MLE maximizes log likelihood; stacked N*J GMM minimizes")
        println(io, "the sum of squared individual choice-indicator-minus-probability moments.")
        println(io, "GMM criterion at MLE start: ", Optim.minimum(gmm1))
        println(io, "GMM criterion at random start: ", Optim.minimum(gmm2))
        println(io, "MLE converged: ", Optim.converged(mle_fit))
        println(io, "GMM convergence flags: ", Optim.converged(gmm1), ", ", Optim.converged(gmm2))
        println(io, "The GMM minimization criterion need not be globally convex.")
        println(io, "Variation by initial values alone is not a proof of nonconvexity.")
        println(io, "Q3: Simulated choices are produced by inverse CDF and Gumbel shocks.")
        println(io, "Both fitted coefficients approximate the assigned true parameters.")
        println(io, "Q5: SMM minimizes squared observed-minus-average-simulated")
        println(io, "choice indicators using fixed simulation draws.")
        println(io, "Reported SMM uses n=", min(smm_n,length(y)),
                " and D=",smm_draws, "; numerical result depends on simulation size.")
        if run_smm
            println(io, "SMM converged: ", Optim.converged(smm_fit))
            println(io, "SMM objective: ", Optim.minimum(smm_fit))
        end
    end
    println("\nSaved Uwin_PS7_results.csv and Uwin_PS7_writeup.txt")
    println("PS7 main completed.")
    return (; ols_gmm=Optim.minimizer(ols_fit), ols_closed,
            mle, gmm_mle,
            gmm_random, smm=smm_est)
end
