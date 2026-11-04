# ============================================================================
# Uwin Ariyarathna | ECON 6343 Econometrics III | Problem Set 8
# SOURCE FILE: functions only; packages are imported by Uwin_script.jl.
# ============================================================================

const ASVAB_COLS = [:asvabAR, :asvabCS, :asvabMK, :asvabNO, :asvabPC, :asvabWK]
const WAGE_COLS = [:black, :hispanic, :female, :schoolt, :gradHS, :grad4yr]
const REQUIRED_COLS = [:logwage; WAGE_COLS; ASVAB_COLS]

"""Read nlsy.csv from an HTTP(S) URL or from a local path."""
function load_data(location::AbstractString)
    if startswith(lowercase(location), "http://") || startswith(lowercase(location), "https://")
        response = HTTP.get(location)
        return CSV.read(IOBuffer(response.body), DataFrame)
    end
    return CSV.read(location, DataFrame)
end

"""Keep a common complete-case sample for comparable Q1-Q6 estimates."""
function clean_data(df::DataFrame)
    missing_cols = setdiff(REQUIRED_COLS, Symbol.(names(df)))
    isempty(missing_cols) || error("Required columns missing: $(join(missing_cols, ", "))")
    out = dropmissing(df, REQUIRED_COLS)
    for col in REQUIRED_COLS
        out[!, col] = Float64.(out[!, col])
    end
    for col in REQUIRED_COLS
        all(isfinite, out[!, col]) || error("Non-finite values in $col")
    end
    return out
end

"""Q1: baseline log(wage) regression."""
function estimate_base_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
end

"""Q2: six-by-six correlation matrix, labeled with the original ASVAB names."""
function compute_asvab_correlations(df::DataFrame)
    C = cor(Matrix{Float64}(df[:, ASVAB_COLS]))
    return DataFrame(C, ASVAB_COLS)
end

"""Q3: baseline specification plus all six ASVAB scores."""
function estimate_full_regression(df::DataFrame)
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr +
                              asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
end

"""Estimate the first principal component of the six ASVAB variables (J x N input)."""
function first_asvab_pca(df::DataFrame)
    A = permutedims(Matrix{Float64}(df[:, ASVAB_COLS]))
    model = MultivariateStats.fit(PCA, A; maxoutdim=1)
    return vec(MultivariateStats.transform(model, A)), model
end

"""Q4: baseline regression plus the first ASVAB principal component."""
function estimate_pca_regression(df::DataFrame)
    score, _ = first_asvab_pca(df)
    work = copy(df)
    work[!, :asvabPCA] = score
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), work)
end

"""Estimate the first factor score, using only named ASVAB columns."""
function first_asvab_factor(df::DataFrame)
    A = permutedims(Matrix{Float64}(df[:, ASVAB_COLS]))
    model = MultivariateStats.fit(FactorAnalysis, A; maxoutdim=1)
    return vec(MultivariateStats.transform(model, A)), model
end

"""Q5: baseline regression plus first estimated ASVAB factor score."""
function estimate_factor_regression(df::DataFrame)
    score, _ = first_asvab_factor(df)
    work = copy(df)
    work[!, :asvabFactor] = score
    return lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactor), work)
end

"""Q6: matrices with intercept first, followed by regressors in assignment order."""
function prepare_factor_matrices(df::DataFrame)
    n = nrow(df)
    X = hcat(ones(n), Matrix{Float64}(df[:, WAGE_COLS]))
    y = Vector{Float64}(df[!, :logwage])
    Xfac = hcat(ones(n), Matrix{Float64}(df[:, [:black, :hispanic, :female]]))
    measurements = Matrix{Float64}(df[:, ASVAB_COLS])
    return X, y, Xfac, measurements
end

"""
Gauss-Legendre nodes and weights for integrating over [-bound, bound].
Uses the Golub-Welsch eigenvalue construction, so no external lgwt.jl is needed.
"""
function gauss_legendre(n::Integer; bound::Real=8.0)
    n > 0 || throw(ArgumentError("n must be positive"))
    bound > 0 || throw(ArgumentError("bound must be positive"))
    off = [k / sqrt(4k^2 - 1) for k in 1:n-1]
    eig = eigen(SymTridiagonal(zeros(Float64, n), off))
    return Float64(bound) .* eig.values, (2 * Float64(bound)) .* abs2.(eig.vectors[1, :])
end

"""
Unpack parameters in the order of the professor's demo:
vec(alpha[L,J]), beta[K], gamma[J], delta[1], log_sigma[J+1].
Log standard deviations guarantee positivity during unconstrained MLE.
The latent factor has variance one; simultaneous sign reversal of all loadings
leaves the likelihood unchanged.
"""
function unpack_factor_parameters(theta::AbstractVector, K::Integer, L::Integer, J::Integer)
    expected = L*J + K + J + 1 + J+1
    length(theta) == expected || throw(DimensionMismatch("Expected $expected parameters, got $(length(theta))"))
    p = L*J
    alpha = reshape(theta[1:p], L, J)
    beta = theta[p+1:p+K]
    gamma = theta[p+K+1:p+K+J]
    delta = theta[p+K+J+1]
    log_sigma = theta[p+K+J+2:end]
    return alpha, beta, gamma, delta, exp.(log_sigma)
end

"""Stable scalar log(exp(a)+exp(b)); compatible with ForwardDiff dual numbers."""
function logaddexp_pair(a::Real, b::Real)
    m = max(a, b)
    return m + log(exp(a-m) + exp(b-m))
end

"""
Negative log-likelihood, integrating out xi ~ N(0,1) by Gauss-Legendre quadrature.
Includes every Gaussian 1/sigma normalization and the N(0,1) factor density.
Works with automatic differentiation and avoids underflow from products of pdfs.
"""
function factor_model(theta::AbstractVector, X::AbstractMatrix, Xfac::AbstractMatrix,
                      Meas::AbstractMatrix, y::AbstractVector,
                      nodes::AbstractVector, weights::AbstractVector)
    N, K = size(X)
    L = size(Xfac, 2)
    size(Xfac, 1) == N && size(Meas, 1) == N && length(y) == N ||
        throw(DimensionMismatch("Inconsistent sample dimensions"))
    J = size(Meas, 2)
    length(nodes) == length(weights) || throw(DimensionMismatch("Quadrature arrays differ"))
    alpha, beta, gamma, delta, sigma = unpack_factor_parameters(theta, K, L, J)
    log2pi = log(2pi)
    baseM = Xfac * alpha
    baseY = X * beta
    acc = nothing
    for r in eachindex(nodes)
        xi = nodes[r]
        # Log likelihood conditional on the one-dimensional latent factor.
        ll = fill(zero(eltype(theta)), N)
        for j in 1:J
            z = (Meas[:, j] .- baseM[:, j] .- gamma[j]*xi) ./ sigma[j]
            ll .-= (0.5*log2pi + log(sigma[j])) .+ 0.5 .* z.^2
        end
        zw = (y .- baseY .- delta*xi) ./ sigma[end]
        ll .-= (0.5*log2pi + log(sigma[end])) .+ 0.5 .* zw.^2
        ll .+= log(weights[r]) - 0.5*log2pi - 0.5*xi^2
        if acc === nothing
            acc = ll
        else
            acc = logaddexp_pair.(acc, ll)
        end
    end
    return -sum(acc)
end

"""Convenience method matching the demo signature with quadrature count R."""
function factor_model(theta::AbstractVector, X::AbstractMatrix, Xfac::AbstractMatrix,
                      Meas::AbstractMatrix, y::AbstractVector, R::Integer)
    nodes, weights = gauss_legendre(R)
    return factor_model(theta, X, Xfac, Meas, y, nodes, weights)
end

"""OLS-based starting values with nonzero factor loadings to avoid zero-loading stationary point."""
function factor_starting_values(df::DataFrame)
    X, y, Xfac, M = prepare_factor_matrices(df)
    alpha = Xfac \ M
    beta = X \ y
    resM = M - Xfac*alpha
    resY = y - X*beta
    sdM = vec(std(resM; dims=1, corrected=false))
    sdY = std(resY; corrected=false)
    gamma = 0.35 .* max.(sdM, 0.05)
    delta = 0.25 * max(sdY, 0.05)
    sigma = vcat(max.(sdM .* 0.85, 0.05), max(sdY*0.85, 0.05))
    return vcat(vec(alpha), beta, gamma, [delta], log.(sigma))
end

"""
Estimate full measurement/wage system by forward-AD L-BFGS MLE.
Returns (parameters, standard_errors, log_likelihood, converged).
SEs use an observed-information Hessian and may be missing (NaN) if singular.
"""
function run_estimation(df::DataFrame, start_vals::AbstractVector;
                        R::Integer=31, max_iterations::Integer=350, show_trace::Bool=false)
    X, y, Xfac, M = prepare_factor_matrices(df)
    nodes, weights = gauss_legendre(R)
    objective = th -> factor_model(th, X, Xfac, M, y, nodes, weights)
    result = Optim.optimize(objective, Float64.(start_vals), Optim.LBFGS(),
                            Optim.Options(iterations=max_iterations,
                                          show_trace=show_trace,
                                          show_every=20, g_tol=1e-5);
                            autodiff=Optim.ADTypes.AutoForwardDiff())
    theta = Optim.minimizer(result)
    se = fill(NaN, length(theta))
    try
        H = ForwardDiff.hessian(objective, theta)
        covtheta = inv(Symmetric(H))
        v = diag(covtheta)
        se = [v_i > 0 ? sqrt(v_i) : NaN for v_i in v]
    catch err
        @warn "Standard errors unavailable (singular/invalid Hessian)" exception=(err, catch_backtrace())
    end
    return theta, se, -Optim.minimum(result), Optim.converged(result)
end

"""Q6: labeled result table, converting log-sigma coefficients and SEs to sigma scale."""
function format_results(theta::AbstractVector, se::AbstractVector, loglike::Real, asvabs::AbstractMatrix)
    J = size(asvabs, 2)
    L, K = 4, 7  # Intercept + 3 measurement controls; intercept + 6 wage controls
    length(theta) == L*J + K + J + 1 + J+1 || throw(DimensionMismatch("Parameter count"))
    rows = DataFrame(equation=String[], variable=String[], coefficient=Float64[], std_error=Float64[])
    pos = 1
    for j in 1:J, v in ("intercept", "black", "hispanic", "female")
        push!(rows, (string(ASVAB_COLS[j]), "alpha_"*v, theta[pos], se[pos]))
        pos += 1
    end
    for v in ("intercept", "black", "hispanic", "female", "schoolt", "gradHS", "grad4yr")
        push!(rows, ("logwage", "beta_"*v, theta[pos], se[pos]))
        pos += 1
    end
    for j in 1:J
        push!(rows, (string(ASVAB_COLS[j]), "gamma_factor_loading", theta[pos], se[pos]))
        pos += 1
    end
    push!(rows, ("logwage", "delta_factor_loading", theta[pos], se[pos]))
    pos += 1
    for v in [string.(ASVAB_COLS); "logwage"]
        sigma = exp(theta[pos])
        push!(rows, (v, "sigma_error_sd", sigma, sigma*se[pos]))
        pos += 1
    end
    return rows
end

"""Create an ordinary regression output table for CSV files."""
function regression_table(model, label::AbstractString)
    ct = coeftable(model)
    return DataFrame(model=fill(String(label), length(coef(model))),
                     variable=string.(coefnames(model)), coefficient=Float64.(coef(model)),
                     std_error=Float64.(stderror(model)))
end
