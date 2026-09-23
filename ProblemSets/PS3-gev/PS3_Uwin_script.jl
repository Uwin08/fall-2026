# Uwin Ariyarathna
# Division of Finance
# ECON 6343: Econometrics III
# Problem Set 3

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables

# Problem Set 3 - Main Script


cd(@__DIR__)
include("PS3_Uwin_source.jl")

# Questions 1, 3, and 4 are estimated by allwrap().
# Question 2 is printed immediately after the multinomial-logit estimates.
theta_hat_mle, nlogit_theta_hat = allwrap()

# Question 2: Interpretation of gamma
# gamma is the coefficient on the alternative-specific expected log-wage difference.
# A one-unit increase in (Z_ij - Z_iJ) changes occupation j's latent utility by gamma,
# holding the individual-specific X variables fixed. The sign of the estimated gamma
# indicates whether a higher expected log wage relative to occupation J raises or lowers
# the model's latent utility for occupation j.
