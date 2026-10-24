# PS7 | Uwin Ariyarathna | Division of Finance
# ECON 6343: Econometrics III
# Package imports and execution are kept in this script.

using Random, LinearAlgebra, Statistics, Distributions, Optim
using DataFrames, DataFramesMeta, CSV, HTTP, GLM, FreqTables

include(joinpath(@__DIR__, "Uwin_source.jl"))

function run_assignment()
    # Full-sample Q1 and Q2; Q3 simulation; Q5 modest SMM demonstration.
    # Increasing smm_n and smm_draws improves precision at substantial runtime cost.
    result = main(sim_n=10_000, smm_n=1_200, smm_draws=100, run_smm=true)
    return result
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_assignment()
end
