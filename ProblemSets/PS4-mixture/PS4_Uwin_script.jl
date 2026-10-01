# Uwin Ariyarathna
# Division of Finance
# ECON 6343: Econometrics III
# Problem Set 4 Script

import Pkg

# Install required packages if they are not already available
required_packages = [
    "Optim",
    "DataFrames",
    "CSV",
    "HTTP",
    "GLM",
    "FreqTables",
    "Distributions",
    "ForwardDiff",
    "ADTypes"
]

for pkg in required_packages
    if isnothing(Base.find_package(pkg))
        Pkg.add(pkg)
    end
end

using Random, LinearAlgebra, Statistics
using Optim, DataFrames, CSV, HTTP, GLM, FreqTables, Distributions, ForwardDiff
using ADTypes: AutoForwardDiff

cd(@__DIR__)

Random.seed!(1234)

include("PS4_Uwin_source.jl")

allwrap()
