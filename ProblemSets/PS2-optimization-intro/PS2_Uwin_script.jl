# ==============================================================================
# ECON 6343: Econometrics III
# Problem Set 2
# Name: Uwin Ariyarathna
# Division of Finance
# University of Oklahoma
#
# File: PS2_Uwin_script.jl
# Purpose: Load required packages, read the source file, and run PS2().
# ==============================================================================

using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables

cd(@__DIR__)

# Read in the source code.
include("PS2_Uwin_source.jl")

# Execute the wrapped Problem Set 2 function.
PS2()
