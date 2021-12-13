using Profile
using LinearAlgebra
using Arpack
using BenchmarkTools
using Plots
using LaTeXStrings
using Test
using HDF5
include("utility_functions.jl")
include("MPO.jl")
include("variational_MPS_algorithm.jl")

# ----------------------------------------------------------------------------------------------------------------------------------

# Checking that the spectrum of the exact Schwinger Hamiltonian agrees with the matrix formed by its MPO

# N = 2
# x = 1.0
# m_g_ratio = 0.5
# l_0 = 0.0
# lambda = 0.0

# mpo = get_Schwinger_Wilson_MPO(N, l_0, x, lambda, m_g_ratio)

# matrix = mpo_to_matrix(mpo)

# matrix_h = get_Schwinger_hamiltonian_matrix(N, l_0, x, lambda, m_g_ratio)

# display(norm(eigvals(matrix_h)-eigvals(matrix)))

# ----------------------------------------------------------------------------------------------------------------------------------

# Checking that the minimum energy from the variational ground state search agrees with the minimum energy from exact diagonalization

# N = 2
# x = 1.0
# m_g_ratio = 0.5
# l_0 = 0.0
# lambda = 0.0
# acc = 10^(-10)
# max_sweeps = 10
# d = 2
# D = 2
# mpo = get_Schwinger_Wilson_MPO(N, l_0, x, lambda, m_g_ratio)
# E_0, mps_ground, sn = variational_ground_state_MPS(2*N, d, D, mpo, acc, max_sweeps)
# println("Minimum energy from variational ground state search: ", E_0)
# matrix = mpo_to_matrix(mpo)
# println("Minimum energy from exact diagonalization: ", minimum(eigvals(matrix)))

# ----------------------------------------------------------------------------------------------------------------------------------

# Check that the penalty term enforces total charge to 0 and checking the local charge MPO

# N = 4
# x = 1.0
# m_g_ratio = 0.5
# l_0 = 0.25 # this is theta/2pi
# lambda = 100.0
# acc = 10^(-10)
# max_sweeps = 10
# d = 2
# D = 10

# penalty_mpo = get_penalty_term_MPO(N, lambda)
# penalty_mpo_matrix = mpo_to_matrix(penalty_mpo)
# penalty_matrix_exact = get_penalty_term_matrix(N, lambda)
# display(eigvals(penalty_mpo_matrix))
# display(eigvals(penalty_matrix_exact))

# mpo = get_Schwinger_Wilson_MPO(N, l_0, x, lambda, m_g_ratio)
# E_0, mps_ground, sn = variational_ground_state_MPS(2*N, d, D, mpo, acc, max_sweeps)

# mpo_penalty = get_penalty_term_MPO(N, lambda)
# mps_after_penalty_mpo = act_mpo_on_mps(mpo_penalty, mps_ground)

# println(get_mpo_expectation_value(2*N, mps_ground, mpo_penalty))
# println(get_spin_half_expectation_value(2*N, mps_ground, "z")) # the total charge operator is sum -g/2 sigma_z

# charge_list = []

# for n in 2:2:2*N

#     charge_mpo = get_local_charge_MPO(N, n)
#     mps_right = act_mpo_on_mps(charge_mpo, mps_ground)
#     append!(charge_list, inner_product_MPS(mps_ground, mps_right))

# end

# display(charge_list)
# println(sum(charge_list))

# l_field = get_electric_field_configuration(N, l_0, mps_ground)

# # This checks Gauss' law: l_field[i] - l_field[i-1] = Q[i]

# display(isapprox(l_field[2]-l_field[1], charge_list[2]))
# display(l_field[2]-l_field[1])
# display(charge_list[2])

# display(sum(get_electric_field_configuration(N, l_0, mps_ground))/N)

# ----------------------------------------------------------------------------------------------------------------------------------

# Generate data for extrapolation to continuum

m_over_g_list = [0.125]
x_list = [1.0, 2.0, 3.0, 4.0, 5.0]
D_list = [10, 20, 30, 40, 50]

# x_list = [1.0, 2.0]
# D_list = [2, 4]

accuracy = 10^(-10)
lambda = 100.0
l_0 = 0.0
max_sweep_number = 10
N = parse(Int, ARGS[1])
N = 8
# println(N)

# N_list = [24, 26, 28, 30, 32]

generate_Schwinger_data(m_over_g_list, x_list, N, D_list, accuracy, lambda, l_0, max_sweep_number)

# fid = h5open("mps.h5", "r")
# g = fid["$(lambda)_$(l_0)_0.125_2.0_8_4"]
# display(read(g["mps_2"]))
# close(fid)

# ----------------------------------------------------------------------------------------------------------------------------------