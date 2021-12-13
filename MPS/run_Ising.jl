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

# The command which you write in a terminal not in REPL to generate the variational_MPS_algorithm.jl.mem file is:
# 
# julia --track-allocation=user variational_MPS_algorithm.jl
#
# Then you run the variational_MPS_algorithm.jl again in the terminal with the command:
#
# julia variational_MPS_algorithm.jl 
#
# and then open the .mem file which will contain the number of memory allocations per line of code.

# function wrapper() # so as to not misallocate and focus on the function we want to probe
# initialize_MPS(4,2,2) # force compilation
# Profile.clear_malloc_data() # clear allocation
# initialize_MPS(4,2,2) # run again without compilation
# end

# wrapper()

# function wrapped()
#     N = 4
#     d = 2
#     D = 2
#     J = -1.0
#     g_x = 0.5
#     g_z = -0.000001
#     mpo = get_Ising_MPO(N, J, g_x, g_z)
#     acc = 10^(-10)
#     max_sweeps = 10
#     E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
    
#     Profile.clear_malloc_data() # clear allocation
    
#     N = 4
#     d = 2
#     D = 2
#     J = -1.0
#     g_x = 0.5
#     g_z = -0.000001
#     mpo = get_Ising_MPO(N, J, g_x, g_z)
#     acc = 10^(-10)
#     max_sweeps = 10
#     E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
# end
    
# wrapped()

# ----------------------------------------------------------------------------------------------------------------------------------

# function wrap()
#     N = 4
#     d = 2
#     D = 2
#     mpo = get_Ising_MPO(N, 1.0, 1.0, 0.1)
#     acc = 10^(-10)
#     max_sweeps = 10
#     E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
# end

# @benchmark wrap()

# ----------------------------------------------------------------------------------------------------------------------------------

# N = 4
# d = 2
# D = 2
# J = -1.0
# g_x = 0.5
# g_z = -0.000001
# mpo = get_Ising_MPO(N, J, g_x, g_z)
# acc = 10^(-10)
# max_sweeps = 10
# E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
# println(inner_product_MPS(mps, mps))
# println("Minimum energy: ", E_optimal)
# println("Number of sweeps performed: ", sweep_number)
# println("Below is the optimal MPS that minimized the energy:")
# display(mps[1][1,:,:])
# display(mps[2][:,1,:])
# total_spin = get_spin_half_expectation_value(N, mps, "z")
# println("Magnetisation per site: ", total_spin/N)
# display(quantum_state_coefficients(mps, N))

# ----------------------------------------------------------------------------------------------------------------------------------

# # Don't do this at home. At home write data in a text file.

# N_list = [40]
# d_list = [2]
# D_list = [20]
# J_list = [-1.0]
# g_x_list = LinRange(0.0, 2.0, 100)
# g_z_list = [-0.1]
# average_spin_list = []
# ground_state_energy_list = []

# for N in N_list
#     for d in d_list
#         for D in D_list
#             for J in J_list
#                 for g_x in g_x_list
#                     for g_z in g_z_list
                    
#                         mpo_1 = get_Ising_MPO(N, J, g_x, g_z)
#                         acc_1 = 10^(-10)
#                         max_sweeps_1 = 10
#                         E_optimal_1, mps_1, sweep_number_1 = variational_ground_state_MPS(N, d, D, mpo_1, acc_1, max_sweeps_1)
#                         total_spin_1 = get_spin_half_expectation_value(N, mps_1, "z")
#                         average_spin = total_spin_1/N
#                         append!(average_spin_list, real(average_spin))
#                         append!(ground_state_energy_list, E_optimal_1)

#                     end
#                 end
#             end 
#         end
#     end
# end

# title_str = "N = 40, d = 2, D = 20, J = -1.0, g_z = -0.1"
# plot(g_x_list, average_spin_list, legend = false, title = latexstring(title_str), titlefontsize = 12) # label = "Magnetisation per site along z-axis"
# xlabel!(L"g_x")
# ylabel!("Magnetisation per site along z-axis")
# savefig("quantum_phase_transition.pdf")

# ----------------------------------------------------------------------------------------------------------------------------------