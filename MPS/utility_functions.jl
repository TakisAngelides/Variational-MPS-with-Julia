using Profile
using LinearAlgebra
using Arpack
using BenchmarkTools
using Plots
using LaTeXStrings
using Test
using HDF5
include("MPO.jl")

@enum Form begin
    left
    right
end

function get_electric_field_configuration(N::Int64, l_0::Float64, mps)::Vector{Float64}
    
    """
    Gets the L_n = l_0 + sum_k=1^N Q_k for the Schwinger model

    Inputs:

    N = number of physical lattice sites

    l_0 = backgroud electric field

    mps = to calculate the electric field configuration of 

    Output:

    electric_field_list = the index of this list is the left edge of the link on which the electric field is described

    """
    
    charge_list = []
    electric_field_list = []

    for n in 2:2:2*N

        charge_mpo = get_local_charge_MPO(N, n)
        mps_right = act_mpo_on_mps(charge_mpo, mps)
        append!(charge_list, real(inner_product_MPS(mps, mps_right)))
    
    end

    for n in 1:N
        
        append!(electric_field_list, l_0 + sum(charge_list[1:n]))
    
    end

    return electric_field_list

end

function operator_to_sparse_matrix(operator, idx::Int64, N::Int64)

    result = ones(1)
    I = [1 0; 0 1]
    
    for i in 1:N
        if i == idx
            result = kron(result, operator)
        else
            result = kron(result, I)
        end
    end

    return result

end

function get_Schwinger_hamiltonian_matrix(N::Int64, l_0::Float64, x::Float64, lambda::Float64, m_g_ratio::Float64)

    A = -2*1im*(sqrt(x)*m_g_ratio + x)
    B = -2*1im*x
    C = (N-1)*l_0^2 + lambda*N/2 + N*(N-1)/4

    Z = [1 0; 0 -1]
    PLUS = [0 1; 0 0]
    MINUS = [0 0; 1 0]

    result = zeros((2^(2*N), 2^(2*N)))

    for n in 1:N

        result += A.*(operator_to_sparse_matrix(PLUS, 2*n-1, 2*N)*operator_to_sparse_matrix(MINUS, 2*n, 2*N))
        result += - A.*(operator_to_sparse_matrix(MINUS, 2*n-1, 2*N)*operator_to_sparse_matrix(PLUS, 2*n, 2*N))

    end

    for n in 1:N-1
    
        result += B.*(operator_to_sparse_matrix(MINUS, 2*n-1, 2*N)*operator_to_sparse_matrix(Z, 2*n, 2*N)operator_to_sparse_matrix(Z, 2*n+1, 2*N)operator_to_sparse_matrix(PLUS, 2*n+2, 2*N))
        result += -B.*(operator_to_sparse_matrix(PLUS, 2*n-1, 2*N)*operator_to_sparse_matrix(Z, 2*n, 2*N)operator_to_sparse_matrix(Z, 2*n+1, 2*N)operator_to_sparse_matrix(MINUS, 2*n+2, 2*N))
    
    end

    for k in 1:N-1

        result += (l_0*(N-k)).*(operator_to_sparse_matrix(Z,2*k-1,2*N) + operator_to_sparse_matrix(Z,2*k,2*N))

    end

    for k in 1:N-1
        for q in k+1:N
        
            result += (0.5*(N-q+lambda)).*(operator_to_sparse_matrix(Z,2*k-1,2*N)*operator_to_sparse_matrix(Z,2*q-1,2*N)+operator_to_sparse_matrix(Z,2*k-1,2*N)*operator_to_sparse_matrix(Z,2*q,2*N)+operator_to_sparse_matrix(Z,2*k,2*N)*operator_to_sparse_matrix(Z,2*q-1,2*N)+operator_to_sparse_matrix(Z,2*k,2*N)*operator_to_sparse_matrix(Z,2*q,2*N))

        end
    end

    for k in 1:N
    
        result = result + (0.5*(N-k+lambda)).*(operator_to_sparse_matrix(Z,2*k-1,2*N)*operator_to_sparse_matrix(Z,2*k,2*N))
    
    end

    result += C.*diagm(ones(2^(2*N)))

    return result

end

function get_penalty_term_matrix(N::Int64, lambda::Float64)

    Z = [1 0; 0 -1]

    result = zeros((2^(2*N), 2^(2*N)))

    for k in 1:N-1
        for q in k+1:N
        
            result += (0.5*(lambda)).*(operator_to_sparse_matrix(Z,2*k-1,2*N)*operator_to_sparse_matrix(Z,2*q-1,2*N)+operator_to_sparse_matrix(Z,2*k-1,2*N)*operator_to_sparse_matrix(Z,2*q,2*N)+operator_to_sparse_matrix(Z,2*k,2*N)*operator_to_sparse_matrix(Z,2*q-1,2*N)+operator_to_sparse_matrix(Z,2*k,2*N)*operator_to_sparse_matrix(Z,2*q,2*N))

        end
    end

    for k in 1:N
    
        result = result + (0.5*(lambda)).*(operator_to_sparse_matrix(Z,2*k-1,2*N)*operator_to_sparse_matrix(Z,2*k,2*N))
    
    end

    result += (lambda*N/2).*diagm(ones(2^(2*N)))

    return result

end

function mpo_to_matrix(mpo::Vector{Array{ComplexF64}})
    """
    Converts an MPO to a matrix by contracting the bond links and reshaping into (sigma_1...sigma_N),(sigma'_1...sigma'_N)
    
    Input:

    mpo = the mpo to be converted to a matrix (Vector of arrays of complex numbers)

    Output:

    result = the matrix with indices (sigma_1...sigma_N),(sigma'_1...sigma'_N) 
    """

    N = length(mpo)

    idx = 2

    result = contraction(mpo[1], (idx,), mpo[2], (1,))

    for i in 3:N
        
        idx = idx + 2
        result = contraction(result, (idx,), mpo[i], (1,))

    end
        
    result = contraction(result, (length(size(result))-2,), ones(1), (1,))
    result = contraction(result, (1,), ones(1), (1,))

    odd = 1:2:2*N-1
    even = 2:2:2*N

    result = permutedims(result, (odd..., even...))

    dims = size(result)
    total_dim = dims[1:N]
    total_dim_dash = dims[N+1:2*N]

    result = reshape(result, (prod(total_dim), prod(total_dim_dash)))

    return result
end

function act_mpo_on_mps(mpo::Vector{Array{ComplexF64}}, mps::Vector{Array{ComplexF64}})::Vector{Array{ComplexF64}}

    """
    Act with an mpo on an mps to produce a new mps with increased bond dimension.

    Inputs:

    mpo = the mpo to act on the mps (Vector of Arrays)

    mps = the mps that the mpo will act on (Vector of Arrays)

    Output:

    result = the new mps with increased bond dimension resulting from acting with the mpo input on the mps input
    """
    
    N = length(mps)

    result = Vector{Array{ComplexF64}}(undef, N)

    for i in 1:N
    
        tmp = contraction(mpo[i], (4,), mps[i], (3,)) # Does contraction of sigma'_i: W_(b_i-1 b_i sigma_i sigma'_i) M_(a_i-1 a_i sigma'_i) = T_(b_i-1 b_i sigma_i a_i-1 a_i)
        tmp = permutedims(tmp, (4, 1, 5, 2, 3)) # T_(a_i-1 b_i-1 a_i b_i sigma_i)
        idx_dims = size(tmp) # returns a tuple of the dimensions of the indices of the tensor T_(a_i-1 b_i-1 a_i b_i sigma_i)
        result[i] = reshape(tmp, (idx_dims[1]*idx_dims[2], idx_dims[3]*idx_dims[4], idx_dims[5])) # merges the bond indices of i-1 together and the bond indices of i together by reshaping the tensor into having indices of higher dimension 

    end

    return result

end

function get_spin_half_expectation_value(N::Int64, mps::Vector{Array{ComplexF64}}, measure_axis::String)::ComplexF64

    """
    Computes the expectation value of the operator

    O^j = sum_i (S^j_i) = sum_i (sigma^j_i/2)

    which gives the total magnetisation along a specific axis j which can be x, y or z and i runs over all sites from 1 to N.

    Inputs:

    N = number of lattice sites (Integer)

    mps = the mps which we will use to calculate <mps|operator|mps> (Vector of 3-tensors)

    measure_axis = which axis to measure the spin (String) takes values "x", "y" or "z"

    Outputs:

    result = total magnetisation of spin 1/2 along a given x, y or z axis (ComplexF64)
    """

    # If we want to measure spin in the x direction we get the MPO operator = sum_i sigma^x_i/2 or y or z equivalently

    if measure_axis == "x"
        mpo = get_spin_half_MPO(N, "x")
    elseif measure_axis == "y"
        mpo = get_spin_half_MPO(N, "y")
    else
        mpo = get_spin_half_MPO(N, "z")
    end

    # Contracts the triple of <mps|mpo|mps> at site 1, then contracts this triple with a dummy 1x1x1 tensor of value 1
    # which will get rid of the trivial indices of the first triple at site 1. The trivial indices are the ones labelled 1,
    # see for example Schollwock equation (192) first bracket.

    triple_1 = contraction(conj!(deepcopy(mps[1])), (3,), mpo[1], (3,))
    triple_1 = contraction(triple_1, (5,), mps[1], (3,))
    dummy_tensor = ones(ComplexF64, 1,1,1)
    result = contraction(dummy_tensor, (1,2,3), triple_1, (1,3,5))

    # Now we compute the triple <mps|mpo|mps> at site i and contract it with the triple at site i-1 which we named result before

    for i in 2:N
    
        triple_i = contraction(conj!(deepcopy(mps[i])), (3,), mpo[i], (3,))
        triple_i = contraction(triple_i, (5,), mps[i], (3,))
        result = contraction(result, (1,2,3), triple_i, (1,3,5))

    end

    return result[1,1,1] # expectation value of total magnetisation with respect to a give x,y or z axis which was a 1x1x1 tensor hence the [1,1,1] index to get a ComplexF64

end

function get_mpo_expectation_value(N::Int64, mps::Vector{Array{ComplexF64}}, mpo::Vector{Array{ComplexF64}})::ComplexF64

    """
    Computes the expectation value of the operator represented by the mpo input

    Inputs:

    N = number of lattice sites (Integer)

    mps = the mps which we will use to calculate <mps|operator|mps> (Vector of 3-tensors)

    mpo = the mpo representing the operator to evaluate the expectation value on

    Outputs:

    result = expectation value of mpo with respect to mps
    """

    # Contracts the triple of <mps|mpo|mps> at site 1, then contracts this triple with a dummy 1x1x1 tensor of value 1
    # which will get rid of the trivial indices of the first triple at site 1. The trivial indices are the ones labelled 1,
    # see for example Schollwock equation (192) first bracket.

    triple_1 = contraction(conj!(deepcopy(mps[1])), (3,), mpo[1], (3,))
    triple_1 = contraction(triple_1, (5,), mps[1], (3,))
    dummy_tensor = ones(ComplexF64, 1,1,1)
    result = contraction(dummy_tensor, (1,2,3), triple_1, (1,3,5))

    # Now we compute the triple <mps|mpo|mps> at site i and contract it with the triple at site i-1 which we named result before

    for i in 2:N
    
        triple_i = contraction(conj!(deepcopy(mps[i])), (3,), mpo[i], (3,))
        triple_i = contraction(triple_i, (5,), mps[i], (3,))
        result = contraction(result, (1,2,3), triple_i, (1,3,5))

    end

    return result[1,1,1] # expectation value of mpo with respect to mps which was a 1x1x1 tensor hence the [1,1,1] index to get a ComplexF64

end

function inner_product_MPS(mps_1::Vector{Array{ComplexF64}}, mps_2::Vector{Array{ComplexF64}})::ComplexF64

    """
    Computes the inner product of two MPS as <mps_1|mps_2>

    Note 1: See Schollwock equation (95)

    Note 2: See my personal website for notes on this function and how the contractions are done in a specific order for efficiency.

    Inputs:

    mps_1 = The bra MPS state of the inner product (Vector)

    mps_2 = The ket MPS state of the inner product (Vector)

    Output:

    result = The complex value of the inner product (Complex)
    """

    # Assert that the number of sites in each MPS are equal

    N = length(mps_1)

    @assert(N == length(mps_2), "The two MPS inputs do not have the same number of sites.")
 
    # conj! takes the input to its complex conjugate not complex transpose. Also note the first contraction which happens 
    # at the very left contracts the first index of the bra matrix with the first index of the ket matrix but these indices are 
    # trivial and set to 1. It also contracts the physical indices of the two aforementioned matrices which gives a new non-trivial result.

    result = contraction(conj!(deepcopy(mps_1[1])), (1, 3), mps_2[1], (1, 3)) # The reason we deepcopy is because mps_1 might point to mps_2 and conj! mutates the input as the ! suggests
    
    for i in 2:N

        result = contraction(result, (2,), mps_2[i], (1,))

        result = contraction(conj!(deepcopy(mps_1[i])), (1, 3), result, (1, 3))
    
    end
    
    return result[1] # results ends up being a 1x1 matrix that is why we index it with [1] to get its value
        
end

function quantum_state_coefficients(mps::Vector{Array{ComplexF64}}, N::Int64)::Array{ComplexF64}

    """
    If we write a quantum state as psi = psi_sigma1,sigma2...|sigma1>|sigma2>... then this function returns the tensor
    psi_sigma1,sigma2 of the psi represented by the input MPS.

    Inputs:

    mps = the mps that represents the quantum state for which we want the coefficients (Vector with elements being 3-tensors ie 3-arrays)

    N = number of lattice sites (Integer)

    Outputs:

    result = coefficients of quantum state namely the psi_sigma1,sigma2,... coefficients (Array of complex floats 64)

    """

    result = contraction(mps[1], (2,), mps[2], (1,))
    for i in 2:N-1
        result = contraction(result, (i+1,), mps[i+1], (1,))
    end

    result = contraction(ones(ComplexF64, 1), (1,), result, (1,))
    result = contraction(ones(ComplexF64, 1), (1,), result, (N,))

    return result

end

function generate_Schwinger_data(m_over_g_list, x_list, N, D_list, accuracy, lambda, l_0, max_sweep_number)

    open("observables_$(N).txt", "a") do file
        h5open("mps_$(N).h5", "cw") do fid

            for mg in m_over_g_list
                for x in x_list
                    for D in D_list

                        mpo = get_Schwinger_Wilson_MPO(N, l_0, x, lambda, mg)
                        E_0, mps, sweeps = variational_ground_state_MPS(2*N, 2, D, mpo, accuracy, max_sweep_number)
                        
                        if !(haskey(fid, "$(lambda)_$(l_0)_$(mg)_$(x)_$(N)_$(D)"))
                            create_group(fid, "$(lambda)_$(l_0)_$(mg)_$(x)_$(N)_$(D)")
                        
                            g = fid["$(lambda)_$(l_0)_$(mg)_$(x)_$(N)_$(D)"]
                            
                            for i in 1:length(mps)
                                
                                g["mps_$(i)"] = mps[i]
                                
                            end
                            
                        end
                        
                        penalty_term_expectation_value = get_mpo_expectation_value(N, mps, get_penalty_term_MPO(N, lambda))
                        
                        write(file, "$(lambda),$(l_0),$(mg),$(x),$(N),$(D),$(sweeps),$(penalty_term_expectation_value),$(E_0)\n")

                    end
                end
            end
        end
    end

end

