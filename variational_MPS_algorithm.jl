using Profile
using LinearAlgebra
include("contraction.jl")

@enum Form begin
    left
    right
end

"""
Initializes a random MPS with open boundary conditions of type Vector{Array{ComplexF64}} where each element of the vector is 
a site where a 3-tensor lives or in this case a 3-array. 

The physical index is always stored last in this 3-array and the first index is the index to the left of
the site, while the second index is the index to the right of the site. 

If we label the first, second and third index of this 3-array
as 1,2,3 and call the array on each site M then schematically storage is done as:

     3
     |
1 -- M -- 2 , where 1 and 2 are the left and right bond indices that control entanglement between sites and 3 is the physical index. 

Note 1: This function assumes both bond indices has the same dimension.

Note 2: The first and last sites have a left and right index respectively which is set to the value 1, ie a trivial index of 0 dimension.

Inputs: 

N = Number of sites (Integer)
d = Physical index dimension (Integer)
D = Bond dimension (Integer)

Output:

Vector{Array} = each element of the vector is a site where a 3-tensor lives or in this case a 3-array, representing an MPS.

"""
function initialize_MPS(N::Int64, d::Int64, D::Int64)::Vector{Array{ComplexF64}}

    mps = Vector{Array{ComplexF64}}(undef, N)
    
    mps[1] = rand(ComplexF64, 1, D, d)
    mps[N] = rand(ComplexF64, D, 1, d)
    for i in 2:N-1
        mps[i] = rand(ComplexF64, D, D, d)
    end
    return mps

end

# # The command to generate the variational_MPS_algorithm.jl.mem file is:
# # 
# # julia --track-allocation=user variational_MPS_algorithm.jl
# #
# # Then you run the variational_MPS_algorithm.jl and then open the .mem file which will contain the number of memory allocations

# function wrapper() # so as to not misallocate and focus on the function we want to probe
#     initialize_MPS(4,2,2) # force compilation
#     Profile.clear_malloc_data() # clear allocation
#     initialize_MPS(4,2,2) # run again without compilation
# end

# wrapper()

"""
Creates the 1D transverse and longitudinal applied magnetic field Ising model with open boundary conditions with the Hamiltonian 
operator H = sum_over_nearest_neighbours(J * Z_i * Z_j) + sum_over_all_sites((g_x * X_i) + (g_z * Z_i)). It stores the MPO as a 
vector of N elements, 1 element for each lattice site, and at site or in other words each element is a 4-tensor or in memory a 
4-array with the indices stored as shown below.

      sigma_i                                              3
         |                                                 |
alpha -- Wi -- beta , which is stored in the order: 1 -- mpo[i] -- 2
         |                                                 |
    sigma_i_dash                                           4      

Note 1: The indices are stored in the order alpha, beta, sigma_i, sigma_i_dash. The first two are the left and right bond indices
and the last two are the physical indices that would connect to the physical indices of the bra (above MPO) and ket (below MPO)
respectively.

Note 2: See notes for this function, including the derivation of the W1, WN and Wi, in Constructing Hamiltonian MPO note in goodnotes
(which should be on my website).

Inputs:

N = Number of physical sites on the lattice (Integer)
J, g_x, g_z = coupling constants in the Hamiltonian (Floats)

Output:

mpo = vector that stores 4-arrays representing the MPO

"""
function get_Ising_MPO(N::Int64, J, g_x, g_z)::Vector{Array{ComplexF64}}

    D = 3 # Bond dimension for the 1D transverse and longitudinal field Ising model is 3

    d = 2 # Physical dimension for the 1D transverse and longitudinal field Ising model is 2

    # zero matrix, identity matrix and Pauli matrices X and Y
    zero = [0 0; 0 0]
    I = [1 0; 0 1]
    X = [0 1; 1 0]
    Z = [1 0; 0 -1]

    mpo = Vector{Array{ComplexF64}}(undef, N) # An MPO is stored as a vector and each element of the vector stores a 

    W1 = [g_x*X+g_z*Z J*Z I] # This is a d x (d*D) matrix - see note 2 in docstrings
    mpo[1] = zeros((1, D, d, d))

    # The following for loop is to place the elements of W1 which is the first site of the MPO in the order alpha, beta, sigma_i, 
    # sigma_i_dash in mpo[1] as shown schematically in the docstring above.
    
    for alpha in [1]
        for sigma_i in 1:d
            for beta in 1:D
                for sigma_i_dash in 1:d

                    # The indexing from RHS to LHS is essentially unmerging the column index of the W1 matrix from a d*D index to
                    # two indices of dimensions D and d

                    mpo[1][alpha, beta, sigma_i, sigma_i_dash] = W1[sigma_i, sigma_i_dash + d*(beta - 1)]
                end
            end
        end
    end

    WN = [I; Z; g_x*X+g_z*Z] # This is a (d*D) x d matrix - see note 2 in docstrings
    mpo[N] = zeros((D, 1, d, d))

    # The following for loop is to place the elements of WN which is the last site of the MPO in the order alpha, beta, sigma_i, 
    # sigma_i_dash in mpo[N] as shown schematically in the docstring above.

    for beta in [1]
        for sigma_i_dash in 1:d
            for alpha in 1:D
                for sigma_i in 1:d

                    # The indexing from RHS to LHS is essentially unmerging the row index of the WN matrix from a d*D index to two
                    # indices of dimensions d and D.

                    mpo[N][alpha, beta, sigma_i, sigma_i_dash] = WN[sigma_i + d*(alpha - 1), sigma_i_dash]
                end
            end
        end
    end


    for i in 2:N-1
        
        Wi = [I zero zero; Z zero zero; g_x*X+g_z*Z J*Z I] # This is a (d*D) x (d*D) matrix - see note 2 in docstrings
        mpo[i] = zeros((D, D, d, d))
        
        for beta in 1:D
            for sigma_i_dash in 1:d
                for alpha in 1:D
                    for sigma_i in 1:d

                        # The indexing from RHS to LHS is essentially unmerging the row and column indices of the Wi matrix from 
                        # d*D x d*D indices to four indices of dimensions D, D, d and d.

                        mpo[i][alpha, beta, sigma_i, sigma_i_dash] = Wi[sigma_i + d*(alpha - 1), sigma_i_dash + d*(beta - 1)]
                    end
                end
            end
        end

    end
    
    return mpo # Vector{Array{ComplexF64, N} where N} and the array at each site is Array{ComplexF64, 4}

end

function gauge_site(form::Form, M_initial::Array)::Tuple{Array, Array} # Julia is call by reference for arrays which are mutable so manipulations on M_initial in this function will reflect on the original unless we remove that reference with eg M = permutedims(M_initial, (1,2,3))

    if form == right # See Schollwock equation (137) for right canonical form

        D_left, D_right, d = size(M_initial) # Dimensions of indices of site represented by M_initial to be SVD decomposed
        # The next line is enough to remove the reference on M_initial so that it does not mutate the original M_initial and just uses its value, hence the gauge_site function does not mutate M_initial at all
        M = permutedims(M_initial, (1,3,2)) # Assumes initial index was left right physical and now M_(a_i-1)(sigma_i)(a_i)
        M = reshape(M, (D_left, d*D_right)) # Merging indices: Prepare as 2 index tensor to give to SVD, M_(a_i-1)(sigma_i)(a_i) -> M_(a_i-1)(sigma_i a_i)
        F = svd(M) # One can recover M by M = U*Diagonal(S)*Vt 
        U = F.U # U_(a_i-1)(s_i-1)
        S = F.S # S_(s_i-1)(s_i-1) although S here is just a vector storing the diagonal elements
        # Note for complex M_initial, the following should be named Vd for V_dagger rather than Vt for V_transpose but we keep it Vt
        Vt = F.Vt # Vt_(s_i-1)(sigma_i a_i)
        Vt = reshape(Vt, (length(S), d, D_right)) # Unmerging indices: Vt_(s_i-1)(sigma_i a_i) -> Vt_(s_i-1)(sigma_i)(a_i)
        B = permutedims(Vt, (1,3,2)) # Vt_(s_i-1)(sigma_i)(a_i) -> B_(s_i-1)(a_i)(sigma_i)
        US = U*Diagonal(S) # US_(a_i-1)(s_i-1)

        return US, B # US_(a_i-1)(s_i-1), B_(s_i-1)(a_i)(sigma_i)

    else # See Schollwock equation (136) for left canonical form

        D_left, D_right, d = size(M_initial) 
        M = permutedims(M_initial, (3, 1, 2)) # M_(a_i-1)(a_i)(sigma_i) -> M_(sigma_i)(a_i-1)(a_i)
        M = reshape(M, (d*D_left, D_right)) # M_(sigma_i)(a_i-1)(a_i) -> M_(sigma_i a_i-1)(a_i)
        F = svd(M)
        U = F.U # U_(sigma_i a_i-1)(s_i)
        S = F.S # S_(s_i)(s_i) although stored as vector here
        Vt = F.Vt # Vt_(s_i)(a_i)
        U = reshape(U, (d, D_left, length(S))) # U_(sigma_i)(a_i-1)(s_i)
        A = permutedims(U, (2, 3, 1)) # A_(a_i-1)(s_i)(sigma_i)
        SVt = Diagonal(S)*Vt # SVt_(s_i)(a_i)

        return A, SVt # A_(a_i-1)(s_i)(sigma_i), SVt_(s_i)(a_i)

    end

end

function gauge_mps!(form::Form, mps::Vector, normalized::Bool, N::Int64) # In Julia, it's a convention to append ! to names of functions that modify their arguments.

    if form == right

        M_tilde = mps[N] # This will not work if the MPS site does not have 3 legs

        for i in N:-1:2 # We start from the right most site and move to the left

            US, mps[i] = gauge_site(right, M_tilde) # US will be multiplied to the M on the left
            M_tilde =  contraction(mps[i-1], (2,), US, (1,)) # M_tilde_(a_i-2)(sigma_i-1)(s_i-1)
            M_tilde = permutedims(M_tilde, (1,3,2)) # Put the physical index to the right most place M_tilde_(a_i-2)(sigma_i-1)(s_i-1) -> M_tilde_(a_i-2)(s_i-1)(sigma_i-1)
            if i == 2
                if normalized # If we require the state to be normalized then we gauge even the first site to be a B tensor so that the whole contraction <psi|psi> collapses to the identity
                    _, mps[1] = gauge_site(right, M_tilde) # The placeholder _ for the value of US tells us that we are discarding that number and so the state is normalized
                else
                    mps[1] = M_tilde # Here we don't enforce a normalization so we dont have to gauge the first site we will just need to contract it with its complex conjugate to get the value for <psi|psi>
                end
            end
        end
    end

    if form == left 

        M_tilde = mps[1]

        for i in 1:(N-1)

            mps[i], SVt = gauge_site(left, M_tilde)
            M_tilde = contraction(SVt, (2,), mps[i+1], (1,)) # M_tilde_(s_i-1)(a_i)(sigma_i) so no permutedims needed here
            if i == (N-1)
                if normalized
                    mps[N], _ = gauge_site(left, M_tilde)
                else
                    mps[N] = M_tilde
                end
            end
        end
    end
end

function inner_product_MPS(mps_1::Vector, mps_2::Vector)::Number # See Schollwock equation (95)

    # Assert that the number of sites in each MPS are equal

    N = length(mps_1)

    @assert(N == length(mps_2), "The two MPS inputs do not have the same number of sites.")
 
    # conj! takes the input to its complex conjugate not complex transpose. Also note the first contraction which happens 
    # at the very left contracts the first index of the bra matrix with the first index of the ket matrix but these indices are 
    # trivial and set to 1. It also contracts the physical indices of the two aforementioned matrices which gives a new non-trivial result.

    result = contraction(conj!(deepcopy(mps_1[1])), (1, 3), mps_2[1], (1, 3)) # The reason we deepcopy is because mps_1 might point to mps_2 and conj! mutates the input as the ! suggests

    for i in 2:N
        
        # TODO: put the following goodnotes on my website
        # See inner_product_MPS function in Code Notes in goodnotes to follow the next two lines (should be on my website)

        result = contraction(result, (2,), mps_2[i], (1,))

        result = contraction(conj!(mps_1[i]), (1, 3), result, (1, 3))
    
    end
    
    return result[1] # results ends up being a 1x1 matrix that is why we index it with [1] to get its value
        
end



