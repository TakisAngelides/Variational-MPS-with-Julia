using Profile
using LinearAlgebra
using Arpack

@enum Form begin
    left
    right
end


function initialize_MPS(N::Int64, d::Int64, D::Int64)::Vector{Array{ComplexF64}}

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
# initialize_MPS(4,2,2) # force compilation
# Profile.clear_malloc_data() # clear allocation
# initialize_MPS(4,2,2) # run again without compilation
# end

# wrapper()

function contraction(A, c_A::Tuple, B, c_B::Tuple)

    """
    The contraction function takes 2 tensors A, B and 2 tuples c_A, c_B and returns
    another tensor after contracting A and B

    A: first tensor
    c_A: indices of A to contract (Tuple of Int64)
    B: second tensor
    c_B: indices of B to contract (Tuple of Int64)

    Note 1: c_A and c_B should be the same length and the first index from c_A should
    have the same dimension as the first index of c_B, the second index from c_A
    should have the same dimension as the second index of c_B and so on.

    Note 2: It is assumed that the first index in c_A is to be contracted with the
    first index in c_B and so on.

    Note 3: If we were instead to use vectors for c_A and c_B, the memory allocation 
    sky rockets and the run time is 10 times slower. Vectors require more memory than
    tuples and run time since tuples are immutable and only store a certain type each time etc.

    Example: If A is a 4-tensor, B is a 3-tensor and I want to contract the first
    index of A with the second index of B and the fourth index of A with the first
    index of B, then the input to the contraction function should be:

    contraction(A, (1, 4), B, (2, 1))

    This will result in a 3-tensor since we have 3 open indices left after the
    contraction, namely second and third indices of A and third index of B

    Code Example:
    # @time begin
    # A = cat([1 2; 3 4], [5 6; 7 8], dims = 3)
    # B = cat([9 11; 11 12], [13 14; 15 16], dims = 3)
    # c_A = (1, 2)
    # c_B = (2, 1)
    # display(contraction(A, c_A, B, c_B))
    # end
    """

    # Get the dimensions of each index in tuple form for A and B

    A_indices_dimensions = size(A) # returns tuple(dimension of index 1 of A, ...)
    B_indices_dimensions = size(B)

    # Get the uncontracted indices of A and B named u_A and u_B. The setdiff
    # returns the elements which are in the first argument and which are not
    # in the second argument.

    u_A = setdiff(1:ndims(A), c_A)
    u_B = setdiff(1:ndims(B), c_B)

    # Check that c_A and c_B agree in length and in each of their entry they
    # have the same index dimension using the macro @assert. Below we also find
    # the dimensions of each index of the uncontracted indices as well as for the
    # contracted ones.

    dimensions_c_A = A_indices_dimensions[collect(c_A)]
    dimensions_u_A = A_indices_dimensions[collect(u_A)]
    dimensions_c_B = B_indices_dimensions[collect(c_B)]
    dimensions_u_B = B_indices_dimensions[collect(u_B)]

    @assert(dimensions_c_A == dimensions_c_B, "Note 1 in the function
    contraction docstring is not satisfied: indices of tensors to be contracted
    should have the same dimensions. Input received: indices of first tensor A
    to be contracted have dimensions $(dimensions_c_A) and indices of second
    tensor B to be contracted have dimensions $(dimensions_c_B).")

    # Permute the indices of A and B so that A has all the contracted indices
    # to the right and B has all the contracted indices to the left.

    # NOTE: The order in which we give the uncontracted indices (in this case
    # they are in increasing order) affects the result of the final tensor. The
    # final tensor will have indices starting from A's indices in increasing
    # ordera and then B's indices in increasing order. In addition c_A and c_B
    # are expected to be given in such a way so that the first index of c_A is
    # to be contracted with the first index of c_B and so on. This assumption is
    # crucial for below, since we need the aforementioned specific order for
    # c_A, c_B in order for the vectorisation below to work.

    A = permutedims(A, (u_A..., c_A...)) # Splat (...) unpacks a tuple in the argument of a function
    B = permutedims(B, (c_B..., u_B...))

    # Reshape tensors A and B so that for A the u_A are merged into 1 index and
    # the c_A are merged into a second index, making A essentially a matrix.
    # The same goes with B, so that A*B will be a vectorised implementation of
    # a contraction. Remember that c_A will form the columns of A and c_B will
    # form the rows of B and since in A*B we are looping over the columns of A
    # with the rows of B it is seen from this fact why the vectorisation works.

    # To see the index dimension of the merged u_A for example you have to think
    # how many different combinations I can have of the individual indices in
    # u_A. For example if u_A = (2, 4) this means the uncontracted indices of A
    # are its second and fourth index. Let us name them alpha and beta
    # respectively and assume that alpha ranges from 1 to 2 and beta from
    # 1 to 3. The possible combinations are 1,1 and 1,2 and 1,3 and 2,1 and 2,2
    # and 2,3 making 6 in total. In general the total dimension of u_A will be
    # the product of the dimensions of its indivual indices (in the above
    # example the individual indices are alpha and beta with dimensions 2 and
    # 3 respectively so the total dimension of the merged index for u_A will
    # be 2x3=6).

    A = reshape(A, (prod(dimensions_u_A), prod(dimensions_c_A)))
    B = reshape(B, (prod(dimensions_c_B), prod(dimensions_u_B)))

    # Perform the vectorised contraction of the indices

    C = A*B

    # Reshape the resulting tensor back to the individual indices in u_A and u_B
    # which we previously merged. This is the unmerging step.

    C = reshape(C, (dimensions_u_A..., dimensions_u_B...))

    return C

end

function get_Ising_MPO(N::Int64, J::Float64, g_x::Float64, g_z::Float64)::Vector{Array{ComplexF64}}

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

    D = 3 # Bond dimension for the 1D transverse and longitudinal field Ising model is 3

    d = 2 # Physical dimension for the 1D transverse and longitudinal field Ising model is 2

    # zero matrix, identity matrix and Pauli matrices X and Y - notice this implicitly assumes d = 2 for our degrees of freedom on a site
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

function get_identity_MPO(N::Int64, d::Int64) # See get_identity_MPO notes on personal website 

    # TODO: add goodnotes to personal website

    mpo = Vector{Array{ComplexF64}}(undef, N)

    on_each_site = reshape(Matrix{ComplexF64}(I, d, d), (1, 1, d, d))

    for i in 1:N
        mpo[i] = on_each_site
    end

    return mpo

end

function gauge_site(form::Form, M_initial::Array)::Tuple{Array, Array} # Julia is call by reference for arrays which are mutable so manipulations on M_initial in this function will reflect on the original unless we remove that reference with eg M = permutedims(M_initial, (1,2,3))

    """
    Gauges a site into left or right canonical form

    Inputs: 

    form = left or right depending on whether we want the site in left or right canonical form (of enumarative type Form)

    M_initial = 3-array to gauge representing the 3-tensor on a given site (Array)

    Output:

    If left: A, SVt # A_(a_i-1)(s_i)(sigma_i), SVt_(s_i)(a_i) and If right: US, B # US_(a_i-1)(s_i-1), B_(s_i-1)(a_i)(sigma_i)
    """

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

    """
    This function calls the function gauge_site for all sites on a lattice putting the MPS in left or right canonical form

    Inputs: 

    form = left or right depending on whether we want left or right canonical form (of enumarative type Form)

    mps = Vector{Array} representing the MPS

    normalized = true or false depending on whether we want the mutated mps to be normalized or not (Boolean)

    N = Number of physical sites on the lattice (Integer)

    Output:

    This function does not return anything. As suggested by the exclamation mark which is conventionally placed in its name (when
    the given function mutates the input), it mutates the input mps.
    """

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

function inner_product_MPS(mps_1::Vector, mps_2::Vector)::ComplexF64 # See Schollwock equation (95)

    """
    Computes the inner product of two MPS as <mps_1|mps_2>

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
        
        # TODO: put the following goodnotes on my website
        # See inner_product_MPS function in Code Notes in goodnotes to follow the next two lines (should be on my website)

        result = contraction(result, (2,), mps_2[i], (1,))

        result = contraction(conj!(deepcopy(mps_1[i])), (1, 3), result, (1, 3))
    
    end
    
    return result[1] # results ends up being a 1x1 matrix that is why we index it with [1] to get its value
        
end

function initialize_L_R_states(mps, mpo, N)

    # We will assume that we start with a left sweep so we need the L_R_states vector to be all R states

    states = Vector{Array{ComplexF64}}(undef, N+1)

    states[1] = ones(ComplexF64, 1, 1, 1)
    states[N+1] = ones(ComplexF64, 1, 1, 1)
    
    for i in N:-1:2

        states[i] = contraction(conj!(deepcopy(mps[i])), (3,), mpo[i], (3,))
        states[i] = contraction(states[i], (5,), mps[i], (3,))
        states[i] = contraction(states[i], (2,4,6), states[i+1], (1,2,3))

    end

    return states

end

function get_Heff(L, W, R)

    Heff = contraction(L, (2,), W, (1,))
    Heff = contraction(Heff, (3,), R, (2,))
    Heff = permutedims(Heff, (3,1,5,4,2,6))
    dimensions = size(Heff)
    Heff = reshape(Heff, (dimensions[1]*dimensions[2]*dimensions[3], dimensions[4]*dimensions[5]*dimensions[6]))

    return Heff, dimensions

end

function get_updated_site(L, W, R)

    Heff, dimensions = get_Heff(L, W, R)
    E, M = eigs(Heff, nev=1, which=:SR) # Understand what this does and the input
    M = reshape(M, (dimensions[1], dimensions[2], dimensions[3])) # M is reshaped in the form sigma_i, a_i-1, a_i
    M = permutedims(M, (2,3,1)) # M is permuted into the form a_i-1, a_i, sigma_i

    return M, E[1]

end

function update_states!(sweep_direction::Form, states, M, W, i)

    site = contraction(conj!(deepcopy(M)), (3,), W, (3,))
    site = contraction(site, (5,), M, (3,))

    if sweep_direction == right # Right moving sweep from left to right
    
        states[i] = contraction(states[i-1], (1,2,3), site, (1,3,5))
    
    else # Left moving sweep from right to left

        states[i] = contraction(site, (2,4,6), states[i+1], (1,2,3))

    end





end

function variational_ground_state_MPS(N::Int64, d::Int64, D::Int64, mpo::Vector, accuracy::Float64, max_sweeps::Int64)

    mps = initialize_MPS(N, d, D)
    gauge_mps!(right, mps, true, N)

    # This are the partial contractions of the initial mps configuration which is contains all B tensors, 
    # it has length N+1 where the first and last elements are 1x1x1 tensors of value 1. The states vector will thus be of the form
    # 1RRRR1 for N = 5.

    states = initialize_L_R_states(mps, mpo, N) 
    E_initial = 10^(-5)
    E_optimal = 0
    sweep_number = 0
    US = 0

    while(true)
        
        E = 0

        # From left to right sweep (right moving sweep or right sweep)

        for i in 1:N-1 # Its up to N-1 here because the left moving sweep will start from N

            L = states[i]
            W = mpo[i]
            R = states[i+1]
            M, _ = get_updated_site(L, W, R)
            mps[i], _ = gauge_site(left, M)
            update_states!(right, states, mps[i], W, i+1) # i+1 because for loop starts from 1 and index 1 in states is the dummy 1x1x1 tensor of value 1 

        end

        for i in N:-1:2 # Its down to 2 here because the right moving sweep will start from 1

            L = states[i]
            W = mpo[i]
            R = states[i+1]
            M, E = get_updated_site(L, W, R)
            US, mps[i] = gauge_site(right, M)
            update_states!(left, states, mps[i], W, i)

        end

        fractional_energy_change = abs((E - E_initial)/E_initial)

        if fractional_energy_change < accuracy

            E_optimal = E
            
            mps[1] = contraction(mps[1], (2,), US, (1,))
            mps[1] = permutedims(mps[1], (1,3,2))

            println("Desired accuracy reached.")

            break
        
        elseif max_sweeps < sweep_number
            
            E_optimal = E

            mps[1] = contraction(mps[1], (2,), US, (1,))
            mps[1] = permutedims(mps[1], (1,3,2))

            println("Maximum number of sweeps reached before desired accuracy.")

            break

        end

        E_initial = E
        sweep_number = sweep_number + 1
    end

    return E_optimal, mps, sweep_number

end

# TODO: Check that get_L_of_Heff and get_R_of_Heff work as expected and derive the identity MPO, 
# write functions to update the vector they return after 1 update

# N = 4
# mpo = get_Ising_MPO(N,1.0,1.0,1.0)
# mps = initialize_MPS(N,2,2)
# gauge_mps!(left, mps, true, N)
# R_vector, result = get_R_of_Heff(mps, mpo, 2, N)
# display(R_vector)

# N = 5
# d = 2
# D = 2
# mpo = get_Ising_MPO(N, 1.0, 1.0, 1.0)
# mps = initialize_MPS(N, d, D)
# gauge_mps!(left, mps, true, N)
# R_vector, result_R = get_R_of_Heff(mps, mpo, 4, N)
# L_vector, result_L = get_L_of_Heff(mps, mpo, 2)
# contraction_00 = contraction(conj!(deepcopy(mps[3])), (3,), mpo[3], (3,)) # a1,a2,s ... b1,b2,s,s' -> a1,a2,b1,b2,s'
# contraction_0 = contraction(contraction_00, (5,), mps[3], (3,)) # a1,a2,b1,b2,s' ... a1',a2',s' -> a1,a2,b1,b2,a1',a2'
# contraction_1 = contraction(result_L, (4,5,6), contraction_0, (1,3,5)) # 1,1,1,a1,b1,a1' ... a1,a2,b1,b2,a1',a2' -> 1,1,1,a2,b2,a2'
# contraction_2 = contraction(contraction_1, (4,5,6), result_R, (1,2,3)) # 1,1,1,a2,b2,a2' ... a2,b2,a2',1,1,1
# display(contraction_2)
# Heff = get_Heff(result_L, mpo[3], result_R)
# display(isapprox(Heff, Heff')) true


N = 4
d = 2
D = 2
mpo = get_Ising_MPO(N, 1.0, 1.0, 0.01)
acc = 10^(-8)
max_sweeps = 100
E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
display(E_optimal)
