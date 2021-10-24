using Profile
using LinearAlgebra
using Arpack
using BenchmarkTools
using Plots
using LaTeXStrings

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
    
    # Tensor at site 1

    mps[1] = rand(ComplexF64, 1, D, d) # random 3-tensor with index dimensions 1, D and d

    # Tensor at site N

    mps[N] = rand(ComplexF64, D, 1, d)

    # Tensors at site 2 to N-1

    for i in 2:N-1
        mps[i] = rand(ComplexF64, D, D, d)
    end
    
    return mps

end

function contraction(A, c_A::Tuple, B, c_B::Tuple)::Array{ComplexF64}

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
    operator 
    
    H = sum_over_nearest_neighbours(J * Z_i * Z_j) + sum_over_all_sites((g_x * X_i) + (g_z * Z_i)). 
    
    It stores the MPO as a vector of N elements, 1 element for each lattice site, and at site or in other words each element is a 
    4-tensor or in memory a 4-array with the indices stored as shown below.

          sigma_i                                              3
             |                                                 |
    alpha -- Wi -- beta , which is stored in the order: 1 -- mpo[i] -- 2
             |                                                 |
        sigma_i_dash                                           4      

    Note 1: The indices are stored in the order alpha, beta, sigma_i, sigma_i_dash. The first two are the left and right bond indices
    and the last two are the physical indices that would connect to the physical indices of the bra (above MPO) and ket (below MPO)
    respectively.

    Note 2: See notes for this function, including the derivation of the W1, WN and Wi, in Constructing Hamiltonian MPO note on my website.

    Note 3: See equation (8) at https://arxiv.org/pdf/1012.0653.pdf.

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
    
    # mpo tensor at site 1
    
    mpo[1] = zeros((1, D, d, d))

    mpo[1][1,1,:,:] = g_x*X+g_z*Z
    mpo[1][1,2,:,:] = J*Z
    mpo[1][1,3,:,:] = I

    # mpo tensor at site N

    mpo[N] = zeros((D, 1, d, d))

    mpo[N][1,1,:,:] = I
    mpo[N][2,1,:,:] = Z
    mpo[N][3,1,:,:] = g_x*X+g_z*Z

    # mpo tensors at sites 2 to N-1

    for i in 2:N-1
        
        mpo[i] = zeros((D, D, d, d))
        
        mpo[i][1,1,:,:] = I
        mpo[i][1,2,:,:] = zero
        mpo[i][1,3,:,:] = zero
        mpo[i][2,1,:,:] = Z
        mpo[i][2,2,:,:] = zero
        mpo[i][2,3,:,:] = zero
        mpo[i][3,1,:,:] = g_x*X+g_z*Z
        mpo[i][3,2,:,:] = J*Z
        mpo[i][3,3,:,:] = I

    end

    return mpo # Vector{Array{ComplexF64, N} where N} and the array at each site is Array{ComplexF64, 4}

end

function get_identity_MPO(N::Int64, d::Int64)::Vector{Array{ComplexF64}}

    """
    Creates the identity MPO for a lattice of N sites with d degrees of freedom per site.

    Note 1: See notes for this function on my personal website on how to derived the MPO representation.

    Inputs:

    N = number of lattice sites (Integer)

    d = number of degrees of freedom per site - eg: d = 2 in the Ising model of spin-1/2 for spin up and spin down along z (Integer)

    Outputs:

    mpo = identity mpo as a vector of N elements and each element is a 4-tensor storing indices in the order shown schematically below

          sigma_i                                              3
             |                                                 |
    alpha -- Wi -- beta , which is stored in the order: 1 -- mpo[i] -- 2
             |                                                 |
         sigma_i_dash                                          4 


    In other words each tensor on each site has indices W_alpha,beta,sigma_i,sigma_i_dash
    """

    mpo = Vector{Array{ComplexF64}}(undef, N)

    on_each_site = reshape(Matrix{ComplexF64}(I, d, d), (1, 1, d, d))

    for i in 1:N
        mpo[i] = on_each_site
    end

    return mpo

end

function get_spin_half_MPO(N::Int64, measure_axis::String)::Vector{Array{ComplexF64}}

    """
    Returns the MPO as a Vector of 4-tensors that represents the operator of total spin 1/2 of the lattice

    O^j = sum_i (S^j_i) = sum_i (sigma^j_i/2) 
    
    where j = x,y,z depending on which axis we want to measure and i runs from 1 to N over lattice sites. 

    Note 1: See my personal website documentation to see how the MPO representation is derived.

    Inputs:

    N = number of lattice sites (Integer)

    measure_axis = which axis to measure the spin (String) takes values "x", "y" or "z"

    Output:

    mpo = vector where each element is a 4-tensor representing the operator as a W tensor
    """

    zero = [0.0 0.0; 0.0 0.0]
    identity = [1.0 0.0;0.0 1.0]

    if measure_axis == "x"
        operator = [0.0 0.5;0.5 0.0] # sigma^x/2 - pauli x operator divided by 2
    elseif measure_axis == "y"
        operator = [0.0 -0.5im;0.5im 0.0] # sigma^y/2 - pauli y operator divided by 2
    else 
        operator = [0.5 0.0;0.0 -0.5] # sigma^z/2 - pauli z operator divided by 2
    end

    mpo = Vector{Array{ComplexF64}}(undef, N)

    D = 2 # Bond dimension of MPO - this will be the dimension of the virtual/bond indices left and right of the tensor in a diagram
    d = 2 # Physical dimension of MPO - this will be the dimension of the physical indices above and below the tensor in a diagram

    # Tensor at site 1

    mpo[1] = zeros(ComplexF64, 1,D,d,d)
    mpo[1][1,1,:,:] = operator
    mpo[1][1,2,:,:] = identity

    # Tensor at site N

    mpo[N] = zeros(ComplexF64, D,1,d,d)
    mpo[N][1,1,:,:] = identity
    mpo[N][2,1,:,:] = operator

    # Tensor at sites 2 to N-1

    tmp = zeros(ComplexF64, D,D,d,d)
    tmp[1,1,:,:] = identity
    tmp[1,2,:,:] = zero
    tmp[2,1,:,:] = operator
    tmp[2,2,:,:] = identity

    for i in 2:N-1
        mpo[i] = tmp
    end

    return mpo

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

function gauge_site(form::Form, M_initial::Array{ComplexF64})::Tuple{Array{ComplexF64}, Array{ComplexF64}}

    """
    Gauges a site into left or right canonical form

    Note 1: See Schollwock equations (136), (137) at link: https://arxiv.org/pdf/1008.3477.pdf

    Inputs: 

    form = left or right depending on whether we want the site in left or right canonical form (of enumarative type Form)

    M_initial = 3-array to gauge representing the 3-tensor on a given site (Array)

    Output:

    If left: A, SVt # A_(a_i-1)(s_i)(sigma_i), SVt_(s_i)(a_i) and If right: US, B # US_(a_i-1)(s_i-1), B_(s_i-1)(a_i)(sigma_i)
    """

    # Julia is call by reference for arrays which are mutable so manipulations on M_initial in this function will reflect on the original unless we remove that reference with eg M = permutedims(M_initial, (1,2,3))

    if form == right # See Schollwock equation (137) for right canonical form (link: https://arxiv.org/pdf/1008.3477.pdf)

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

function gauge_mps!(form::Form, mps::Vector{Array{ComplexF64}}, normalized::Bool, N::Int64)

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

    # In Julia, it's a convention to append ! to names of functions that modify their arguments.

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

function initialize_L_R_states(mps::Vector{Array{ComplexF64}}, mpo::Vector{Array{ComplexF64}}, N::Int64)::Vector{Array{ComplexF64}}

    """
    Creates a vector called states which holds the partial contractions of the left and right components of the effective Hamiltonian 
    which emerges each time we are trying to update a site with the eigensolver (see Schollwock Figure 38, 39 and equation (210)). We
    store in the first and last elements of the vector tensors of dimension 1x1x1 with value 1 so as to contract with the trivial 
    indices that stick out at the very left and very right of an <mps|mpo|mps> diagram (see for example Schollwock equation (192) which
    contains 3 trivial indices labeled 1 on the A tensor and W tensor that live on site 1).

    Inputs:

    mps = This will be the initial mps once we start the algorithm and its a vector of tensors of complex numbers 

    mpo = The mpo Hamiltonian we are investigating which is a vector of tensors of complex numbers

    N = number of lattice sites (Integer)

    Outputs:

    Below is a schematic example of what the output vector will store for N = 5 where R_1 is the right most site - which is site N - 
    triple of bra mps, mpo and ket mps as shown in Schollwock Figure 39. Then R_2 will be the contraction of R_1 with the triple at 
    site N-1 of bra mps, mpo and ket mps and so on up to the triple at site 2. We go down to site 2 and not site 1 because we will 
    start the algorithm by trying to update site 1 using the eigensolver and so the effective Hamiltonian as shown in Schollwock
    Figure 38 will consist of the contraction of the triples found at site 2,3,4,...,N.

    states = [1x1x1 of value 1, R_4, R_3, R_2, R_1, 1x1x1 of value 1] 
    """

    # We will assume that we start with a left sweep so we need the L_R_states vector to be all R states initially

    states = Vector{Array{ComplexF64}}(undef, N+1) # This vector will hold the partial contractions of the left and right parts of the effective Hamiltonian see for example Schollwock Figure 38, 39

    states[1] = ones(ComplexF64, 1, 1, 1)
    states[N+1] = ones(ComplexF64, 1, 1, 1)
    
    for i in N:-1:2

        states[i] = contraction(conj!(deepcopy(mps[i])), (3,), mpo[i], (3,))
        states[i] = contraction(states[i], (5,), mps[i], (3,))
        states[i] = contraction(states[i], (2,4,6), states[i+1], (1,2,3)) # Remember for loop index is going downwards so i+1 was the previous result in the for loop

    end

    return states

end

function get_Heff(L::Array{ComplexF64}, W::Array{ComplexF64}, R::Array{ComplexF64})::Tuple{Array{ComplexF64}, NTuple{6, Int64}}

    """
    Calculates the effective Hamiltonian as shown in Schollwock Figure 38 and returns it as a matrix with indices grouped exactly as
    shown in Schollwock below equation (209), namely H_(sigma_l,a_l-1,a_l)(sigma_l_dash,a_l-1_dash,a_l_dash). It also returns the 
    dimensions of the indices found in the effective Hamiltonian before we reshape it to a matrix. The dimensions of the indices are
    in the order sigma_l, a_l-1, a_l, sigma_l_dash, a_l-1_dash, a_l_dash.

    Note 1: Where we specify the return type in the function signature we are hard coding that the effective Hamiltonian will have initially 6 open indices before reshaping it (see NTuple{6, Int64}).

    Inputs:

    L = the fully contracted left part of the effective Hamiltonian - see Schollwock equation (192) (3-tensor array with indices in the order a_l-1,b_l-1,a_l-1_dash) - note site l is the site we are trying to update with the eigensolver

    W = the mpo tensor at the site l we are going to update with the eigensolver (4-tensor array with indices b_l-1,b_l,sigma_l,sigma_l_dash)

    R = the fully contracted right part of the effective Hamiltonian - see Schollwock equation (193) (3-tensor array with indices in the order a_l,b_l,a_l_dash)

    Outputs:

    Heff, dimensions = effective Hamiltonian as matrix of two indices H_(sigma_l,a_l-1,a_l)(sigma_l_dash,a_l-1_dash,a_l_dash), dimensions of the indices in the order: sigma_l, a_l-1, a_l, sigma_l_dash, a_l-1_dash, a_l_dash
    """

    Heff = contraction(L, (2,), W, (1,))
    Heff = contraction(Heff, (3,), R, (2,))
    Heff = permutedims(Heff, (3,1,5,4,2,6))
    dimensions = size(Heff)
    Heff = reshape(Heff, (dimensions[1]*dimensions[2]*dimensions[3], dimensions[4]*dimensions[5]*dimensions[6]))

    return Heff, dimensions

end

function get_updated_site(L::Array{ComplexF64}, W::Array{ComplexF64}, R::Array{ComplexF64})::Tuple{Array{ComplexF64}, ComplexF64}

    """
    We give the eigensolver the effective Hamiltonian as a matrix with 2 indices and we get back the lowest eigenvalue and the lowest 
    eigenvector of this matrix and we name them E and M. This M will update the site we are trying to minimize the energy with.

    Inputs:

    L = the fully contracted left part of the effective Hamiltonian - see Schollwock equation (192) (3-tensor array with indices in the order a_l-1,b_l-1,a_l-1_dash) - note site l is the site we are trying to update with the eigensolver

    W = the mpo tensor at the site l we are going to update with the eigensolver (4-tensor array with indices b_l-1,b_l,sigma_l,sigma_l_dash)

    R = the fully contracted right part of the effective Hamiltonian - see Schollwock equation (193) (3-tensor array with indices in the order a_l,b_l,a_l_dash)
    
    Outputs:

    M, E[1] = updates site of the mps with indices a_l-1, a_l, sigma_l (see Schollwock above equation (210)), ground state energy approximation
    """

    Heff, dimensions = get_Heff(L, W, R)
    E, M = eigs(Heff, nev=1, which=:SR) # nev = 1 => it will return only 1 number of eigenvalues, SR => compute eigenvalues which have the smallest real part (ie the ground state energy and upwards depending on nev)
    M = reshape(M, (dimensions[1], dimensions[2], dimensions[3])) # M is reshaped in the form sigma_i, a_i-1, a_i
    M = permutedims(M, (2,3,1)) # M is permuted into the form a_i-1, a_i, sigma_i

    return M, E[1]

end

function update_states!(sweep_direction::Form, states::Vector{Array{ComplexF64}}, M::Array{ComplexF64}, W::Array{ComplexF64}, i::Int64)

    """
    Mutates the states vector which holds the partial contractions for the effective Hamiltonian shown in Schollwock Figure 38. We have
    just optimised the tensor at site i and we are calculating the triple of bra mps, mpo and ket mps as shown in Schollwock Figure 39
    and contracting it with element i-1 in the states vector if we are growing the L part of the effective Hamiltonian (ie sweeping 
    from left to right) or we are contracting it with element i+1 in the states vector if we are growing the R part of the effective
    Hamiltonian (ie sweeping from right to left).

    Inputs:

    sweep_direction = which way we are currently sweeping towards (Form enumarative type)

    states = holds the partial contractions for the effective Hamiltonian

    M = the tensor which was recently returned by the eigensolver which optimized the tensor on the site of the mps 

    W = the mpo at the site we just updated

    i = lattice index of the site on the mps we just updated

    Outputs:

    This function does not return anything. As suggested by the exclamation mark which is conventionally placed in its name (when
    the given function mutates the input), it mutates the states vector.
    """

    site = contraction(conj!(deepcopy(M)), (3,), W, (3,))
    site = contraction(site, (5,), M, (3,))

    if sweep_direction == right # Right moving sweep from left to right
    
        states[i] = contraction(states[i-1], (1,2,3), site, (1,3,5))
    
    else # Left moving sweep from right to left

        states[i] = contraction(site, (2,4,6), states[i+1], (1,2,3))

    end
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

function variational_ground_state_MPS(N::Int64, d::Int64, D::Int64, mpo::Vector{Array{ComplexF64}}, accuracy::Float64, max_sweeps::Int64)::Tuple{ComplexF64, Vector{Array{ComplexF64}}, Int64}

    """
    This is the main function which implements the variational MPS ground state algorithm described in Schollwock section 6.3.
        
    Inputs:

    N = number of lattice sites (Integer)

    d = number of degrees of freedom on each site - eg d = 2 if we have only spin up, spin down (Integer)

    D = bond dimension which controls the entanglement of the mps state (Integer)

    mpo = The Hamiltonian we are investigating in the form of an mpo

    accuracy = We are trying to find the mps that will give the smallest energy and we stop the search once the fractional change in energy is less than the accuracy (Float)
    
    max_sweeps = number of maximum sweeps we should perform if the desired accuracy is not reached and the algorithm does not stop because it reached the desired accuracy

    Outputs:

    E_optimal, mps, sweep_number = minimum energy we reached, mps that reached this minimum energy which approximates the ground state of the mpo Hamiltonian we gave as input, number of sweeps we performed when we stopped the algorithm

    """

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

# ----------------------------------------------------------------------------------------------------------------------------------

# The command to generate the variational_MPS_algorithm.jl.mem file is:
# 
# julia --track-allocation=user variational_MPS_algorithm.jl
#
# Then you run the variational_MPS_algorithm.jl and then open the .mem file which will contain the number of memory allocations

# function wrapper() # so as to not misallocate and focus on the function we want to probe
# initialize_MPS(4,2,2) # force compilation
# Profile.clear_malloc_data() # clear allocation
# initialize_MPS(4,2,2) # run again without compilation
# end

# wrapper()

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

# N = 2
# d = 2
# D = 2
# J = -1.0
# g_x = 0.0
# g_z = 0.1
# mpo = get_Ising_MPO(N, J, g_x, g_z)
# acc = 10^(-10)
# max_sweeps = 10
# E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
# println("Minimum energy: ", E_optimal)
# println("Number of sweeps performed: ", sweep_number)
# println("Below is the optimal MPS that minimized the energy:")
# display(mps[1][1,:,:])
# display(mps[2][:,1,:])
# total_spin = get_spin_half_expectation_value(N, mps, "z")
# println("Magnetisation per site: ", total_spin/N)
# psi = contraction(mps[1], (2,), mps[2], (1,))
# # psi = contraction(zeros(ComplexF64, 1, 1), (1,2), psi, (1,3))
# display(psi[1,:,1,:])

# ----------------------------------------------------------------------------------------------------------------------------------

# N_list = [4]
# d_list = [2]
# D_list = [2]
# J_list = LinRange(1.0, 2.0,100)
# g_x_list = [0.0]
# g_z_list = [1.0]
# average_spin_list = []
# ground_state_energy_list = []

# for N in N_list
#     for d in d_list
#         for D in D_list
#             for J in J_list
#                 for g_x in g_x_list
#                     for g_z in g_z_list
                    
#                         mpo = get_Ising_MPO(N, J, g_x, g_z)
#                         acc = 10^(-10)
#                         max_sweeps = 10
#                         E_optimal, mps, sweep_number = variational_ground_state_MPS(N, d, D, mpo, acc, max_sweeps)
#                         total_spin = get_spin_half_expectation_value(N, mps, "z")
#                         average_spin = total_spin/N
#                         append!(average_spin_list, real(average_spin))
#                         append!(ground_state_energy_list, E_optimal)

#                     end
#                 end
#             end 
#         end
#     end
# end

# N, d, D, J, g_x = 4,2,2,1.0,0.0
# title_str = "N = $N, d = $d, D = $D, J = $J, g_x = $g_x"
# plot(g_z_list, average_spin_list, label = "Spin along z per site", title = latexstring(title_str), titlefontsize = 12)
# xlabel!(L"g_z")
# ylabel!("Spin along z per site")

# plot(J_list, average_spin_list)

# ----------------------------------------------------------------------------------------------------------------------------------
