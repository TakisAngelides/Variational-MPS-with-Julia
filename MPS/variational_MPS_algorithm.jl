using Profile
using LinearAlgebra
using Arpack
using BenchmarkTools
using Plots
using LaTeXStrings
using Test
using HDF5
include("utility_functions.jl")

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

        # @test length(S) == min(D_left, d*D_right) # property of SVD, note S is returned as a vector not a diagonal matrix

        # Note for complex M_initial, the following should be named Vd for V_dagger rather than Vt for V_transpose but we keep it Vt
        Vt = F.Vt # Vt_(s_i-1)(sigma_i a_i)
        Vt = reshape(Vt, (length(S), d, D_right)) # Unmerging indices: Vt_(s_i-1)(sigma_i a_i) -> Vt_(s_i-1)(sigma_i)(a_i)
        B = permutedims(Vt, (1,3,2)) # Vt_(s_i-1)(sigma_i)(a_i) -> B_(s_i-1)(a_i)(sigma_i)
        US = U*Diagonal(S) # US_(a_i-1)(s_i-1)

        # @test isapprox(contraction(B, (2,3), conj!(deepcopy(B)), (2,3)), I) # right canonical form property

        return US, B # US_(a_i-1)(s_i-1), B_(s_i-1)(a_i)(sigma_i)

    else # See Schollwock equation (136) for left canonical form

        D_left, D_right, d = size(M_initial) 
        M = permutedims(M_initial, (3, 1, 2)) # M_(a_i-1)(a_i)(sigma_i) -> M_(sigma_i)(a_i-1)(a_i)
        M = reshape(M, (d*D_left, D_right)) # M_(sigma_i)(a_i-1)(a_i) -> M_(sigma_i a_i-1)(a_i)
        F = svd(M)
        U = F.U # U_(sigma_i a_i-1)(s_i)
        S = F.S # S_(s_i)(s_i) although stored as vector here

        # @test length(S) == min(d*D_left, D_right) # property of SVD, note S is returned as a vector not a diagonal matrix

        Vt = F.Vt # Vt_(s_i)(a_i)
        U = reshape(U, (d, D_left, length(S))) # U_(sigma_i)(a_i-1)(s_i)
        A = permutedims(U, (2, 3, 1)) # A_(a_i-1)(s_i)(sigma_i)
        SVt = Diagonal(S)*Vt # SVt_(s_i)(a_i)

        # @test isapprox(contraction(conj!(deepcopy(A)), (1,3), A, (1,3)), I) # left canonical form property

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
            M_tilde = contraction(mps[i-1], (2,), US, (1,)) # M_tilde_(a_i-2)(sigma_i-1)(s_i-1)
            M_tilde = permutedims(M_tilde, (1,3,2)) # Put the physical index to the right most place M_tilde_(a_i-2)(sigma_i-1)(s_i-1) -> M_tilde_(a_i-2)(s_i-1)(sigma_i-1)
            if i == 2
                if normalized # If we require the state to be normalized then we gauge even the first site to be a B tensor so that the whole contraction <psi|psi> collapses to the identity
                    _, mps[1] = gauge_site(right, M_tilde) # The placeholder _ for the value of US tells us that we are discarding that number and so the state is normalized just like we would divide psi by sqrt(a) when <psi|psi> = a
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
    E, M = eigs(Heff, nev=1, which=:SR) # nev = 1 => it will return only 1 number of eigenvalues, SR => compute eigenvalues which have the smallest real part (ie the ground state energy and upwards depending on nev), also note M'*M = 1.0+0.0im
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

    mps = initialize_MPS(N, d, D) # Initialize a random mps
    gauge_mps!(right, mps, true, N) # Put it in right canonical form and normalize it

    # This are the partial contractions of the initial mps configuration which is contains all B tensors, 
    # it has length N+1 where the first and last elements are 1x1x1 tensors of value 1. The states vector will thus be of the form
    # 1RRRR1 for N = 5.

    states = initialize_L_R_states(mps, mpo, N) # Holds partial contractions needed for each site's H_eff
    E_initial = 10^(-5) # Will hold the ground state energy approximation of the previous full sweep
    E_optimal = 0 # Will hold the final best ground state energy approximation once the algorithm is finished
    sweep_number = 0 # Counts the number of full sweeps performed
    US = 0 # Will hold the residual matrices from SVD when we put a newly updated tensor in right canonical form while sweeping from right to left

    while(true)
        
        E = 0 # Will hold the ground state energy apprxoimation right after a full sweep finishes

        # From left to right sweep (right moving sweep or right sweep)

        for i in 1:N-1 # Its up to N-1 here because the left moving sweep will start from N

            L = states[i]
            W = mpo[i]
            R = states[i+1]
            M, _ = get_updated_site(L, W, R)
            mps[i], _ = gauge_site(left, M)
            update_states!(right, states, mps[i], W, i+1) # i+1 because for loop starts from 1 and index 1 in states is the dummy 1x1x1 tensor of value 1 

        end

        for i in N:-1:2 # Lower limit is 2 here because the right moving sweep will start from 1

            L = states[i]
            W = mpo[i]
            R = states[i+1]
            M, E = get_updated_site(L, W, R)
            US, mps[i] = gauge_site(right, M) # We only use US after this for loop to restore normalization to the mps state
            update_states!(left, states, mps[i], W, i)

        end
        
        fractional_energy_change = abs((E - E_initial)/E_initial)

        if fractional_energy_change < accuracy

            E_optimal = E
            
            # To restore the normalization of the mps

            # We start with a normalized mps and each time we update a tensor we replace it with a normalized one such that the mps 
            # normalization is maintained. The fact that we discard residual matrices from the SVD that put the updated tensors into 
            # a particular gauge changes the mps and thus destroys its normalization. However these residual matrices would have been 
            # multiplied on the next site which is about to get updated next and replaced with a normalized tensor, 
            # restoring theoverall mps normalization. Just as we are about to stop sweeping, we need to multiply onto the tensor 
            # on site 1 the lastresidual matrices from gauging site 2 so as to not change the mps, 
            # maintaining in this way its normalization.

            mps[1] = contraction(mps[1], (2,), US, (1,))
            mps[1] = permutedims(mps[1], (1,3,2))

            # println("Desired accuracy reached.")

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
