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
"""
function contraction(A, c_A::Tuple, B, c_B::Tuple)

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
    to be contracted have dimensions $(A_indices_vec[c_A]) and indices of second
    tensor B to be contracted have dimensions $(B_indices_vec[c_B]).")

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

# Example -------------------------------------------------------------------
# @time begin
# A = cat([1 2; 3 4], [5 6; 7 8], dims = 3)
# B = cat([9 11; 11 12], [13 14; 15 16], dims = 3)
# c_A = (1, 2)
# c_B = (2, 1)
# display(contraction(A, c_A, B, c_B))
# end
# ---------------------------------------------------------------------------
