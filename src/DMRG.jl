"""
    dmrg_sweep(initial, mpo[, optimization])
"""
function dmrg_sweep(initial::MPS{T},
                    mpo::MPO{T};
                    optimization::Symbol=:twosite) where {T<:Union{Float64, Complex128}}

    result = initial
    # one full sweep

    # make the right tensor

    # make the left tensor

    # find the new matrices
    new_matrices = dmrg_lanczos(left_tensor, mp, right_tensor, input_vector)

    # perform svd

    # push the new matrices

    # do till end

    return result
end

"""
    dmrg_matvec(left, mpo, right, v)

This is matrix vector multiplication of the DMRG-MPS, we need to
define this multiplication in a specific order for tensor contraction
to get the best performance.

"""
function dmrg_matvec(left,
                     mpo,
                     right,
                     v)

end

function dmrg_lanczos(lefttensor,
                      mpo,
                      righttensor,
                      input_vector)

end
