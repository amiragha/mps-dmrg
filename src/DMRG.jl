"""
    dmrg_sweep_twosite(mps, mpo)
"""
function dmrg_sweep_twosite(initial::MPS{T},
                            mpo::MPO{T}) where {T<:Union{Float64, Complex128}}

    newmps = mps

    Lx = mps.length
    bond = 1
    left  = ones(T, 1,1,1)
    right = ones(T, 1,1,1)
    mats = mps.matrices[site]
    for site=Lx:-1:3
        mats = mps.matrices[site]
        mato = mpo.matrices[site]

        @tensor right[lu,lm,ld] :=  conj(mats)[ld,rd,d'] *
            (mato[d',lm,d,rm] * (mats[lu,ru,d] * right[ru,rm,rd]))
    end

    ## QQQ? Is this the best way?!
    function dmrg_matvec!(result::Vector{T},
                          input::Vector{T})

        reshape(input , diml, d, d, dimr)
        reshape(result, diml, d, d, dimr)

        @tensor result[ld, d1',d2',rd] := left[lu,lm,ld] * (
            (mpo.matrices[bond][d1',lm,d1,mm] * mpo.matrices[bond+1][d2',mm,d2,rm]) *
            (input[lu,d1,d2,ru] * right[ru,rm,rd]))

        reshape(result, diml*d*d*dimr)
        nothing
    end

    evals, evecs = (dmrg_matvec!, normalize(randn(diml*d*d*dimr)), true)

    println(evals)

    # perform svd

    # push the new matrices

    return result
end
