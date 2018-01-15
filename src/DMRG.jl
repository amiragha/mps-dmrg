"""
    dmrg_sweep_twosite(mps, mpo)
"""
function dmrg_sweep_twosite!(mps::MPS{T},
                             mpo::MPO{T};
                             verbose::Bool=false) where {T<:Union{Float64, Complex128}}

    Lx = mps.length
    physd = mps.phys_dim

    left  = ones(T, 1,1,1)
    for bond=1:Lx-2
        (verbose) && println("Minimizing sites ", bond, " and ", bond+1, " ...")
        right = ones(T, 1,1,1)
        for i=Lx:-1:bond+2
            mats = mps.matrices[i]
            mato = mpo.matrices[i]
            @tensor right[lu,lm,ld] :=  conj(mats)[ld,rd,d'] *
                (mato[d',lm,d,rm] * (mats[lu,ru,d] * right[ru,rm,rd]))
        end

        diml = mps.dims[bond]
        dimr = mps.dims[bond+2]

        ## QQQ? Is this the best way?!
        matvec!(v, v0) = dmrg_matvec_twosite!(v, left, mpo.matrices[bond], mpo.matrices[bond+1], right, v0)
        @tensor twosite[l,d1,d2,r] := (mps.matrices[bond])[l,m,d1] * (mps.matrices[bond+1])[m,r,d2]
        evals, evecs = eigsfn(matvec!, reshape(twosite, diml*physd*physd*dimr), true, nev=1, which=:SR)

        (verbose) && println("Energy is ", evals[1])

        fact = svdfact(reshape(evecs[:, 1], diml*physd, dimr*physd), thin=true)
        S, n, ratio = truncate(fact[:S])
        U = fact[:U][:,1:n]
        Vt = fact[:Vt][1:n,:]

        mps.dims[bond+1] = n
        U = reshape(transpose(U), n, diml, physd)
        mps.matrices[bond] = permutedims(U, [2,1,3])
        mps.matrices[bond+1] = reshape(diagm(S) * Vt, n, dimr, physd)
        mps.center = bond+1
        mats = mps.matrices[bond]
        mato = mpo.matrices[bond]
        @tensor left[ru,rm,rd] :=  conj(mats)[ld,rd,d'] *
                (mato[d',lm,d,rm] * (mats[lu,ru,d] * left[lu,lm,ld]))
    end

    #@show dims_are_consistent(mps)

    right  = ones(T, 1,1,1)
    for bond=Lx-1:-1:1
        (verbose) && println("Minimizing sites ", bond, " and ", bond+1, " ...")
        left = ones(T, 1,1,1)
        for i=1:bond-1
            mats = mps.matrices[i]
            mato = mpo.matrices[i]
            @tensor left[ru,rm,rd] :=  conj(mats)[ld,rd,d'] *
                (mato[d',lm,d,rm] * (mats[lu,ru,d] * left[lu,lm,ld]))
        end

        diml = mps.dims[bond]
        dimr = mps.dims[bond+2]

        ## QQQ? Is this the best way?!
        matvec!(v, v0) = dmrg_matvec_twosite!(v, left, mpo.matrices[bond], mpo.matrices[bond+1], right, v0)
        @tensor twosite[l,d1,d2,r] := (mps.matrices[bond])[l,m,d1] * (mps.matrices[bond+1])[m,r,d2]
        evals, evecs = eigsfn(matvec!, reshape(twosite, diml*physd*physd*dimr), true, nev=1, which=:SR)

        (verbose) && println("Energy is ", evals[1])

        fact = svdfact(reshape(evecs[:, 1], diml*physd, dimr*physd), thin=true)
        S, n, ratio = truncate(fact[:S])
        U = fact[:U][:,1:n]
        Vt = fact[:Vt][1:n,:]

        mps.dims[bond+1] = n
        U = reshape(transpose(U * diagm(S)), n, diml, physd)
        mps.matrices[bond] = permutedims(U, [2,1,3])
        mps.matrices[bond+1] = reshape(Vt, n, dimr, physd)
        mps.center = bond
        mats = mps.matrices[bond+1]
        mato = mpo.matrices[bond+1]
        @tensor right[lu,lm,ld] :=  conj(mats)[ld,rd,d'] *
                (mato[d',lm,d,rm] * (mats[lu,ru,d] * right[ru,rm,rd]))

    end
    nothing
end

### TODO: what is wrong with types for vectors? Apparently the
### _aupd_wrapper is calling with subarrays?!
function dmrg_matvec_twosite!(result,#::Vector{T},
                              left::Array{T, 3},
                              mpoten1::Array{T, 4},
                              mpoten2::Array{T, 4},
                              right::Array{T, 3},
                              inputv#::Vector{T}
                              ) where {T<:Union{Float64, Complex128}}

    diml = size(left, 1)
    dimr = size(right, 1)
    physd = size(mpoten1, 1)
    @assert  diml*physd*physd*dimr == length(inputv) == length(result)
    inputv = reshape(inputv, diml, physd, physd, dimr)

    ### TODO: explain the indexes.
    @tensor v[ld, d1',d2',rd] := left[lu,lm,ld] * (
        (mpoten1[d1',lm,d1,mm] * mpoten2[d2',mm,d2,rm]) *
        (inputv[lu,d1,d2,ru] * right[ru,rm,rd]))

    result[:] = reshape(v, diml*physd*physd*dimr)
end
