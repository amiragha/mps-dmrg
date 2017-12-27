    using TensorOperations

## QQQ? should I define these as UInt instead? how?
# The MPS type
mutable struct MPS{T<:Number}
    length   :: Int64
    phys_dim :: Int64
    dims     :: Vector{Int64}
    matrices :: Vector{Array{T, 3}}
    center   :: Int64
end

####################
### constructors ###
####################

### TODO: write a constructor for an arbitrary classical configuration
function MPS{T}(Lx::Int64,
                d::Int64=2,
                noise::Float64=0.0) where {T<:Number}

    # I think the normalization factor is necessary so that the result
    # of contraction is not a very large number (d^L in this case)
    matrices = [ sqrt(1/d) * ones(T, 1, 1, d) + noise * rand(T, 1, 1, d)
                 for i=1:Lx ]

    dims = ones(Int64, Lx+1)

    ## QQQ?: normalize, probably due to application of noise?!
    ## QQQ?: why should I call center_at!() here?
    center_at!(state, Lx)
    MPS{T}(Lx, d, dims, matrices, Lx)
end

# constructor from occupied/unoccupied configuration vector{Int64} (d is 2)
function MPS{T}(Lx::Int64,
                configuration::Vector{Int64},
                noise::Float64=0.0) where {T<:Number}

    matrices = [ zeros(T, 1, 1, 2) + noise * rand(T,1,1,2)
                 for i=1:Lx ]

    ## NOTE: A binary (0 and 1) configuration vector{Int64} is
    ## assumed! Should I enforce this?
    for site=1:Lx
        matrices[site][1, 1, configuration[site]+1] = 1.
    end

    dims = ones(Int64, Lx+1)

    ## QQQ?: normalize, probably due to application of noise?!
    ## QQQ?: why there is a need to call center_at!()
    canonicalize_at!(matrices, Lx)
    MPS{T}(Lx, 2, dims, matrices, Lx)
end

## QQQ? Can I make a constructor from sparse matrices? at least for
## special cases ??
### TODO: make a constructor from sparse matrices
### TODO: make a constructor from matrices with symmetries

#constructor from a ketstate given in Ising configuration
function MPS(Lx::Int64,
             d::Int64,
             ketstate::Vector{T}) where {T<:Number}

    @assert length(ketstate) == d^Lx
    matrices = Array{T,3}[]

    dims = zeros(Int64, Lx+1)
    dims[1] = 1
    rdim = d^(Lx-1)

    # state_matrix is now : d*dims[1] x rdim
    state_matrix = transpose(reshape(ketstate,
                                     rdim, d*dims[1]))

    for link=2:Lx-1
        fact = svdfact(state_matrix, thin=true)
        S, n, ratio =  Tmp.truncate(fact[:S])
        dims[link] = n

        U = fact[:U][:,1:n]
        # 1- U is : dims[link-1]*d x dims[link]
        # 2- tranpose : dims[link] x dims[link-1]*d
        # 3- reshape : dims[link]*d x dims[link-1]
        # 4- transpose : dims[link-1] x dims[link]*d
        # 5- reshape : dims[link-1] x dims[link] x d
        ## QQQ? Is there a better faster way, does it matter?
        push!(matrices, reshape(transpose(reshape(transpose(U),
                                                  dims[link]*d,
                                                  dims[link-1])),
                                dims[link-1], dims[link], d))

        # 1- S*Vt is : dims[link] x rdim
        # 2- transpose : rdim x dims[link]
        rdim = div(rdim , d)
        # 3- reshape: rdim x d*dims[link]
        # 4- transpose: d*dims[link] x rdim
        state_matrix = transpose(reshape(transpose(
            diagm(S) * fact[:Vt][1:n,:]), rdim, d*dims[link]))
    end

    fact = svdfact(state_matrix, thin=true)
    S, n, ratio =  Tmp.truncate(fact[:S])
    dims[Lx] = n

    U = fact[:U][:,1:n]
    push!(matrices, reshape(transpose(reshape(transpose(U),
                                              dims[Lx]*d,
                                              dims[Lx-1])),
                            dims[Lx-1], dims[Lx], d))

    dims[Lx+1] = 1
    push!(matrices, reshape(diagm(S) * fact[:Vt],
                            dims[Lx], dims[Lx+1], d))

    return MPS{T}(Lx, d, dims, matrices, Lx)
end

################################
### conversion and promotion ###
################################

### TODO: fix and test this convert function
## QQQ? How can this help coding other methods?!
function convert(::MPS{Complex128},
                 mps::MPS{Float64})

    return MPS{Complex128}(mps.length,
                           mps.phys_dim,
                           mps.dims,
                           convert(Vector{Array{Complex128,3}}, mps.matrices),
                           mps.center)
end

#######################################
### MPS center manipulation methods ###
#######################################

"""
    canonicalize_push_rightstep!(matrices, site)

perform a single right step of the canonicalization procedure of a
vector of matrices at `site`. This is perform an SVD on the matrices
at `site`, and then multiply the singluar values and right unitary
matricx ``V^{\dagger}`` to matrices of `site+1`. This ensures the
matrices of `site` are left(!) isometric.

"""
function canonicalize_push_rightstep!(matrices::Vector{Array{T, 3}},
                                      site::Int64) where {T<:Number}
    Lx = length(matrices)
    if site < Lx
        a = matrices[site]
        dims = size(a)
        @tensor a[i,d,j] := a[i,j,d]
        fact = svdfact(reshape(a, dims[1]*dims[3], dims[2]))
        U = reshape(transpose(fact[:U]), dims[2], dims[1], dims[3])
        @tensor U[i,j,k] := U[j,i,k]
        matrices[site] = U

        dims = size(matrices[site+1])

        @tensor matrices[site+1][i,j,d] := (diagm(fact[:S]) * fact[:Vt])[i, k] *
            matrices[site+1][k,j,d]
    end
    return
end

"""
    canonicalize_push_leftstep!(matrices, site)

perform a single left step of the canonicalization procedure of a
vector of matrices at `site`. This is perform an SVD on the matrices
at `site`, and then multiply the left unitary matricx ``U`` and
singular values to matrices of `site-1`. This ensures the matrices of
`site` are right(!) isometric.

"""
function canonicalize_push_leftstep!(matrices::Vector{Array{T, 3}},
                                     site::Int64) where {T<:Number}
    if site > 0
        a = matrices[site]
        dims = size(a)
        fact = svdfact(reshape(a, dims[1], dims[2] * dims[3]))
        matrices[site] = reshape(fact[:Vt], dims[1], dims[2], dims[3])

        dims = size(matrices[site-1])
        @tensor matrices[site-1][i,j,d] := matrices[site-1][i,k,d] *
            (fact[:U] * diagm(fact[:S]))[k,j]

    end
end

"""
    recenter_at!(matrices, center)

canonicalize at `center` from the both ends of an MPS.

"""
function canonicalize_at!(matrices::Vector{Array{Complex128, 3}},
                          center::Int64)
    Lx = length(matrices)
    @assert center > 0 && center < Lx + 1

    for site=1:center_index-1
        canonicalize_to_right!(matrices, site)
    end

    for site=Lx:-1:center_index+1
        canonicalize_to_left!(matrices, site)
    end
end

"""
    move_center!(mps, new_center)

move center of mps to a new location at `new_center`.

"""
function move_center!(mps::MPS{T},
                      new_center::Int64) where {T<:Number}
    Lx= mps.length
    @assert new_center > 0 && new_center <= Lx

    ## NOTE: it is assumed that the center is correct!
    center = mps.center
    if new_center > center
        for p=center:new_center-1
            canonicalize_push_rightstep!(mps.matrices, p)
        end
    elseif new_center < center
        for p=center:-1:new_center+1
            canonicalize_push_leftstep!(mps.matrices, p)
        end
    end
    mps.center = new_center
end

# #########################################################
# ### MPS measurement and operator applications methods ###
# #########################################################

function norm(mps::MPS{T}) where {T<:Number}
    d = mps.phys_dim
    Lx = mps.length

    m0 = mps.dims[1]
    result = ones(T, m0, m0)
    for site=1:Lx
        mat = mps.matrices[site]

        # the ":=" sign is needed to redefine result every time!
        @tensor begin
            result[ru, rd] := ((result[lu, ld] * mat[lu, ru, d])
                               * conj(mat)[ld, rd, d])
        end
    end

    ## NOTE: the result is a matrix not a number. Here it is
    ## guaranteed to be real.
    return result
end

"""
    measure(mps, operator, location)

local operator measurement at one location.
"""
function measure(mps::MPS{T},
                 operator::Matrix{T},
                 location::Int64) where {T<:Number}
    d = mps.phys_dim
    Lx = mps.length
    @assert (location <= Lx) && (location > 0)
    @assert size(operator) == (d, d)

    m0 = mps.dims[1]
    result = ones(T, m0, m0)

    for site=1:Lx
        mat = mps.matrices[site]
        if site == location
            @tensor begin
                result[ru, rd] := ((result[lu, ld] * mat[lu, ru, du])
                                   * operator[dd, du]
                                   * conj(mat)[ld, rd, dd])
            end
        else
            # the ":=" sign is needed to redefine result every time!
            @tensor begin
                result[ru, rd] := ((result[lu, ld] * mat[lu, ru, d])
                                   * conj(mat)[ld, rd, d])
            end
        end
    end

    ## NOTE: result is <psi|psi> which is a dims[1] x dims[L+1] matrix
    ## that is guaranteed to be real.
    return result

end

"""
    measure(mps, operator, locations)

local operator measurement at a set of locations. currently just
calles the local measure function. TODO a faster way of computation
with less number of contractions.

"""
function measure(mps::MPS{T},
                 operator::Matrix{T},
                 locations::Vector{Int64}) where {T<:Number}

    n = length(locations)
    result = Vector{Matrix{T}}(n)
    for i = 1:n
        result[i] = measure(mps, operator, locations[i])
    end
    return result
end

"""
    measure(mps, operators, locations)

correlation measurements of a set of operators at distinct set of
locations. The locations are expected to be in strictly ascending
order.

"""
function measure(mps::MPS{T},
                 operators::Vector{Matrix{T}},
                 locations::Vector{Int64}) where {T<:Number}

    Lx = mps.length
    n = length(locations)
    @assert length(operators) == n
    @assert is_strictly_ascending(locations)
    @assert locations[1] > 0 && locations[n] <= Lx

    m0 = mps.dims[1]
    result = ones(T, m0, m0)

    index = 1
    for site=1:Lx
        mat = mps.matrices[site]
        if (index <= n) && (site == locations[index])
            @tensor begin
                result[ru, rd] := ((result[lu, ld] * mat[lu, ru, du])
                                   * (operators[index])[dd, du]
                                   * conj(mat)[ld, rd, dd])
            end
            index += 1
        else
            # the ":=" sign is needed to redefine result every time!
            @tensor begin
                result[ru, rd] := ((result[lu, ld] * mat[lu, ru, d])
                                   * conj(mat)[ld, rd, d])
            end
        end
    end
    return result
end

"""
    measure(mps, operators, mode)

correlation measurements of a set of operators at all distinct
locations possible. The parameter `mode` has two possible values, the
`:half` (default) measure at all possible /descending/ locations,
`:full` measures in all possible locations.

"""
function measure(mps::MPS{T},
                 operators::Vector{Matrix{T}},
                 mode::Symbol=:half) where {T<:Number}

    Lx = mps.length
    n = length(operators)

    result = T[]

    for locations in all_combinations(Lx, n, mode)
        push!(result, measure(mps, operators, locations)[1,1])
    end

    return result
end

"""
    measure(mps, mpo)

measures the expectation value of a matrix product operator `mpo`.

"""
function measure(mps::MPS{T},
                 mpo::MPO{T}) where {T<:Number}
    d = mps.phys_dim
    Lx = mps.length
    @assert (Lx == mpo.length) && (d == mpo.phys_dim)

    m0 = mps.dims[1]
    o0 = mpo.dims[1]
    result = ones(T, m0, m0, o0)

    ## NOTE: This is not the fastest contraction possible, but is not
    ## that bad!
    for site=1:Lx
        mats = mps.matrices[site]
        mato = mpo.matrices[site]
        # the ":=" sign is needed to redefine result every time!
        @tensor begin
            result[ru,rd,or] := (((result[lu,ld,ol] * mats[lu,ru,du])
                                  * mato[dd,ol,du,or])
                                 * conj(mats)[ld, rd, dd])
        end
    end

    ## NOTE: result is <psi|O|psi> which is a dims[1] x dims[L+1]
    ## matrix that is guaranteed to be real if the MPO is Hermitian
    return result
end

"""
    apply_nn_unitary!(mps, l, operator[, max_dim [, center_to]])

applies a unitray matrix `operator` which is `2d x 2d` matrix to site
`l` and `l+1` of the `mps`. The `max_dim` operator chooses the max
possible size of dimension of the new mps at bond between `l` and
`l+1`, The singular values are push to either `:left` or `:right`
(default) matrices using the variable `push_to`.

"""
function apply_nn_unitary!(mps      ::MPS{T},
                           l        ::Int64,
                           operator ::Matrix{T},
                           max_dim  ::Int64=mps.dim[l],
                           push_to=:right) where {T<:Number}

    ## QQQ?: does this unitary ever end up being actually complex in
    ## the Fishman approach?

    @assert mps.center == l || mps.center == l+1
    @assert l < mps.length-1

    d = mps.phys_dim

    ## NOTE: the unitary matrix is (2d)x(2d) dimensional and is acting
    ## on the physical dimension of MPS at 2 sites: l, l+1. The acting
    ## indeces are on the columns and l before l+1
    tensorO = reshape(operator,d,d,d,d)

    one = mps.matrices[l]
    two = mps.matrices[l+1]


    chi_l = mps.dims[l]
    chi_m = mps.dims[l+1]
    chi_r = mps.dims[l+2]

    @tensor R[a,i,b,j] := tensorO[i,j,k,l] * (one[a,b,k] * two[a,b,l])
    ## Q?: should be possible to make the reshaped version of R
    ## directly (how to get rid of indeces in TensorOperations)!

    fact = svdfact(reshape(R, chi_l*d, chi_r*d), thin=true)

    S, n, ratio = truncate(fact[:S], max_dim)

    U = fact[:U][:,1:n]
    Vt = fact[:Vt][1:n,:]
    ## QQQ? do we need to normalize S here?

    if (push_to == :right)
        U = reshape(transpose(U, n, chi_l, d))
        @tensor U[i,j,k] := U[j,i,k]
        mps.matrices[l]   = U
        mps.matrices[l+1] = reshape(diagm(S) * Vt, n, chi_r, d)
        mps.center = l+1
    elseif (push_to == :left)
        U = reshape(transpose(U * diagm(S), n, chi_l, d))
        @tensor U[i,j,k] := U[j,i,k]
        mps.matrices[l]   = U
        mps.matrices[l+1] = reshape(Vt          , n, chi_r, d)
        mps.center = l
    else error("invalid push_to value : ", push_to)
    end
end

"""
    calculate_entanglement_spectrum_at(mps, bond)

calculates the entanglement specturm at the given `bond` if the
orthogonality center of `mps`is located on the right site of the bond.

"""
function calculate_entanglement_spectrum_at(mps::MPS{T},
                                            bond::Int64) where {T<:Number}
    Lx = mps.length
    @assert bond == mps.center

    M1 = mps.matrices[bond]
    M2 = mps.matrices[bond+1]
    @tensor tmp[a,d,c,d'] := M1[a,b,d] * M2[b,c,d']

    diml = mps.dims[bond]
    dimr = mps.dims[bond+2]
    d = mps.phys_dim

    ### TODO: Is there an easier way to find singular values without
    ### calculating the vectors?

    ## NOTE: the square of the singular values are the entanglement
    ## spectrum because it is defined based on the reduced density
    return svdfact(reshape(tmp, diml*d, dimr*d))[:S].^2
end

"""
    entanglements!(mps)

calculates the entanglement specturm at each bond of MPS from left to
right by moving the center to each site and calling
`calculate_entanglement_spectrum_at` on the bond connecting that site
and the right neighbor. Finally it restores the original center, so
technically it leaves the MPS invariant.

"""

function entanglements!(mps::MPS{T}) where {T<:Number}

    initial_center = mps.center

    result = Vector{Float64}[]

    for bond = 1:mps.length-1
        move_center!(mps, bond)
        push!(result,
              calculate_entanglement_spectrum_at(mps, bond))
    end

    move_center!(mps, initial_center)
    return result
end

###########################
### some test functions ###
###########################

## NOTE: this function is only meant to be used for testing purposes
## because it is very expensive!
function mps2ketstate(mps::MPS{T}) where {T<:Number}

    Lx = mps.length
    d = mps.phys_dim

    ketstate = zeros(T, d^Lx)

    for ketindex::Int64 = 0:d^Lx-1
        resolve::Int64 = ketindex
        amplitude = T[1]

        ### TODO: there should be a better way to enumerate through
        ### ketindex without generating the configuration from scratch
        ### every time, but does it matter?!
        for i::Int64 = Lx:-1:1
            amplitude = mps.matrices[i][:,:,(resolve % d) + 1] * amplitude
            resolve = div(resolve, d)
        end
        ketstate[ketindex+1] = amplitude[1]
    end

    return ketstate
end

"""
    dispaly_matrices(mps[, range])

display all the matrices of the `mps` in `range`. This is useful only
for testing or educational purposes.

"""
function display_matrices(mps::MPS{T},
                          range::UnitRange{Int64}=1:16) where {T<:Number}
    for site=range
        for s=1:mps.phys_dim
            display("Matrix $site $(s-1)")
            display(mps.matrices[site][:,:,s])
        end
    end
end
################################
### read and write functions ###
################################

function save(mps::MPS{T}) where {T<:Number}
    return 0
end

function load(mps::MPS{T}) where {T<:Number}
    return 0
end

function load_alps_checkpoint()
    return 0
end

###########################
### Multi-MPS functions ###
###########################

"""
    overlap(mps1, mps2)

calculates the overlap between two MPS. Note that it is not divided by
the norm of the two MPSs, so it returned value is the overlap of the
two states multiplied by the norm of each.

"""
function overlap(mps1::MPS{S},
                 mps2::MPS{T}) where {S<:Number,T<:Number}
    d = mps1.phys_dim
    Lx = mps1.length
    @assert d == mps2.phys_dim && mps2.length == Lx

    result = ones(T, mps1.dims[1], mps2.dims[1])
    for site=1:Lx
        mat1 = mps1.matrices[site]
        mat2 = mps2.matrices[site]

        @tensor begin
            result[ru, rd] := ((result[lu,ld] * mat1[lu,ru,d])
                               * conj(mat2)[ld,rd,d])
        end
    end

    # NOTE: result is a matrix not a number.
    return result
end
