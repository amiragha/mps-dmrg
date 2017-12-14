    using TensorOperations

## QQQ? should I define these as UInt instead? how?
# The MPS type
mutable struct MPS{T<:Number}
    length :: Int64
    phys_dim :: Int64
    dims :: Vector{Int64}
    matrices :: Vector{Array{T, 3}}
    center :: Int64
end

####################
### constructors ###
####################

# function MPS(Lx::UInt64,
#              chi::UInt64,
#              d::UInt64,
#              configuration::Vector{Vector{Complex128}},
#              noise::Float64)

#     @assert length(configuration) == L
#     ### TODO: check for configuration or even better make it a structure
#     state = [ reshape(configuration[i], 1, d, 1) + noise * rand(Float64, 1, d, 1)
#               for i=1:Lx ]

#     ## QQQ?: normalize, probably due to the application of noise?!
#     center_at!(mps, Lx)
#     MPS(Lx, chi, d, state)
# end

### TODO: merge this constructor with the above one for an arbitrary
### classical configuration
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
    center_at!(state, Lx)
    MPS{T}(Lx, 2, dims, matrices, Lx)
end

## QQQ? Can I make a constructor from sparse matrices? at least for
## special cases ??
### TODO: make a constructor from sparse matrices

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

#######################################
### MPS center manipulation methods ###
#######################################

function canonicalize_to_right!(state::Vector{Array{Complex128, 3}},
                                site::Int64)
    Lx = length(state)
    if site < Lx
        a = state[site]
        dims = size(a)
        @tensor a[i,d,j] := a[i,j,d]
        fact = svdfact(reshape(a, dims[1]*dims[3], dims[2]))
        U = reshape(transpose(fact[:U]), dims[2], dims[1], dims[3])
        @tensor U[i,j,k] := U[j,i,k]
        state[site] = U

        dims = size(state[site+1])

        @tensor state[site+1][i,j,d] := (diagm(fact[:S]) * fact[:Vt])[i, k] *
            state[site+1][k,j,d]
    end
    return
end

function canonicalize_to_left!(state::Vector{Array{Complex128, 3}},
                               site::Int64)
    if site > 0
        a = state[site]
        dims = size(a)
        fact = svdfact(reshape(a, dims[1], dims[2] * dims[3]))
        state[site] = reshape(fact[:Vt], dims[1], dims[2], dims[3])

        dims = size(state[site-1])
        @tensor state[site-1][i,j,d] := state[site-1][i,k,d] *
            (fact[:U] * diagm(fact[:S]))[k,j]

    end
end

function center_at!(state::Vector{Array{Complex128, 3}},
                    center_index::Int64)
    Lx = length(state)
    @assert center_index > 0 && center_index < Lx + 1

    for site=1:center_index-1
        canonicalize_to_right!(state, site)
    end

    for site=Lx:-1:center_index+1
        canonicalize_to_left!(state, site)
    end
end

# function center_at!(mps::MPS,
#                     center_index::Int64)
#     center_at(mps.state, center_index)
#     mps.center = center_index
# end

# ## QQQ? describe the significance of center changing in detail?
# function move_center!(mps::MPS,
#                       new_center::Int64)
#     @assert new_center > 0 && new_center < mps.length + 1

#     ## NOTE: Assume center is already at point current_center
#     current_center = mps.center

#     if new_center < current_center
#         for site = current_center:-1:new_center+1
#             canonicalize_to_left!(mps.state, site)
#         end
#     elseif new_center > current_center
#         for site = current_center:new_center-1
#             canonicalize_to_right!(mps.state, site)
#         end
#     end

#     mps.center = new_center
# end

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
            result[ru, rd] := (result[lu, ld] * mat[lu, ru, d]) * conj(mat)[ld, rd, d]
        end
    end

    ## NOTE: result is <psi|psi> which is a dims[1] x dims[L+1] matrix
    ## that is guaranteed to be real.
    return real(result)
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
    return real(result)

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
    apply_nn_unitary!(mps, l, U[, center_to])

applies a unitray matrix `U` which is `2d x 2d` matrix to site `l` and
`l+1` of the `mps`. It choose the new canonicalization center to be
by the `center_to` variable.

"""
function apply_nn_unitary!(mps      ::MPS{T},
                           l        ::Int64,
                           operator ::Matrix{Complex128},
                           max_dim  ::Int64,
                           center_to=:right) where {T<:Number}

    ## QQQ?: does this unitary ever end up being actually complex in
    ## the Fishman approach?

    @assert 0 < l < mps.length-1

    d = mps.phys_dim

    ## NOTE: the unitary matrix is (2d)x(2d) dimensional and is acting
    ## on the physical dimension of MPS at 2 sites: l, l+1
    tensorO = reshape(operator,d,d,d,d)
    ### TODO: explain how all these reshapes work correctly for future!

    one = mps.matrices[l]
    two = mps.matrices[l+1]


    chi_l = dims[l]
    chi_m = dims[l+1]
    chi_r = dims[l+2]

    @tensor R[alpha,i,beta,j] := tensorO[i,j,k,l] * (one[alpha,gamma,k] * two[gamma,beta,l])
    ## Q?: should be possible to make the reshaped version of R directly!

    fact = svdfact(reshape(R, chi_l*d, chi_r*d), thin=true)

    S, n, ratio = truncate(fact[:S], max_dim)

    U = fact[:U][:,1:n]
    Vt = fact[:Vt][1:n,:]
    ## QQQ? do we need to normalize S here?

    if (center_to == :right)
        mps.matrices[l]   = reshape(U            , chi_l ,n, d)
        mps.matrices[l+1] = reshape(diagm(S) * Vt, n, chi_r, d)
        mps.center = l+1
    elseif (center_to == :left)
        mps.matrices[l]   = reshape(U * diagm(S), chi_l, n, d)
        mps.matrices[l+1] = reshape(Vt          , n, chi_r, d)
        mps.center = l
    end
end

function entropies(mps::MPS{T}) where {T<:Number}

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

################################
### read and write functions ###
################################

function save(mps::MPS{T}) where {T<:Number}

end

function load(mps::MPS{T}) where {T<:Number}

end
