    using TensorOperations

## QQQ? should I define these as UInt instead? how?
# The MPS type
mutable struct MPS
    length :: Int64
    phys_dim :: Int64
    dims :: Vector{Int64}
    matrices :: Vector{Array{Complex128, 3}}
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

# function MPS(Lx::UInt64,
#              chi::UInt64,
#              d::UInt64=2,
#              noise::Float64=0.0)

#     state = [ sqrt(1/d) * ones(Complex128, 1, d, 1) + noise * rand(Complex128, 1, d, 1)
#               for i=1:Lx ]
#     ## QQQ?: normalize, probably due to application of noise?!
#     center_at!(state, Lx)
#     MPS(Lx, chi, d, state, Lx)
# end

# # contructor from occupied/unoccupied configuration vector (d is 2)
# function MPS(Lx::UInt64,
#              chi::UInt64,
#              configuration::Vector{UInt64},
#              noise::Float64=0.0)

#     mps_noise = noise * rand(Complex128, 1, 2, 1)

#     state = [ zeros(Complex128, 1, 2, 1) + noise * rand(Complex128,1,2,1)
#               for i=1:Lx ]
#     ## NOTE: A binary (0 and 1) configuration files is assumed!
#     for site=1:Lx
#         state[site][1, configuration[site]+1, 1] = 1
#     end

#     ## Q?: normalize, probably due to application of noise?!
#     center_at!(state, Lx)
#     MPS(Lx, chi, 2, state, Lx)
# end

#constructor from a ketstate given in Ising configuration
function MPS(Lx::Int64,
             d::Int64,
             ketstate::Vector{Complex128})

    @assert length(ketstate) == d^Lx
    matrices = Array{Complex128,3}[]

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

    return MPS(Lx, d, dims, matrices, Lx)
end
#######################################
### MPS center manipulation methods ###
#######################################

# function canonicalize_to_right!(state::Vector{Array{Complex128, 3}},
#                                 site::Int64)
#     Lx = length(state)
#     if site < Lx
#         a = state[site]
#         dims = size(a)
#         fact = svdfact(reshape(a, dims[1] * dims[2], dims[3]))
#         state[site] = reshape(fact[:U], dims[1], dims[2], dims[3])

#         dims = size(state[site+1])
#         state[site+1] = reshape( diagm(fact[:S]) * fact[:Vt] *
#                                  reshape(state[site+1], dims[1], dims[2] * dims[3]),
#                                  dims[1], dims[2], dims[3] )
#     end
#     return
# end

# function canonicalize_to_left!(state::Vector{Array{Complex128, 3}},
#                                site::Int64)
#     if site > 0
#         a = state[site]
#         dims = size(a)
#         fact = svdfact(reshape(a, dims[1], dims[2] * dims[3]))
#         state[site] = reshape(fact[:Vt], dims[1], dims[2], dims[3])

#         dims = size(state[site-1])
#         state[site-1] = reshape(reshape(state[site-1], dims[1]*dims[2], dims[3]) *
#                                 fact[:U] * diagm(fact[:S]),
#                                 dims[1], dims[2], dims[3])
#     end
# end

# function center_at!(state::Vector{Array{Complex128, 3}},
#                     center_index::Int64)
#     Lx = length(state)
#     @assert center_index > 0 && center_index < Lx + 1

#     for site=1:center_index-1
#         canonicalize_to_right!(state, site)
#     end

#     for site=Lx:-1:center_index+1
#         canonicalize_to_left!(state, site)
#     end
# end

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

function norm(mps::MPS)
    d = mps.phys_dim
    Lx = mps.length

    m0 = mps.dims[1]
    result = ones(Complex128, m0, m0)

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

local operator measurement
"""
function measure(mps::MPS,
                 operator::Matrix{Complex128},
                 location::Int64)
    d = mps.phys_dim
    Lx = mps.length
    @assert (location =< Lx) && (location > 0)
    @assert size(operator) == (d, d)

    m0 = mps.dims[1]
    result = ones(Complex128, m0, m0)

    for site=1:Lx
        mat = mps.matrices[site]
        if site == location
            @tensor begin
                result[ru, rd] := ((result[lu, ld] * mat[lu, ru, du])
                                   * operator[dd, du]
                                   * conj(mat)[ld, rd, dl])
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

function measure(mps::MPS,
                 mpo::MPO)
    d = mps.phys_dim
    Lx = mps.length
    @assert (Lx == mpo.length) && (d == mpo.phys_dim)

    m0 = mps.dims[1]
    o0 = mpo.dims[1]
    result = ones(Complex128, m0, m0, o0)

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

# function apply_2site_unitary!(mps::MPS,
#                               l::Int64,
#                               U::Matrix{Complex128},
#                               center_to=:right)

#     ## Q?: does this unitary ever end up being actually complex in
#     ## the Fishman approach?

#     @assert 0 < l < mps.length

#     d = mps.phys_dim

#     ## NOTE: the unitary matrix is (2d)x(2d) dimensional and is acting
#     ## on the physical dimension of MPS at 2 sites: l, l+1
#     tensorU = reshape(U,d,d,d,d)
#     ### TODO: explain how all these reshapes work correctly for future!

#     one = mps.state[l]
#     two = mps.state[l+1]

#     chi_l = size(one)[1]
#     chi_m = size(one)[3]
#     chi_r = size(two)[3]

#     @tensor R[alpha,i,j,beta] := tensorU[i,j,k,l] * (one[alpha,k,gamma] * two[gamma,l,beta])
#     ## Q?: should be possible to make the reshaped version of R directly!

#     fact = svdfact(reshape(R, chi_l*d, d*chi_r), thin=true)
#     U  = fact[:U]
#     S  = fact[:S]
#     Vt = fact[:Vt]

#     chi_new = min( mps.max_bond_dim, sum(S .> S[1]*1.e-14) )

#     if chi_new < min(chi_l*d, d*chi_r)
#         S  = S[1:chi_new]
#         U  = U[:, 1:chi_new]
#         Vt = Vt[1:chi_new, :]
#         ## possibly normalize?!
#     end

#     if (center_to == :right)
#         mps.state[l]   = reshape(U            , chi_l  , d, chi_new)
#         mps.state[l+1] = reshape(diagm(S) * Vt, chi_new, d, chi_r)
#         mps.center = l+1
#     elseif (center_to == :left)
#         mps.state[l]   = reshape(U * diagm(S), chi_l  , d, chi_new)
#         mps.state[l+1] = reshape(Vt          , chi_new, d, chi_r)
#         mps.center = l
#     end

# end

function entropies(mps::MPS)

end

###########################
### some test functions ###
###########################

## NOTE: this function is only meant to be used for testing purposes
## because it is very expensive!
function mps2ketstate(mps::MPS)

    Lx = mps.length
    d = mps.phys_dim

    ketstate = zeros(Complex128, d^Lx)

    for ketindex::Int64 = 0:d^Lx-1
        resolve::Int64 = ketindex
        amplitude = Complex128[1]

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

function save(mps::MPS)

end

function load(mps::MPS)

end
