    using TensorOperations

# The MPS type
mutable struct MPS
    length :: Int64
    max_bond_dim :: Int64
    phys_dim :: Int64
    state :: Vector{Array{Complex128, 3}}
    center :: Int64
end

####################
### constructors ###
####################

function MPS(Lx, chi, d, configuration::Vector{Vector{Complex128}}, noise::Float64)

    @assert length(configuration) == L
    ### TODO: check for configuration or even better make it a structure
    state = [ reshape(configuration[i], 1, d, 1) + noise * rand(Float64, 1, d, 1)
              for i=1:Lx ]

    ## QQQ?: normalize, probably due to the application of noise?!
    center_at!(mps, Lx)
    MPS(Lx, chi, d, state)
end

function MPS(Lx, chi, d=2, noise::Float64=0.0)
    state = [ sqrt(1/d) * ones(Complex128, 1, d, 1) + noise * rand(Complex128, 1, d, 1)
              for i=1:Lx ]
    ## QQQ?: normalize, probably due to application of noise?!
    center_at!(state, Lx)
    MPS(Lx, chi, d, state, Lx)
end

# contructor from occupied/unoccupied configuration vector (d is 2)
function MPS(Lx, chi, configuration::Vector{Int}, noise::Float64=0.0)
    mps_noise = noise * rand(Complex128, 1, 2, 1)

    state = [ zeros(Complex128, 1, 2, 1) + noise * rand(Complex128,1,2,1)
              for i=1:Lx ]
    ## NOTE: A binary (0 and 1) configuration files is assumed!
    for site=1:Lx
        state[site][1, configuration[site]+1, 1] = 1
    end

    ## Q?: normalize, probably due to application of noise?!
    center_at!(state, Lx)
    MPS(Lx, chi, 2, state, Lx)
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
        fact = svdfact(reshape(a, dims[1] * dims[2], dims[3]))
        state[site] = reshape(fact[:U], dims[1], dims[2], dims[3])

        dims = size(state[site+1])
        state[site+1] = reshape( diagm(fact[:S]) * fact[:Vt] *
                                 reshape(state[site+1], dims[1], dims[2] * dims[3]),
                                 dims[1], dims[2], dims[3] )
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
        state[site-1] = reshape(reshape(state[site-1], dims[1]*dims[2], dims[3]) *
                                fact[:U] * diagm(fact[:S]),
                                dims[1], dims[2], dims[3])
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

function center_at!(mps::MPS,
                    center_index::Int64)
    center_at(mps.state, center_index)
    mps.center = center_index
end

## QQQ? describe the significance of center changing in detail?
function move_center!(mps::MPS,
                      new_center::Int64)
    @assert new_center > 0 && new_center < mps.length + 1

    ## NOTE: Assume center is already at point current_center
    current_center = mps.center

    if new_center < current_center
        for site = current_center:-1:new_center+1
            canonicalize_to_left!(mps.state, site)
        end
    elseif new_center > current_center
        for site = current_center:new_center-1
            canonicalize_to_right!(mps.state, site)
        end
    end

    mps.center = new_center
end

#########################################################
### MPS measurement and operator applications methods ###
#########################################################

function apply_2site_unitary!(mps::MPS,
                              l::Int64,
                              U::Matrix{Complex128},
                              center_to=:right)

    ## Q?: does this unitary ever end up being actually complex in the
    ## Fishman approach?

    @assert 0 < l < mps.length

    d = mps.phys_dim

    ## NOTE: the unitary matrix is (2d)x(2d) dimensional and is acting
    ## on the physical dimension of MPS at 2 sites: l, l+1
    tensorU = reshape(U,d,d,d,d)
    ### TODO: explain how all these reshapes work correctly for future!

    one = mps.state[l]
    two = mps.state[l+1]

    chi_l = size(one)[1]
    chi_m = size(one)[3]
    chi_r = size(two)[3]

    @tensor R[alpha,i,j,beta] := tensorU[i,j,k,l] * (one[alpha,k,gamma] * two[gamma,l,beta])
    ## Q?: should be possible to make the reshaped version of R directly!

    fact = svdfact(reshape(R, chi_l*d, d*chi_r), thin=true)
    U  = fact[:U]
    S  = fact[:S]
    Vt = fact[:Vt]

    chi_new = min( mps.max_bond_dim, sum(S .> S[1]*1.e-14) )

    if chi_new < min(chi_l*d, d*chi_r)
        S  = S[1:chi_new]
        U  = U[:, 1:chi_new]
        Vt = Vt[1:chi_new, :]
        ## possibly normalize?!
    end

    if (center_to == :right)
        mps.state[l]   = reshape(U            , chi_l  , d, chi_new)
        mps.state[l+1] = reshape(diagm(S) * Vt, chi_new, d, chi_r)
        mps.center = l+1
    elseif (center_to == :left)
        mps.state[l]   = reshape(U * diagm(S), chi_l  , d, chi_new)
        mps.state[l+1] = reshape(Vt          , chi_new, d, chi_r)
        mps.center = l
    end

end

function measure_entropies(mps::MPS)

end

function measure_heisenberg(mps::MPS)

end
