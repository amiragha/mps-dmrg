using TensorOperations

# The MPS type
mutable struct MPS
    length :: Int64
    max_bond_dim :: Int64
    phys_dim :: Int64
    state :: Vector{Array{Complex128, 3}}
end

####################
### constructors ###
####################

function MPS(Lx, chi, d, configuration::Vector{Vector{Complex128}}, noise::Float64)

    @assert length(configuration) == L
    ### TODO: check for configuration or even better make it a structure

    mps_noise = noise * rand(Float64, 1, d, 1)
    state = [ reshape(configuration[i], 1, d, 1) + mps_noise for i=1:Lx ]
    ## Q?: normalize, probably due to the application of noise?!
    MPS(Lx, chi, d, state)
end

function MPS(Lx, chi, d=2, noise::Float64=0.0)
    mps_noise = noise * rand(Complex128, 1, d, 1)
    state = [ sqrt(1/d) * ones(Complex128, 1, d, 1) + mps_noise for i=1:Lx ]
    ## Q?: normalize, probably due to application of noise?!
    MPS(Lx, chi, d, state)
end

################################
### MPS manipulation methods ###
################################

function apply_2site_unitary!(mps::MPS, l::Int64, U::Matrix{Complex128})

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

    standard = :right
    if (standard == :right)
        mps.state[l]   = reshape(U            , chi_l  , d, chi_new)
        mps.state[l+1] = reshape(diagm(S) * Vt, chi_new, d, chi_r)
        mps.canonical_point = l+1
    elseif (standard == :left)
        mps.state[l]   = reshape(U * diagm(S), chi_l  , d, chi_new)
        mps.state[l+1] = reshape(Vt          , chi_new, d, chi_r)
        mps.canonical_point = l
    end

end

function measure_entropies(mps::MPS)

end

function measure_heisenberg(mps::MPS)

end
