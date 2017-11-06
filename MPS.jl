using TensorOperations

# The MPS type
mutable struct MPS
    length :: Int64
    bond_dim :: Int64
    phys_dim :: Int64
    state :: Vector{Array{Complex128, 3}}
end

####################
### constructors ###
####################

function MPS(L, chi, d, configuration::Vector{Vector{Complex128}}, noise::Float64)

    @assert length(configuration) == L
    ### TODO: check for configuration or even better make it a structure

    mps_noise = noise * rand(Float64, 1, d, 1)
    state = [ reshape(configuration[i], 1, d, 1) + mps_noise for i=1:L ]
    ## Q?: normalize, probably due to the application of noise?!
    MPS(L, chi, d, state)
end

function MPS(L, chi, d=2, noise::Float64=0.0)
    mps_noise = noise * rand(Complex128, 1, d, 1)
    state = [ sqrt(1/d) * ones(Complex128, 1, d, 1) + mps_noise for i=1:L ]
    ## Q?: normalize, probably due to application of noise?!
    MPS(L, M, d, state)
end

################################
### MPS manipulation methods ###
################################

function apply_2site_unitary!(mps::MPS, l::Int64, U::Matrix{Complex128})

    ## Q?: does this unitary ever end up being actually complex in the
    ## Fishman approach?

    @assert 0 < l < mps.L

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

    # 1. truncate based on a threshold
    # 2. find new bond dimensions
    # 3. update the states of the MPS based on the standard(canonicalization)

end
