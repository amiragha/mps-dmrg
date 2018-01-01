"""
    generateMPS(gmps, max_dim)

Construct a matrix product state with a maximum bond dimension
`max_dim` from the Fishman gates `gmps`.

Since the unitary gates orthogonalize the unitary matrix, if they are
applied in revese order to the configuration state (which is the state
in the occupation basis) they will generate the state in the original
spatial basis. So, a classical MPS from the configurations is made and
then the gates are applied in reverse order.

"""
function generateMPS(gmps::FishmanGates,
                     max_dim::Int64)

    mps = MPS{Complex128}(gmps.Lx, 2, gmps.configuration)

    ugate_manybody = eye(Complex128, 4, 4)

    ## NOTE: the gates are applies in reverse order
    for n=length(gmps.locations):-1:1
        site = gmps.locations[n]
        theta = gmps.thetas[n]
        ugate_manybody[2:3, 2:3] =
            [
                cos(theta) sin(theta);
                -sin(theta) cos(theta)
            ]
        ### TODO: explain why we need to move center of MPS
        move_center!(mps, site)
        apply_twosite_operator!(mps, site, ugate_manybody, max_dim)
    end
    return mps
end

"""
    generateMPS_twoflavors(gmps, max_dim)

two flavors!
"""
function generateMPS_twoflavors(gmps::FishmanGates,
                                max_dim::Int64)

    mps = MPS{Complex128}(gmps.Lx, 4, 3*gmps.configuration)

    ugate_manybody = eye(Complex128, 4, 4)

    ## NOTE: the gates are applies in reverse order
    for n=length(gmps.locations):-1:1
        site = gmps.locations[n]
        theta = gmps.thetas[n]
        ugate_manybody[2:3, 2:3] =
            [
                cos(theta) sin(theta);
                -sin(theta) cos(theta)
            ]
        ugate_manybody_twoflavor = generate_twoflavor_ugate(ugate_manybody)

        ### TODO: explain why we need to move center of MPS
        move_center!(mps, site)
        apply_twosite_operator!(mps, site, ugate_manybody_twoflavor, max_dim)
    end
    return mps
end

function generate_twoflavor_ugate(ugate::Matrix{T}) where {T<:Union{Float64,Complex128}}
    result = weavekron(ugate, ugate, 2, 2, 2)

    ## TODO: explain the following indeces
    # apply negative signs imposed by Fock space convention
    # result[6, 10]  *= -1.0
    # result[7, 10]  *= -1.0
    # result[10, 6]  *= -1.0
    # result[10, 7]  *= -1.0
    # result[14, 15] *= -1.0
    # result[15, 14] *= -1.0

    for j=1:16
        for i=1:16
            if xor(i in [10,12,14,16], j in [10,12,14,16])
                result *= -1.0
            end
        end
    end

    return result
end

function genMPSfromGMERA(gmera::GMERA)
    return MPS(4,2)
end
