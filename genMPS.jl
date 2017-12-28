function generateMPS(gmps::FishmanGates,
                     max_chi::Int64,
                     verbose::Bool=true)

    ## TODO: explain why the MPS need to be initialized by the same
    ## configuration obtained through GMPS procedure?
    mps = MPS{Complex128}(gmps.Lx, gmps.configuration)

    ugate_manybody = eye(Complex128, 4, 4)

    ## NOTE: The gates should be applied in opposite order compared to
    ## how they were obtained for GMPS, TODO: explain why!
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
        apply_twosite_operator!(mps, site, ugate_manybody, max_chi)
    end
    return mps
end

function genMPSfromGMERA(gmera::GMERA)
    return MPS(4,2)
end
