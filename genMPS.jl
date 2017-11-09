function genMPSfromGMPS(gmps::GMPS, chi::Int, verbose=true)
    ## QQQ? Why the MPS need to be initialized by the same
    ## configuration obtained through GMPS procedure?
    mps = MPS(gmps.Lx, chi, gmps.configuration)
    (verbose) && display(mps)

    ugate_manybody = eye(4, 4)

    ## NOTE: The gates should be applied in opposite order compared to
    ## how they were obtained for GMPS
    for nngate in gmps.gates
        theta = nngate.theta
        ugate_mabybody[2:3, 2:3] =
            [
                cos(theta) sin(theta);
                -sin(theta) cos(theta)
            ]
        move_center_MPS!(mps, ugate.site)
        apply_2site_unitary!(mps, ugate_manybody)
    end
    return mps
end

function genMPSfromGMERA(gmera::GMERA)
    return MPS(4,2)
end
