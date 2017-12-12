function gutzwiller_project(mps::MPS)
    ## NOTE: currently the assumption is that the mps is for one free
    ## half-filled fermion species. Here I extend it to two (spin up
    ## and down) identical versions and then Gutzwiller project it,
    ## which means ignoring the double and single occupancy.
    projected_mps = mps
    return projected_mps
end
