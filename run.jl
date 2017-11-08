function runstuff(Lx::Int, chi::Int, verbose=true)
    @assert Lx % 2 == 0

    (verbose) && println("making correlation matrix...")
    corr_matrix = correlation_matrix(Lx, div(Lx, 2))
    (verbose) && display(corr_matrix)

    (verbose) && println("generating the gates(GMPS)...")
    gmps = makeGMPS!(corr_matrix, 1.e-8)

    (verbose) && println("forming MPS from the GMPS gates...")
    mps = genMPSfromGMPS(gmps, chi)
    measure_entropies(mps)
    measure_heisenberg(mps)

    (verbose) && println("Gutzwiller projection and reporting...")
    gmps = gutzwiller_project(mps)
    measure_entropies(gmps)
    measure_heisenberg(gmps)

end
