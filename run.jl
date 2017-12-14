function runstuff(Lx::Int64,
                  chi::Int64,
                  verbose::Bool=true)

    @assert Lx % 2 == 0

    (verbose) && println("making correlation matrix...")
    corr_matrix = correlation_matrix(Lx)
    (verbose) && display(corr_matrix)

    (verbose) && println("generating the gates(GMPS)...")
    gmps = makeGMPS!(corr_matrix, 1.e-8)

    (verbose) && println("forming MPS from the GMPS gates...")
    mps = generateMPS(gmps, chi)

    (verbose) && println("performing measurement on the MPS...")
    entropies(mps)
    measure(mps)

    (verbose) && println("Gutzwiller projection and reporting...")
    gmps = gutzwiller_project(mps)
    measure_entropies(gmps)
    measure_heisenberg(gmps)

end
