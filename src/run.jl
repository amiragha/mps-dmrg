function rungutz(Lx::Int64,
                 max_dim::Int64;
                 verbose::Bool=true,
                 compare::Bool=false)

    @assert Lx % 2 == 0
    (verbose) && println("run Gutzwiller GDMRG of Lx = ",
                         Lx, " with max_dim = ", max_dim)

    corr_matrix = correlation_matrix(hamiltonian_matrix(Lx), div(Lx,2))
    gmps = makeGMPS!(corr_matrix, 1.e-8)

    twoflavormps = generateMPS_twoflavors(gmps, max_dim)
    mpsgutz = Tmp.gutzwiller_project(twoflavormps)

    (verbose) && println("dims are ", mpsgutz.dims)
    # (verbose) && println("saving the mps...")
    # save(mpsgutz, string("mps", Dates.format(now(), "yymmdd:HH:MM:SS"),
    #                      ",", "gutzwiller,chainLx=", "Lx", ".h5"))

    gutznorm = real(norm(mpsgutz)[1,1])
    (verbose) && println("mps norm2 is ", gutznorm)

    (verbose) && println("measurements...")
    measgutz_zz = real.(measure(mpsgutz, [sz_half, sz_half]))/gutznorm
    measgutz_pm = real.(measure(mpsgutz, [sp_half, sm_half]))/gutznorm

    mpo = MPO{Float64}(Lx, 2)
    egutz = real(measure(mpsgutz, mpo)/gutznorm[1,1])

    overlap_gutzheis = 1.4320821209129546e-5

    if compare
        eheis, mpsheis = mpsheisenberg(Lx)
        measheis_zz = measure(mpsheis, [sz_half, sz_half])
        measheis_pm = measure(mpsheis, [sp_half, sm_half])
        overlap_gutzheis = real(overlap(mpsheis, mpsgutz)[1,1,1]/sqrt(gutznorm))
        vneheis = entropy(entanglements!(mpsheis))

        filename = string("mps,measurements",
                          Dates.format(now(), "yymmdd:HH:MM:SS"),
                          ",", "heisenberg,chainLx=", Lx, ".h5")

        h5open(filename, "w") do file
            @write file measheis_zz
            @write file measheis_pm
            @write file eheis
            @write file vneheis
        end
    end

    vnegutz = entropy(entanglements!(mpsgutz))

    filename = string("mps,measurements",
                      Dates.format(now(), "yymmdd:HH:MM:SS"),
                      ",", "gutzwiller,chainLx=", Lx, ".h5")
    h5open(filename, "w") do file
        @write file measgutz_zz
        @write file measgutz_pm
        @write file egutz
        @write file vnegutz
        @write file overlap_gutzheis
    end
    (verbose) && println("Finished with everything!")
    nothing
end

function mpsheisenberg(Lx::Int64;
                       method::Symbol=:ED,
                       savemps::Bool=true)

    Hhs, indeces = Tmp.xxz_szblock(Lx)
    eheis, v = eigs(Hhs, nev=1, which=:SR)

    vheis = full(sparsevec(indeces, v[:,1], 2^Lx))

    mps = MPS(Lx, 2, vheis)
    if savemps
        #save(mpsgutz, string("mps", Dates.format(now(), "yymmdd:HH:MM:SS"),
        #                     ",", "heisenberg,chainLx=", "Lx", ".h5"))
    end

    return eheis, mps
end
