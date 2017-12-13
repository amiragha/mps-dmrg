# Define operators as sparse for less memory usage

"""
    heisenberg(Lx[, boundary])

calculates a full heisenberg Hamiltonian matrix in the Ising basis for
a system of length `Lx`. The possible boundary conditions are `:open`
(default) and `:periodic`.

"""
function heisenberg(Lx::Int64,
                    boundary::Symbol=:open)
    @assert Lx > 1

    Sz = sparse(Float64[.5 0; 0 -.5])
    Sp = sparse(Float64[ 0 1; 0  0])
    Sm = sparse(Float64[ 0 0; 1  0])
    I2 = sparse(eye(Float64, 2))

    heis_term = 0.5 * (kron(Sp, Sm) + kron(Sm, Sp)) + kron(Sz,Sz)

    Hmat = heis_term
    if Lx > 2
    # recursive generation of open chain
        for i=3:Lx
            Hmat = kron(Hmat, I2) + kron(eye(2^(i-2)), heis_term)
        end
    end

    if boundary == :open
        return Hmat
    elseif boundary == :periodic
        Hmat = Hmat +
            kron(kron(Sp,eye(2^(Lx-2))), Sm) +
            kron(kron(Sm,eye(2^(Lx-2))), Sp) +
            kron(kron(Sz,eye(2^(Lx-2))), Sz)
        return Hmat
    else
        error("unrecognized boundary condition", boundary)
    end

end
