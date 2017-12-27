"""
    heisenberg(Lx[, boundary[, method]])

calculates a full heisenberg Hamiltonian matrix in the Ising basis for
a system of length `Lx`. The possible boundary conditions are `:open`
(default) and `:periodic`. The possible methods are `:explicit` that
makes the full heisenberg hamiltonian from operators and `:enumerate`
which goes through all states of the Hilbert space and implements the
action of Hamiltonian on them.

"""
function heisenberg(Lx::Int64,
                    boundary::Symbol=:open,
                    method::Symbol=:explicit)
    if method == :explicit
        return heisenberg_explicit(Lx, boundary)
    elseif method == :enumerate
        return heisenberg_enumerate(Lx, boundary)
    else
        error("unrecognized generation method :", method)
    end
end

function heisenberg_explicit(Lx::Int64,
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
            0.5 * kron(kron(Sp,eye(2^(Lx-2))), Sm) +
            0.5 * kron(kron(Sm,eye(2^(Lx-2))), Sp) +
            kron(kron(Sz,eye(2^(Lx-2))), Sz)
        return Hmat
    else
        error("unrecognized boundary condition :", boundary)
    end
end

"""
    heisenberg_enumerate(Lx[, boundary])

Generates a full heisenberg hamiltonian for a system of size `Lx`. It
works because a single term of the Heisenberg operator can be written
in terms of the swap operator ``P_{ij}`` as follows
```math
h_{ij} = \\frac{1}{2} P_{ij} - \\frac{1}{4}I.
```

"""
function heisenberg_enumerate(Lx::Int64,
                              boundary::Symbol=:open)
    @assert Lx > 1

    Hmat = spzeros(2^Lx,2^Lx)

    if boundary == :open
        bond_range = 0:Lx-2
    elseif boundary == :periodic
        bond_range = 0:Lx-1
    else
        error("unrecognized boundary condition :", boundary)
    end

    ## NOTE: In the code below for simplicity and dealing with bit
    ## shifts, the states and sites are starting from `0` not the
    ## usual Julia indexing that starts from `1`, be careful!

    # For all states:
    for state=0:2^Lx-1
        # For all bond terms of the Hamiltonian
        for site=bond_range
            site_nn = (site + 1) % Lx
            # if parallel do:
            if xor((state >> site) & 1,  (state >> site_nn) & 1) == 0
                Hmat[state+1, state+1] += 0.25
                # if not parallel do:
            else
                Hmat[state+1, state+1] -= 0.25
                flipped_state = xor(state, 1 << site | 1 << site_nn)
                Hmat[state+1, flipped_state+1] += 0.5
            end
        end
    end
    return Hmat
end
