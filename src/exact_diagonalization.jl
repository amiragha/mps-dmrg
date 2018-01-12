"""
    xxz_explicit(Lx[, boundary])

explicitely construct the full XXZ Hamiltonian using the
operator definition. Note that this is slow and uses a lot of memory.

"""
function xxz_explicit(Lx::Int64,
                      delta::Float64=1.0;
                      boundary::Symbol=:open)
    @assert Lx > 1

    Sz = sparse(Float64[.5 0; 0 -.5])
    Sp = sparse(Float64[ 0 1; 0  0])
    Sm = sparse(Float64[ 0 0; 1  0])
    I2 = sparse(eye(Float64, 2))

    heis_term = 0.5 * (kron(Sp, Sm) + kron(Sm, Sp)) + delta * kron(Sz,Sz)

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
            delta * kron(kron(Sz,eye(2^(Lx-2))), Sz)
        return Hmat
    else
        error("unrecognized boundary condition :", boundary)
    end
end

"""
    xxz_enumerate(Lx[, delta[, boundary]])

Generates a full XXZ hamiltonian for a system of size `Lx`. It
works because a single term of the Heisenberg operator can be written
in terms of the swap operator ``P_{ij}`` as follows
```math
h_{ij} = \\frac{1}{2} P_{ij} - \\frac{1}{4}I.
```

"""
function xxz_enumerate(Lx::Int64,
                       delta::Float64=1.0;
                       boundary::Symbol=:open)
    @assert Lx > 1

    if boundary == :open
        bond_range = 0:Lx-2
    elseif boundary == :periodic
        bond_range = 0:Lx-1
    else
        error("unrecognized boundary condition :", boundary)
    end

    I = Int32[]
    J = Int32[]
    V = Float64[]

    ## NOTE: In the code below for simplicity and dealing with bit
    ## shifts, the states and sites are starting from `0` not the
    ## usual Julia indexing that starts from `1`, be careful!

    ### TODO: think of a better hint or even reserve space
    sizehint!(I, Lx * 2^Lx )
    sizehint!(J, Lx * 2^Lx)
    sizehint!(V, Lx * 2^Lx)

    # For all states:
    for state=0:2^Lx-1
        # For all bond terms of the Hamiltonian
        for site=bond_range
            site_nn = (site + 1) % Lx
            # if parallel do:
            if xor((state >> site) & 1,  (state >> site_nn) & 1) == 0
                push!(I, state+1)
                push!(J, state+1)
                push!(V, delta * 0.25)
                # if not parallel do:
            else
                push!(I, state+1)
                push!(J, state+1)
                push!(V, - delta * 0.25)
                flipped_state = xor(state, 1 << site | 1 << site_nn)
                push!(I, state+1)
                push!(J, flipped_state+1)
                push!(V, 0.5)
            end
        end
    end

    return sparse(I, J, V, 2^Lx, 2^Lx, +)
end

"""
    XXZ(Lx[, boundary[, method]])

calculates a full XXZ Hamiltonian matrix in the Ising basis for
a system of length `Lx`. The possible boundary conditions are `:open`
(default) and `:periodic`. The possible methods are `:explicit` that
makes the full XXZ hamiltonian from operators and `:enumerate`
which goes through all states of the Hilbert space and implements the
action of Hamiltonian on them.

"""
function xxz(Lx::Int64,
             delta::Float64=1.0;
             boundary::Symbol=:open,
             method::Symbol=:enumerate)
    if method == :explicit
        return xxz_explicit(Lx, delta, boundary=boundary)
    elseif method == :enumerate
        return xxz_enumerate(Lx, delta, boundary=boundary)
    else
        error("unrecognized generation method :", method)
    end
end

"""
    xxz_szblock(Lx[, delta[, boundary[, sz_total]]])


Generates the XXZ Hamiltonian of size `Lx` in the given
`sz_total` block. This is possible because ``\left[ S^z_{\text{tot}},
H\right]`` and its eigenvalues can be used to label separate
magnetization blocks in the Hamiltonian matrix and each block can be
diagonalized separately to significantly reduce time complexity. Each
`sz_total` block consists of Ising vectors with exactly `M = (sz_total
+ Lx)/2` spins up, so the size of the block is ``\binom{L}{M}``. Using
Sterling formula the size of the largest block, ``M=L/2``, is
``√{π L/2}`` times smaller than the full Hilbert space, which
For typical sizes of ED calculation ,``L∼ 30``, is roughly about
``∼\!7`` times.

"""
function xxz_szblock(Lx::Int64,
                     delta::Float64=1.0;
                     boundary::Symbol=:open,
                     sz_total::Int64=(Lx % 2))
    @assert Lx > 1
    @assert (Lx-sz_total) % 2 == 0

    if boundary == :open
        bond_range = 0:Lx-2
    elseif boundary == :periodic
        bond_range = 0:Lx-1
    else
        error("unrecognized boundary condition :", boundary)
    end

    M = div(Lx + sz_total, 2)
    block_size = binomial(Lx, M)

    I = Int32[]
    J = Int32[]
    V = Float64[]

    ## NOTE: In the code below for simplicity and dealing with bit
    ## shifts, the states and sites are starting from `0` not the
    ## usual Julia indexing that starts from `1`, be careful!

    # store state with magnetization M
    szblock_states = Vector{Int64}(block_size)
    index = 0
    for state=0:2^Lx-1
        if count_ones(state) == M
            index += 1
            szblock_states[index] = state
        end
    end

    # For all states (of magnetization M):
    for state_index=1:block_size
        state = szblock_states[state_index]
        # For all bond terms of the Hamiltonian
        for site=bond_range
            site_nn = (site + 1) % Lx
            # if parallel do:
            if xor((state >> site) & 1, (state >> site_nn) & 1) == 0
                push!(I, state_index)
                push!(J, state_index)
                push!(V, delta * 0.25)
                # if not parallel do:
            else
                push!(I, state_index)
                push!(J, state_index)
                push!(V, - delta * 0.25)

                flipped_state = xor(state, 1 << site | 1 << site_nn)
                flipped_state_index = searchsortedfirst(szblock_states, flipped_state)
                push!(I, state_index)
                push!(J, flipped_state_index)
                push!(V, 0.5)
            end
        end
    end

    return sparse(I, J, V, block_size, block_size, +), (szblock_states .+ 1)
end

function groundstate_xxz_szblock(Lx::Int64,
                                 delta::Float64=1.0;
                                 boundary::Symbol=:open,
                                 sz_total::Int64=(Lx % 2))
    Hhs, indeces = xxz_szblock(Lx, delta, boundary=boundary, sz_total=sz_total)
    eheis, v = eigs(Hhs, nev=1, which=:SR)

    return eheis[1], v[:, 1]
end
