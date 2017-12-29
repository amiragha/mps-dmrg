### TODO: extend to a more general model based system.
function hamiltonian_matrix(Lx         ::Int64,
                            t          ::T=1.0,
                            mu         ::T=0.0,
                            boundary   ::Symbol=:open) where {T<:Union{Float64,Complex128}}

    hmatrix = diagm(mu .* ones(T, Lx)) +
        diagm(t .* ones(T, Lx-1), 1) +
        diagm(conj(t) .* ones(T, Lx-1), -1)

    if boundary == :open
        return hmatrix
    elseif boundary == :periodic
        hmatrix[1, Lx] = t
        hmatrix[Lx, 1] = conj(t)
        return hmatrix
    else
        error("unrecognized boundary condition : ", boundary)
    end
end

"""
    correlation_matrix(Lx, n_occupied[, t[, mu[, boundary]]])

calculates the two-body correlation matrix lambda ``Λ = ⟨a^†_i a_j⟩``
by full diagonalization of given Hamiltonian matrix.

Here, the correlation matrix Λ is made from its diagonal form Γ, which
has n_occupied 1s--corresponding to the lowest eigenvalues--and the
rest of the diagonal are 0s. We have ``Λ = U^{*} Γ U^{T}``

"""
function correlation_matrix(hmatrix::Matrix{Float64},
                            n_occupied ::Int64)

    Lx = size(hmatrix)[1]
    @assert 0 < n_occupied && n_occupied <= Lx

    vecs = eigfact(Symmetric(hmatrix), Lx-n_occupied+1:Lx)[:vectors]
    return vecs * transpose(vecs)
end

function correlation_matrix(hmatrix::Matrix{Complex128},
                            n_occupied ::Int64)

    Lx = size(hmatrix)[1]
    @assert 0 < n_occupied && n_occupied <= size(hmatrix)[1]

    vecs = eigfact(Hermitian(hmatrix), Lx-n_occupied+1:Lx)[:vectors]
    return conj(vecs) * transpose(vecs)
end
