function correlation_matrix(Lx, n_occupied, t=1.0, mu=0.0)

    @assert 0 < n_occupied && n_occupied <= Lx

    # create Hamiltonian matrix for hopping fermions
    Hmatrix = SymTridiagonal(mu .* ones(Lx), t .* ones(Lx-1))

    ## NOTE: the correlation matrix \lambda is made from its diagonal
    ## form \Gamma, which has n_occupied 1s--corresponding to the
    ## lowest eigenvalues--and the rest of the diagonal are 0s. We
    ## have $\Lambda = U^{*} \Gamma U^{T}$
    vecs = eigfact(Hmatrix, 1:n_occupied)[:vectors]

    return conj(vecs) * transpose(vecs)

end
