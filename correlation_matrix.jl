"""
    correlation_matrix(Lx, n_occupied[, t[, mu[, boundary]]])

calculates the two-body correlation matrix lambda using full
diagonalization "\lambda = \langle a^{dagger}_ia_j \rangle". Support
for `:open` default and `:periodic` boundary conditions.

"""
function correlation_matrix(Lx::Int64,
                            n_occupied::Int64,
                            t::Float64=1.0,
                            mu::Float64=0.0,
                            boundary::Symbol=:open)

    @assert 0 < n_occupied && n_occupied <= Lx

    if boundary == :open
        Hmatrix = SymTridiagonal(mu .* ones(Lx), t .* ones(Lx-1))
    elseif boundary == :periodic
        Hmatrix = Matrix(SymTridiagonal(mu .* ones(Lx), t .* ones(Lx-1)))
        Hmatrix[Lx, 1] = t
        Hmatrix[1, Lx] = t
        Hmatrix = Hermitian(Hmatrix)
    else
        error("unrecognized boundary condition : ", boundary)
    end

    ## NOTE: the correlation matrix \lambda is made from its diagonal
    ## form \Gamma, which has n_occupied 1s--corresponding to the
    ## lowest eigenvalues--and the rest of the diagonal are 0s. We
    ## have $\Lambda = U^{*} \Gamma U^{T}$
    vecs = eigfact(Hmatrix, 1:n_occupied)[:vectors]

    return conj(vecs) * transpose(vecs)
end
