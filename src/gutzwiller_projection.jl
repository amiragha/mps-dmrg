function gutzwiller_project(mps::MPS{T}) where {T<:Union{Float64,Complex128}}

    @assert mps.phys_dim == 4
    Lx = mps.length

    matrices = Array{T, 3}[]
    for site=1:Lx
        push!(matrices, mps.matrices[site][:,:,2:3])
    end

    canonicalize_at!(matrices, div(Lx+1,2), svtruncate=true)

    dims = Int64[]
    push!(dims,size(matrices[1])[1])
    for n=1:mps.length
        push!(dims, size(matrices[n])[2])
    end

    return MPS{T}(Lx, 2, dims, matrices, Lx)
end
