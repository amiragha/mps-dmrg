function truncate(svector::Vector{Float64},
                  max_dim::Int64=length(svector),
                  threshold::Float64=1.e-15)

    ## NOTE: S is assumed be all non-negative and sorted in
    ## descending order
    n = min(max_dim, sum(svector .> svector[1]*threshold))
    return svector[1:n], n, sum(svector[1:n])/sum(svector)
end

function is_strictly_ascending(v::Vector{T}) where {T<:Number}
    for i=1:length(v)-1
        if v[i] >= v[i+1]
            return false
        end
    end
    return true
end

"""
    entorpy(spectrum[, alpha])

calculate the entropy of a vector of numbers. If `alpha=1` (the
default) it calculates the usual Shannon (Von-Neumann) entropy and if
`alpha > 1` calculates the Renyi entorpy.

"""
function entropy(spectrum::Vector{T},
                 alpha::Int64=1) where {T<:Number}

    s = normalize(spectrum)
    if alpha == 1
        return - sum(s .* log2.(s))
    else
        return log2(sum(s.^alpha))/(1-alpha)
    end
end

function entropy(spectrums::Vector{Vector{T}},
                 alpha::Int64=1) where {T<:Number}
    result = T[]
    for i=1:length(spectrums)
        push!(result, entropy(spectrums[i], alpha))
    end
    return result
end

"""
    all_combinations(Lx, n)

calculate all possible combinations of `n` locations in `Lx`
slots. The returned value is a list (with size `binomial(Lx, n)`) of
vector of locations.

"""
function all_combinations(Lx::Int64,
                          n::Int64,
                          mode::Symbol)

    @assert n > 0 && n < Lx

    combinations = Vector{Int64}[]
    if mode == :half

        ### TODO: clean-up and redo this algorithm! is there a better,
        ### more efficient way?
        current = collect(1:n)
        push!(combinations, current)
        for i=1:binomial(Lx, n)-1

            # Find the first from end that is not in its final
            # position
            l=n
            for l=n:-1:1
                if current[l] != Lx-(n-l); break; end
            end

            nextc=ones(Int64,n)

            # increment it and replace others
            for i=1:l-1
                nextc[i] = current[i]
            end
            nextc[l] = current[l]+1
            for i=l+1:n
                nextc[i] = nextc[i-1]+1
            end
            push!(combinations, nextc)
            current = nextc
        end
    else
        error("only half is currently implemented!")
    end

    return combinations
end

"""
    weavekron(M1, M2, d1, d2, Lx)

the weave Kronecker tensor product of matrices `M1` and `M2`. It is
assumed that `M1` and `M2` are each the `Lx`th tensor powers of a
small vector space (site) of size `d1` and `d2` respectively. The
output is the tensor product of the two matrices such that

"""
function weavekron(M1::Matrix{T}, M2::Matrix{T},
                   d1::Int64, d2::Int64, Lx::Int64) where {T<:Number}

    @assert size(M1) == (d1 ^ Lx, d1 ^ Lx)
    @assert size(M2) == (d2 ^ Lx, d2 ^ Lx)

    d = d1 * d2
    dim_full = d ^ Lx
    result = Matrix{T}(dim_full, dim_full)

    for j=1:dim_full
        jc1, jc2 = index2config(j-1, d1, d2, Lx)
        l = config2index(jc1, d1) + 1
        n = config2index(jc2, d2) + 1
        for i=1:dim_full
            ic1, ic2 = index2config(i-1, d1, d2, Lx)
            k = config2index(ic1, d1) + 1
            m = config2index(ic2, d2) + 1
            result[i, j] = M1[k,l] * M2[m,n]
        end
    end

    return result
end

function weavekron(V1::Vector{T}, V2::Vector{T},
                   d1::Int64, d2::Int64, Lx::Int64) where {T<:Number}

    @assert length(V1) == d1 ^ Lx
    @assert length(V2) == d2 ^ Lx

    d = d1 * d2
    dim_full = d ^ Lx
    result = Vector{T}(dim_full)

    for i=1:dim_full
        ic1, ic2 = index2config(i-1, d1, d2, Lx)
        k = config2index(ic1, d1) + 1
        l = config2index(ic2, d2) + 1
        result[i] = V1[k] * V2[l]
    end

    return result
end

function config2index(config::Vector{Int64}, d::Int64)
    @assert all((-1 .< config) .& (config .< d))
    index = config[1]
    for site = 2:length(config)
        index = d * index + config[site]
    end
    return index
end

function index2config(index::Int64, d::Int64, Lx::Int64)
    @assert -1 < index && index < d^Lx
    config = Vector{Int64}(Lx)
    s = index
    r = 0
    for site = Lx:-1:1
        s, config[site] = divrem(s, d)
    end
    return config
end

function index2config(index::Int64, d1::Int64, d2::Int64, Lx::Int64)
    d = d1*d2
    @assert -1 < index && index < d^Lx
    config1 = Vector{Int64}(Lx)
    config2 = Vector{Int64}(Lx)
    s = index
    c = 0
    for site = Lx:-1:1
        s, r = divrem(s, d)
        config1[site], config2[site] = divrem(r, d2)
    end
    return config1, config2
end
