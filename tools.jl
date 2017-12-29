function truncate(svector::Vector{Float64},
                  max_dim::Int64=length(svector),
                  threshold::Float64=1.e-14)

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
default) it calculates the usual Shannon (Von-Neumann) entorpy and if
`alpha > 1` calculates the Renyi entorpy.

"""
function entropy(spectrum::Vector{T},
                 alpha::Int64=1) where {T<:Number}
    if alpha == 1
        return - sum(spectrum .* log2.(spectrum))
    else
        return log2(sum(spectrum.^alpha))/(1-alpha)
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
