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
