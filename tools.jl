function truncate(svector::Vector{Float64},
                  threshold::Float64=1.e-15)
    ## NOTE: S is assumed be all non-negative and sorted in
    ## descending order
    n=1
    sum_upto = svector[n]
    for n=2:length(svector)
        if svector[n] > (sum_upto * threshold)
            sum_upto += svector[n]
        else break
        end
    end
    return svector[1:n], n, sum_upto/sum(svector)
end

function is_strictly_ascending(v::Vector{T}) where {T<:Number}
    for i=1:length(v)-1
        if v[i] >= v[i+1]
            return false
        end
    end
    return true
end
