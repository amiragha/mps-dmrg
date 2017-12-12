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
