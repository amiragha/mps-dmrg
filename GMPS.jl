# The GMPS class containing gates. (Fishman and White)
struct NNGate
    site::Int
    theta::Float64
end

## QQQ? does GMPS have to be mutable?
mutable struct GMPS
    Lx::Int
    configuration::Vector{Int}
    gates::Vector{NNGate}
end

function makeGMPS!(corr_matrix, threshold=1.e-8, verbose=true)

    Lx = size(corr_matrix)[1]

    configuration = Vector{Int}(Lx)
    gates = NNGate[]

    for site=1:Lx

        block_end = site
        evalue = corr_matrix[site, site]
        corr_block = [evalue]
        vals = corr_block

        # abs function is there just in case but should be removed
        delta = min(abs(evalue), abs(1-evalue))

        while (delta > threshold && block_end < Lx)
            block_end += 1
            ## TODO: better slicing method (probably arrayviews or sub)
            corr_block = corr_matrix[site:block_end, site:block_end]

            vals = eigvals(corr_block)

            ## The abs function is there just in case! but should be removed!
            delta = min(abs(minimum(vals)), abs(1-maximum(vals)))
        end

        # println(site, " B = ", block_end-site+1, ", delta = ", delta)
        # println(vals)

        if block_end > site

            block_size = block_end - site + 1
            block_range = site:block_end

            # find the closest eigenvalue to occupied/unoccupied and
            # the corresponding eigenvector, $v$.
            v_index = indmax(abs.(vals - 0.5))
            v = eigvecs(corr_block)[:,v_index]

            evalue = vals[v_index]
            # find the set of block_size-1 unitary gate that
            # diagonalize the correlation block
            ## TODO: how to organize and store the gates?
            ugate_block = eye(block_size, block_size)
            for pivot = block_size-1:-1:1
                ## QQQ? Is it safe to use the atan fn here?
                theta = atan(v[pivot + 1]/ v[pivot])
                ugate = eye(block_size, block_size)
                ugate[pivot:pivot+1, pivot:pivot+1] =
                    [
                        cos(theta) -sin(theta);
                        sin(theta)  cos(theta)
                    ]
                push!(gates, NNGate(site+pivot-1, theta))
                ## QQQ? acting on the vectors on left! Why is this a
                ## better choice?
                v = transpose(transpose(v) * ugate)

                ugate_block = ugate_block * ugate
            end

            ## QQQ? There should be a faster way! Do I even need to
            ## rotate the whole matrix? Is it faster to rotate per
            ## block or per NN ugate?
            ugate_extended = eye(Lx, Lx)
            ugate_extended[block_range, block_range] = ugate_block
            corr_matrix= ugate_extended' * corr_matrix * ugate_extended
        end
        (verbose) && display(corr_matrix)
        configuration[site] = round(evalue)
    end
    (verbose) && display(diag(corr_matrix.'))
    (verbose) && display(configuration)
    return GMPS(Lx, configuration, gates)
end
