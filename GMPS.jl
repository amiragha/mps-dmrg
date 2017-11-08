# The GMPS class containing gates. (Fishman and White)
mutable struct GMPS
    gates
    size
    # other stuff
end

function makeGMPS!(corr_matrix, threshold=1.e-8)

    Lx = size(corr_matrix)[1]

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
        println(vals)

        if block_end > site

            block_size = block_end - site + 1
            block_range = site:block_end

            # find the closest eigenvalue to occupied/unoccupied and
            # the corresponding eigenvector, $v$.
            v_index = indmax(abs.(vals - 0.5))
            v = eigvecs(corr_block)[:,v_index]

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
                ## QQQ? acting on the vectors on left! Why is this a
                ## better choice?
                v = transpose(transpose(v) * ugate)

                ### UNCOMMENT FOR TEST ###
                ugate_block = ugate_block * ugate

            end
            ### UNCOMMENT FOR TEST ###
            ugate_extended = eye(Lx, Lx)
            ugate_extended[block_range, block_range] = ugate_block
            corr_matrix= ugate_extended' * corr_matrix * ugate_extended
        end
        ### UNCOMMENT FOR TEST ###
        display(corr_matrix)
    end
    diag(corr_matrix.')
end
