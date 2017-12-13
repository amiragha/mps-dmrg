# The MPO class
struct MPO{T<:Number}
    length :: Int64
    phys_dim :: Int64
    dims :: Vector{Int64}
    matrices :: Vector{Array{T,4}}
end

####################
### constructors ###
####################

function MPO{T}(Lx::Int64,
                d::Int64) where {T<:Number}
    # just make the Heisenberg Hamiltonian MPO
    Sp = T[0.  1.; 0.  0.]
    Sm = T[0.  0.; 1.  0.]
    Sz = T[.5  0.; 0. -.5]
    I2 = eye(T, 2)

    mat = zeros(T, d, 5, d, 5)
    mat[:,1,:,1] = I2
    mat[:,2,:,1] = 0.5 * Sp
    mat[:,3,:,1] = 0.5 * Sm
    mat[:,4,:,1] = Sz

    mat[:,5,:,2] = Sm
    mat[:,5,:,3] = Sp
    mat[:,5,:,4] = Sz
    mat[:,5,:,5] = I2

    matrices = Array{T,4}[]
    dims = zeros(Int64, Lx+1)

    dims[1] = 1
    push!(matrices, mat[:,5:5,:,:])
    dims[2] = 5
    for site=2:Lx-1
        push!(matrices, mat)
        dims[site+1] = 5
    end
    push!(matrices, mat[:,:,:,1:1])
    dims[Lx+1] = 1

    return MPO{T}(Lx, d, dims, matrices)
end
