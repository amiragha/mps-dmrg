# The MPO class
struct MPO
    length :: Int64
    phys_dim :: Int64
    dims :: Vector{Int64}
    matrices :: Vector{Array{Complex128,4}}
end

####################
### constructors ###
####################

function MPO(Lx::Int64,
             d::Int64)
    # just make the Heisenberg Hamiltonian MPO
    Sp = Complex128[0.  1.; 0.  0.]
    Sm = Complex128[0.  0.; 1.  0.]
    Sz = Complex128[.5  0.; 0. -.5]
    I2 = eye(Complex128, 2)

    mat = zeros(Complex128, d, 5, d, 5)
    mat[:,1,:,1] = I2
    mat[:,2,:,1] = 0.5 * Sp
    mat[:,3,:,1] = 0.5 * Sm
    mat[:,4,:,1] = Sz

    mat[:,5,:,2] = Sm
    mat[:,5,:,3] = Sp
    mat[:,5,:,4] = Sz
    mat[:,5,:,5] = I2

    matrices = Array{Complex128,4}[]
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

    return MPO(Lx, d, dims, matrices)
end
