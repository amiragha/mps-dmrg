using Base.Test
import Tmp

#srand(1911)

# test ket states and MPS
d, Lx = rand(2:4, 2)
ketstate = rand(Complex128, d^Lx)
mps = Tmp.MPS(Lx, d, ketstate)
@test ketstate ≈ Tmp.mps2ketstate(mps)
@test norm(ketstate)^2 ≈ Tmp.norm(mps)[1,1]
@test ketstate ≈ Tmp.mps2ketstate(mps)

# test MPO measure
H = Tmp.heisenberg(4)
e, v = eigs(H, nev=1, which=:SR)
mps = Tmp.MPS(4, 2, v[:,1])
mpo = Tmp.MPO{Float64}(4, 2)
@test Tmp.measure(mps, mpo) ≈ e
