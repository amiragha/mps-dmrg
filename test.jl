using Base.Test
import Tmp

#srand(1911)

d, Lx = rand(2:4, 2)
ketstate = rand(Complex128, d^Lx)
mps = Tmp.MPS(Lx, d, ketstate)

@test ketstate ≈ Tmp.mps2ketstate(mps)
@test norm(ketstate)^2 ≈ Tmp.mps_norm(mps)[1,1]
@test ketstate ≈ Tmp.mps2ketstate(mps)
