# spin-1/2
const sz_half = Float64[.5 0.;0. -.5]
const sp_half = Float64[.0 1.;0. 0.]
const sm_half = transpose(sp_half)

# spin-1
const sz_one = Float64[1. 0. 0.; 0. 0. 0.; 0. 0. -1.]
const sp_one = Float64[0. 1. 0.; 0. 0. 1.; 0. 0.  0.] * sqrt(2)
const sm_one = transpose(sp_one)
