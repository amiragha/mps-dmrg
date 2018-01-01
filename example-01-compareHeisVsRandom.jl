using Plots
using LaTeXStrings
font = Plots.font("Helvetica", 12)
pyplot(guidefont=font, xtickfont=font, ytickfont=font, legendfont=font)

import Tmp
sz_half = Tmp.sz_half
sp_half = Tmp.sp_half
sm_half = Tmp.sm_half

sz_one = Tmp.sz_one
sp_one = Tmp.sp_one
sm_one = Tmp.sm_one


## NOTE: don't go over Lx=20
Lx = 16

# The Bethe chain ground state
H, indeces = Tmp.xxz_szblock(Lx)
eheis, v = eigs(H, nev=1, which=:SR)
vheis = full(sparsevec(indeces, v[:,1], 2^Lx))

mps = Tmp.MPS(Lx, 2, vheis);

# random state
randmps = Tmp.MPS(Lx, 2, normalize(complex.(randn(2^Lx), randn(2^Lx))))

# calculate the overlap
overlap = Tmp.overlap(mps, randmps)[1]
@show abs(overlap), angle(overlap)

correlations = Tmp.measure(mps, [sz_half,sz_half], :half)
p1 = plot(correlations[1:Lx-1], label="Heis GS", title="correlations")

correlations_random = Tmp.measure(randmps, [sz_half,sz_half], :half)
p2 = plot(real(correlations_random[1:Lx-1]), label="random", title="correlations")
plot!(p2, imag(correlations_random[1:Lx-1]), label="imag")

display(
    plot(p1,p2, layout=(1,2), size=(800,200), show=true)
)

vnes_mps = Tmp.entropy(Tmp.entanglements!(mps))
vnes_randmps = Tmp.entropy(Tmp.entanglements!(randmps))
p3 = plot(vnes_mps, label="Heis GS", title="Entaglement entropies")
p4 = plot(vnes_randmps, label="random", title="Entanglement entropies")
plot!(p4, [1, Lx/2, Lx-1], [1, Lx/2, 1], label="linear")

display(
    plot(p3,p4, layout=(1,2), size=(800,200), show=true)
)
