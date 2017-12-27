using Base.Test
import Tmp

#srand(1911)

@testset "stuff" begin

    d, Lx = 2, 8

    # A uniformly chosen random state as MPS
    ketstate = complex.(randn(d^Lx), randn(d^Lx))
    randmps = Tmp.MPS(Lx, d, ketstate)

    # The Bethe chain ground state as MPS
    H = Tmp.heisenberg(Lx)
    eheis, vheis = eigs(H, nev=1, which=:SR)
    mps = Tmp.MPS(Lx, 2, vheis[:,1])

    @testset "ketstate" begin
        @test norm(ketstate)^2 ≈ Tmp.norm(randmps)[1,1]
        @test ketstate ≈ Tmp.mps2ketstate(randmps)
    end

    @testset "overlap" begin
        @test Tmp.overlap(mps, mps) ≈ Tmp.norm(mps)
        @test Tmp.overlap(mps, randmps) ≈ conj.(Tmp.overlap(randmps, mps))
    end

    @testset "measure" begin
        sz = Float64[0.5 0; 0 -0.5]
        @test Tmp.measure(mps, sz, collect(1:Lx)) ≈ zeros(Lx) atol=1.e-12

        mpo = Tmp.MPO{Float64}(Lx, 2)
        @test Tmp.measure(mps, mpo) ≈ eheis
    end

    @testset "center" begin
        # TODO:
        # test canonicalize to right and left
        # and center_at (how?)
        Tmp.move_center!(randmps, Lx-1)
        Tmp.move_center!(randmps, Lx-2)
        Tmp.move_center!(randmps, 1)
        Tmp.move_center!(mps, Lx)
        @test Tmp.mps2ketstate(randmps) ≈ ketstate
    end
end

#############################################
### tests for auxilary functions in tools ###
#############################################

@testset "tools" begin
    @testset "locations" begin
        @test Tmp.all_combinations(4, 1, :half) == [[1],[2],[3],[4]]
        @test Tmp.all_combinations(4, 2, :half) == [[1,2],[1,3],[1,4],
                                                   [2,3],[2,4],[3,4]]
    end
end
