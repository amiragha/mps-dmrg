using Base.Test
import Tmp

#srand(1911)

###########################
### tests for MPS stuff ###
###########################

@testset "MPS stuff" begin

    d, Lx = 2, 8

    # A uniformly chosen random state as MPS
    ketstate = complex.(randn(d^Lx), randn(d^Lx))
    randmps = Tmp.MPS(Lx, d, ketstate)

    # The Bethe chain ground state as MPS
    H = Tmp.xxz(Lx)
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

@testset "apply twosite" begin

    Lx, l = 8, 3
    sz = Float64[0.5 0; 0 -0.5]

    H = Tmp.xxz(Lx)
    eheis, vheis = eigs(H, nev=1, which=:SR)
    mps = Tmp.MPS(Lx, 2, vheis[:,1])

    mpscopy = deepcopy(mps)
    Tmp.move_center!(mps, l)
    Tmp.apply_twosite_operator!(mps, l, kron(sz,sz))

    @test Tmp.dims_are_consistent(mps)
    @test Tmp.overlap(mpscopy, mps) ≈ Tmp.measure(mpscopy, [sz,sz], [l,l+1])

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

########################################
### tests for exact diagonalizations ###
########################################

@testset "exact diagonalization" begin

    @testset "full xxz generation methods" begin
        @test Tmp.xxz(2, 1.0, :open, :explicit)     ≈ Tmp.xxz(2, 1.0, :open, :enumerate)
        @test Tmp.xxz(3, 1.0, :open, :explicit)     ≈ Tmp.xxz(3, 1.0, :open, :enumerate)
        @test Tmp.xxz(4, 1.0, :open, :explicit)     ≈ Tmp.xxz(4, 1.0, :open, :enumerate)
        @test Tmp.xxz(2, 1.0, :periodic, :explicit) ≈ Tmp.xxz(2, 1.0, :periodic, :enumerate)
        @test Tmp.xxz(3, 1.0, :periodic, :explicit) ≈ Tmp.xxz(3, 1.0, :periodic, :enumerate)
        @test Tmp.xxz(4, 1.0, :periodic, :explicit) ≈ Tmp.xxz(4, 1.0, :periodic, :enumerate)
    end

    @testset "szblock vs full odd" begin

        Lx=3
        efull, vfull = eigs(Tmp.xxz(Lx), nev=1, which=:SR)

        Hsz_p, indeces_p = Tmp.xxz_szblock(Lx, 1.0, :open, 1)
        esz_p, vsz_p = eigs(Hsz_p, nev=1, which=:SR)
        Hsz_m, indeces_m = Tmp.xxz_szblock(Lx, 1.0, :open, -1)
        esz_m, vsz_m = eigs(Hsz_m, nev=1, which=:SR)

        @test efull ≈ esz_m ≈ esz_p

        v_p = full(sparsevec(indeces_p, vsz_p[:,1], 2^Lx))
        v_m = full(sparsevec(indeces_m, vsz_m[:,1], 2^Lx))

        @test sqrt(dot(vfull[:,1], v_p)^2 + dot(vfull[:,1],v_m)^2) ≈ 1.0
    end

    @testset "szblock vs full even" begin
        Lx=4
        efull, vfull = eigs(Tmp.xxz(Lx), nev=1, which=:SR)

        Hsz, indeces = Tmp.xxz_szblock(Lx)
        esz, vsz = eigs(Hsz, nev=1, which=:SR)

        @test efull ≈ esz

        v = full(sparsevec(indeces, vsz[:,1], 2^Lx))

        @test abs(dot(vfull[:,1], v)) ≈ 1.0

    end
end
