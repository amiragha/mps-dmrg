using Base.Test
import Tmp

sz_half = Tmp.sz_half
sp_half = Tmp.sp_half
sm_half = Tmp.sm_half

sz_one = Tmp.sz_one
sp_one = Tmp.sp_one
sm_one = Tmp.sm_one

srand(1911)

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

    @testset "ketstate to MPS to ketstate" begin
        @test norm(ketstate)^2 ≈ Tmp.norm(randmps)[1,1]
        @test ketstate ≈ Tmp.mps2ketstate(randmps)
    end

    @testset "overlap of two MPS" begin
        @test Tmp.overlap(mps, mps) ≈ Tmp.norm(mps)
        @test Tmp.overlap(mps, randmps) ≈ conj.(Tmp.overlap(randmps, mps))
    end

    @testset "operator measurements" begin
        @test Tmp.measure(mps, sz_half, collect(1:Lx)) ≈ zeros(Lx) atol=1.e-12

        mpo = Tmp.MPO{Float64}(Lx, 2)
        @test Tmp.measure(mps, mpo) ≈ eheis
    end

    @testset "MPS center manipulations" begin
        ### TODO: write more tests for here!
        Tmp.move_center!(randmps, Lx-1)
        Tmp.move_center!(randmps, Lx-2)
        Tmp.move_center!(randmps, 1)
        Tmp.move_center!(randmps, Lx)
        @test Tmp.mps2ketstate(randmps) ≈ ketstate

        Tmp.canonicalize_at!(randmps.matrices, Lx)
        @test Tmp.mps2ketstate(randmps) ≈ ketstate
    end

    @testset "direct product: spin-1/2 and spin-1" begin

        Lx = 4
        d1, d2 = 2, 3

        ket1 = complex.(randn(d1^Lx), randn(d1^Lx))
        ket2 = complex.(randn(d2^Lx), randn(d2^Lx))

        mps1 = Tmp.MPS(Lx, d1, ket1)
        mps2 = Tmp.MPS(Lx, d2, ket2)

        ## NOTE: here the "prod_mps" is the direct prodcut of the two
        ## random mps while "mps_prod" is the mps made from the direct
        ## product of the two random kets

        prod_mps = Tmp.directproduct(mps1, mps2)

        @test prod_mps.phys_dim == d1 * d2
        @test Tmp.dims_are_consistent(prod_mps)

        sz_prod = kron(sz_half, sz_one)
        sp_prod = kron(sp_half, sp_one)
        sm_prod = kron(sm_half, sm_one)

        flatten(x) = collect(Base.Iterators.flatten(x))

        measures_sz1 = flatten(Tmp.measure(mps1, sz_half, collect(1:Lx)))
        measures_sz2 = flatten(Tmp.measure(mps2, sz_one, collect(1:Lx)))
        measures_sz3 = flatten(Tmp.measure(prod_mps, sz_prod, collect(1:Lx)))

        measures_szsz1 = flatten(Tmp.measure(mps1, [sz_half,sz_half]))
        measures_szsz2 = flatten(Tmp.measure(mps2, [sz_one,sz_one]))
        measures_szsz3 = flatten(Tmp.measure(prod_mps, [sz_prod,sz_prod]))

        measures_spsm1 = flatten(Tmp.measure(mps1, [sp_half,sm_half]))
        measures_spsm2 = flatten(Tmp.measure(mps2, [sp_one,sm_one]))
        measures_spsm3 = flatten(Tmp.measure(prod_mps, [sp_prod,sm_prod]))

        @test measures_sz3 ≈ measures_sz1 .* measures_sz2
        @test measures_szsz3 ≈ measures_szsz1 .* measures_szsz2
        @test measures_spsm3 ≈ measures_spsm1 .* measures_spsm2

        # make an MPS explicitely from the direct product of the kets
        ket_prod = Tmp.weavekron(ket1, ket2, d1, d2, Lx)
        mps_prod = Tmp.MPS(Lx, d1*d2, ket_prod, svtruncate=false)

        @test mps_prod.dims == prod_mps.dims
        @test mps_prod.center == prod_mps.center

        @test flatten(Tmp.measure(mps_prod, sz_prod, collect(1:Lx))) ≈ measures_sz3
        @test flatten(Tmp.measure(mps_prod, [sz_prod, sz_prod])) ≈ measures_szsz3
        @test flatten(Tmp.measure(mps_prod, [sp_prod, sm_prod])) ≈ measures_spsm3
    end
end

@testset "apply twosite" begin

    Lx, l = 8, 3

    H = Tmp.xxz(Lx)
    eheis, vheis = eigs(H, nev=1, which=:SR)
    mps = Tmp.MPS(Lx, 2, vheis[:,1])

    mpscopy = deepcopy(mps)
    Tmp.move_center!(mps, l)
    Tmp.apply_twosite_operator!(mps, l, kron(sz_half,sz_half))

    @test Tmp.dims_are_consistent(mps)
    @test Tmp.overlap(mpscopy, mps) ≈ Tmp.measure(mpscopy, [sz_half,sz_half], [l,l+1])

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

    @testset "index <-> config" begin
        c1 = [0, 1, 1, 0]
        c2 = [1, 2, 0, 1]
        @test Tmp.index2config(46, 3, 4) == c2
        @test Tmp.config2index(c2, 3) == 46
        @test (c1, c2) == Tmp.index2config(Tmp.config2index(3 * c1 .+ c2, 2*3),
                                           2, 3, 4)
    end

    @testset "weavekron" begin
        v11 = rand(2)
        v12 = rand(2)
        v21 = rand(3)
        v22 = rand(3)
        @test kron(kron(v11,v21),kron(v12,v22)) ≈
            Tmp.weavekron(kron(v11,v12), kron(v21,v22),2, 3, 2)
        m11 = rand(2, 2)
        m12 = rand(2, 2)
        m21 = rand(3, 3)
        m22 = rand(3, 3)
        @test kron(kron(m11,m21),kron(m12,m22)) ≈
            Tmp.weavekron(kron(m11,m12), kron(m21,m22),2, 3, 2)
    end

end

########################################
### tests for exact diagonalizations ###
########################################

@testset "ED stuff" begin

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

##############################
### Fishman approach tests ###
##############################

@testset "Fishman" begin

    Lx = 8

    # XY spin model
    H = Tmp.xxz(Lx, 0.0)
    exy, vxy = eigs(H, nev=1, which=:SR)
    mpsxy = Tmp.MPS(Lx, 2, vxy[:,1])

    # free fermions
    corr_matrix = Tmp.correlation_matrix(Tmp.hamiltonian_matrix(Lx), div(Lx,2))
    gmps = Tmp.makeGMPS!(corr_matrix, 1.e-8)
    mpsff = Tmp.generateMPS(gmps, 2^div(Lx,2))

    @testset "general" begin

        evals = eigvals(corr_matrix)
        threshold = 1.e-15
        @test all((0 - threshold .<= evals) .& (evals .<= 1 + threshold))

        @test Tmp.dims_are_consistent(mpsff)
    end

    @testset "compare XY with Free Fermions" begin
        @test Tmp.measure(mpsff, [sz_half,sz_half]) ≈ Tmp.measure(mpsxy, [sz_half,sz_half])
        @test Tmp.measure(mpsff, [sp_half,sm_half]) ≈ Tmp.measure(mpsxy, [sp_half,sm_half])

        xympo = Tmp.MPO{Float64}(Lx, 2, 0.0)
        @test Tmp.measure(mpsxy, xympo) ≈ Tmp.measure(mpsff, xympo)
    end
end
