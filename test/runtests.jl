using PairCorrelationFunction
using Test, Plots

@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(isequal(x.$n, y.$n)), (a,b)->:($a && $b), fieldnames(x))
    else
        :(x == y)
    end
end

@testset "PairCorrelationFunction.jl" begin
    # 2D testing
    xlims = (-450.0, 450.0)
    ylims = (-450.0, 450.0)
    radii = 0:20.0:1280.0
    constants = Constants(xlims, ylims, radii)

    centers_matrix = [
        20.0 30.0;
        -200.0 100.0;
        300.0 -400.0;
        0.0 0.0;
    ]

    targets_matrix = [
        10.0 20.0;
        -150.0 50.0;
        250.0 -350.0;
        5.0 5.0;
        100.0 200.0;
        -300.0 400.0;
        400.0 -400.0
    ]

    g = pcf(centers_matrix, targets_matrix, constants)
    @test all(g.g .>= 0.0 .|| isnan.(g.g))

    Base.show(stdout, MIME"text/plain"(), g)

    g_self = pcf(centers_matrix, constants)
    @test all(g_self.g .>= 0.0 .|| isnan.(g_self.g))

    dr = 20.0
    constants_v2 = Constants(xlims, ylims, dr)
    @test constants ≂ constants_v2
    g_v2 = pcf(centers_matrix, targets_matrix, constants_v2)
    @test g ≂ g_v2
    @test all(g_v2.g .>= 0.0 .|| isnan.(g_v2.g))

    pcfplot(g)
    pcfplot(g.g)
    pcfplot(constants.radii, g_self)
    pcfplot([g, g_self])
    pcfplot(constants.radii[1:end-1], [g, g_self])
    
    G = hcat(g, g_self)
    Base.show(stdout, MIME"text/plain"(), G)
    pcfplot(G)
    pcfplot(G.g)
    pcfplot([G, G], clims=(0.0, 2.0))
    pcfplot([G.g, G.g])
    pcfplot([100.0, 200.0], G)
    pcfplot([100.0, 200.0], constants.radii, G)
    @test_throws ArgumentError pcfplot("not_correct_input")

    #3D testing
    zlims = (-450.0, 450.0)
    radii = 0:20.0:1560.0
    constants_3d = Constants(xlims, ylims, zlims, radii)
    centers_matrix_3d = [
        20.0 30.0 40.0;
        -200.0 100.0 -50.0;
        300.0 -400.0 200.0;
        0.0 0.0 0.0;
    ]
    targets_matrix_3d = [
        10.0 20.0 30.0;
        -150.0 50.0 -25.0;
        250.0 -350.0 100.0;
        5.0 5.0 5.0;
        100.0 200.0 -100.0;
        -300.0 400.0 150.0;
        400.0 -400.0 -200.0
    ]

    g3 = pcf(centers_matrix_3d, targets_matrix_3d, constants_3d)
    @test all(g3.g .>= 0.0 .|| isnan.(g3.g))

    constants_3d_v2 = Constants(xlims, ylims, zlims, dr)
    @test constants_3d ≂ constants_3d_v2
    @test g3 ≂ pcf(centers_matrix_3d, targets_matrix_3d, constants_3d_v2)
    
    #! test that the error is thrown when the centers are outside the limits
    centers_matrix_3d = [
        -500.0 500.0 500.0
    ]

    @test_throws AssertionError pcf(centers_matrix_3d, targets_matrix_3d, constants_3d)

    #! do PCF that we can compute by hand to verify
    xlims = (0.0, 1.0)
    ylims = (0.0, 1.0)
    radii = 0:5.0:10.0
    constants = Constants(xlims, ylims, radii)
    centers_matrix = [
        0.5 0.5
    ]
    targets_matrix = [
        0.25 0.25
    ]
    g = pcf(centers_matrix, targets_matrix, constants)
    @test all(g.g .>= 0.0 .|| isnan.(g.g))
    @test g.g[1] == 1.0
    @test all(isnan.(g.g[2:end]) .|| g.g[2:end] .== 0.0)

    Base.show(stdout, MIME"text/plain"(), constants)

    #! test when g is constant
    G_uniform = deepcopy(G)
    G_uniform.g .= 1.0
    pcfplot(G_uniform)

end
