using PairCorrelationFunction
using Test

@generated function ≂(x, y)
    if !isempty(fieldnames(x)) && x == y
        mapreduce(n -> :(x.$n == y.$n), (a,b)->:($a && $b), fieldnames(x))
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
    g_self = pcf(centers_matrix, constants)

    dr = 20.0
    constants_v2 = Constants(xlims, ylims, dr)
    @test constants ≂ constants_v2
    @test isequal(g, pcf(centers_matrix, targets_matrix, constants_v2))

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
    constants_3d_v2 = Constants(xlims, ylims, zlims, dr)
    @test constants_3d ≂ constants_3d_v2
    @test isequal(g3, pcf(centers_matrix_3d, targets_matrix_3d, constants_3d_v2))
    
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
    @test g[1] == 1.0
    @test all(isnan.(g[2:end]) .|| g[2:end] .== 0.0)

    Base.show(stdout, MIME"text/plain"(), constants)

end
