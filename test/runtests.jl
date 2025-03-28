using PairCorrelationFunction
using Test

@testset "PairCorrelationFunction.jl" begin
    # 2D testing
    xlims = (-450.0, 450.0)
    ylims = (-450.0, 450.0)
    radii = 0:20.0:1300.0
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

    #3D testing
    zlims = (-450.0, 450.0)
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

    @test_throws ErrorException pcf(centers_matrix_3d, targets_matrix_3d, constants_3d)
end
