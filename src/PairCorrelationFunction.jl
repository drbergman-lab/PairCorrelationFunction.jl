module PairCorrelationFunction

using NaNStatistics, RecipesBase, PlotUtils

export pcf, Constants, pcfplot

VERSION >= v"1.11" && include("public.julia")

"""
    PCFResult

A struct to hold the results of the pair correlation function calculation.

For a single sample/timepoint, the `g` field will be a vector of pair correlation function values.
For multiple samples/timepoints, the `g` field will be a matrix where each column corresponds to a sample/timepoint.

The `hcat` function is overloaded to allow for easy concatenation of multiple `PCFResult` objects. See example below.

# Fields
- `radii::AbstractVector{<:Real}`: The range of radii used for the pair correlation function.
- `g::AbstractArray{Float64}`: The pair correlation function values for each radius, possibly over many samples/timepoints.

# Example
```jldoctest
using PairCorrelationFunction
radii = [0.0, 1.0, 2.0]
g = [0.5, 1.2]
PairCorrelationFunction.PCFResult(radii, g)
# output
PCFResult:
  Radii: 0.0 - 2.0 with 2 annuli
  g: 0.5 - 1.2 (min - max)
```
```jldoctest
using PairCorrelationFunction
radii = 0.0:1.0:2.0
result_1 = PairCorrelationFunction.PCFResult(radii, [0.5, 1.2])
result_2 = PairCorrelationFunction.PCFResult(radii, [0.6, 1.3])
result_3 = PairCorrelationFunction.PCFResult(radii, [0.8, 1.4])
hcat(result_1, result_2, result_3)
# output
PCFResult:
  #timepoints: 3
  Radii: 0.0 - 2.0 with 2 annuli
  g: 0.5 - 1.4 (min - max)
```
"""
struct PCFResult
    radii::AbstractVector{<:Real}
    g::AbstractArray{Float64}

    function PCFResult(radii::AbstractRange{<:Real}, g::AbstractArray{<:Real})
        return PCFResult(collect(radii), g)
    end
    
    function PCFResult(radii::AbstractVector{<:Real}, g::AbstractArray{<:Real})
        @assert radii[1] == 0.0 "The first radius must be 0.0. Got $(radii[1])"
        @assert length(radii) == size(g, 1) + 1 "The number of radii must be one more than the number of g values. Got $(length(radii)) and $(size(g, 1))"
        @assert ndims(g) <= 2 "The g values must be a 1D or 2D array. Got $(ndims(g))D."
        new(radii, g)
    end
        
end

function Base.hcat(gs::Vararg{PCFResult})
    @assert all([collect(g.radii) == collect(gs[1].radii) for g in gs]) "All pair correlation function outputs must have the same radii. Got $(collect.(gs))"
    PCFResult(gs[1].radii, hcat([g.g for g in gs]...))
end

function Base.show(io::IO, ::MIME"text/plain", pcf_result::PCFResult)
    println(io, "PCFResult:")
    if size(pcf_result.g, 2) > 1
        println(io, "  #timepoints: $(size(pcf_result.g, 2))")
    end
    println(io, "  Radii: $(pcf_result.radii[1]) - $(pcf_result.radii[end]) with $(length(pcf_result.radii)-1) annuli")
    temp = filter(!isnan, pcf_result.g)
    println(io, "  g: $(minimum(temp)) - $(maximum(temp)) (min - max)")
end

"""
    Constants(xlims::Tuple{Float64, Float64}, ylims::Tuple{Float64, Float64}[, zlims::Tuple{Float64, Float64}], dr::Real)

A struct to hold the constants needed for the pair correlation function calculation.

The vector of radii that define the concentric annuli must have constant spacing.
You may either pass in the spacing `dr` or a range of radii in the form `r0:dr:rf`.

# Fields
- `grid_size::NTuple{N, Float64}`: The size of the grid in each dimension.
- `base_point::NTuple{N, Float64}`: The base point of the grid to be used to calculate distance to the boundary.
- `domain_volume::Float64`: The volume of the domain.
- `radii::AbstractRange{<:Real}`: The range of radii to be used for the pair correlation function.
- `radii2::AbstractVector{<:Real}`: The squared values of the radii.

# Examples
```jldoctest
using PairCorrelationFunction
xlims = (-450.0, 450.0)
ylims = (-450.0, 450.0)
radii = 0:20.0:1300.0
constants = Constants(xlims, ylims, radii)
# output
Constants for 2D pair correlation function:
  grid_size: (900.0, 900.0)
  base_point: (-450.0, -450.0)
  domain_volume: 810000.0
  radii: 0.0 - 1300.0
  #annuli: 65
```
```jldoctest
using PairCorrelationFunction
xlims = (-450.0, 450.0)
ylims = (-450.0, 450.0)
zlims = (-450.0, 450.0)
dr = 100.0
constants = Constants(xlims, ylims, zlims, dr)
# output
Constants for 3D pair correlation function:
  grid_size: (900.0, 900.0, 900.0)
  base_point: (-450.0, -450.0, -450.0)
  domain_volume: 7.29e8
  radii: 0.0 - 1600.0
  #annuli: 16
```
"""
struct Constants{N}
    grid_size::NTuple{N,Float64}
    base_point::NTuple{N,Float64}
    domain_volume::Float64
    radii::AbstractRange{<:Real}
    radii2::AbstractVector{<:Real}

    function Constants(grid_size::NTuple{N,Float64}, base_point::NTuple{N,Float64}, domain_volume::Float64, radii) where {N}
        @assert radii[1] == 0.0 "The first radius must be 0.0. Got $(radii[1])"
        new{N}(grid_size, base_point, domain_volume, radii, radii .^ 2)
    end

    function Constants(grid_size::NTuple{N,Float64}, base_point::NTuple{N,Float64}, domain_volume::Float64, dr::Real) where {N}
        max_radii_multiples = (sqrt(sum(grid_size .^ 2))) / dr |> ceil
        radii = 0:dr:(max_radii_multiples*dr)
        return Constants(grid_size, base_point, domain_volume, radii)
    end

    function Constants(xlims::Tuple{Float64,Float64}, ylims::Tuple{Float64,Float64}, radii)
        grid_size = [L[2] - L[1] for L in [xlims, ylims]] |> Tuple
        base_point = [minimum(L) for L in [xlims, ylims]] |> Tuple
        domain_volume = prod(grid_size)
        return Constants(grid_size, base_point, domain_volume, radii)
    end

    function Constants(xlims::Tuple{Float64,Float64}, ylims::Tuple{Float64,Float64}, zlims::Tuple{Float64,Float64}, radii)
        grid_size = [L[2] - L[1] for L in [xlims, ylims, zlims]] |> Tuple
        base_point = [minimum(L) for L in [xlims, ylims, zlims]] |> Tuple
        domain_volume = prod(grid_size)
        return Constants(grid_size, base_point, domain_volume, radii)
    end
end

Base.ndims(::Constants{N}) where {N} = N

function Base.show(io::IO, ::MIME"text/plain", constants::Constants)
    println(io, "Constants for $(ndims(constants))D pair correlation function:")
    println(io, "  grid_size: $(constants.grid_size)")
    println(io, "  base_point: $(constants.base_point)")
    println(io, "  domain_volume: $(constants.domain_volume)")
    println(io, "  radii: $(constants.radii |> first) - $(constants.radii |> last)")
    println(io, "  #annuli: $(length(constants.radii)-1)")
end

"""
    pcf(centers::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}, constants::Constants)

Calculate the pair correlation function for a set of centers and targets.

For each point in the `centers` matrix, compute the distance to each point in the `targets` matrix.
Bin these distances by the radii defined in the `constants` object.
If `targets` is not provided, the function will use `centers` as both centers and targets.
Technically, this is the traditional pcf.
The version with `centers` and `targets` is the cross-PCF.

# Arguments
- `centers::AbstractMatrix{<:Real}`: A matrix of centers, where each row is a center.
- `targets::AbstractMatrix{<:Real}`: A matrix of targets, where each row is a target.
- `constants::Constants`: A `Constants` object containing the grid size, base point, domain volume, and radii.

# Returns
- `PCFResult`: A struct with the radii and the pair correlation function values.
"""
function pcf(centers::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}, constants::Constants)
    N, volumes, n_targets = pcf_binning(centers, targets, constants)
    return pcf_calculate(N, volumes, n_targets, constants)
end

function pcf(centers::AbstractMatrix{<:Real}, constants::Constants)
    N, volumes, n_targets = pcf_binning(centers, centers, constants)
    N[1] -= 1 #! do not count the center point as one of its targets
    n_targets -= 1 #! make sure to normalize by the number of targets that could be found by any given center
    return pcf_calculate(N, volumes, n_targets, constants)
end

function pcf_binning(centers::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}, constants::Constants)
    n_targets = size(targets, 1)
    N = zeros(Int, length(constants.radii) - 1)
    volumes = zeros(Float64, length(constants.radii) - 1)
    for (i, center) in enumerate(eachrow(centers))
        distances = (targets .- center') .^ 2 |> x -> sum(x, dims=2) .|> sqrt |> vec
        _histcounts_include_min!(N, distances, constants.radii)

        relative_center = vec(center) .- constants.base_point

        @assert all(relative_center .>= 0) && all(relative_center .<= constants.grid_size) """
        Center point is outside the domain:
            $(vec(center)) ∉ $(join(["[$(join(constants.base_point[i] .+ [0, constants.grid_size[i]],", "))]" for i in 1:ndims(constants)], " x "))
        """

        volumes .+= computeVolume(relative_center, constants)
    end
    return N, volumes, n_targets
end

pcf_calculate(N, volumes, n_targets, constants) = PCFResult(constants.radii, (N ./ volumes) ./ (n_targets / constants.domain_volume))

"""
    _histcounts_include_min!(N::Vector{Int}, x::Vector{Float64}, xedges::AbstractRange{<:Real})

Like `histcounts!`, but includes the minimum edge of the histogram in the first bin.

This function relies on `xedges[1]==0.0` and the input `x` being all distances to guarantee no value is less than 0.
Users can pass in radii with any upper bound and nothing stops the targets from being outside the domain, so we cannot guarantee that the maximum value of `x` is less than or equal to the maximum value of `xedges`.
"""
function _histcounts_include_min!(N::Vector{Int}, x::Vector{Float64}, xedges::AbstractRange{<:Real})
    @assert firstindex(N) === 1

    # What is the size of each bin?
    nbins = length(xedges) - 1
    xmin, xmax = extrema(xedges)
    didx = nbins / (xmax - xmin)

    # Make sure we don't have a segfault by filling beyond the length of N
    @assert length(N) == nbins "length(N) != nbins; got length(N)=$(length(N)) and nbins=$(nbins)"

    # Loop through each element of x
    @inbounds for xi in x
        if xi == xmin
            N[1] += 1
            continue
        end
        i_ = (xi - xmin) * didx
        if 0 < i_ <= nbins
            i = ceil(Int, i_)
            N[i] += 1
        end
    end
end

function computeVolume(center::AbstractVector{<:Real}, constants::Constants{2})
    x, y = center
    A = zeros(Float64, length(constants.radii))
    for x₀ in [x, constants.grid_size[1] - x]
        for y₀ in [y, constants.grid_size[2] - y]
            addQuarterArea!(A, (x₀, y₀), constants)
        end
    end
    return diff(A)
end

function addQuarterArea!(A::Vector{Float64}, center::Tuple{Float64,Float64}, constants::Constants{2})
    r, r2 = constants.radii, constants.radii2
    x, y = (minimum(center), maximum(center)) #! this algorithm works by assuming that x<=y
    rb = r2 .<= x^2 + y^2 #! rvalues that mean the shell is still bounded by the origin
    rx = rb .&& r .> x #! rvalues that mean the shell has hit the y-axis
    ry = rb .&& r .> y #! rvalues that mean the shell has hit the x-axis

    A[rb] .+= 0.25 * π * r2[rb] #! add the full area of the shell before correcting for going outside the domain

    x1 = sqrt.(r2[ry] .- y^2) #! x-values where the shell hits the y-axis
    y1 = sqrt.(r2[rx] .- x^2) #! y-values where the shell hits the x-axis

    A[rx] -= 0.5 * (r2[rx] .* atan.(y1 ./ x) - x * y1) #! correct for crossing y-axis
    A[ry] -= 0.5 * (r2[ry] .* atan.(x1 ./ y) - y * x1) #! correct for crossing x-axis

    A[.!rb] .+= x * y
end

function computeVolume(center::AbstractVector{<:Real}, constants::Constants{3})
    x, y, z = center
    A = zeros(Float64, length(constants.radii))
    for x₀ in [x, constants.grid_size[1] - x]
        for y₀ in [y, constants.grid_size[2] - y]
            for z₀ in [z, constants.grid_size[3] - z]
                addOctantVolume!(A, (x₀, y₀, z₀), constants)
            end
        end
    end
    return diff(A)
end

function addOctantVolume!(V::Vector{Float64}, center::NTuple{3,Float64}, constants::Constants{3})
    r, r2 = constants.radii, constants.radii2
    x, y, z = sort([center...]) #! this algorithm works by assuming that x<=y<=z

    x2, y2, z2 = x^2, y^2, z^2
    x3, y3, z3 = x * x2, y * y2, z * z2
    xy, xz, yz = x * y, x * z, y * z

    rx = r .> x #! rvalues that mean the shell has hit the yz-plane
    ry = r .> y #! rvalues that mean the shell has hit the xz-plane
    rz = r .> z #! rvalues that mean the shell has hit the xy-plane
    rxy = r .> sqrt(x2 + y2) #! rvalues that mean the shell has hit the z-axis
    rxz = r .> sqrt(x2 + z2) #! rvalues that mean the shell has hit the y-axis
    ryz = r .> sqrt(y2 + z2) #! rvalues that mean the shell has hit the x-axis
    rb = r .> sqrt(x2 + y2 + z2) #! rvalues that mean the shell no longer bounded by the origin

    V[.!rx] .+= (1 / 6) * π * (r[.!rx] .^ 3) #! volume of octant before reaching wall in x direction

    I0 = rx .&& .!ry #! indices of radii before hitting y wall
    V[I0] .+= (1 / 12) * π * (3 * r[I0] .^ 2 .- x2) #! volume of octant before reaching wall in y direction

    #! split depending on whether shells first reach z wall or xy edge
    #! next, the growing shell could either hit the z wall or reach the edge
    #! where the x and y walls meet; the other of these two must come next
    if z2 <= x2 + y2 #! reach wall in z direction first
        I1 = ry .&& .!rz #! indices of radii before hitting z wall
        I2 = rz .&& .!rxy #! indices of radii after hitting z wall and before reaching xy edge
        ri = r[I2] #! radii after hitting z wall and before reaching xy edge
        ri2 = r2[I2]

        V[I2] .+= (1 / 12) * π * (ri2 .* (3 * (x + y + z) .- 4 * ri) .- (x3 + y3 + z3)) #! volume at radii between hitting z wall and reaching xy edge

    else #! reaches xy edge first
        I1 = ry .&& .!rxy #! indices of radii before hitting xy edge
        I2 = rxy .&& .!rz #! indices of radii after hitting xy edge and before reaching z wall
        ri = r[I2] #! radii after hitting xy edge and before reaching z wall
        ri2 = r2[I2] #! precompute radius squared
        z1 = sqrt.(ri2 .- (x2 + y2)) #! useful value to precompute

        V[I2] .+= (1 / 6) * (2 * xy * z1 .+
                             atan.(y ./ z1) .* x .* (3 * ri2 .- x2) +
                             atan.(x ./ z1) .* y .* (3 * ri2 .- y2) +
                             -2 * atan.(xy ./ (ri .* z1)) .* ri2 .* ri
        ) #! volume at radii between reaching xy edge and hitting z wall
    end

    ri = r[I1] #! radii before z wall or xy edge (whichever came first)
    ri2 = r2[I1] #! precompute radius squared

    V[I1] .+= (1 / 12) * π * (ri2 .* (3 * (x + y) .- 2 * ri) .- (x3 + y3)) #! volume before reaching z wall or xy edge

    #! go from further of z wall and xy edge up to xz edge
    I3 = rz .&& rxy .&& .!rxz #! indices of radii after reaching both z wall and xy edge before hitting xz edge
    ri = r[I3] #! radii after reaching both z wall and xy edge before hitting xz edge
    ri2 = r2[I3] #! precompue radius squared
    z1 = sqrt.(ri2 .- (x2 + y2)) #! precompute a useful quantity

    V[I3] .+= (1 / 12) * π * (ri2 .* (3 * z .- 2 * ri) .- z3) +
              (1 / 6) * (
        2 * xy * z1 .+
        x * (3 * ri2 .- x2) .* atan.(y ./ z1) +
        y * (3 * ri2 .- y2) .* atan.(x ./ z1) +
        -2 * ri .* ri2 .* atan.(xy ./ (ri .* z1))
    ) #! volume between z wall/xy edge and xz edge

    #! go from xz edge up to yz edge
    I4 = rxz .&& .!ryz #! indices of radii after reaching xz edge and before reaching yz edge
    ri = r[I4] #! radii after reaching xz edge and before reaching yz edge
    ri2 = r2[I4] #! precompue radius squared
    y1 = sqrt.(ri2 .- (x2 + z2)) #! precompute a useful quantity
    z1 = sqrt.(ri2 .- (x2 + y2)) #! precompute a useful quantity

    V[I4] .+= (1 / 6) * (
        2 * x * (z * y1 + y * z1) .+
        (atan.(y ./ z1) - atan.(y1 ./ z)) .* x .* (3 * ri2 .- x2) +
        atan.(x ./ z1) .* y .* (3 * ri2 .- y2) +
        atan.(x ./ y1) .* z .* (3 * ri2 .- z2) -
        2 * ri2 .* ri .* (atan.(xy ./ (ri .* z1)) + atan.(xz ./ (ri .* y1)))
    )

    #! go from yz edge up to the corner and then we're out of the woods!!
    I5 = ryz .&& .!rb #! indices of radii after reaching yz edge and before reaching the corner
    ri = r[I5] #! radii after reaching yz edge and before reaching the corner
    ri2 = r2[I5] #! precompue radius squared

    x1 = sqrt.(ri2 .- (y2 + z2)) #! precompute a useful quantity
    y1 = sqrt.(ri2 .- (x2 + z2)) #! precompute a useful quantity
    z1 = sqrt.(ri2 .- (x2 + y2)) #! precompute a useful quantity

    V[I5] .+= (1 / 6) * (
        2 * (xy * z1 + xz * y1 + yz * x1) .+
        x * (3 * ri2 .- x2) .* (atan.(z ./ y1) - atan.(z1 ./ y)) +
        y * (3 * ri2 .- y2) .* (atan.(z ./ x1) - atan.(z1 ./ x)) +
        z * (3 * ri2 .- z2) .* (atan.(y ./ x1) - atan.(y1 ./ x)) +
        2 * ri2 .* ri .* (atan.(x * z1 ./ (ri .* y)) - atan.(xz ./ (ri .* y1)) + atan.(y * z1 ./ (ri .* x)) - atan.(yz ./ (ri .* x1)))
    )

    #! going beyond the corner
    V[rb] .+= xy * z #! you're welcome to explore this area, but there's not much of interest beyond...
end

@userplot PCFPlot

"""
    pcfplot

Plot the pair correlation function.

# Arguments
The function accepts 1, 2, or 3 arguments.
In all three cases, the objects containing the PCF values are in the final of these arguments either as single instances or as vectors of instances.

## Vectors
If the PCF values are all vectors, this will assume that these are independent realizations of the same process and will plot a mean and standard deviation.
In this case, an optional first argument can be passed to specify the radii.
If omitted, then the radii will be inferred from the `PCFResult` objects or, in the case `Vector{<:Real}` are passed in, the radii will be the vector indices, i.e. `1:length(g)`.

## Matrices
If the PCF values are all matrices, this will assume that each column is a different timepoint and will plot a heatmap of the mean across all samples (each matrix representing a timeseries sample).
In this case, two optional first arguments can be passed to specify the timepoints and radii, respectively.
If providing these arguments, the radii argument can be omitted ONLY IF the PCF values are passed in as `PCFResult` objects, i.e., they have the radii stored in the `radii` field.
This ordering is because we assume that the time will be displayed on the x-axis.
To swap the axes, transpose the matrices; then, pass in the timepoints as the second argument and the radii as the first argument.
If neither are passed in, the radii will be inferred as for the vectors; timepoints will be the indices of the columns of the matrices, i.e. `1:size(g, 2)`.

# Optional arguments
When plotting a heatmap, i.e., matrices, `pcfplot` will use the `:tofino` colorscheme by default.
This can be changed by passing in a keyword argument `colorscheme` with the desired colorscheme.
See the `Plots`-supported [color schemes](https://docs.juliaplots.org/latest/generated/colorschemes/) for more options.
So long as `color` is not user-defined, then `pcfplot` use this scheme and will further set the color scheme to transition at the pcf value 1 to highlight the transition from depletion to enrichment.

# Examples
```julia
using PairCorrelationFunction
using Plots
g = [0.5; 1.2; 0.8; 0.7; 0.6; 1.6]
pcfplot(g)
```
```julia
using PairCorrelationFunction
using Plots
g1 = [0.5; 1.2; 0.8; 0.7; 0.6; 1.6]
g2 = [0.9; 1.0; 1.1; 1.2; 1.3; 1.4]
pcfplot([g1, g2])
```
```julia
using PairCorrelationFunction
using Plots
g = [0.5 1.2 0.8;
     0.7 0.6 1.6;
     0.9 1.0 1.1;
     1.2 1.3 1.4;
     1.5 1.6 1.7]
pcfplot(g; colorscheme=:cork)
```
```julia
using PairCorrelationFunction
using Plots
result = PairCorrelationFunction.PCFResult(0:20.0:100.0, [0.1; 0.2; 0.3; 0.4; 0.5])
pcfplot(result)
```
```julia
using PairCorrelationFunction
using Plots
result1 = PairCorrelationFunction.PCFResult(0:20.0:100.0, [0.1; 0.2; 0.3; 0.4; 0.5])
result2 = PairCorrelationFunction.PCFResult(0:20.0:100.0, [0.6; 0.7; 0.8; 0.9; 1.0])
RESULT = hcat(result1, result2)
pcfplot([100.0, 200.0], RESULT) # use radii in RESULT for y-axis
pcfplot([100.0, 200.0], 0:20.0:100.0, RESULT) # explicitly pass in radii
```
"""
pcfplot

@recipe function f(p::PCFPlot)
    r, t, g = processPCFPlotArguments(p.args...)

    if isnothing(t)
        y = reduce(hcat, g) |> x -> nanmean(x; dims=2)
        @series begin
            if length(g) > 1
                ribbon := reduce(hcat, g) |> x -> nanstd(x; dim=2)
            end
            label --> missing
            r, y
        end
        @series begin
            linestyle := :dash
            linecolor := :black
            label := missing
            xlabel --> "radii"
            ylabel --> "PCF"
            [r[1], r[end]], ones(Int, 2)
        end
    else
        z = cat(g...; dims=3) |> x -> nanmean(x; dim=3)
        one_point_fn = (l, u) -> (1 - l) / (u - l)
        if :clims in keys(plotattributes)
            one_point = one_point_fn(plotattributes[:clims]...)
        else
            l = nanminimum(z)
            u = nanmaximum(z)
            if l == u
                #! if all values are the same, then set the clims to (0.0, 2.0)
                one_point = 0.5
                plotattributes[:clims] = (0.0, 2.0)
            else
                one_point = one_point_fn(l, u)
            end
        end
        z[isnan.(z)] .= -Inf #! set NaN values to -Inf so the heatmap can happen and show the color as low as possible
        colorscheme = :colorscheme in keys(plotattributes) ? plotattributes[:colorscheme] : :tofino
        @series begin
            seriestype := :heatmap
            color --> cgrad(colorscheme, [one_point])
            xlabel --> "time"
            ylabel --> "radii"
            t, r, z
        end
    end
end

function processPCFPlotArguments(args...)
    @assert length(args) <= 3 "Too many arguments passed to pcfplot. Expected 1, 2, or 3 arguments. Got $(length(args))"
    if length(args) == 1
        r, t, g = processPCFValuesArgument(args[1])
       
    elseif length(args) == 2
        r, t, g = processPCFValuesArgument(args[2])
        if size(g[1], 2) == 1 #! vectors of pcf values passed in
            r, t = args[1], nothing
        else #! matrices of pcf values passed in
            @assert length(args[1]) == length(t) "Timepoints passed in as the first argument must match the number of columns in the second argument(s). Expected $(length(t)) and got $(length(args[1]))"
            r, t = r, args[1]
        end
        
    else
        t = args[1]
        r = args[2]
        _, _, g = processPCFValuesArgument(args[3])
        @assert length(t) == size(g[1], 2) "The second argument must be a vector of timepoints that matches the number of columns in the third argument."
    end
    r = collect(r)
    processPCFPlotR!(r, g[1])
    
    return r, t, g
end

function processPCFValuesArgument(input)
    if input isa PCFResult
        r = input.radii[2:end]
        t = size(input.g, 2) == 1 ? nothing : 1:size(input.g, 2)
        g = [input.g]
    elseif input isa AbstractVector{PCFResult}
        r = input[1].radii[2:end]
        t = size(input[1].g, 2) == 1 ? nothing : 1:size(input[1].g, 2)
        g = [pcf_result.g for pcf_result in input]
        @assert all([size(x) == size(g[1]) for x in g]) "All pair correlation function outputs must have the same size. Got $(size.(g))"
    elseif input isa AbstractArray{<:Real}
        @assert ndims(input) <= 2 "The input must be a 1D or 2D array. Got $(ndims(input))D."
        r = 1:size(input, 1)
        t = size(input, 2) == 1 ? nothing : 1:size(input, 2)
        g = [input]
    elseif input isa AbstractVector{<:AbstractArray{<:Real}}
        @assert ndims(input[1]) <= 2 "The input must be a 1D or 2D array. Got $(ndims(input[1]))D."
        r = 1:size(input[1], 1)
        t = size(input[1], 2) == 1 ? nothing : 1:size(input[1], 2)
        g = input
        @assert all([size(x) == size(g[1]) for x in g]) "All pair correlation function outputs must have the same size. Got $(size.(g))"
    else
        throw(ArgumentError("Invalid argument type: $(typeof(input)). Expected PCFResult, AbstractVector{PCFResult}, AbstractMatrix{<:Real}, or AbstractVector{<:AbstractMatrix{<:Real}}."))
    end
    return r, t, g
end

function processPCFPlotR!(r, g)
    if length(r) == size(g, 1) + 1
        popfirst!(r)
    else
        @assert length(r) == size(g, 1) "The first argument must be a vector of radii that either matches the number of rows in the second argument or is one longer than the number of rows in the second argument."
    end
end

end