module PairCorrelationFunction

using NaNStatistics

export pcf, Constants

"""
    Constants(xlims::Tuple{Float64, Float64}, ylims::Tuple{Float64, Float64}[, zlims::Tuple{Float64, Float64}], dr::Real)
    Constants(xlims::Tuple{Float64, Float64}, ylims::Tuple{Float64, Float64}[, zlims::Tuple{Float64, Float64}], radii::AbstractRange{<:Real})

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
  domain_volume: 729000000.0
  radii: 0.0 - 1600.0
  #annuli: 16
"""
struct Constants{N}
    grid_size::NTuple{N, Float64}
    base_point::NTuple{N, Float64}
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

    function Constants(xlims::Tuple{Float64, Float64}, ylims::Tuple{Float64, Float64}, radii)
        grid_size = [L[2]-L[1] for L in [xlims, ylims]] |> Tuple
        base_point = [minimum(L) for L in [xlims, ylims]] |> Tuple
        domain_volume = prod(grid_size)
        return Constants(grid_size, base_point, domain_volume, radii)
    end

    function Constants(xlims::Tuple{Float64, Float64}, ylims::Tuple{Float64, Float64}, zlims::Tuple{Float64, Float64}, radii)
        grid_size = [L[2] - L[1] for L in [xlims, ylims, zlims]] |> Tuple
        base_point = [minimum(L) for L in [xlims, ylims, zlims]] |> Tuple
        domain_volume = prod(grid_size)
        return Constants(grid_size, base_point, domain_volume, radii)
    end
end

Base.ndims(::Constants{N}) where N = N

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
- `pcf::Vector{Float64}`: A vector of normalized target densities for each annulus around all centers.
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
    n_centers = size(centers, 1)
    n_targets = size(targets, 1)
    distances = zeros(Float64, n_centers, n_targets)
    volumes = zeros(Float64, length(constants.radii) - 1)
    for (i, center) in enumerate(eachrow(centers))
        distances[i, :] = (targets .- center') .^ 2 |> x -> sum(x, dims=2)
        relative_center = vec(center) .- constants.base_point
        
        @assert all(relative_center .>= 0) && all(relative_center .<= constants.grid_size) """
        Center point is outside the domain:
            $(vec(center)) ∉ $(join(["[$(join(constants.base_point[i] .+ [0, constants.grid_size[i]],", "))]" for i in 1:ndims(constants)], " x "))
        """

        volumes .+= computeVolume(relative_center, constants)
    end
    N, _ = histcountindices(vec(distances .|> sqrt), constants.radii)
    return N, volumes, n_targets
end

pcf_calculate(N, volumes, n_targets, constants) = (N ./ volumes) ./ (n_targets / constants.domain_volume)

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

function addQuarterArea!(A::Vector{Float64}, center::Tuple{Float64, Float64}, constants::Constants{2})
    r, r2 = constants.radii, constants.radii2
    x, y = (minimum(center), maximum(center)) #! this algorithm works by assuming that x<=y
    rb = r2 .<= x^2 + y^2 #! rvalues that mean the shell is still bounded by the origin
    rx = rb .&& r .> x #! rvalues that mean the shell has hit the y-axis
    ry = rb .&& r .> y #! rvalues that mean the shell has hit the x-axis

    A[rb] .+= 0.25 * π * r2[rb] #! add the full area of the shell before correcting for going outside the domain

    x1 = sqrt.(r2[ry] .- y^2) #! x-values where the shell hits the y-axis
    y1 = sqrt.(r2[rx] .- x^2) #! y-values where the shell hits the x-axis

    A[rx] -= 0.5 * (r2[rx] .* atan.(y1 ./ x) - x*y1) #! correct for crossing y-axis
    A[ry] -= 0.5 * (r2[ry] .* atan.(x1 ./ y) - y*x1) #! correct for crossing x-axis

    A[.!rb] .+= x*y
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
    x3, y3, z3 = x*x2, y*y2, z*z2
    xy, xz, yz = x*y, x*z, y*z

    rx = r .> x #! rvalues that mean the shell has hit the yz-plane
    ry = r .> y #! rvalues that mean the shell has hit the xz-plane
    rz = r .> z #! rvalues that mean the shell has hit the xy-plane
    rxy = r .> sqrt(x2 + y2) #! rvalues that mean the shell has hit the z-axis
    rxz = r .> sqrt(x2 + z2) #! rvalues that mean the shell has hit the y-axis
    ryz = r .> sqrt(y2 + z2) #! rvalues that mean the shell has hit the x-axis
    rb = r .> sqrt(x2 + y2 + z2) #! rvalues that mean the shell no longer bounded by the origin

    V[.!rx] .+= (1/6) * π * (r[.!rx] .^ 3) #! volume of octant before reaching wall in x direction

    I0 = rx .&& .!ry #! indices of radii before hitting y wall
    V[I0] .+= (1/12) * π * (3*r[I0] .^ 2 .- x2) #! volume of octant before reaching wall in y direction

    #! split depending on whether shells first reach z wall or xy edge
    #! next, the growing shell could either hit the z wall or reach the edge
    #! where the x and y walls meet; the other of these two must come next
    if z2 <= x2+y2 #! reach wall in z direction first
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
    V[rb] .+= xy*z #! you're welcome to explore this area, but there's not much of interest beyond...
end
end