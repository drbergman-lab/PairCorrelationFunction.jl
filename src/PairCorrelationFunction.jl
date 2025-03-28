module PairCorrelationFunction

using NaNStatistics

export pcf, Constants

struct Constants{N}
    grid_size::NTuple{N, Float64}
    base_point::NTuple{N, Float64}
    domain_volume::Float64
    radii::AbstractRange{<:Real}
    nr::Int
    radii2::AbstractVector{<:Real}

    function Constants(grid_size::NTuple{N, Float64}, base_point::NTuple{N, Float64}, domain_volume::Float64, radii) where N
        new{N}(grid_size, base_point, domain_volume, radii, length(radii), radii .^ 2)
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

function pcf(centers::AbstractMatrix{<:Real}, targets::AbstractMatrix{<:Real}, constants::Constants)
    n_centers = size(centers, 1)
    n_targets = size(targets, 1)
    distances = zeros(Float64, n_centers, n_targets)
    volumes = zeros(Float64, constants.nr - 1)
    for (i, center) in enumerate(eachrow(centers))
        distances[i, :] = (targets .- center') .^ 2 |> x -> sum(x, dims=2)
        volumes .+= computeVolume(vec(center), constants)
    end
    println("volumes = $volumes")
    N, _ = histcountindices(vec(distances .|> sqrt), constants.radii)
    return (N./volumes) ./ (n_targets/constants.domain_volume)
end

function computeVolume(center::AbstractVector{<:Real}, constants::Constants{2})
    x, y = center .- constants.base_point
    A = zeros(Float64, constants.nr)
    for x₀ in [x, constants.grid_size[1] - x]
        for y₀ in [y, constants.grid_size[2] - y]
            addQuarterArea!(A, (x₀, y₀), constants)
        end
    end
    return diff(A)
end

function addQuarterArea!(A::Vector{Float64}, center::Tuple{Float64, Float64}, constants::Constants{2})
    r, r2 = constants.radii, constants.radii2
    x, y = (minimum(center), maximum(center)) #! this algorithm works by assuming that x0<=y0
    rb = r2 .<= x^2 + y^2 #! rvalues that mean the shell is still bounded by the origin
    rx = rb .&& r .> x #! rvalues that mean the shell has hit the y-axis
    ry = rb .&& r .> y #! rvalues that mean the shell has hit the x-axis

    A[rb] += 0.25 * π * r2[rb] #! add the full area of the shell before correcting for going outside the domain

    x1 = sqrt.(r2[ry] .- y^2) #! x-values where the shell hits the y-axis
    y1 = sqrt.(r2[rx] .- x^2) #! y-values where the shell hits the x-axis

    A[rx] -= 0.5 * (r2[rx] .* atan.(y1 ./ x) - x*y1) #! correct for crossing y-axis
    A[ry] -= 0.5 * (r2[ry] .* atan.(x1 ./ y) - y*x1) #! correct for crossing x-axis

    A[.!rb] .+= x*y
end

function computeVolume(center::Vector{Float64}, constants::Constants{3})
    throw(ErrorException("Not implemented"))
end

end