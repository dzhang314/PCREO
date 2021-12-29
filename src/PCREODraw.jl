using ColorTypes: RGB
using GeometryBasics: meta, SimplexFace, Mesh, Point3f, Vec3f
using GLMakie: Scene, lines!, mesh!,
    VideoStream, cameracontrols, update_cam!, recordframe!, save
using MultiFloats: Float64x2
using StaticArrays: SVector, dot, cross, norm
using UUIDs: UUID

push!(LOAD_PATH, @__DIR__)
using PCREO


function nearest_neighbor_distance(
    points::AbstractVector{SVector{N,T}},
    pairs::AbstractVector{Tuple{Int,Int}}
) where {T,N}
    result = typemax(T)
    for (i, j) in pairs
        @inbounds dist = norm(points[i] - points[j])
        result = min(result, dist)
    end
    return result
end


function farthest_neighbor_distance(
    points::AbstractVector{SVector{N,T}},
    pairs::AbstractVector{Tuple{Int,Int}}
) where {T,N}
    result = zero(T)
    for (i, j) in pairs
        @inbounds dist = norm(points[i] - points[j])
        result = max(result, dist)
    end
    return result
end


normalize(x::SVector{N,T}) where {T,N} = x / norm(x)


function point_markers(points::AbstractVector{SVector{3,T}},
                       m::Int, r::Float32) where {T}
    n = length(points)
    coords = Vector{SVector{3,Float32}}(undef, (m+1)*n)
    faces = Matrix{Int}(undef, m*n, 3)
    for i = 1 : n
        a_ = normalize(points[i])
        b_ = normalize(cross(a_, ifelse(a_[3] * a_[3] < T(0.5),
            SVector{3,T}(zero(T), zero(T), one(T)),
            SVector{3,T}(zero(T), one(T), zero(T))
        )))
        c = Float32.(cross(a_, b_))
        b = Float32.(b_)
        a = Float32.(a_)
        coords[(m+1)*(i-1)+1] = 1.003f0 * a
        @simd ivdep for j = 1 : m
            t = (2.0f0 * Float32(j)) / Float32(m)
            st, ct = sincospi(t)
            coords[(m+1)*(i-1)+j+1] = a + r*(ct*b + st*c)
        end
        @simd ivdep for j = 1 : m-1
            faces[m*(i-1)+j,1] = (m+1)*(i-1)+1
            faces[m*(i-1)+j,2] = (m+1)*(i-1)+j+1
            faces[m*(i-1)+j,3] = (m+1)*(i-1)+j+2
        end
        faces[m*i,1] = (m+1)*(i-1)+1
        faces[m*i,2] = (m+1)*(i-1)+m+1
        faces[m*i,3] = (m+1)*(i-1)+2
    end
    return (coords, faces)
end


function spherical_line_impl!(coords::AbstractVector{SVector{3,Float32}},
                              a::SVector{3,T}, b::SVector{3,T},
                              r::T, n::Int) where {T}
    if n == 0
        push!(coords, Float32.(r * a))
    else
        ab = normalize(a + b)
        spherical_line_impl!(coords,  a, ab, r, n - 1)
        spherical_line_impl!(coords, ab,  b, r, n - 1)
    end
    return nothing
end


function spherical_line!(coords::AbstractVector{SVector{3,Float32}},
                         a::SVector{3,T}, b::SVector{3,T},
                         r::T, n::Int) where {T}
    @assert n >= 0
    na = normalize(a)
    nb = normalize(b)
    spherical_line_impl!(coords, na, nb, r, n)
    push!(coords, Float32.(r * b))
    push!(coords, SVector{3,Float32}(NaN32, NaN32, NaN32))
    return nothing
end


function spherical_triangle_impl!(
    coords::AbstractVector{SVector{3,Float32}},
    colors::AbstractVector{RGB{U}},
    a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T},
    color::RGB{U}, n::Int
) where {T,U}
    if n == 0
        push!(coords, Float32.(a))
        push!(coords, Float32.(b))
        push!(coords, Float32.(c))
        push!(colors, color)
        push!(colors, color)
        push!(colors, color)
    else
        ab = normalize(a + b)
        bc = normalize(b + c)
        ca = normalize(c + a)
        spherical_triangle_impl!(coords, colors,  a, ab, ca, color, n - 1)
        spherical_triangle_impl!(coords, colors,  b, bc, ab, color, n - 1)
        spherical_triangle_impl!(coords, colors,  c, ca, bc, color, n - 1)
        spherical_triangle_impl!(coords, colors, ab, bc, ca, color, n - 1)
    end
    return nothing
end


function spherical_triangle!(coords::AbstractVector{SVector{3,Float32}},
                             colors::AbstractVector{RGB{U}},
                             a::SVector{3,T}, b::SVector{3,T}, c::SVector{3,T},
                             color::RGB{U}, n::Int) where {T,U}
    @assert n >= 0
    orientation = dot(a, cross(b - a, c - a))
    if !signbit(orientation)
        spherical_triangle_impl!(coords, colors,
                                 normalize(a), normalize(b), normalize(c),
                                 color, n)
    else
        spherical_triangle_impl!(coords, colors,
                                 normalize(a), normalize(c), normalize(b),
                                 color, n)
    end
    return nothing
end


const DEGREE_COLOR = Dict{Int,RGB{Float32}}(
    3 => RGB{Float32}(1.0f0, 0.5f0, 0.0f0),
    4 => RGB{Float32}(1.0f0, 1.0f0, 0.0f0),
    5 => RGB{Float32}(1.0f0, 0.0f0, 0.0f0),
    6 => RGB{Float32}(0.0f0, 1.0f0, 0.0f0),
    7 => RGB{Float32}(0.0f0, 0.0f0, 1.0f0),
    8 => RGB{Float32}(1.0f0, 0.0f0, 1.0f0),
)


function spherical_mesh(points::AbstractVector{SVector{3,T}},
                        degrees::AbstractDict{Int,Int},
                        facets::AbstractVector{Vector{Int}},
                        facet_centers::AbstractVector{SVector{3,T}},
                        adjacent_facets::AbstractVector{Tuple{Int,Int}},
                        n::Int) where {T}
    coords = SVector{3,Float32}[]
    colors = RGB{Float32}[]
    for (i, j) in adjacent_facets
        for k in intersect(facets[i], facets[j])
            spherical_triangle!(coords, colors,
                                points[k], facet_centers[i], facet_centers[j],
                                DEGREE_COLOR[degrees[k]], n)
        end
    end
    return Mesh(
        meta(Point3f.(coords); normals=Vec3f.(coords), color=colors),
        [SimplexFace{3,Int}(i, i+1, i+2) for i = 1 : 3 : length(coords)]
    )
end


function save_pcreo_movie(moviepath::AbstractString, rec::PCREORecord)

    points = to_point_vector(Val{3}(), rec.points)
    facet_centers = [
        spherical_circumcenter(points, facet)
        for facet in rec.facets
    ]

    degrees = incidence_degrees(rec.facets)
    adjacent_vertices, adjacent_facets = adjacency_structure(rec.facets)
    nndist = nearest_neighbor_distance(points, adjacent_vertices)
    fndist = farthest_neighbor_distance(points, adjacent_vertices)

    m = 0
    while fndist / 2^m >= Float64x2(0.125)
        m += 1
    end

    line_coordinates = SVector{3,Float32}[]
    for (i, j) in adjacent_facets
        spherical_line!(line_coordinates, facet_centers[i], facet_centers[j],
                        Float64x2(1.003), m)
    end

    scene = Scene(resolution=(800, 800))
    mesh!(scene, point_markers(points, 25, Float32(nndist) / 12.0f0)...;
          show_axis=false, color=:black, shading=false)
    lines!(scene, line_coordinates;
           linewidth=min(10.0f0*Float32(nndist), 3.0f0))
    mesh!(scene,
          spherical_mesh(points, degrees, rec.facets,
                         facet_centers, adjacent_facets, m);
          shading=true)

    stream = VideoStream(scene; framerate=60)
    cam = cameracontrols(scene)

    for t in range(0.0f0, 2.0f0; length=450)
        st, ct = sincospi(t)
        cam.eyeposition[] = 20.0f0 * normalize(SVector{3,Float32}(
            ct, st, 0.6f0 * sinpi(2.0f0 * t)
        ))
        cam.zoom_mult[] = 0.132f0
        cam.far[] = 30.0f0
        update_cam!(scene, cam)
        recordframe!(stream)
    end

    save(moviepath, stream; compression=0, framerate=60)
    return nothing
end


function main(a::Int, b::Int)

    @assert isdir(ENV["PCREO_DATABASE_DIRECTORY"])
    @assert isdir(ENV["PCREO_IMAGE_DIRECTORY"])

    for num_dir in lsdir(ENV["PCREO_DATABASE_DIRECTORY"]; join=true)
        num_str = basename(num_dir)
        n = parse(Int, num_str)
        if n % b == a
            for entry_dir in lsdir(num_dir; join=true)
                uuid = UUID(basename(entry_dir))
                moviename = "PCREO-03-$num_str-$uuid.webm"
                moviepath = joinpath(ENV["PCREO_IMAGE_DIRECTORY"], moviename)
                if isfile(moviepath)
                    println("$moviepath already exists.")
                    flush(stdout)
                else
                    println("Generating PCREO movie $moviepath...")
                    flush(stdout)
                    filename = "PCREO-03-$num_str-$uuid.csv"
                    save_pcreo_movie(
                        moviepath,
                        PCREORecord(joinpath(entry_dir, filename))
                    )
                end
            end
        end
    end
end


main(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
