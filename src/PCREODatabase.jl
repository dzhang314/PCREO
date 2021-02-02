using MultiFloats: Float64x3
using StaticArrays: SVector
using UUIDs: UUID

push!(LOAD_PATH, @__DIR__)
using PCREO
using PointGroups


const UUID_REGEX = Regex("[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-" *
                         "[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")


function to_point_vector(points::AbstractMatrix{T}) where {T}
    dimension, num_points = size(points)
    @assert dimension == 3
    return [SVector{3,T}(view(points, :, i)) for i = 1 : num_points]
end


function add_to_database!(filepath::String)
    filename = basename(filepath)
    uuid = UUID(match(UUID_REGEX, filename).match)
    record = PCREORecord(filepath)
    dim_str = lpad(record.dimension, 2, '0')
    num_str = lpad(record.num_points, 8, '0')
    num_dir = joinpath(PCREO_DATABASE_DIRECTORY, num_str)
    if !isdir(num_dir)
        mkdir(num_dir)
    end
    for entry_name in readdir(num_dir)
        entry_dir = joinpath(num_dir, entry_name)
        reference_name = "PCREO-$dim_str-$num_str-$entry_name.csv"
        reference = PCREORecord(joinpath(entry_dir, reference_name))
        if isapprox(record.energy, reference.energy;
                    rtol=Float64x3(2^-80))
            if isometric(to_point_vector(record.points),
                         to_point_vector(reference.points), 2^-40)
                mv(filepath, joinpath(entry_dir, filename))
                return joinpath(entry_dir, filename)
            end
        end
    end
    new_dir = joinpath(num_dir, string(uuid))
    mkdir(new_dir)
    mv(filepath, joinpath(new_dir, filename))
    return joinpath(new_dir, filename)
end


function main()
    while true
        filenames = filter(startswith("PCREO-"),
                           readdir("D:\\Data\\PCREOOutput"))
        if length(filenames) > 0
            src = joinpath("D:\\Data\\PCREOOutput", rand(filenames))
            dst = add_to_database!(src)
            println(src, " => ", dst)
        end
    end
end


main()
