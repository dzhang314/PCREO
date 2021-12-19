using MultiFloats: Float64x3
using StaticArrays: SVector
using UUIDs: UUID

push!(LOAD_PATH, @__DIR__)
using PCREO
using PointGroups


@assert isdir(ENV["PCREO_DATABASE_DIRECTORY"])


lsdir(path...) = filter(!startswith('.'),
    readdir(joinpath(path...); sort=false))


const PCREO_CACHE = Dict{Int,Vector{Tuple{Float64x3,UUID}}}()


println("Building PCREO cache...")
flush(stdout)
for num_dir in lsdir(ENV["PCREO_DATABASE_DIRECTORY"])
    println("Scanning directory $num_dir...")
    flush(stdout)
    num_points = parse(Int, num_dir)
    @assert !haskey(PCREO_CACHE, num_points)
    entry_list = Tuple{Float64x3,UUID}[]
    for entry_name in lsdir(ENV["PCREO_DATABASE_DIRECTORY"], num_dir)
        reference_name = "PCREO-03-$num_dir-$entry_name.csv"
        reference = PCREORecord(joinpath(
            ENV["PCREO_DATABASE_DIRECTORY"], num_dir,
            entry_name, reference_name))
        push!(entry_list, (reference.energy, UUID(entry_name)))
    end
    sort!(entry_list)
    PCREO_CACHE[num_points] = entry_list
end
println("Finished building PCREO cache.")
flush(stdout)


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
    num_dir = joinpath(ENV["PCREO_DATABASE_DIRECTORY"], num_str)
    if !isdir(num_dir)
        mkdir(num_dir)
    end
    for (reference_energy, reference_uuid) in PCREO_CACHE[record.num_points]
        if isapprox(record.energy, reference_energy, rtol=Float64x3(2^-80))
            entry_dir = joinpath(num_dir, string(reference_uuid))
            reference = PCREORecord(joinpath(entry_dir,
                "PCREO-$dim_str-$num_str-$reference_uuid.csv"))
            @assert reference.energy == reference_energy
            if isometric(to_point_vector(record.points),
                         to_point_vector(reference.points), 2^-40)
                newpath = joinpath(entry_dir, filename)
                mv(filepath, newpath)
                return newpath
            end
        end
    end
    push!(PCREO_CACHE[record.num_points], (record.energy, uuid))
    sort!(PCREO_CACHE[record.num_points])
    new_dir = joinpath(num_dir, string(uuid))
    mkdir(new_dir)
    newpath = joinpath(new_dir, filename)
    mv(filepath, newpath)
    return newpath
end


function main(n_min::Int, n_max::Int)
    while true
        filenames = filter(startswith("PCREO-03-"),
                           readdir("/Users/dzhang314/pcreo-output-icme"))
        if length(filenames) > 0
            filename = rand(filenames)
            num_points = parse(Int, filename[10:17])
            if n_min <= num_points <= n_max
                src = joinpath("/Users/dzhang314/pcreo-output-icme", filename)
                try
                    dst = add_to_database!(src)
                    println(src, " => ", dst)
                catch e
                    if e isa AssertionError
                        println(filename, ": ", e)
                    else
                        rethrow(e)
                    end
                end
            end
        else
            sleep(1.0)
        end
    end
end


main(parse(Int, ARGS[1]), parse(Int, ARGS[2]))
