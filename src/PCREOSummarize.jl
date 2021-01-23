using Printf
using ProgressMeter

push!(LOAD_PATH, @__DIR__)
using PCREO


function count_database_entries()
    result = 0
    dirnames = readdir(PCREO_DATABASE_DIRECTORY)
    prog = Progress(length(dirnames); desc="Scanning PCREO database... ")
    for dirname in dirnames
        result += length(readdir(joinpath(PCREO_DATABASE_DIRECTORY, dirname)))
        next!(prog)
    end
    println("Found $result database entries.")
    return result
end


function verify_unique_energies()
    count = count_database_entries()
    prog = Progress(count; desc="Verifying energy uniqueness... ")
    for dirname in readdir(PCREO_DATABASE_DIRECTORY)
        data = PCREORecord[]
        for fname in readdir(joinpath(PCREO_DATABASE_DIRECTORY, dirname))
            push!(data, PCREORecord(joinpath(
                PCREO_DATABASE_DIRECTORY, dirname, fname)))
            next!(prog)
        end
        energies = [record.energy for record in data]
        @assert length(unique!(energies)) == 1
    end
    println("Verified that all isometric database" *
            " entries have identical energies.")
end


function main()
    # verify_unique_energies()
    summary_data = Dict{Int,Vector{Tuple{Float64,String,Int}}}()
    @showprogress for dirname in readdir(PCREO_DATABASE_DIRECTORY)
        dirpath = joinpath(PCREO_DATABASE_DIRECTORY, dirname)
        filenames = readdir(dirpath)
        count = length(filenames)
        record = PCREORecord(joinpath(dirpath, first(filenames)))
        if record.num_points in keys(summary_data)
            push!(summary_data[record.num_points],
                  (record.energy, dirname, count))
        else
            summary_data[record.num_points] = [(record.energy, dirname, count)]
        end
    end
    for k in sort!(collect(keys(summary_data)))
        v = summary_data[k]
        sort!(v)
        total_count = sum(n for (_, _, n) in v)
        if length(v) > 1
            total_energy = sum(e * n for (e, _, n) in v)
            mean_energy = total_energy / total_count
            total_deviation = sum((e - mean_energy)^2 * n for (e, _, n) in v)
            stdev = sqrt(total_deviation / (total_count - 1))
        else
            mean_energy = v[1][1]
            stdev = 1.0
        end
        println("\n", k, " =======================================",
                "============================= ", total_count)
        for (e, s, n) in v
            z = (e - mean_energy) / stdev
            @printf("% 8.4f    %s    %4d\n", z, s, n)
        end
    end
end


main()
