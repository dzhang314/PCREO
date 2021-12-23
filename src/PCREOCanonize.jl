# Read all entries in PCREO_DATABASE_DIRECTORY, compute canonical
# representatives of their nearest-neighbor graphs, and store the resulting
# graphs in PCREO_GRAPH_DIRECTORY in graph6 format. Does not recompute graphs
# that already exist in PCREO_GRAPH_DIRECTORY.


using Base.Threads: @threads, threadid, nthreads
using ProgressMeter

push!(LOAD_PATH, @__DIR__)
using PCREO


function main()

    @assert isdir(ENV["PCREO_DATABASE_DIRECTORY"])
    @assert isdir(ENV["PCREO_GRAPH_DIRECTORY"])

    for num_dir in lsdir(ENV["PCREO_DATABASE_DIRECTORY"]; join=true)

        println("Reading PCREO database directory $num_dir...")
        flush(stdout)

        entries = String[]

        @showprogress for entry_dir in lsdir(num_dir; join=true)
            for entry_path in lsdir(entry_dir; join=true)
                push!(entries, entry_path)
            end
        end

        println("Generating canonical nearest-neighbor graphs for $num_dir...")
        flush(stdout)

        p = Progress(length(entries))
        @threads for entry_path in entries
            entry_name = basename(entry_path)
            @assert endswith(entry_name, ".csv")
            graph_path = joinpath(ENV["PCREO_GRAPH_DIRECTORY"],
                                  entry_name[1:end-3] * "g6")
            if !isfile(graph_path)
                write(graph_path, canonical_graph6(entry_path))
            end
            sleep(0.01)
            next!(p)
        end
    end
end


main()
