# Read all entries in PCREO_DATABASE_DIRECTORY, compute canonical
# representatives of their nearest-neighbor graphs, and store the resulting
# graphs in PCREO_GRAPH_DIRECTORY in graph6 format. Does not recompute graphs
# that already exist in PCREO_GRAPH_DIRECTORY.


using Base.Threads: @threads, threadid, nthreads
using ProgressMeter
using UUIDs: UUID


push!(LOAD_PATH, @__DIR__)
using PCREO


function get_or_compute_canonical_graph(entrypath::AbstractString,
                                        graphpath::AbstractString)
    if isfile(graphpath)
        return readchomp(graphpath)
    else
        result = canonical_graph6(entrypath)
        write(graphpath, result, '\n')
        return result
    end
end


function main()

    println("Using $(nthreads()) threads.")
    flush(stdout)

    @assert isdir(ENV["PCREO_DATABASE_DIRECTORY"])
    @assert isdir(ENV["PCREO_GRAPH_DIRECTORY"])

    for num_dir in lsdir(ENV["PCREO_DATABASE_DIRECTORY"]; join=true)

        println("Reading PCREO database directory $num_dir...")
        flush(stdout)

        entry_paths = [
            entry_path
            for entry_dir in lsdir(num_dir; join=true)
            for entry_path in lsdir(entry_dir; join=true)
        ]

        n = length(entry_paths)
        canonical_graphs = Vector{String}(undef, n)
        uuids = Vector{UUID}(undef, n)

        println("Loading canonical nearest-neighbor graphs...")
        flush(stdout)

        p = Progress(n; dt=0.05, desc=basename(num_dir),
                        output=stdout, showspeed=true)

        @threads for i = 1 : n
            @inbounds entry = entry_paths[i]
            path = splitpath(entry)
            @inbounds uuids[i] = UUID(path[end-1])
            @inbounds name = path[end]
            @assert endswith(name, ".csv")
            @inbounds canonical_graphs[i] = get_or_compute_canonical_graph(
                entry,
                joinpath(ENV["PCREO_GRAPH_DIRECTORY"], name[1:end-3] * "g6")
            )
            next!(p)
        end

        flush(stdout)
        println("Verifying uniqueness...")
        flush(stdout)

        graph_dict = Dict{UUID,Set{String}}()
        for (graph, uuid) in zip(canonical_graphs, uuids)
            if haskey(graph_dict, uuid)
                push!(graph_dict[uuid], graph)
            else
                new_set = Set{String}()
                push!(new_set, graph)
                graph_dict[uuid] = new_set
            end
        end

        for (uuid, graph_set) in graph_dict
            @assert length(graph_set) == 1
        end

        @assert allunique(
            graph
            for (uuid, graph_set) in graph_dict
            for graph in graph_set
        )
    end
end


main()
