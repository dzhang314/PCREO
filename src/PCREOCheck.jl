# Read all entries in PCREO_DATABASE_DIRECTORY and verify that they are
# syntactically valid PCREO files. This performs some basic sanity checks
# (e.g., floating-point numbers and integers can be correctly parsed, the
# advertised dimension and number of points in each configuration are correct)
# but does not recompute points configurations, energies, or nearest neighbors.


using ProgressMeter

push!(LOAD_PATH, @__DIR__)
using PCREO


function main()
    @assert isdir(ENV["PCREO_DATABASE_DIRECTORY"])
    for num_dir in lsdir(ENV["PCREO_DATABASE_DIRECTORY"]; join=true)
        println(num_dir)
        flush(stdout)
        @showprogress for entry_dir in lsdir(num_dir)
            for record_path in lsdir(num_dir, entry_dir; join=true)
                old = read(record_path, String)
                new = string(PCREORecord(record_path))
                if length(old) != length(new)
                    error(record_path)
                end
                for i = 1 : length(new)
                    if !(old[i] == new[i] ||
                         (old[i:i+3] == "-0.0" && new[i:i+3] == "+0.0"))
                        error(record_path)
                    end
                end
                write(record_path, new)
            end
        end
    end
end


main()
