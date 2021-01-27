using Printf: @sprintf

push!(LOAD_PATH, @__DIR__)
using PCREO

const IMAGE_URL_PREFIX = "http://web.stanford.edu/~dkzhang/PCREOImages/"
const DATABASE_URL_PREFIX = "http://web.stanford.edu/~dkzhang/PCREODatabase/"
const HTML_OUTPUT_DIRECTORY = "C:\\Users\\Zhang\\Documents\\GitHub\\my-madoko-docs\\PersonalWebsite\\PCREOTables"


function point_group_string(points)
    group_name = point_group(points)
    if group_name == "C_1"
        return ""
    elseif '_' in group_name
        a, b = split(group_name, '_')
        return "$a<sub>$b</sub>"
    end
    return group_name
end


function sigfigs_string(x::Float64)
    @assert x >= 0.0
    if x == 0.0
        return "0.000000000000"
    elseif x < 10.0
        return @sprintf("%.12f", x)
    elseif x < 100.0
        return @sprintf("%.11f", x)
    elseif x < 1000.0
        return @sprintf("%.10f", x)
    elseif x < 10000.0
        return @sprintf("%.9f", x)
    elseif x < 100000.0
        return @sprintf("%.8f", x)
    elseif x < 1000000.0
        return @sprintf("%.7f", x)
    elseif x < 10000000.0
        return @sprintf("%.6f", x)
    elseif x < 100000000.0
        return @sprintf("%.5f", x)
    elseif x < 1000000000.0
        return @sprintf("%.4f", x)
    elseif x < 10000000000.0
        return @sprintf("%.3f", x)
    elseif x < 100000000000.0
        return @sprintf("%.2f", x)
    elseif x < 1000000000000.0
        return @sprintf("%.1f", x)
    else
        @assert false
    end
end


function percentage_string(x::Float64)
    @assert 0.0 <= x <= 100.0
    if x == 0.0
        return "0.000%"
    elseif x == 100.0
        return "100.0%"
    elseif x < 10.0
        return @sprintf("%.3f", x) * '%'
    else
        return @sprintf("%.2f", x) * '%'
    end
end


const TABLE_HEADER = """
<table class="table table-bordered table-hover table-sm">
  <thead>
    <tr>
      <th scope="col">Image</th>
      <th scope="col">Z-score</th>
      <th scope="col">Energy</th>
      <th scope="col">Sym.</th>
      <th scope="col">Frequency</th>
      <th scope="col">PCREO Database ID</th>
    </tr>
  </thead>
  <tbody>
"""


const TABLE_FOOTER = """
  </tbody>
</table>
"""


function main()
    for num_points = 50 : 99
        ids = filter(startswith("PCREO-03-" * lpad(num_points, 4, '0')),
                     readdir(PCREO_DATABASE_DIRECTORY))
        @assert !isempty(ids)
        records = Dict(id => PCREORecord(id) for id in ids)
        energies = sort!([
            (records[id].energy, id, length(readdir(
                joinpath(PCREO_DATABASE_DIRECTORY, id))))
            for id in ids])
        total_count = sum(n for (_, _, n) in energies)
        if length(energies) > 1
            total_energy = sum(n * e for (e, _, n) in energies)
            mean_energy = total_energy / total_count
            total_deviation = sum(n * (e - mean_energy)^2
                                  for (e, _, n) in energies)
            stdev_energy = sqrt(total_deviation / (total_count - 1))
        else
            mean_energy = first(energies)[1]
            stdev_energy = 1.0
        end
        table_entries = String[]
        for (energy, id, freq) in energies
            z_score = (energy - mean_energy) / stdev_energy
            z_score_string = @sprintf("%+.3f", z_score)
            defects = defect_classes(records[id].facets)
            defect_strings = [
                string(num) * "&#8239;&times;&#8239;" * html_defect_string(shapes)
                for (num, shapes) in defects]
            push!(table_entries, """
                <tr height="30px">
                  <td rowspan="2"><a href="$IMAGE_URL_PREFIX$id.gif">
                    <img src="$IMAGE_URL_PREFIX$id.gif" width="100px">
                  </a></td>
                  <td>$z_score_string</td>
                  <td>$(sigfigs_string(energy))</td>
                  <td>$(point_group_string(records[id].points))</td>
                  <td>$freq ($(percentage_string(100 * freq / total_count)))</td>
                  <td class="text-monospace">
                    <a href="$DATABASE_URL_PREFIX$id.txt">$(id[15:end])</a>
                  </td>
                </tr>
                <tr>
                  <td colspan="5">$(join(defect_strings, "&emsp;"))</td>
                </tr>
            """)
        end
        open(joinpath(HTML_OUTPUT_DIRECTORY, "$num_points.html"), "w+") do io
            println(io, "<h5>$num_points Points</h5>")
            print(io, TABLE_HEADER)
            for entry in table_entries
                print(io, entry)
            end
            print(io, TABLE_FOOTER)
        end
    end
end


main()
