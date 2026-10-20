# Fill the one missed entropy series, then rebuild combined plots from TSV/CSV.
using Trixi
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Plots
using DelimitedFiles

const OUTDIR = joinpath(@__DIR__, "out_sweep_64")
const POLYDEGS = (1, 2, 3, 4)

elixir = joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl")
trixi_include(@__MODULE__, elixir;
              polydeg = 1, mortar_type = "entropy", mortar_nodes = "gauss",
              reverse_quad = "gauss", checkerboard = false,
              shock_capturing = "none", tspan = (0.0, 0.4),
              saveat_times = 0.02, initial_refinement_level = 4,
              make_plots = false)

sol_ = @invokelatest getfield(Main, :sol)
semi_ = @invokelatest getfield(Main, :semi)
tvec = collect(sol_.t)
Svec = [Trixi.integrate(Trixi.entropy, u, semi_) for u in sol_.u]
S0, Send = Svec[1], Svec[end]
delta_S = Send - S0
println("filled density_wave_conforming_none_gauss_N1  ΔS = ", delta_S)

case = "density_wave_conforming_none_gauss_N1"
open(joinpath(OUTDIR, "entropy_series.tsv"), "a") do io
    for (t, S) in zip(tvec, Svec)
        println(io, join((case, "density_wave", "conforming", "none", "gauss",
                          "1", string(t), string(S), string(S - S0)), '\t'))
    end
end

eoc_path = joinpath(OUTDIR, "eoc.csv")
lines = readlines(eoc_path)
open(eoc_path, "w") do io
    for line in lines
        if startswith(line, case * ",")
            parts = split(line, ',')
            # status, ΔS, S0, Send sit at the end: ...,status, means..., ΔS, S0, Send, elapsed
            parts[7] = "ok"
            parts[end - 3] = string(delta_S)
            parts[end - 2] = string(S0)
            parts[end - 1] = string(Send)
            println(io, join(parts, ','))
        else
            println(io, line)
        end
    end
end

# Rebuild combined plots from TSV
raw = readdlm(joinpath(OUTDIR, "entropy_series.tsv"), '\t', header = true)
data, header = raw
# columns: case problem mesh capturing nodes polydeg t S delta_S
entropy_series = Dict{String, NamedTuple}()
for i in 1:size(data, 1)
    c = String(data[i, 1])
    t = Float64(data[i, 7])
    S = Float64(data[i, 8])
    if !haskey(entropy_series, c)
        entropy_series[c] = (problem = String(data[i, 2]),
                             mesh = String(data[i, 3]),
                             capturing = String(data[i, 4]),
                             nodes = String(data[i, 5]),
                             polydeg = Int(data[i, 6]),
                             t = Float64[], S = Float64[])
    end
    s = entropy_series[c]
    push!(s.t, t)
    push!(s.S, S)
end

eoc_raw = readdlm(eoc_path, ',', header = true)
ed, eh = eoc_raw
eoc_rows = NamedTuple[]
for i in 1:size(ed, 1)
    push!(eoc_rows,
          (problem = String(ed[i, 2]), mesh = String(ed[i, 3]),
           capturing = String(ed[i, 4]), nodes = String(ed[i, 5]),
           polydeg = Int(ed[i, 6]), mean_l2 = Float64[ed[i, 8], ed[i, 9],
                                                      ed[i, 10], ed[i, 11]]))
end

default(linewidth = 1.6, legendfontsize = 7, tickfontsize = 8,
        guidefontsize = 9, titlefontsize = 10)
node_style = Dict("gauss" => :solid, "lgl" => :dash)
poly_color = Dict(1 => :black, 2 => :steelblue, 3 => :darkorange, 4 => :firebrick)

function plot_entropy_problem(problem)
    fig = plot(layout = (2, 2), size = (1400, 1000),
               plot_title = "$problem  entropy  ΔS(t) = ∫S(t) − ∫S(0)  (level 4)")
    for (mesh_name, capturing, i) in (("conforming", "none", 1),
                                      ("conforming", "ecav", 2),
                                      ("checkerboard", "none", 3),
                                      ("checkerboard", "ecav", 4))
        plt = fig[i]
        plot!(plt; xlabel = "t", ylabel = "ΔS",
              title = "$mesh_name  $capturing", legend = :outertopright)
        hline!(plt, [0.0]; color = :gray, linestyle = :dot, label = "")
        for nodes_name in ("gauss", "lgl"), N in POLYDEGS
            c = "$(problem)_$(mesh_name)_$(capturing)_$(nodes_name)_N$(N)"
            series = get(entropy_series, c, nothing)
            series === nothing && continue
            isempty(series.t) && continue
            plot!(plt, series.t, series.S .- series.S[1];
                  color = poly_color[N], linestyle = node_style[nodes_name],
                  label = "$(nodes_name) N=$N")
        end
    end
    return fig
end

function plot_entropy_grid(problem)
    fig = plot(layout = (4, 2), size = (1400, 1600),
               plot_title = "$problem  entropy ΔS(t)  each N / nodes")
    idx = 1
    for N in POLYDEGS, nodes_name in ("gauss", "lgl")
        plt = fig[idx]
        plot!(plt; xlabel = "t", ylabel = "ΔS",
              title = "$nodes_name  N=$N", legend = :outertopright)
        hline!(plt, [0.0]; color = :gray, linestyle = :dot, label = "")
        for mesh_name in ("conforming", "checkerboard"),
            capturing in ("none", "ecav")
            c = "$(problem)_$(mesh_name)_$(capturing)_$(nodes_name)_N$(N)"
            series = get(entropy_series, c, nothing)
            series === nothing && continue
            isempty(series.t) && continue
            ls = mesh_name == "conforming" ? :solid : :dash
            col = capturing == "ecav" ? :firebrick : :steelblue
            plot!(plt, series.t, series.S .- series.S[1];
                  color = col, linestyle = ls,
                  label = "$mesh_name $capturing")
        end
        idx += 1
    end
    return fig
end

function plot_eoc_problem(problem)
    fig = plot(layout = (2, 2), size = (1400, 1000),
               plot_title = "$problem  mean L² ρ EOC  (levels 3→5, t=0.1)")
    Ns = collect(POLYDEGS)
    for (mesh_name, capturing, i) in (("conforming", "none", 1),
                                      ("conforming", "ecav", 2),
                                      ("checkerboard", "none", 3),
                                      ("checkerboard", "ecav", 4))
        plt = fig[i]
        plot!(plt; xlabel = "N", ylabel = "mean L² ρ EOC",
              title = "$mesh_name  $capturing", legend = :topleft,
              xticks = Ns, ylims = (0, 6))
        plot!(plt, Ns, Ns .+ 1; color = :gray, linestyle = :dot, label = "N+1")
        for nodes_name in ("gauss", "lgl")
            y = Float64[]
            for N in POLYDEGS
                row = nothing
                for r in eoc_rows
                    if r.problem == problem && r.mesh == mesh_name &&
                       r.capturing == capturing && r.nodes == nodes_name &&
                       r.polydeg == N
                        row = r
                        break
                    end
                end
                push!(y, row === nothing ? NaN : row.mean_l2[1])
            end
            plot!(plt, Ns, y; marker = :circle,
                  color = nodes_name == "gauss" ? :steelblue : :darkorange,
                  linestyle = nodes_name == "gauss" ? :solid : :dash,
                  label = nodes_name)
        end
    end
    return fig
end

per_case_dir = joinpath(OUTDIR, "entropy_per_case")
mkpath(per_case_dir)
plt = plot(tvec, Svec; xlabel = "t", ylabel = "∫ S dV", title = case,
           legend = false, size = (700, 420))
savefig(plt, joinpath(per_case_dir, "$(case)_S.png"))
plt = plot(tvec, Svec .- S0; xlabel = "t", ylabel = "ΔS = ∫S(t) − ∫S(0)",
           title = case, legend = false, size = (700, 420))
hline!(plt, [0.0]; color = :gray, linestyle = :dot)
savefig(plt, joinpath(per_case_dir, "$(case)_dS.png"))

for problem in ("density_wave", "isentropic_vortex")
    fig = plot_entropy_problem(problem)
    savefig(fig, joinpath(OUTDIR, "entropy_$(problem).png"))
    savefig(fig, joinpath(OUTDIR, "entropy_$(problem).pdf"))
    fig = plot_entropy_grid(problem)
    savefig(fig, joinpath(OUTDIR, "entropy_grid_$(problem).png"))
    savefig(fig, joinpath(OUTDIR, "entropy_grid_$(problem).pdf"))
    fig = plot_eoc_problem(problem)
    savefig(fig, joinpath(OUTDIR, "eoc_$(problem).png"))
    savefig(fig, joinpath(OUTDIR, "eoc_$(problem).pdf"))
end
println("replotted")
