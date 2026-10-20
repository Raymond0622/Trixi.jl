# 64-case EOC + entropy-integral sweep:
#   {density_wave, isentropic_vortex}
#   × {conforming, checkerboard}
#   × {ecav, none}
#   × {gauss, gauss_lobatto}
#   × polydeg {1, 2, 3, 4}
#
# EOC: levels 3→5, tspan (0, 0.1), save only the final time.
# Entropy series: level 4, tspan (0, 0.4), saveat 0.02.
#
# Run with:
#   JULIA_NUM_THREADS=1 julia --project=run examples/tree_2d_dgsem/run_sweep_64_eoc_entropy.jl

using Trixi
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Printf
using Plots

const OUTDIR = joinpath(@__DIR__, "out_sweep_64")
mkpath(OUTDIR)

const PROBLEMS = (
    ("density_wave", joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl")),
    ("isentropic_vortex", joinpath(@__DIR__, "elixir_ecav_2d_isentropic_vortex.jl")),
)
const MESHES = ((false, "conforming"), (true, "checkerboard"))
const CAPTURINGS = ("none", "ecav")
const NODES = (("gauss", "gauss", "gauss"),
               ("gauss_lobatto", "gauss_lobatto", "lgl"))
const POLYDEGS = (1, 2, 3, 4)

const EOC_ITERS = 3
const EOC_LEVEL = 3
const EOC_TSPAN = (0.0, 0.1)
const ENTROPY_LEVEL = 4
const ENTROPY_TSPAN = (0.0, 0.4)
const ENTROPY_SAVEAT = 0.02

function case_kwargs(polydeg, mortar_nodes, reverse_quad, checkerboard, shock_capturing;
                     tspan, saveat_times, initial_refinement_level)
    return (polydeg = polydeg,
            mortar_type = "entropy",
            mortar_nodes = mortar_nodes,
            reverse_quad = reverse_quad,
            checkerboard = checkerboard,
            shock_capturing = shock_capturing,
            tspan = tspan,
            saveat_times = saveat_times,
            initial_refinement_level = initial_refinement_level,
            make_plots = false)
end

function mean_eoc(eocs, kind, var)
    col = eocs[kind][:, var]
    return sum(col) / length(col)
end

function extract_entropy_series(sol, semi)
    t = collect(sol.t)
    S = [Trixi.integrate(Trixi.entropy, u, semi) for u in sol.u]
    return t, S
end

cases = collect(Iterators.product(PROBLEMS, MESHES, CAPTURINGS, NODES, POLYDEGS))
n_cases = length(cases)
@assert n_cases == 64

eoc_csv = joinpath(OUTDIR, "eoc.csv")
entropy_tsv = joinpath(OUTDIR, "entropy_series.tsv")
summary_txt = joinpath(OUTDIR, "summary.txt")
open(eoc_csv, "w") do io
    println(io,
            "case,problem,mesh,capturing,nodes,polydeg,status,",
            "mean_l2_rho,mean_l2_rho_v1,mean_l2_rho_v2,mean_l2_rho_e,",
            "mean_linf_rho,",
            "eoc_l2_rho_1,eoc_l2_rho_2,",
            "err_l2_rho_lev0,err_l2_rho_lev1,err_l2_rho_lev2,",
            "delta_S,S0,Send,elapsed_s")
end
open(entropy_tsv, "w") do io
    println(io, "case\tproblem\tmesh\tcapturing\tnodes\tpolydeg\tt\tS\tdelta_S")
end

entropy_series = Dict{String, NamedTuple}()
eoc_rows = NamedTuple[]

wall0 = time()
for (icase, ((problem, elixir), (checkerboard, mesh_name), capturing,
             (mortar_nodes, reverse_quad, nodes_name), polydeg)) in enumerate(cases)
    case = "$(problem)_$(mesh_name)_$(capturing)_$(nodes_name)_N$(polydeg)"
    println()
    println("="^100)
    println("CASE $icase / $n_cases  $case")
    println("="^100)
    flush(stdout)

    t_case = time()
    status = "ok"
    mean_l2 = fill(NaN, 4)
    mean_linf_rho = NaN
    eoc_l2_rho = fill(NaN, 2)
    err_l2_rho = fill(NaN, 3)
    S0 = NaN
    Send = NaN
    delta_S = NaN
    tvec = Float64[]
    Svec = Float64[]

    try
        kw_eoc = case_kwargs(polydeg, mortar_nodes, reverse_quad, checkerboard,
                             capturing;
                             tspan = EOC_TSPAN, saveat_times = EOC_TSPAN[2],
                             initial_refinement_level = EOC_LEVEL)
        eocs, errorsmatrix = convergence_test(elixir, EOC_ITERS; kw_eoc...)
        for v in 1:4
            mean_l2[v] = mean_eoc(eocs, :l2, v)
        end
        mean_linf_rho = mean_eoc(eocs, :linf, 1)
        eoc_l2_rho[1] = eocs[:l2][1, 1]
        eoc_l2_rho[2] = eocs[:l2][2, 1]
        err_l2_rho[1] = errorsmatrix[:l2][1, 1]
        err_l2_rho[2] = errorsmatrix[:l2][2, 1]
        err_l2_rho[3] = errorsmatrix[:l2][3, 1]

        kw_ent = case_kwargs(polydeg, mortar_nodes, reverse_quad, checkerboard,
                             capturing;
                             tspan = ENTROPY_TSPAN, saveat_times = ENTROPY_SAVEAT,
                             initial_refinement_level = ENTROPY_LEVEL)
        trixi_include(@__MODULE__, elixir; kw_ent...)
        # Look up `sol`/`semi` in the latest world. Passing the names as
        # arguments to `@invokelatest` still evaluates them in the caller.
        tvec, Svec = extract_entropy_series(@invokelatest(getfield(Main, :sol)),
                                            @invokelatest(getfield(Main, :semi)))
        S0 = Svec[1]
        Send = Svec[end]
        delta_S = Send - S0
    catch err
        status = "FAIL: $(typeof(err)): $(err)"
        @error "case failed" case exception = (err, catch_backtrace())
    end

    elapsed = time() - t_case
    entropy_series[case] = (problem = problem, mesh = mesh_name,
                            capturing = capturing, nodes = nodes_name,
                            polydeg = polydeg, t = tvec, S = Svec,
                            status = status)
    row = (case = case, problem = problem, mesh = mesh_name,
           capturing = capturing, nodes = nodes_name, polydeg = polydeg,
           status = status, mean_l2 = copy(mean_l2),
           mean_linf_rho = mean_linf_rho, eoc_l2_rho = copy(eoc_l2_rho),
           err_l2_rho = copy(err_l2_rho), delta_S = delta_S, S0 = S0,
           Send = Send, elapsed = elapsed)
    push!(eoc_rows, row)

    open(eoc_csv, "a") do io
        @printf(io,
                "%s,%s,%s,%s,%s,%d,%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6e,%.6e,%.6e,%.6e,%.16e,%.16e,%.1f\n",
                case, problem, mesh_name, capturing, nodes_name, polydeg,
                replace(status, ',' => ';'),
                mean_l2[1], mean_l2[2], mean_l2[3], mean_l2[4], mean_linf_rho,
                eoc_l2_rho[1], eoc_l2_rho[2],
                err_l2_rho[1], err_l2_rho[2], err_l2_rho[3],
                delta_S, S0, Send, elapsed)
    end
    open(entropy_tsv, "a") do io
        if !isempty(tvec)
            for (t, S) in zip(tvec, Svec)
                println(io,
                        join((case, problem, mesh_name, capturing, nodes_name,
                              string(polydeg), string(t), string(S),
                              string(S - Svec[1])), '\t'))
            end
        end
    end

    remaining = n_cases - icase
    rate = (time() - wall0) / icase
    @printf("  status=%s  mean L2 ρ EOC=%.3f  ΔS=%.3e  elapsed=%.0fs  ETA=%.0f min\n",
            status, mean_l2[1], delta_S, elapsed, remaining * rate / 60)
    flush(stdout)
end

# ---------------------------------------------------------------------------
# Plots: entropy ΔS(t) and EOC vs N
# ---------------------------------------------------------------------------
default(linewidth = 1.6, legendfontsize = 7, tickfontsize = 8,
        guidefontsize = 9, titlefontsize = 10)

node_style = Dict("gauss" => :solid, "lgl" => :dash)
poly_color = Dict(1 => :black, 2 => :steelblue, 3 => :darkorange, 4 => :firebrick)

function plot_entropy_problem(problem)
    fig = plot(layout = (2, 2), size = (1400, 1000),
               plot_title = "$problem  entropy  ΔS(t) = ∫S(t) − ∫S(0)  (level $ENTROPY_LEVEL)")
    panels = (("conforming", "none", 1),
              ("conforming", "ecav", 2),
              ("checkerboard", "none", 3),
              ("checkerboard", "ecav", 4))
    for (mesh_name, capturing, i) in panels
        plt = fig[i]
        plot!(plt; xlabel = "t", ylabel = "ΔS",
              title = "$mesh_name  $capturing", legend = :outertopright)
        hline!(plt, [0.0]; color = :gray, linestyle = :dot, label = "")
        for nodes_name in ("gauss", "lgl"), N in POLYDEGS
            case = "$(problem)_$(mesh_name)_$(capturing)_$(nodes_name)_N$(N)"
            series = get(entropy_series, case, nothing)
            if series === nothing || isempty(series.t)
                continue
            end
            dS = series.S .- series.S[1]
            plot!(plt, series.t, dS;
                  color = poly_color[N], linestyle = node_style[nodes_name],
                  label = "$(nodes_name) N=$N")
        end
    end
    return fig
end

function plot_entropy_grid(problem)
    # One panel per N × nodes, 4 curves (mesh × capturing) — 8 panels.
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
            case = "$(problem)_$(mesh_name)_$(capturing)_$(nodes_name)_N$(N)"
            series = get(entropy_series, case, nothing)
            if series === nothing || isempty(series.t)
                continue
            end
            dS = series.S .- series.S[1]
            ls = mesh_name == "conforming" ? :solid : :dash
            col = capturing == "ecav" ? :firebrick : :steelblue
            plot!(plt, series.t, dS; color = col, linestyle = ls,
                  label = "$mesh_name $capturing")
        end
        idx += 1
    end
    return fig
end

function plot_eoc_problem(problem)
    fig = plot(layout = (2, 2), size = (1400, 1000),
               plot_title = "$problem  mean L² ρ EOC  (levels $EOC_LEVEL→$(EOC_LEVEL + EOC_ITERS - 1), t=$(EOC_TSPAN[2]))")
    panels = (("conforming", "none", 1),
              ("conforming", "ecav", 2),
              ("checkerboard", "none", 3),
              ("checkerboard", "ecav", 4))
    Ns = collect(POLYDEGS)
    for (mesh_name, capturing, i) in panels
        plt = fig[i]
        plot!(plt; xlabel = "N", ylabel = "mean L² ρ EOC",
              title = "$mesh_name  $capturing", legend = :topleft,
              xticks = Ns, ylims = (0, 6))
        plot!(plt, Ns, Ns .+ 1; color = :gray, linestyle = :dot,
              label = "N+1")
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

for (problem, _) in PROBLEMS
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

# Per-case entropy PNG (64 files)
per_case_dir = joinpath(OUTDIR, "entropy_per_case")
mkpath(per_case_dir)
for (case, series) in entropy_series
    if isempty(series.t)
        continue
    end
    dS = series.S .- series.S[1]
    plt = plot(series.t, series.S; xlabel = "t", ylabel = "∫ S dV",
               title = case, legend = false, size = (700, 420))
    savefig(plt, joinpath(per_case_dir, "$(case)_S.png"))
    plt = plot(series.t, dS; xlabel = "t", ylabel = "ΔS = ∫S(t) − ∫S(0)",
               title = case, legend = false, size = (700, 420))
    hline!(plt, [0.0]; color = :gray, linestyle = :dot)
    savefig(plt, joinpath(per_case_dir, "$(case)_dS.png"))
end

# Text summary
open(summary_txt, "w") do io
    println(io, "64-case EOC + entropy sweep")
    println(io, "EOC: level $EOC_LEVEL, $EOC_ITERS iterations, tspan $EOC_TSPAN")
    println(io, "Entropy: level $ENTROPY_LEVEL, tspan $ENTROPY_TSPAN, saveat $ENTROPY_SAVEAT")
    println(io, "Wall time: $(round((time() - wall0) / 60, digits = 1)) min")
    println(io)
    for (problem, _) in PROBLEMS
        println(io, "="^100)
        println(io, problem)
        println(io, "="^100)
        @printf(io, "%-14s %-12s %-6s %3s  %8s %8s %8s %8s  %12s %12s\n",
                "mesh", "capturing", "nodes", "N", "EOC L2ρ", "EOC L2m1",
                "EOC L2m2", "EOC L2e", "ΔS", "status")
        for r in eoc_rows
            r.problem == problem || continue
            @printf(io, "%-14s %-12s %-6s %3d  %8.3f %8.3f %8.3f %8.3f  %12.3e %s\n",
                    r.mesh, r.capturing, r.nodes, r.polydeg,
                    r.mean_l2[1], r.mean_l2[2], r.mean_l2[3], r.mean_l2[4],
                    r.delta_S, r.status)
        end
        println(io)
    end
end

println()
println("Wrote $OUTDIR")
println(read(summary_txt, String))
