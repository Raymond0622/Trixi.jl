using Printf
using Trixi

# Density-wave checkerboard, no ECAV (hyperbolic only). polydeg = 3.
elixir = joinpath(@__DIR__, "elixir_ecav_2d_density_wave.jl")
iterations = 3
outdir = joinpath(@__DIR__, "out_density_wave_no_ecav")
mkpath(outdir)

cases = (
    (:l2, "l2", "gauss_lobatto", "gauss_lobatto", "MortarL2"),
    (:entropy_lgl, "entropy", "gauss_lobatto", "gauss_lobatto", "Entropy LGL"),
    (:entropy_gauss, "entropy", "gauss", "gauss", "Entropy Gauss"),
)

println("Convergence (shock_capturing = none, fluxdiff flux_ranocha, checkerboard density wave)")
means = Dict{Symbol, Any}()
for (key, mortar_type, nodes, reverse_quad, title) in cases
    println("="^80)
    println(title)
    println("="^80)
    eocs, _ = convergence_test(elixir, iterations;
                               shock_capturing = "none",
                               volume_form = "fluxdiff",
                               mortar_type = mortar_type,
                               mortar_nodes = nodes,
                               reverse_quad = reverse_quad,
                               time_integrator = "ck54",
                               cfl = 0.5,
                               tspan = (0.0, 0.1))
    means[key] = Trixi.calc_mean_convergence(eocs)
end

println("\n", "="^80)
println("Mean EOC (polydeg = 3, expected ≈ 4), density wave, no ECAV, fluxdiff")
println("="^80)
println("                    rho     rho_v1  rho_v2  rho_e")
@printf("L2  MortarL2        %.2f    %.2f    %.2f    %.2f\n", means[:l2][:l2]...)
@printf("L2  Entropy LGL     %.2f    %.2f    %.2f    %.2f\n", means[:entropy_lgl][:l2]...)
@printf("L2  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:entropy_gauss][:l2]...)
@printf("L∞  MortarL2        %.2f    %.2f    %.2f    %.2f\n", means[:l2][:linf]...)
@printf("L∞  Entropy LGL     %.2f    %.2f    %.2f    %.2f\n", means[:entropy_lgl][:linf]...)
@printf("L∞  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:entropy_gauss][:linf]...)

open(joinpath(outdir, "eoc.txt"), "w") do io
    println(io, "Mean EOC density wave no ECAV")
    println(io, "l2_l2 ", join(means[:l2][:l2], " "))
    println(io, "entropy_lgl_l2 ", join(means[:entropy_lgl][:l2], " "))
    println(io, "entropy_gauss_l2 ", join(means[:entropy_gauss][:l2], " "))
    println(io, "l2_linf ", join(means[:l2][:linf], " "))
    println(io, "entropy_lgl_linf ", join(means[:entropy_lgl][:linf], " "))
    println(io, "entropy_gauss_linf ", join(means[:entropy_gauss][:linf], " "))
end

# Entropy integral vs time at the elixir default mesh (level 3 checkerboard).
println("\nEntropy integral time series (no ECAV, initial_refinement_level = 3)")
series = Dict{Symbol, NamedTuple}()
for (key, mortar_type, nodes, reverse_quad, title) in cases
    case_dir = joinpath(outdir, String(key))
    mkpath(case_dir)
    try
        trixi_include(elixir;
                      shock_capturing = "none",
                      volume_form = "fluxdiff",
                      mortar_type = mortar_type,
                      mortar_nodes = nodes,
                      reverse_quad = reverse_quad,
                      time_integrator = "ck54",
                      cfl = 0.5,
                      saveat = collect(range(0.0, 0.2; length = 41)))
        times = collect(sol.t)
        entropies = [Trixi.integrate(entropy, u, semi) for u in sol.u]
        series[key] = (t = times, S = entropies, title = title)
        println(title, ": n = ", length(times),
                "  S(0) = ", entropies[1],
                "  S(end) = ", entropies[end],
                "  ΔS = ", entropies[end] - entropies[1])
    catch e
        @warn "entropy series failed" title exception = (e, catch_backtrace())
    end
end

open(joinpath(outdir, "entropy_series.tsv"), "w") do io
    println(io, "case\tt\tentropy")
    for key in (:l2, :entropy_lgl, :entropy_gauss)
        haskey(series, key) || continue
        s = series[key]
        for (t, S) in zip(s.t, s.S)
            println(io, s.title, "\t", t, "\t", S)
        end
    end
end

try
    using Plots
    plt = plot(xlabel = "t", ylabel = "∫ S dV",
               title = "Density wave entropy integral (no ECAV, checkerboard, level 3)",
               legend = :best)
    for key in (:l2, :entropy_lgl, :entropy_gauss)
        haskey(series, key) || continue
        s = series[key]
        plot!(plt, s.t, s.S, label = s.title)
    end
    pngpath = joinpath(outdir, "entropy_integral.png")
    savefig(plt, pngpath)
    println("saved ", pngpath)
catch e
    println("Plots.jl not available; TSV written. (", sprint(showerror, e), ")")
end
