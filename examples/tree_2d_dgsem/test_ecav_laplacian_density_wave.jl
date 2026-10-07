using LinearAlgebra: I
using Printf
using Trixi

# Isolate the ECAV viscous operator from `rhs_combined!` in
# `dg_2d_artificial_viscosity.jl`, with coefficient left at 1 (skip
# `calc_ecav_coefficients!`). Same calls, same equation objects:
#   transform_variables!(..., equations_parabolic)          # cons → entropy
#   calc_gradient!(..., equations_parabolic, ...)           # BR1 ∇w
#   calc_parabolic_fluxes!(..., equations_artificial_viscosity, ...)
#       # g = κ (∂u/∂w) ∇w, κ = 1
#   skip accum_viscous_fluxes!                              # NS μ = 0, not ECAV
#   calc_divergence!(..., equations_parabolic, ...)
#   apply_jacobian_parabolic! then negate                   # combined-RHS −1/J
#
# Full divergence: g = (g_x, g_y), one `calc_divergence!`.
# Density-wave analytic: (∂u/∂w)∇w = ∇u, so the density slot is Δρ = ρ_xx + ρ_yy.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
eqs_av = Trixi.default_artificial_viscosity(equations)

function initial_condition_density_wave_ecav(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 1 + convert(RealT, 0.5) * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e_total = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end

analytic_rho_x(x, y) = π * cospi(2 * (x + y))
analytic_rho_y(x, y) = π * cospi(2 * (x + y))
analytic_rho_xx(x, y) = -2 * π^2 * sinpi(2 * (x + y))
analytic_rho_yy(x, y) = -2 * π^2 * sinpi(2 * (x + y))
analytic_laplace_rho(x, y) = analytic_rho_xx(x, y) + analytic_rho_yy(x, y)

function make_semi(; checkerboard, mortar_type, initial_refinement_level = 7)
    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)
    surface_flux = flux_ranocha
    volume_integral = VolumeIntegralWeakForm()
    if mortar_type == "l2"
        mortar = MortarL2(basis)
    else
        mortar = MortarEntropy(basis; nodes = :gauss,
                               reverse_quadrature = :gauss)
    end
    surface_integral = SurfaceIntegralWeakFormGauss(surface_flux)
    solver = DGSEM(basis, surface_integral, volume_integral, mortar)
    solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()

    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level = initial_refinement_level,
                    n_cells_max = 400_000, periodicity = true)
    if checkerboard
        level = mesh.tree.levels[first(Trixi.leaf_cells(mesh.tree))]
        dx = (coordinates_max[1] - coordinates_min[1]) / 2^level
        cells_to_refine = Int[]
        for cell_id in Trixi.leaf_cells(mesh.tree)
            x, y = Trixi.cell_coordinates(mesh.tree, cell_id)
            ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
            iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
            if iseven(ix + iy)
                push!(cells_to_refine, cell_id)
            end
        end
        Trixi.refine!(mesh.tree, cells_to_refine)
    end

    VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
    filter = ones(polydeg + 1)
    return SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition_density_wave_ecav,
                                                 solver;
                                                 VDM = VDM, filter = filter,
                                                 ecav_choice = :ecav,
                                                 combine_rhs = Trixi.True(),
                                                 equations_artificial_viscosity = eqs_av,
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
end

function weighted_errors(num, exact, semi)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    l2acc = 0.0
    linf = 0.0
    for element in Trixi.eachelement(dg, cache)
        J1 = inv(cache.elements.inverse_jacobian[element])
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            w = dg.basis.weights[i] * dg.basis.weights[j] * J1^2
            err = abs(num[i, j, element] - exact[i, j, element])
            l2acc += w * err^2
            linf = max(linf, err)
        end
    end
    return sqrt(l2acc), linf
end

function fill_exact!(dest, analytic, semi)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    x = cache.elements.node_coordinates
    for element in Trixi.eachelement(dg, cache)
        for j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            dest[i, j, element] = analytic(x[1, i, j, element], x[2, i, j, element])
        end
    end
    return dest
end

function reset_surface_cache!(cache)
    fill!(cache.elements.surface_flux_values, 0)
    fill!(cache.interfaces.u, 0)
    return nothing
end

# Full ECAV viscous operator (no directional split, κ = 1).
function full_divergence!(du, gradients, u_transformed, u, semi, eqs_av)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av, dg, cache)
    Trixi.set_zero!(du, dg, cache)
    Trixi.calc_divergence!(du, flux_parabolic, u, mesh, equations_parabolic,
                           semi.boundary_conditions_parabolic, dg,
                           semi.solver_parabolic, cache, 0.0)
    Trixi.apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    return nothing
end

function run_case(title; checkerboard, mortar_type, initial_refinement_level)
    println("="^72)
    println(title)
    println("="^72)
    semi = make_semi(checkerboard = checkerboard, mortar_type = mortar_type,
                     initial_refinement_level = initial_refinement_level)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi);
    println("level = ", initial_refinement_level,
            "  nelements = ", Trixi.nelements(dg, cache),
            "  nmortars = ", Trixi.nmortars(dg, cache))
    println("equations_parabolic = ", typeof(semi.equations_parabolic).name.name)
    println("equations_AV       = ",
            typeof(semi.artificial_viscosity.equations_artificial_viscosity).name.name)
    println("mortar             = ", typeof(dg.mortar).name.name,
            "  nodes = ", Trixi.mortar_nodes(dg.mortar))

    u_ode = Trixi.compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    du = Trixi.wrap_array(similar(u_ode), semi)

    (; u_transformed, gradients) = semi.cache_parabolic.parabolic_container
    eqs_av_semi = semi.artificial_viscosity.equations_artificial_viscosity

    Trixi.transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
    Trixi.calc_gradient!(gradients, u_transformed, 0.0, mesh, equations_parabolic,
                         semi.boundary_conditions_parabolic, dg,
                         semi.solver_parabolic, cache)

    n = Trixi.nnodes(dg)
    ne = Trixi.nelements(dg, cache)
    flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av_semi, dg, cache)
    gx, gy = flux_parabolic

    rho_x_num = Array{Float64}(undef, n, n, ne)
    rho_y_num = similar(rho_x_num)
    rho_x_ex = similar(rho_x_num)
    rho_y_ex = similar(rho_x_num)
    div_num = similar(rho_x_num)
    div_ex = similar(rho_x_num)
    rho_x_num .= gx[1, :, :, :]
    rho_y_num .= gy[1, :, :, :]
    fill_exact!(rho_x_ex, analytic_rho_x, semi)
    fill_exact!(rho_y_ex, analytic_rho_y, semi)
    fill_exact!(div_ex, (x, y) -> -analytic_laplace_rho(x, y), semi)

    full_divergence!(du, gradients, u_transformed, u, semi, eqs_av_semi)
    div_num .= -du[1, :, :, :]

    errors = Dict{String, NamedTuple}()
    @printf("%-10s  %12s  %12s  %10s  %10s  %12s  %12s  vs\n",
            "field", "L2", "L∞", "rel L2", "rel L∞", "max|num|", "max|ex|")
    names = ("ρ_x", "ρ_y", "Δρ")
    nums = (rho_x_num, rho_y_num, div_num)
    exs = (rho_x_ex, rho_y_ex, div_ex)
    labels = ("analytic ρ_x", "analytic ρ_y", "analytic −Δρ")
    for (name, num, ex, lab) in zip(names, nums, exs, labels)
        l2, linf = weighted_errors(num, ex, semi)
        l2ex, _ = weighted_errors(ex, zero(ex), semi)
        rel_l2 = l2 / max(l2ex, eps())
        rel_inf = linf / max(maximum(abs, ex), eps())
        errors[name] = (l2 = l2, linf = linf, rel_l2 = rel_l2, rel_inf = rel_inf)
        @printf("%-10s  %12.4e  %12.4e  %10.3e  %10.3e  %12.4e  %12.4e  %s\n",
                name, l2, linf, rel_l2, rel_inf, maximum(abs, num),
                maximum(abs, ex), lab)
    end
    return (; title, errors, div_num, div_ex, semi, initial_refinement_level)
end

function nodal_xy(semi)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    x = cache.elements.node_coordinates
    return vec(x[1, :, :, :]), vec(x[2, :, :, :])
end

function bin_abs_heatmap(x, y, val; n = 128, xlim = (-1.0, 1.0), ylim = (-1.0, 1.0))
    dx = (xlim[2] - xlim[1]) / n
    dy = (ylim[2] - ylim[1]) / n
    Z = fill(NaN, n, n)
    for k in eachindex(val)
        isfinite(val[k]) || continue
        i = clamp(floor(Int, (x[k] - xlim[1]) / dx) + 1, 1, n)
        j = clamp(floor(Int, (y[k] - ylim[1]) / dy) + 1, 1, n)
        a = abs(val[k])
        Z[j, i] = isnan(Z[j, i]) ? a : max(Z[j, i], a)
    end
    xs = range(xlim[1] + dx / 2, xlim[2] - dx / 2; length = n)
    ys = range(ylim[1] + dy / 2, ylim[2] - dy / 2; length = n)
    return xs, ys, Z
end

function bin_signed_absmax(x, y, val; n = 128, xlim = (-1.0, 1.0), ylim = (-1.0, 1.0))
    dx = (xlim[2] - xlim[1]) / n
    dy = (ylim[2] - ylim[1]) / n
    Z = fill(NaN, n, n)
    for k in eachindex(val)
        isfinite(val[k]) || continue
        i = clamp(floor(Int, (x[k] - xlim[1]) / dx) + 1, 1, n)
        j = clamp(floor(Int, (y[k] - ylim[1]) / dy) + 1, 1, n)
        if isnan(Z[j, i]) || abs(val[k]) > abs(Z[j, i])
            Z[j, i] = val[k]
        end
    end
    xs = range(xlim[1] + dx / 2, xlim[2] - dx / 2; length = n)
    ys = range(ylim[1] + dy / 2, ylim[2] - dy / 2; length = n)
    return xs, ys, Z
end

function bin_mean(x, y, val; n = 128, xlim = (-1.0, 1.0), ylim = (-1.0, 1.0))
    dx = (xlim[2] - xlim[1]) / n
    dy = (ylim[2] - ylim[1]) / n
    acc = zeros(n, n)
    cnt = zeros(Int, n, n)
    for k in eachindex(val)
        isfinite(val[k]) || continue
        i = clamp(floor(Int, (x[k] - xlim[1]) / dx) + 1, 1, n)
        j = clamp(floor(Int, (y[k] - ylim[1]) / dy) + 1, 1, n)
        acc[j, i] += val[k]
        cnt[j, i] += 1
    end
    Z = fill(NaN, n, n)
    for j in 1:n, i in 1:n
        if cnt[j, i] > 0
            Z[j, i] = acc[j, i] / cnt[j, i]
        end
    end
    xs = range(xlim[1] + dx / 2, xlim[2] - dx / 2; length = n)
    ys = range(ylim[1] + dy / 2, ylim[2] - dy / 2; length = n)
    return xs, ys, Z
end

cases = [
    ("conforming MortarEntropy", false, "entropy"),
    ("checkerboard MortarL2", true, "l2"),
    ("checkerboard MortarEntropy", true, "entropy"),
]

results = NamedTuple[]
# Compile viscous kernels on a tiny mesh first. Julia 1.12 can SIGSEGV/SIGILL
# while inferring `calc_divergence!` if the first mesh is already level 7.
run_case("warmup"; checkerboard = false, mortar_type = "entropy",
         initial_refinement_level = 2)
GC.gc()
for (title, cb, mt) in cases
    push!(results,
          run_case(title; checkerboard = cb, mortar_type = mt,
                   initial_refinement_level = 7))
    GC.gc()
end

outdir = joinpath(@__DIR__, "out_ecav_laplacian_density_wave")
mkpath(outdir)

open(joinpath(outdir, "errors.txt"), "w") do io
    println(io, "field\tcase\tL2\tLinf")
    for r in results
        e = r.errors["Δρ"]
        println(io, "Δρ\t", r.title, "\tlevel ", r.initial_refinement_level,
                "\t", e.l2, "\t", e.linf)
    end
end

try
    using Plots
    default(fontfamily = "sans-serif")

    grids = []
    for r in results
        x, y = nodal_xy(r.semi)
        err = vec(r.div_num) .- vec(r.div_ex)
        xs, ys, Z = bin_abs_heatmap(x, y, err)
        push!(grids, (r.title, xs, ys, Z, err, x, y))
    end
    nplots = length(grids)
    plt = plot(layout = (1, max(nplots, 1)), size = (560 * max(nplots, 1), 520),
               plot_title = "max |Δρ − analytic| in each bin (level 7)")
    for (k, (title, xs, ys, Z, err, x, y)) in enumerate(grids)
        finite = filter(isfinite, vec(Z))
        clim = isempty(finite) ? (0.0, 1.0) : (0.0, maximum(finite))
        heatmap!(plt[k], xs, ys, Z;
                 aspect_ratio = :equal, xlims = (-1, 1), ylims = (-1, 1),
                 clims = clim, c = :inferno, colorbar = true,
                 xlabel = "x", ylabel = k == 1 ? "y" : "",
                 title = title, framestyle = :box)
        finite_e = filter(isfinite, err)
        imax = argmax(abs.(replace(err, NaN => 0.0)))
        @printf("%-28s  Δρ  min|e|=%.3e  max|e|=%.3e  at (%.3f, %.3f)\n",
                title, minimum(abs, finite_e), maximum(abs, finite_e),
                x[imax], y[imax])
    end
    pngpath = joinpath(outdir, "heatmap_div.png")
    savefig(plt, pngpath)
    println("saved ", pngpath)

    r = findfirst(rr -> startswith(rr.title, "conforming"), results)
    if r !== nothing
        r = results[r]
        x, y = nodal_xy(r.semi)
        xs, ys, Zex = bin_mean(x, y, vec(r.div_ex))
        _, _, Znum = bin_mean(x, y, vec(r.div_num))
        _, _, Zerr = bin_signed_absmax(x, y, vec(r.div_num) .- vec(r.div_ex))
        clim = extrema(filter(isfinite, vcat(vec(Zex), vec(Znum))))
        errmax = maximum(z -> isnan(z) ? 0.0 : abs(z), Zerr)
        plt = plot(layout = (1, 3), size = (1680, 520),
                   plot_title = "full divergence Δρ  (conforming, level 7)")
        heatmap!(plt[1], xs, ys, Zerr; aspect_ratio = :equal, xlims = (-1, 1),
                 ylims = (-1, 1), clims = (-errmax, errmax), c = :RdBu, colorbar = true,
                 xlabel = "x", ylabel = "y",
                 title = "error  (max $(round(errmax; sigdigits = 3)))",
                 framestyle = :box)
        heatmap!(plt[2], xs, ys, Zex; aspect_ratio = :equal, xlims = (-1, 1),
                 ylims = (-1, 1), clims = clim, c = :RdBu, colorbar = false,
                 xlabel = "x", ylabel = "", title = "analytic Δρ",
                 framestyle = :box)
        heatmap!(plt[3], xs, ys, Znum; aspect_ratio = :equal, xlims = (-1, 1),
                 ylims = (-1, 1), clims = clim, c = :RdBu, colorbar = true,
                 xlabel = "x", ylabel = "", title = "conforming MortarEntropy",
                 framestyle = :box)
        pngpath = joinpath(outdir, "heatmap_conforming_vs_analytic.png")
        savefig(plt, pngpath)
        println("saved ", pngpath)
    end

    r_ent = findfirst(rr -> rr.title == "checkerboard MortarEntropy", results)
    if r_ent !== nothing
        r_ent = results[r_ent]
        x, y = nodal_xy(r_ent.semi)
        xs, ys, Zex = bin_mean(x, y, vec(r_ent.div_ex))
        _, _, Znum = bin_mean(x, y, vec(r_ent.div_num))
        _, _, Zerr = bin_signed_absmax(x, y, vec(r_ent.div_num) .- vec(r_ent.div_ex))
        clim = extrema(filter(isfinite, vcat(vec(Zex), vec(Znum))))
        errmax = maximum(z -> isnan(z) ? 0.0 : abs(z), Zerr)
        plt = plot(layout = (1, 3), size = (1680, 520),
                   plot_title = "full divergence Δρ  (level 7)")
        heatmap!(plt[1], xs, ys, Zerr; aspect_ratio = :equal, xlims = (-1, 1),
                 ylims = (-1, 1), clims = (-errmax, errmax), c = :RdBu, colorbar = true,
                 xlabel = "x", ylabel = "y",
                 title = "error  (max $(round(errmax; sigdigits = 3)))",
                 framestyle = :box)
        heatmap!(plt[2], xs, ys, Zex; aspect_ratio = :equal, xlims = (-1, 1),
                 ylims = (-1, 1), clims = clim, c = :RdBu, colorbar = false,
                 xlabel = "x", ylabel = "", title = "analytic Δρ",
                 framestyle = :box)
        heatmap!(plt[3], xs, ys, Znum; aspect_ratio = :equal, xlims = (-1, 1),
                 ylims = (-1, 1), clims = clim, c = :RdBu, colorbar = true,
                 xlabel = "x", ylabel = "", title = "checkerboard MortarEntropy",
                 framestyle = :box)
        pngpath = joinpath(outdir, "heatmap_analytic_vs_mortarentropy_div.png")
        savefig(plt, pngpath)
        println("saved ", pngpath)
    end
catch e
    println("Plots.jl not available (", sprint(showerror, e), ")")
    showerror(stdout, e, catch_backtrace())
end
