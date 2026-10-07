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
# Directional split: after AV fluxes, keep only g_x or only g_y, then diverge.
# Cubic polynomial: (∂u/∂w)∇w = ∇u, so the density slot is ρ_x / ρ_xx.
# ρ = 5 + x³ + x²y + xy² + y³  (offset 5 so ρ > 0 on [-1,1]²).

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)
prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
eqs_av = Trixi.default_artificial_viscosity(equations)

function initial_condition_density_wave_ecav(x, t, equations)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    # ρ = 5 + Σ_{i=1}^{4} x^{i-1} y^{4-i} = 5 + y³ + x y² + x² y + x³
    rho = convert(RealT, 5)
    for i in 1:4
        rho = rho + x[1]^(i - 1) * x[2]^(4 - i)
    end
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e_total = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end

analytic_rho_x(x, y) = 3 * x^2 + 2 * x * y + y^2
analytic_rho_y(x, y) = x^2 + 2 * x * y + 3 * y^2
analytic_rho_xx(x, y) = 6 * x + 2 * y
analytic_rho_yy(x, y) = 2 * x + 6 * y

checkerboard = true;
mortar_type = "l2"
initial_refinement_level = 7;
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
semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                        initial_condition_density_wave_ecav,
                        solver;
                        VDM = VDM, filter = filter,
                        ecav_choice = :ecav,
                        combine_rhs = Trixi.True(),
                        equations_artificial_viscosity = eqs_av,
                        solver_parabolic = solver_parabolic,
                        boundary_conditions = (boundary_condition_periodic,
                                            boundary_condition_periodic))

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

# One second-derivative component only: rebuild viscous fluxes from ∇w,
# keep g_x or g_y, drop the other, then one `calc_divergence!` into a fresh `du`.
function d2rho_one_direction!(du, direction, gradients, u_transformed, u, semi, eqs_av)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av, dg, cache)
    fx, fy = flux_parabolic
    if direction == 1
        fill!(fy, 0)
    else
        fill!(fx, 0)
    end
    reset_surface_cache!(cache)
    # `Trixi.set_zero!` uses `@threaded` over a PtrArray and SIGSEGVs on large
    # MortarL2 checkerboard meshes under Julia 1.12.
    du .= 0
    Trixi.calc_divergence!(du, flux_parabolic, u, mesh, equations_parabolic,
                           semi.boundary_conditions_parabolic, dg,
                           semi.solver_parabolic, cache, 0.0)
    Trixi.apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    return nothing
end

# Compile viscous kernels on a tiny conforming mesh first. Julia 1.12 can
# SIGILL/SIGSEGV while inferring `calc_divergence!` if the first mesh is
# already the level-7 checkerboard.
let
    tiny_mesh = TreeMesh(coordinates_min, coordinates_max;
                         initial_refinement_level = 2, n_cells_max = 1_000,
                         periodicity = true)
    tiny_solver = DGSEM(basis, surface_integral, volume_integral, MortarL2(basis))
    tiny_semi = SemidiscretizationArtificialViscosity(tiny_mesh,
                                                      (equations, equations_parabolic),
                                                      initial_condition_density_wave_ecav,
                                                      tiny_solver;
                                                      VDM = VDM, filter = filter,
                                                      ecav_choice = :ecav,
                                                      combine_rhs = Trixi.True(),
                                                      equations_artificial_viscosity = eqs_av,
                                                      solver_parabolic = solver_parabolic,
                                                      boundary_conditions = (boundary_condition_periodic,
                                                                             boundary_condition_periodic))
    tiny_mesh_, _, tiny_dg, tiny_cache = Trixi.mesh_equations_solver_cache(tiny_semi)
    u_tiny_ode = Trixi.compute_coefficients(0.0, tiny_semi)
    u_tiny = Trixi.wrap_array(u_tiny_ode, tiny_semi)
    du_tiny = Trixi.wrap_array(similar(u_tiny_ode), tiny_semi)
    (; u_transformed, gradients) = tiny_semi.cache_parabolic.parabolic_container
    Trixi.transform_variables!(u_transformed, u_tiny, tiny_mesh_, equations_parabolic,
                               tiny_dg, tiny_cache)
    Trixi.calc_gradient!(gradients, u_transformed, 0.0, tiny_mesh_, equations_parabolic,
                         tiny_semi.boundary_conditions_parabolic, tiny_dg,
                         tiny_semi.solver_parabolic, tiny_cache)
    d2rho_one_direction!(du_tiny, 1, gradients, u_transformed, u_tiny, tiny_semi, eqs_av)
end
GC.gc()

mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi);
println("level = ", initial_refinement_level,
        "  nelements = ", Trixi.nelements(dg, cache),
        "  nmortars = ", Trixi.nmortars(dg, cache))
println("equations_parabolic = ", typeof(semi.equations_parabolic).name.name)
println("equations_AV       = ",
        typeof(semi.artificial_viscosity.equations_artificial_viscosity).name.name)


u_ode = Trixi.compute_coefficients(0.0, semi)
u = Trixi.wrap_array(u_ode, semi)
du_xx = Trixi.wrap_array_native(similar(u_ode), semi)
du_yy = Trixi.wrap_array_native(similar(u_ode), semi)

(; u_transformed, gradients) = semi.cache_parabolic.parabolic_container
eqs_av_semi = semi.artificial_viscosity.equations_artificial_viscosity

Trixi.transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
Trixi.calc_gradient!(gradients, u_transformed, 0.0, mesh, equations_parabolic,
                     semi.boundary_conditions_parabolic, dg,
                     semi.solver_parabolic, cache)

# ρ_x is the density slot of g = (∂u/∂w)∇w, not of ∇w.
flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                             eqs_av_semi, dg, cache)
gx, gy = flux_parabolic

n = Trixi.nnodes(dg)
ne = Trixi.nelements(dg, cache)
rho_x_num = Array{Float64}(undef, n, n, ne)
rho_y_num = similar(rho_x_num)
rho_x_ex = similar(rho_x_num)
rho_y_ex = similar(rho_x_num)
rho_x_num .= gx[1, :, :, :]
rho_y_num .= gy[1, :, :, :]
fill_exact!(rho_x_ex, analytic_rho_x, semi)
fill_exact!(rho_y_ex, analytic_rho_y, semi)

@printf("%-10s  %12s  %12s  %10s  %10s  %12s  %12s  vs\n",
        "field", "L2", "L∞", "rel L2", "rel L∞", "max|num|", "max|ex|")
for (name, num, ex, lab) in (("ρ_x", rho_x_num, rho_x_ex, "analytic ρ_x"),
                             ("ρ_y", rho_y_num, rho_y_ex, "analytic ρ_y"))
    l2, linf = weighted_errors(num, ex, semi)
    l2ex, _ = weighted_errors(ex, zero(ex), semi)
    rel_l2 = l2 / max(l2ex, eps())
    rel_inf = linf / max(maximum(abs, ex), eps())
    @printf("%-10s  %12.4e  %12.4e  %10.3e  %10.3e  %12.4e  %12.4e  %s\n",
            name, l2, linf, rel_l2, rel_inf, maximum(abs, num),
            maximum(abs, ex), lab)
end

# using Plots
# plot(ScalarPlotData2D(rho_x_num, semi; variable_name = "ρ_x  (g_x density)"))
# plot!(getmesh(PlotData2D(u_ode, semi)))

rho_xx_num = Array{Float64}(undef, n, n, ne);
rho_yy_num = similar(rho_xx_num);
rho_xx_ex = similar(rho_xx_num);
rho_yy_ex = similar(rho_xx_num);
fill_exact!(rho_xx_ex, (x, y) -> -analytic_rho_xx(x, y), semi);
fill_exact!(rho_yy_ex, (x, y) -> -analytic_rho_yy(x, y), semi);

d2rho_one_direction!(du_xx, 1, gradients, u_transformed, u, semi, eqs_av_semi)
rho_xx_num .= -du_xx[1, :, :, :]
d2rho_one_direction!(du_yy, 2, gradients, u_transformed, u, semi, eqs_av_semi)
rho_yy_num .= -du_yy[1, :, :, :]
 
names = ("ρ_xx", "ρ_yy")
nums = (rho_xx_num, rho_yy_num)
exs = (rho_xx_ex, rho_yy_ex)
labels_ex = ("analytic −ρ_xx", "analytic −ρ_yy")
errors = Dict{String, NamedTuple}()
@printf("%-10s  %12s  %12s  %10s  %10s  %12s  %12s  vs\n",
        "field", "L2", "L∞", "rel L2", "rel L∞", "max|num|", "max|ex|")
for (name, num, ex, lab) in zip(names, nums, exs, labels_ex)
    l2, linf = weighted_errors(num, ex, semi)
    l2ex, _ = weighted_errors(ex, zero(ex), semi)
    rel_l2 = l2 / max(l2ex, eps())
    rel_inf = linf / max(maximum(abs, ex), eps())
    errors[name] = (l2 = l2, linf = linf, rel_l2 = rel_l2, rel_inf = rel_inf)
    @printf("%-10s  %12.4e  %12.4e  %10.3e  %10.3e  %12.4e  %12.4e  %s\n",
            name, l2, linf, rel_l2, rel_inf, maximum(abs, num),
            maximum(abs, ex), lab)
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
