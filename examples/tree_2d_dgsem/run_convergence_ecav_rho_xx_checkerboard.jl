using LinearAlgebra: I
using Printf
using Trixi

# Checkerboard ρ_x and ρ_xx: MortarL2 vs MortarEntropy (Gauss nodes, Gauss-mass reverse).

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
analytic_rho_xx(x, y) = -2 * π^2 * sinpi(2 * (x + y))

function make_semi(; checkerboard, mortar_type, initial_refinement_level)
    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)
    surface_flux = flux_ranocha
    volume_integral = VolumeIntegralWeakForm()
    if mortar_type == "l2"
        mortar = MortarL2(basis)
    else
        mortar = MortarEntropy(basis; nodes = :gauss, reverse_quadrature = :gauss)
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
    du .= 0
    Trixi.calc_divergence!(du, flux_parabolic, u, mesh, equations_parabolic,
                           semi.boundary_conditions_parabolic, dg,
                           semi.solver_parabolic, cache, 0.0)
    Trixi.apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
    return nothing
end

function rho_errors(semi)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u_ode = Trixi.compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    du_xx = Trixi.wrap_array_native(similar(u_ode), semi)
    (; u_transformed, gradients) = semi.cache_parabolic.parabolic_container
    eqs_av_semi = semi.artificial_viscosity.equations_artificial_viscosity
    Trixi.transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
    Trixi.calc_gradient!(gradients, u_transformed, 0.0, mesh, equations_parabolic,
                         semi.boundary_conditions_parabolic, dg,
                         semi.solver_parabolic, cache)
    flux_parabolic = semi.cache_parabolic.parabolic_container.flux_parabolic
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av_semi, dg, cache)
    gx, _ = flux_parabolic
    n = Trixi.nnodes(dg)
    ne = Trixi.nelements(dg, cache)
    rho_x_num = Array{Float64}(undef, n, n, ne)
    rho_x_ex = similar(rho_x_num)
    rho_xx_num = similar(rho_x_num)
    rho_xx_ex = similar(rho_x_num)
    rho_x_num .= gx[1, :, :, :]
    fill_exact!(rho_x_ex, analytic_rho_x, semi)
    fill_exact!(rho_xx_ex, (x, y) -> -analytic_rho_xx(x, y), semi)
    d2rho_one_direction!(du_xx, 1, gradients, u_transformed, u, semi, eqs_av_semi)
    rho_xx_num .= -du_xx[1, :, :, :]
    x_l2, x_inf = weighted_errors(rho_x_num, rho_x_ex, semi)
    xx_l2, xx_inf = weighted_errors(rho_xx_num, rho_xx_ex, semi)
    return x_l2, x_inf, xx_l2, xx_inf
end

function eoc(err_coarse, err_fine)
    return log(err_coarse / err_fine) / log(2)
end

function print_eoc_row(level, ne, nm, l2, eoc_l2, linf, eoc_linf)
    if isnan(eoc_l2)
        @printf("%5d  %10d  %8d  %12.4e  %8s  %12.4e  %8s\n",
                level, ne, nm, l2, "—", linf, "—")
    else
        @printf("%5d  %10d  %8d  %12.4e  %8.2f  %12.4e  %8.2f\n",
                level, ne, nm, l2, eoc_l2, linf, eoc_linf)
    end
end

try
    tiny = make_semi(; checkerboard = false, mortar_type = "l2",
                     initial_refinement_level = 3)
    rho_errors(tiny)
catch
end

levels = 3:7
mortars = (("l2", "MortarL2"), ("entropy", "MortarEntropy"))
results = Dict{String, Any}()

for (mortar_type, title) in mortars
    println("="^80)
    println("checkerboard  ", title, "  (polydeg = 3)")
    println("="^80)
    x_rows = NamedTuple[]
    xx_rows = NamedTuple[]
    prev_x_l2 = prev_x_inf = prev_xx_l2 = prev_xx_inf = NaN
    println("ρ_x")
    @printf("%5s  %10s  %8s  %12s  %8s  %12s  %8s\n",
            "level", "nelements", "nmortars", "L2", "EOC L2", "L∞", "EOC L∞")
    for level in levels
        semi = make_semi(; checkerboard = true, mortar_type = mortar_type,
                         initial_refinement_level = level)
        mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
        ne = Trixi.nelements(dg, cache)
        nm = Trixi.nmortars(dg, cache)
        x_l2, x_inf, xx_l2, xx_inf = rho_errors(semi)
        eoc_x_l2 = isnan(prev_x_l2) ? NaN : eoc(prev_x_l2, x_l2)
        eoc_x_inf = isnan(prev_x_inf) ? NaN : eoc(prev_x_inf, x_inf)
        eoc_xx_l2 = isnan(prev_xx_l2) ? NaN : eoc(prev_xx_l2, xx_l2)
        eoc_xx_inf = isnan(prev_xx_inf) ? NaN : eoc(prev_xx_inf, xx_inf)
        push!(x_rows, (level = level, nelements = ne, nmortars = nm,
                       l2 = x_l2, linf = x_inf, eoc_l2 = eoc_x_l2, eoc_linf = eoc_x_inf))
        push!(xx_rows, (level = level, nelements = ne, nmortars = nm,
                        l2 = xx_l2, linf = xx_inf, eoc_l2 = eoc_xx_l2, eoc_linf = eoc_xx_inf))
        print_eoc_row(level, ne, nm, x_l2, eoc_x_l2, x_inf, eoc_x_inf)
        prev_x_l2, prev_x_inf = x_l2, x_inf
        prev_xx_l2, prev_xx_inf = xx_l2, xx_inf
    end
    println()
    println("ρ_xx")
    @printf("%5s  %10s  %8s  %12s  %8s  %12s  %8s\n",
            "level", "nelements", "nmortars", "L2", "EOC L2", "L∞", "EOC L∞")
    for r in xx_rows
        print_eoc_row(r.level, r.nelements, r.nmortars, r.l2, r.eoc_l2, r.linf, r.eoc_linf)
    end
    results[mortar_type] = (rho_x = x_rows, rho_xx = xx_rows)
    println()
end

println("="^80)
println("MortarEntropy-only run (compare printed digits to MortarL2 above)")
println("="^80)
