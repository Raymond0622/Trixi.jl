using LinearAlgebra: I
using Printf
using Trixi

# Cubic ρ_x / ρ_xx BR1 test. Run with a single thread or Julia 1.12 can
# SIGSEGV in the first viscous compile and look like the script just stopped:
#   JULIA_NUM_THREADS=1 julia --project=. examples/tree_2d_dgsem/run_convergence_ecav_rho_x_checkerboard.jl
#
# Cubic polynomial ρ = 5 + x³ + x²y + xy² + y³ (in P³).

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
    # Offset 5 so ρ > 0 on [-1,1]² (the degree-3 part reaches −4).
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
analytic_rho_xx(x, y) = 6 * x + 2 * y

function make_semi(; checkerboard, mortar_type, initial_refinement_level)
    polydeg = 3
    basis = LobattoLegendreBasis(polydeg)
    surface_flux = flux_ranocha
    volume_integral = VolumeIntegralWeakForm()
    if mortar_type == "l2"
        mortar = MortarL2(basis)
        surface_integral = SurfaceIntegralWeakForm(surface_flux)
    elseif mortar_type == "l2_gauss"
        mortar = MortarL2(basis)
        surface_integral = SurfaceIntegralWeakFormGauss(surface_flux)
    elseif mortar_type == "entropy_lgl"
        mortar = MortarEntropy(basis; nodes = :gauss_lobatto,
                               reverse_quadrature = :gauss_lobatto)
        surface_integral = SurfaceIntegralWeakForm(surface_flux)
    elseif mortar_type == "entropy_gauss_sat"
        mortar = MortarEntropy(basis; nodes = :gauss,
                               reverse_quadrature = :gauss)
        surface_integral = SurfaceIntegralWeakFormGauss(surface_flux)
    else
        mortar = MortarEntropy(basis; nodes = :gauss,
                               reverse_quadrature = :gauss_lobatto)
        surface_integral = SurfaceIntegralWeakForm(surface_flux)
    end
    solver = DGSEM(basis, surface_integral, volume_integral, mortar)
    solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()

    coordinates_min = (-1.0, -1.0)
    coordinates_max = (1.0, 1.0)
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level = initial_refinement_level,
                    n_cells_max = 400_000, periodicity = false)
    if checkerboard
        # Interior checkerboard only: leave the outer ring unrefined so every
        # Dirichlet face is 1:1 (no hanging nodes on the domain boundary).
        level = mesh.tree.levels[first(Trixi.leaf_cells(mesh.tree))]
        n_base = 2^level
        dx = (coordinates_max[1] - coordinates_min[1]) / n_base
        cells_to_refine = Int[]
        for cell_id in Trixi.leaf_cells(mesh.tree)
            x, y = Trixi.cell_coordinates(mesh.tree, cell_id)
            ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
            iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
            on_boundary = ix == 0 || ix == n_base - 1 ||
                          iy == 0 || iy == n_base - 1
            if iseven(ix + iy) && !on_boundary
                push!(cells_to_refine, cell_id)
            end
        end
        Trixi.refine!(mesh.tree, cells_to_refine)
    end

    VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
    filter = ones(polydeg + 1)
    bc = BoundaryConditionDirichlet(initial_condition_density_wave_ecav)
    boundary_conditions = (; x_neg = bc, x_pos = bc, y_neg = bc, y_pos = bc)
    return SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition_density_wave_ecav,
                                                 solver;
                                                 VDM = VDM, filter = filter,
                                                 ecav_choice = :ecav,
                                                 combine_rhs = Trixi.True(),
                                                 equations_artificial_viscosity = eqs_av,
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_conditions,
                                                                        boundary_conditions))
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

function rho_x_fields(semi)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    u_ode = Trixi.compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
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
    num = Array{Float64}(undef, n, n, ne)
    ex = similar(num)
    num .= gx[1, :, :, :]
    fill_exact!(ex, analytic_rho_x, semi)
    return num, ex
end

function gauss_surface_rho_x_errors(num, semi)
    _, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    n = Trixi.nnodes(dg)
    coords = cache.elements.node_coordinates
    g_nodes, g_weights = Trixi.gauss_nodes_weights(n)
    I_lg = Trixi.polynomial_interpolation_matrix(dg.basis.nodes, g_nodes)
    l2acc = 0.0
    linf = 0.0
    for element in Trixi.eachelement(dg, cache)
        J1 = inv(cache.elements.inverse_jacobian[element])
        for direction in 1:4
            if direction == 1
                ρ = num[1, :, element]; x = coords[1, 1, :, element]; y = coords[2, 1, :, element]
            elseif direction == 2
                ρ = num[n, :, element]; x = coords[1, n, :, element]; y = coords[2, n, :, element]
            elseif direction == 3
                ρ = num[:, 1, element]; x = coords[1, :, 1, element]; y = coords[2, :, 1, element]
            else
                ρ = num[:, n, element]; x = coords[1, :, n, element]; y = coords[2, :, n, element]
            end
            ρg, xg, yg = I_lg * ρ, I_lg * x, I_lg * y
            for k in 1:n
                err = abs(ρg[k] - analytic_rho_x(xg[k], yg[k]))
                l2acc += g_weights[k] * J1 * err^2
                linf = max(linf, err)
            end
        end
    end
    return sqrt(l2acc), linf
end

function rho_x_errors(semi)
    num, ex = rho_x_fields(semi)
    vol_l2, vol_linf = weighted_errors(num, ex, semi)
    gauss_l2, gauss_linf = gauss_surface_rho_x_errors(num, semi)
    return vol_l2, vol_linf, gauss_l2, gauss_linf
end

function rho_xx_errors(semi)
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
    n = Trixi.nnodes(dg)
    ne = Trixi.nelements(dg, cache)
    num = Array{Float64}(undef, n, n, ne)
    ex = similar(num)
    fill_exact!(ex, (x, y) -> -analytic_rho_xx(x, y), semi)
    d2rho_one_direction!(du_xx, 1, gradients, u_transformed, u, semi, eqs_av_semi)
    num .= -du_xx[1, :, :, :]
    return weighted_errors(num, ex, semi)
end

function eoc(err_coarse, err_fine)
    return log(err_coarse / err_fine) / log(2)
end

println("warmup (Julia 1.12: use JULIA_NUM_THREADS=1)...")
flush(stdout)

mortar_specs = Dict(
    "entropy" => ("entropy", "MortarEntropy Gauss LGL SAT M_f=W_LGL"),
    "entropy_gauss_sat" => ("entropy_gauss_sat", "MortarEntropy Gauss SAT"),
    "l2" => ("l2", "MortarL2 LGL SAT"),
    "l2_gauss" => ("l2_gauss", "MortarL2 Gauss SAT"),
    "entropy_lgl" => ("entropy_lgl", "MortarEntropy LGL"),
)
mortar_key = get(ENV, "MORTAR", "entropy")
mortars = (mortar_specs[mortar_key],)
warmup_type = mortars[1][1]

try
    tiny = make_semi(; checkerboard = true, mortar_type = warmup_type,
                     initial_refinement_level = 3)
    rho_x_errors(tiny)
    println("  warmup ", warmup_type, " ok")
    flush(stdout)
catch e
    println("  warmup ", warmup_type, " ", typeof(e), " (ignored)")
    flush(stdout)
end

levels = 3:7
results = Dict{String, Any}()

function print_eoc_row(level, ne, nm, l2, eoc_l2, linf, eoc_linf)
    if isnan(eoc_l2)
        @printf("%5d  %10d  %8d  %12.4e  %8s  %12.4e  %8s\n",
                level, ne, nm, l2, "—", linf, "—")
    else
        @printf("%5d  %10d  %8d  %12.4e  %8.2f  %12.4e  %8.2f\n",
                level, ne, nm, l2, eoc_l2, linf, eoc_linf)
    end
end

for (mortar_type, title) in mortars
    println("="^80)
    println("checkerboard  ", title, "  ρ_x  cubic polynomial  (polydeg = 3)")
    if mortar_type == "l2"
        println("MortarL2, LGL SAT (f*/ω)")
    elseif mortar_type == "l2_gauss"
        println("MortarL2, Gauss SAT  Q = W_LGL^{-1} I_{L→G}^T W_G I_{L→G}")
    elseif mortar_type == "entropy_lgl"
        println("MortarEntropy nodes=:gauss_lobatto, reverse M_m=W_LGL, LGL SAT")
    elseif mortar_type == "entropy_gauss_sat"
        println("MortarEntropy nodes=:gauss, M_f=W_G, Gauss SAT  L = I_{L→G}^T W_G")
    else
        println("MortarEntropy nodes=:gauss, M_f=W_LGL, LGL SAT (f*/ω)")
    end
    println("="^80)
    vol_rows = NamedTuple[]
    prev_vol_l2 = prev_vol_linf = NaN
    println("ρ_x  volume LGL")
    @printf("%5s  %10s  %8s  %12s  %8s  %12s  %8s\n",
            "level", "nelements", "nmortars", "L2", "EOC L2", "L∞", "EOC L∞")
    for level in levels
        semi = make_semi(; checkerboard = true, mortar_type = mortar_type,
                         initial_refinement_level = level)
        mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
        ne = Trixi.nelements(dg, cache)
        nm = Trixi.nmortars(dg, cache)
        vol_l2, vol_linf, _, _ = rho_x_errors(semi)
        eoc_vol_l2 = isnan(prev_vol_l2) ? NaN : eoc(prev_vol_l2, vol_l2)
        eoc_vol_linf = isnan(prev_vol_linf) ? NaN : eoc(prev_vol_linf, vol_linf)
        push!(vol_rows, (level = level, nelements = ne, nmortars = nm,
                         l2 = vol_l2, linf = vol_linf,
                         eoc_l2 = eoc_vol_l2, eoc_linf = eoc_vol_linf))
        print_eoc_row(level, ne, nm, vol_l2, eoc_vol_l2, vol_linf, eoc_vol_linf)
        flush(stdout)
        prev_vol_l2, prev_vol_linf = vol_l2, vol_linf
    end
    results[mortar_type] = (volume = vol_rows,)
    println()
end

println("="^80)
println("Mean EOC levels 3→7  checkerboard  ρ_x  (polydeg = 3; cubic ρ, expected ≈ 3)")
println("="^80)
for (mortar_type, title) in mortars
    rows = results[mortar_type][:volume]
    eocs_l2 = [r.eoc_l2 for r in rows if isfinite(r.eoc_l2)]
    eocs_linf = [r.eoc_linf for r in rows if isfinite(r.eoc_linf)]
    @printf("%-28s  L2  %.2f    L∞  %.2f\n", title,
            sum(eocs_l2) / length(eocs_l2),
            sum(eocs_linf) / length(eocs_linf))
end
