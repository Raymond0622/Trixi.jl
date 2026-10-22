# Gauss (g, ∇v) vs (v, ∇·g) on conforming and checkerboard TreeMesh{3}.
# Same polynomial degree, level, and LDG scheme as examples/tree_2d_dgsem/check_gauss_ibp.jl.
using LinearAlgebra: I, dot
using Printf
using Trixi

gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)
prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
eqs_av = Trixi.default_artificial_viscosity(equations)

function initial_condition_poly(x, t, equations)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    v3 = convert(RealT, 0.3)
    # All 10 homogeneous degree-3 monomials. Offset 15 keeps rho >= 5 on [-1, 1]^3.
    rho = convert(RealT, 15)
    for i in 0:3, j in 0:(3 - i)
        k = 3 - i - j
        rho += x[1]^i * x[2]^j * x[3]^k
    end
    p = 20
    rho_e = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2 + v3^2)
    return SVector(rho, rho * v1, rho * v2, rho * v3, rho_e)
end

polydeg = 3
initial_refinement_level = 3
coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)

surface_flux = flux_ranocha
solver_parabolic = Trixi.ParabolicFormulationLocalDG()

function make_solver(basis, mortar)
    return DGSEM(basis, SurfaceIntegralWeakForm(surface_flux), VolumeIntegralWeakForm(),
                 mortar)
end

gauss_basis = GaussLegendreBasis(polydeg)
gauss_solver = make_solver(gauss_basis,
                           MortarEntropy(gauss_basis; nodes = :gauss,
                                         reverse_quadrature = :gauss))
lgl_basis = LobattoLegendreBasis(polydeg)
lgl_solver = make_solver(lgl_basis,
                         MortarEntropy(lgl_basis; nodes = :gauss_lobatto,
                                       reverse_quadrature = :gauss_lobatto))

VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
filter = ones(polydeg + 1)

function checkerboard_mesh()
    mesh = TreeMesh(coordinates_min, coordinates_max;
                    initial_refinement_level = initial_refinement_level,
                    periodicity = true)
    level = mesh.tree.levels[first(Trixi.leaf_cells(mesh.tree))]
    dx = (coordinates_max[1] - coordinates_min[1]) / 2^level
    cells_to_refine = Int[]
    for cell_id in Trixi.leaf_cells(mesh.tree)
        x, y, z = Trixi.cell_coordinates(mesh.tree, cell_id)
        ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
        iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
        iz = round(Int, (z - coordinates_min[3]) / dx - 0.5)
        if iseven(ix + iy + iz)
            push!(cells_to_refine, cell_id)
        end
    end
    Trixi.refine!(mesh.tree, cells_to_refine)
    return mesh
end

function make_semi(mesh, solver)
    return SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                  initial_condition_poly, solver;
                                                  VDM = VDM, filter = filter,
                                                  ecav_choice = :ecav,
                                                  combine_rhs = Trixi.True(),
                                                  equations_artificial_viscosity = eqs_av,
                                                  solver_parabolic = solver_parabolic,
                                                  boundary_conditions = (boundary_condition_periodic,
                                                                         boundary_condition_periodic))
end

function element_g_dot_gradv(flux_parabolic, gradients, semi)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    gx, gy, gz = flux_parabolic
    wx, wy, wz = gradients
    D = zeros(Trixi.nelements(dg, cache))
    for element in Trixi.eachelement(dg, cache)
        J = Trixi.volume_jacobian(element, mesh, cache)
        acc = 0.0
        for k in Trixi.eachnode(dg), j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            g1 = Trixi.get_node_vars(gx, equations, dg, i, j, k, element)
            g2 = Trixi.get_node_vars(gy, equations, dg, i, j, k, element)
            g3 = Trixi.get_node_vars(gz, equations, dg, i, j, k, element)
            d1 = Trixi.get_node_vars(wx, equations, dg, i, j, k, element)
            d2 = Trixi.get_node_vars(wy, equations, dg, i, j, k, element)
            d3 = Trixi.get_node_vars(wz, equations, dg, i, j, k, element)
            w = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]
            acc += (dot(g1, d1) + dot(g2, d2) + dot(g3, d3)) * w * J
        end
        D[element] = acc
    end
    return D
end

function element_v_dot_du(du, u_transformed, semi)
    mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    R = zeros(Trixi.nelements(dg, cache))
    for element in Trixi.eachelement(dg, cache)
        J = Trixi.volume_jacobian(element, mesh, cache)
        acc = 0.0
        for k in Trixi.eachnode(dg), j in Trixi.eachnode(dg), i in Trixi.eachnode(dg)
            v_node = Trixi.get_node_vars(u_transformed, equations, dg, i, j, k, element)
            du_node = Trixi.get_node_vars(du, equations, dg, i, j, k, element)
            w = dg.basis.weights[i] * dg.basis.weights[j] * dg.basis.weights[k]
            acc += dot(v_node, du_node) * w * J
        end
        R[element] = acc
    end
    return R
end

function reset_surface_cache!(cache)
    fill!(cache.elements.surface_flux_values, 0)
    fill!(cache.interfaces.u, 0)
    return nothing
end

function report_g_gradv(label, semi)
    mesh, _, dg, cache = Trixi.mesh_equations_solver_cache(semi)
    println("\n======== ", label, " ========")
    println("basis = ", typeof(dg.basis).name.name,
            "  mortar nodes = ", Trixi.mortar_nodes(dg.mortar))
    println("parabolic scheme = ", semi.solver_parabolic)
    println("nelements = ", Trixi.nelements(dg, cache),
            "  nmortars = ", Trixi.nmortars(dg, cache))
    u_ode = Trixi.compute_coefficients(0.0, semi)
    u = Trixi.wrap_array(u_ode, semi)
    du_ode = similar(u_ode)
    du = Trixi.wrap_array_native(du_ode, semi)
    (; u_transformed, gradients, flux_parabolic) = semi.cache_parabolic.parabolic_container
    eqs_av_semi = semi.artificial_viscosity.equations_artificial_viscosity
    Trixi.transform_variables!(u_transformed, u, mesh, equations_parabolic, dg, cache)
    Trixi.calc_gradient!(nothing, gradients, u_transformed, 0.0, mesh,
                         equations_parabolic, semi.boundary_conditions_parabolic, dg,
                         semi.solver_parabolic, cache)
    Trixi.calc_parabolic_fluxes!(flux_parabolic, gradients, u_transformed, mesh,
                                 eqs_av_semi, dg, cache)
    Dvol = element_g_dot_gradv(flux_parabolic, gradients, semi)
    reset_surface_cache!(cache)
    Ddiv = GC.@preserve du_ode begin
        du .= 0
        Trixi.calc_divergence!(du, flux_parabolic, u, mesh, equations_parabolic,
                               semi.boundary_conditions_parabolic, dg,
                               semi.solver_parabolic, cache, 0.0)
        Trixi.apply_jacobian_parabolic!(du, mesh, equations_parabolic, dg, cache)
        element_v_dot_du(du, u_transformed, semi)
    end
    mismatch = Dvol .+ Ddiv
    println("min/max (g, ∇v)            = ", minimum(Dvol), " / ", maximum(Dvol))
    println("max |(v, ∇·g)|             = ", maximum(abs, Ddiv))
    println("max |(g, ∇v) + (v, ∇·g)|   = ", maximum(abs, mismatch))
    println("mean |(g, ∇v) + (v, ∇·g)|  = ", sum(abs, mismatch) / length(mismatch))
    println("sum_e (g, ∇v)              = ", sum(Dvol))
    println("sum_e (v, ∇·g)             = ", sum(Ddiv))
    println("sum_e [(g, ∇v) + (v, ∇·g)] = ", sum(mismatch))
    nshow = min(8, length(Dvol))
    println("first ", nshow, " elements: (g,∇v), (v,∇·g), leftover")
    for e in 1:nshow
        @printf("  e=%-4d  % .6e  % .6e  % .6e\n", e, Dvol[e], Ddiv[e], mismatch[e])
    end
    return nothing
end

conf_mesh = TreeMesh(coordinates_min, coordinates_max;
                     initial_refinement_level = initial_refinement_level,
                     periodicity = true)
cb_mesh = checkerboard_mesh()
report_g_gradv("conforming Gauss MortarEntropy", make_semi(conf_mesh, gauss_solver))
report_g_gradv("checkerboard Gauss MortarEntropy", make_semi(cb_mesh, gauss_solver))
report_g_gradv("conforming LGL MortarEntropy", make_semi(conf_mesh, lgl_solver))
report_g_gradv("checkerboard LGL MortarEntropy", make_semi(cb_mesh, lgl_solver))
