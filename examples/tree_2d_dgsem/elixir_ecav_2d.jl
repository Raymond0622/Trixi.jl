using OrdinaryDiffEqSSPRK
using OrdinaryDiffEqLowStorageRK
using LinearAlgebra: I
using Trixi

###############################################################################
# Checkerboard TreeMesh. `problem = "riemann"` or `"modified_sod"`.
# `shock_capturing = "ecav"` uses entropy-correction AV; `"adaptive"` uses the
# entropy-correction FV volume switch instead.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()

"""
    initial_condition_riemann1(coords, t, equations::CompressibleEulerEquations2D)

2D Riemann (checkerboard) initial condition.
"""
function initial_condition_riemann1(coords, t, equations::CompressibleEulerEquations2D)
    x, y = coords

    if x < 0.5
        if y < 0.5
            rho = 0.8;
            v1 = 0
            v2 = 0;
            p = 1
        else 
            rho = 1
            v1 = 3 / sqrt(17)
            v2 = 0
            p = 1
        end
    else 
        if y < 0.5
            rho = 1
            v1 = 0
            v2 = 3/sqrt(17)
            p = 1
        else
            rho = 17 * 0.03125
            v1 = 0
            v2 = 0;
            p = 0.4;
        end
    end

    return prim2cons(SVector(rho, v1, v2, p), equations);
end

"""
    initial_condition_modified_sod_2d(x, t, equations)

Toro modified Sod (Sec. 6.4) extruded uniformly in `y`: left sonic rarefaction,
contact, and shock. Discontinuity at `x = 0.3`.
"""
function initial_condition_modified_sod_2d(x, t, equations)
    if x[1] < 0.3
        return prim2cons(SVector(1.0, 0.75, 0.0, 1.0), equations)
    else
        return prim2cons(SVector(0.125, 0.0, 0.0, 0.1), equations)
    end
end

function Trixi.compute_coefficients!(backend::Nothing, u,
                                     func::typeof(initial_condition_riemann1), t,
                                     mesh::TreeMesh{2}, equations, dg::DG, cache)
    Trixi.@threaded for element in eachelement(dg, cache)
        for j in eachnode(dg), i in eachnode(dg)
            x_node = Trixi.get_node_coords(cache.elements.node_coordinates, equations, dg,
                                           i, j, element)
            if i == 1 # left boundary node
                x_node = SVector(nextfloat(x_node[1]), x_node[2])
            elseif i == nnodes(dg) # right boundary node
                x_node = SVector(prevfloat(x_node[1]), x_node[2])
            end
            if j == 1 # bottom boundary node
                x_node = SVector(x_node[1], nextfloat(x_node[2]))
            elseif j == nnodes(dg) # top boundary node
                x_node = SVector(x_node[1], prevfloat(x_node[2]))
            end

            u_node = func(x_node, t, equations)
            Trixi.set_node_vars!(u, u_node, equations, dg, i, j, element)
        end
    end
end

# `"riemann"` or `"modified_sod"`. String so `trixi_include` can override it.
problem = "modified_sod"
if problem == "modified_sod"
    initial_condition = initial_condition_modified_sod_2d
    periodicity = (false, true)
    boundary_conditions_hyperbolic = (;
                                      x_neg = BoundaryConditionDirichlet(initial_condition),
                                      x_pos = boundary_condition_do_nothing,
                                      y_neg = boundary_condition_periodic,
                                      y_pos = boundary_condition_periodic)
    # Same named tuple works for entropy-gradient / divergence BCs (`mu = 0`).
    boundary_conditions_parabolic = boundary_conditions_hyperbolic
else
    initial_condition = initial_condition_riemann1
    periodicity = true
    boundary_conditions_hyperbolic = Trixi.boundary_condition_periodic
    boundary_conditions_parabolic = Trixi.boundary_condition_periodic
end

polydeg = 3
basis = LobattoLegendreBasis(polydeg)
surface_flux = FluxLaxFriedrichs(max_abs_speed)
volume_flux = flux_central

# "ecav"     → entropy-correction AV + standard DG volume integral
# "adaptive" → VolumeIntegralAdaptive (entropy-correction FV switch), hyperbolic only
# String so `trixi_include` / `convergence_test` can override it.
shock_capturing = "ecav"
if shock_capturing == "adaptive"
    indicator_ec = IndicatorEntropyCorrection(equations, basis)
    volume_integral = VolumeIntegralAdaptive(indicator_ec,
                                             VolumeIntegralWeakForm(),
                                             VolumeIntegralPureLGLFiniteVolume(surface_flux))
else
    volume_integral = VolumeIntegralWeakForm()
end

# LGL mortar traces / Riemann (hanging-face LGL nodes line up). Reverse L² uses
# Gauss quadrature, same sandwich as `MortarL2`.
# `mortar_nodes = "gauss"` → Gauss traces/Riemann, L² reverse
# `P = M_f^{-1} I^T M_m` with lumped `M_f = W_LGL`, `M_m = W_G / 2`.
# SAT is always LGL `SurfaceIntegralWeakForm` (`f*/ω`).
# Use a String so `trixi_include` can override without turning `:gauss` into
# the bare name `gauss`.
mortar_nodes = "gauss_lobatto"
if mortar_nodes == "gauss"
    mortar = MortarEntropy(basis; nodes = :gauss)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
else
    mortar = MortarEntropy(basis; nodes = :gauss_lobatto)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
end
solver = DGSEM(basis, surface_integral, volume_integral, mortar)


coordinates_min = (0.0, 0.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                n_cells_max = 400_000, periodicity = periodicity)

# Checkerboard inside each quadrant. Cells that touch a quadrant boundary stay
# coarse so those faces are conforming (same level on both sides):
#   x = 0.5 (ix = n/2-1 and n/2), y = 0.5 (iy = n/2-1 and n/2),
#   and the periodic wraps x = 0/1, y = 0/1 (which also glue different quadrants).
n_base = 2^initial_refinement_level
dx = (coordinates_max[1] - coordinates_min[1]) / n_base
ix_jump = (n_base ÷ 2 - 1, n_base ÷ 2)  # faces at x = 0.5
iy_jump = (n_base ÷ 2 - 1, n_base ÷ 2)  # faces at y = 0.5
cells_to_refine = Int[]
for cell_id in Trixi.leaf_cells(mesh.tree)
    x, y = Trixi.cell_coordinates(mesh.tree, cell_id)
    ix = round(Int, (x - coordinates_min[1]) / dx - 0.5)
    iy = round(Int, (y - coordinates_min[2]) / dx - 0.5)
    on_quadrant_boundary = ix in (0, n_base - 1, ix_jump...) ||
                           iy in (0, n_base - 1, iy_jump...)
    on_quadrant_boundary = false  # full checkerboard: hanging faces on the slips
    if iseven(ix + iy) && !on_quadrant_boundary
        push!(cells_to_refine, cell_id)
    end
end
Trixi.refine!(mesh.tree, cells_to_refine)

if shock_capturing == "ecav"
    # Identity filter: required by the constructor, unused for ECAV-only.
    VDM = Matrix{Float64}(I, polydeg + 1, polydeg + 1)
    filter = ones(polydeg + 1)
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 VDM = VDM, filter = filter,
                                                 ecav_choice = :ecav,
                                                 combine_rhs = Trixi.True(),
                                                 solver_parabolic = solver_parabolic,
                                                 boundary_conditions = (boundary_conditions_hyperbolic,
                                                                        boundary_conditions_parabolic))
else
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_conditions_hyperbolic)
end

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.2)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)
save_solution = SaveSolutionCallback(interval = 500,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)
stepsize_callback = StepsizeCallback(cfl = 0.5)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        save_solution)

# local_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1e-1, 5.0e-6),
#                                                      variables = (Trixi.density, pressure))
# global_limiter! = PositivityPreservingLimiterLiuZhang(local_limiter!, semi;
#                                                       record_davis_yin_iterations = true)

# sol = solve(ode, SSPRK43(; stage_limiter! = global_limiter!,
#                             step_limiter! = global_limiter!);
#             abstol = 1e-8, reltol = 1e-6,
#             saveat = 0.05,
#             ode_default_options()..., callback = callbacks)

sol = solve(ode, SSPRK43();
            abstol = 1e-8, reltol = 1e-6,
            saveat = 0.05,
            ode_default_options()..., callback = callbacks)

using Plots
pd = PlotData2D(sol)
plot(getmesh(pd), title = "mesh")
savefig("mesh.png")
plot(pd["rho"], title = "rho at t = $(round(sol.t[end]; digits = 3))")
plot!(getmesh(pd))
savefig("rho.png")

###############################################################################
# Domain-integrated entropy at each saved solution time (same quadrature as AnalysisCallback)
entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
        legend = false, title = "entropy integral")
savefig("entropy_integral.png")

