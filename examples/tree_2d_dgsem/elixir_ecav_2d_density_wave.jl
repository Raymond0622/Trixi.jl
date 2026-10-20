using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi

###############################################################################
# Density wave with ECAV on a checkerboard TreeMesh.
# Physical NS viscosity is off (`mu = 0`). The exact solution is time-dependent
# so `convergence_test` can use a short tspan.
#
# Mortar / capturing switches (strings so `trixi_include` / `convergence_test`
# can override them):
#   mortar_type      = "entropy" | "l2"
#   mortar_nodes     = "gauss_lobatto" | "gauss"   (entropy only)
#   reverse_quad     = "gauss" | "gauss_lobatto"   (LGL entropy mortars only)
#   shock_capturing  = "ecav" | "none"
# `mortar_nodes = "gauss"`: GaussLegendre volume nodes, MortarEntropy Gauss
# traces/Riemann, Gauss L2 reverse (`P = W_G^{-1} I^T (W_G/2)`), and the native
# Gauss SAT (`SurfaceIntegralWeakForm` with `M^{-1} L^T`).
# `mortar_nodes = "gauss_lobatto"`: LGL volume + LGL SAT; `reverse_quad` selects
# the large-face reverse mass. `reverse_quad` is not named `reverse_quadrature`
# so `trixi_include` cannot rewrite the `MortarEntropy` keyword.

gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

prandtl_number() = 0.73
mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = Trixi.ParabolicFormulationBassiRebay1()
solver_parabolic = Trixi.ParabolicFormulationLocalDG()

# Trixi's stock wave uses amplitude 0.98 (`rho` down to 0.02). Entropy mortar
# traces then fail in ECAV `entropy2cons` (`(-V5)^γ` with V5 > 0).
function initial_condition_density_wave_ecav(x, t, equations::CompressibleEulerEquations2D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    rho = 1 + convert(RealT, 0.1) * sinpi(2 * (x[1] + x[2] - t * (v1 + v2)))
    rho_v1 = rho * v1
    rho_v2 = rho * v2
    p = 20
    rho_e_total = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2)
    return SVector(rho, rho_v1, rho_v2, rho_e_total)
end
initial_condition = initial_condition_density_wave_ecav

polydeg = 2
volume_flux = flux_ranocha
surface_flux = flux_ranocha
#volume_integral = VolumeIntegralFluxDifferencing(volume_flux)

mortar_type = "entropy"
mortar_nodes = "gauss"
reverse_quad = "gauss"
if mortar_type == "l2"
    basis = LobattoLegendreBasis(polydeg)
    mortar = MortarL2(basis)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    volume_integral = VolumeIntegralWeakForm()
elseif mortar_nodes == "gauss"
    basis = GaussLegendreBasis(polydeg)
    mortar = MortarEntropy(basis; nodes = :gauss, reverse_quadrature = :gauss)
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    volume_integral = VolumeIntegralWeakForm()
else
    basis = LobattoLegendreBasis(polydeg)
    mortar = MortarEntropy(basis; nodes = :gauss_lobatto,
                           reverse_quadrature = Symbol(reverse_quad))
    surface_integral = SurfaceIntegralWeakForm(surface_flux)
    volume_integral = VolumeIntegralWeakForm()
end
solver = DGSEM(basis, surface_integral, volume_integral, mortar)

# Mesh switch (so `trixi_include` / `convergence_test` can override it):
#   checkerboard = true  → 2:1 hanging faces (nmortars > 0)
#   checkerboard = false → uniform conforming TreeMesh
checkerboard = true;

coordinates_min = (-1.0, -1.0)
coordinates_max = (1.0, 1.0)
initial_refinement_level = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

if checkerboard
    # Read the level from the tree so `convergence_test` overrides of
    # `initial_refinement_level` still produce a scaled checkerboard.
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

# "ecav" → entropy-correction AV (combined hyperbolic + viscous RHS, mortar ψ
#          residual). "none" → hyperbolic only.
shock_capturing = "ecav"
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
                                                 boundary_conditions = (boundary_condition_periodic,
                                                                        boundary_condition_periodic))
else
    semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
                                        boundary_conditions = boundary_condition_periodic)
end

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     extra_analysis_integrals = (entropy,))
alive_callback = AliveCallback(analysis_interval = analysis_interval)
stepsize_callback = StepsizeCallback(cfl = 0.8)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

# `dt` is overwritten by `StepsizeCallback`. Do not omit that callback with
# CarpenterKennedy2N54 — a raw `dt = 1` is far too large for this mesh.
saveat_times = tspan[2]
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, saveat = saveat_times,
            ode_default_options()..., callback = callbacks)

make_plots = true
if make_plots
    using Plots
    pd = PlotData2D(sol)
    plot(getmesh(pd), title = "mesh")

    entropy_integral = [Trixi.integrate(entropy, u, semi) for u in sol.u]
    println("entropy integral  S(0) = ", entropy_integral[1],
            "  S(end) = ", entropy_integral[end],
            "  ΔS = ", entropy_integral[end] - entropy_integral[1])
    plot(sol.t, entropy_integral, xlabel = "t", ylabel = "∫ S dV / |Ω|",
         legend = false, title = "entropy integral")
    savefig("entropy_integral.png")
end
