using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using LinearAlgebra: I
using Trixi

###############################################################################
# 3D density wave. Physical NS viscosity is off (`mu = 0`). The exact solution
# is the initial condition evaluated at time t, so `convergence_test` can use
# a short tspan.
#
# Switches (strings / Bool so `trixi_include` can override them):
#   mortar_nodes     = "gauss_lobatto" | "gauss"
#   shock_capturing  = "ecav" | "none"
#   checkerboard     = false | true
# `mortar_nodes = "gauss"`: GaussLegendre volume nodes and MortarEntropy.
# `mortar_nodes = "gauss_lobatto"`: LGL volume nodes and MortarEntropy
# (`cons2entropy` on the face, forward map, `entropy2cons`).

gamma = 1.4
equations = CompressibleEulerEquations3D(gamma)

mu() = 0.0
equations_parabolic = CompressibleNavierStokesDiffusion3D(equations, mu = mu(),
                                                          Prandtl = 0.73,
                                                          gradient_variables = GradientVariablesEntropy())
solver_parabolic = ParabolicFormulationLocalDG()

function initial_condition_density_wave_ecav(x, t, equations::CompressibleEulerEquations3D)
    RealT = eltype(x)
    v1 = convert(RealT, 0.1)
    v2 = convert(RealT, 0.2)
    v3 = convert(RealT, 0.3)
    rho = 1 + convert(RealT, 0.1) * sinpi(2 * (x[1] + x[2] + x[3] - t * (v1 + v2 + v3)))
    p = 20
    rho_e = p / (equations.gamma - 1) + 0.5f0 * rho * (v1^2 + v2^2 + v3^2)
    return SVector(rho, rho * v1, rho * v2, rho * v3, rho_e)
end
initial_condition = initial_condition_density_wave_ecav

polydeg = 2
mortar_nodes = "gauss"
shock_capturing = "ecav"
checkerboard = true
surface_flux = flux_lax_friedrichs

if mortar_nodes == "gauss"
    basis = GaussLegendreBasis(polydeg)
    mortar = MortarEntropy(basis; nodes = :gauss, reverse_quadrature = :gauss)
else
    basis = LobattoLegendreBasis(polydeg)
    mortar = MortarEntropy(basis; nodes = :gauss_lobatto,
                           reverse_quadrature = :gauss_lobatto)
end
solver = DGSEM(basis, SurfaceIntegralWeakForm(surface_flux), VolumeIntegralWeakForm(),
               mortar)

coordinates_min = (-1.0, -1.0, -1.0)
coordinates_max = (1.0, 1.0, 1.0)
initial_refinement_level = 2
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = true)

if checkerboard
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
end

if shock_capturing == "ecav"
    n = polydeg + 1
    semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                 initial_condition, solver;
                                                 VDM = Matrix{Float64}(I, n, n),
                                                 filter = ones(n),
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
tspan = (0.0, 10.0)
saveat_times = tspan[2]
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval = analysis_interval,
                                     save_analysis = false)
alive_callback = AliveCallback(analysis_interval = 100)
stepsize_callback = StepsizeCallback(cfl = 0.8)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0, saveat = saveat_times,
            ode_default_options()..., callback = callbacks)
