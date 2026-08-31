using OrdinaryDiffEqLowStorageRK
using Trixi

###############################################################################
# 1:1 p-nonconforming interface: two conforming StructuredMesh views glued at x = 1.
# Left half uses polydeg 3, right half polydeg 7. Faces match geometrically;
# quadrature does not.
#
# Neighbor traces stay on their native LGL nodes. The Riemann flux is evaluated
# on a Gauss mortar quadrature (different nodes), then L²-projected back onto
# each element's original face quadrature (`BoundaryConditionCoupledPMortar`).

equations = CompressibleEulerEquations2D(1.4)
initial_condition = initial_condition_convergence_test
coupling_function = (x, u, equations_other, equations_own) -> u

polydeg_left = 3
polydeg_right = 7
solver_left = DGSEM(polydeg = polydeg_left,
                    surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))
solver_right = DGSEM(polydeg = polydeg_right,
                     surface_flux = FluxLaxFriedrichs(max_abs_speed_naive))

coordinates_min = (0.0, 0.0)
coordinates_max = (2.0, 2.0)
cells_per_dimension = (16, 16)
parent_mesh = StructuredMesh(cells_per_dimension, coordinates_min, coordinates_max,
                             periodicity = true)

mesh_left = StructuredMeshView(parent_mesh; indices_min = (1, 1), indices_max = (8, 16))
mesh_right = StructuredMeshView(parent_mesh; indices_min = (9, 1),
                                indices_max = (16, 16))

boundary_conditions_left = (;
                            x_neg = BoundaryConditionCoupledPMortar(2,
                                                                    (:end, :i_forward),
                                                                    Float64,
                                                                    coupling_function,
                                                                    solver_left.basis,
                                                                    solver_right.basis),
                            x_pos = BoundaryConditionCoupledPMortar(2,
                                                                    (:begin, :i_forward),
                                                                    Float64,
                                                                    coupling_function,
                                                                    solver_left.basis,
                                                                    solver_right.basis),
                            y_neg = boundary_condition_periodic,
                            y_pos = boundary_condition_periodic)
boundary_conditions_right = (;
                             x_neg = BoundaryConditionCoupledPMortar(1,
                                                                    (:end, :i_forward),
                                                                    Float64,
                                                                    coupling_function,
                                                                    solver_right.basis,
                                                                    solver_left.basis),
                             x_pos = BoundaryConditionCoupledPMortar(1,
                                                                    (:begin, :i_forward),
                                                                    Float64,
                                                                    coupling_function,
                                                                    solver_right.basis,
                                                                    solver_left.basis),
                             y_neg = boundary_condition_periodic,
                             y_pos = boundary_condition_periodic)

semi_left = SemidiscretizationHyperbolic(mesh_left, equations, initial_condition,
                                         solver_left;
                                         source_terms = source_terms_convergence_test,
                                         boundary_conditions = boundary_conditions_left)
semi_right = SemidiscretizationHyperbolic(mesh_right, equations, initial_condition,
                                          solver_right;
                                          source_terms = source_terms_convergence_test,
                                          boundary_conditions = boundary_conditions_right)
semi = SemidiscretizationCoupled(semi_left, semi_right)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
analysis_callback_left = AnalysisCallback(semi_left, interval = 100)
analysis_callback_right = AnalysisCallback(semi_right, interval = 100)
analysis_callback = AnalysisCallbackCoupled(semi, analysis_callback_left,
                                            analysis_callback_right)
alive_callback = AliveCallback(analysis_interval = 100)
cfl = 0.5
stepsize_callback = StepsizeCallback(cfl = cfl)
callbacks = CallbackSet(summary_callback, analysis_callback, alive_callback,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = 1.0,
            ode_default_options()..., callback = callbacks)
