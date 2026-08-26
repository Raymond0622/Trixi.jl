using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using StartUpDG
using Trixi

###############################################################################
# semidiscretization of the ideal compressible Navier-Stokes equations

function initial_condition_kelvin_helmholtz_instability(x, t,
                                                        equations::CompressibleEulerEquations2D)
    # change discontinuity to tanh
    # typical resolution 128^2, 256^2
    # domain size is [-1,+1]^2
    slope = 15
    B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
    rho = 0.5 + 0.75 * B
    v1 = 0.5 * (B - 1)
    v2 = 0.1 * sin(2 * pi * x[1])
    p = 1.0
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

coordinates_min = (-1.0, -1.0) # minimum coordinates (min(x), min(y))
coordinates_max = (1.0, 1.0) # maximum coordinates (max(x), max(y))
tspan = (0.0, 5.0)

function init_random(x, t, equations) 
    return prim2cons(SVector(2+rand(), rand(), rand(), 2+rand()), equations)
end
initial_condition = initial_condition_kelvin_helmholtz_instability
#initial_condition = init_random
periodicity = (true, true)
@inline mu() = 0.0 # Re = 200
prandtl_number() = 0.73

equations = CompressibleEulerEquations2D(1.4)
equations_parabolic = CompressibleNavierStokesDiffusion2D(equations, mu = mu(),
                                                          Prandtl = prandtl_number(),
                                                          gradient_variables = GradientVariablesEntropy())

dg = DGSEM(polydeg = 3, surface_flux = FluxLaxFriedrichs(max_abs_speed),
           volume_integral = VolumeIntegralWeakForm())
#         #    volume_integral = VolumeIntegralFluxDifferencing(flux_shima_etal))

initial_refinement_level = 5
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level = initial_refinement_level,
                periodicity = periodicity, n_cells_max = 400_000)

#solver_parabolic = ParabolicFormulationBassiRebay1()
solver_parabolic = ParabolicFormulationLocalDG()
    N = Trixi.polydeg(dg)
    filter = [((i - 1)/N)^2 for i in 1:N+1]
    #filter = [(i - 1) < 5 ? 0 : 1 for i in 1:N+1]
    #filter = [1 for i in 1:N+1]
    nodes = Trixi.get_nodes(dg.basis)
    VDM = vandermonde(Line(), N, nodes)
    ecav_choice = :ecav
semi = SemidiscretizationArtificialViscosity(mesh, (equations, equations_parabolic),
                                                VDM, filter,
                                                initial_condition, dg;
                                                combine_rhs = Trixi.True(),
                                                solver_parabolic = solver_parabolic)


# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver;
#                     boundary_conditions = boundary_condition_periodic)


###############################################################################
# ODE solvers, callbacks etc.
# Create ODE problem with time span `tspan`
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()
alive_callback = AliveCallback(alive_interval = 100)
callbacks = CallbackSet(summary_callback, alive_callback)
# analysis_interval = 1000
# analysis_callback = AnalysisCallback(semi, interval = analysis_interval)
# callbacks = CallbackSet(summary_callback, alive_callback, analysis_callback) #, amr_callback)

###############################################################################
# run the simulation

# stage_limiter! = PositivityPreservingLimiterZhangShu(thresholds = (1.0e-6, 1.0e-6),
#                                                      variables = (Trixi.density, pressure))
# solver = SSPRK43(stage_limiter!, stage_limiter!)
solver = SSPRK43()

sol = solve(ode, solver; abstol = 1e-6, reltol = 1e-4, adaptive=true,# dt = 1e-8,
            ode_default_options()..., callback = callbacks)
