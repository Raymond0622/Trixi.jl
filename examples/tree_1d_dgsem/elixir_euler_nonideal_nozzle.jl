using OrdinaryDiffEqLowStorageRK
using Trixi
using Trixi: ForwardDiff

# if subsonic, the specified outlet pressure 
# msut be the ambient pressure (p_star)
# for the nozzle problem
p_inflow() = 1e6;
p_outflow() = 0.5e6;
# https://www.sciencedirect.com/science/article/pii/S0029549310005157?ref=pdf_download&fr=RR-2&rr=9bcddc6449a3290b
function nozzle_outlet_subsonic(uA_ll, p_star, equations::NonIdealQuasiCompressibleEulerEquations1D)
    model = equations.eos 
    (; pInf, gamma, q) = model
    a_rho, a_rho_v, a_rho_E, a = uA_ll
    v_ll = a_rho_v / a_rho 
    p_ll = pressure(uA_ll, equations)
    c_ll = speed_of_sound(uA_ll, equations) 
    rho_ll = a_rho / a

    rho_star = rho_ll + (p_star - p_ll)/c_ll^2
    z_ll = rho_ll * c_ll
    v_star = v_ll + (p_ll - p_star)/z_ll

    rho_e_star = (p_star + gamma * pInf)/(gamma - 1) + rho_star * q
    a_rho_E_star = a * (rho_e_star + 0.5 *rho_star * v_star^2)
    return SVector(rho_star * a, 
        a * rho_star * v_star, 
        a_rho_E_star, a)
end

# inlet BC stagnation. It seems like it's a isentropic and isenthalpic 
# process which yields various equalities to solve for the
# the surface variables. see above reference as well
# this is going to follow convention of the paper where
# u_rr is actually the surface value (uM) and we need to find u_star
# which is (uP) in DG notation
# 0 subscript represents x=-∞ stagnation values which are given to us
function inlet_stagnation_nozzle(uA_rr, 
        p_0, T_0, equations::NonIdealQuasiCompressibleEulerEquations1D)
    (; pInf, gamma, q, cv) = equations.equation_of_state
    rho_0 = (p_0 + pInf)/((gamma - 1)*cv*T_0)
    a_rho, a_rho_v, a_rho_E, a = u_rr
    rho_rr = a_rho / a;
    c_rr = speed_of_sound(uA_rr, equations)
    p_rr = pressure(uA_rr, equations)
    vel_rr = a_rho_v/a_rho
    z_rr = rho_rr * c_rr
    g_ratio = (gamma/(gamma - 1))
    K = (p_0 + pInf)/(rho_0^gamma)
    H_bar = gamma *(p_0 + pInf)/(rho_0 *(gamma - 1))
    
    #following the reference convention
    # we use u_star as the velocity at surface (uP point)
    f(u_star) = K * ((1/g_ratio * (1/K))^(g_ratio)) *
        (H_bar- 0.5 * u_star^2)^g_ratio  - pInf - p_rr + z_rr * (vel_rr - u_star)
    
    dfdu(u_star) = -(K * ((1/g_ratio * (1/K))^(g_ratio)
            * (g_ratio) * u_star *
            (H_bar - 0.5 * u_star^2)^(g_ratio/gamma)) + z_rr)
    
    u_star = vel_rr
    iter = 0
    diff = f(u_star)
    while (abs(diff) > 100*eps() && iter < 100) 
        u_star = u_star - f(u_star)/(dfdu(u_star))
        diff = f(u_star);
        iter+= 1
    end
    p_star = p_rr - z_rr *(vel_rr - u_star)
    rho_star = ((p_star + pInf)/K)^(1/gamma)
    rho_e_star = (p_star + gamma * pInf)/(gamma - 1) +rho_star * q
    return SVector(a * rho_star, a * rho_star * u_star, 
        a * (rho_e_star + 0.5 * rho_star * u_star^2), a)
end

eos = StiffenedGas();
volume_flux = flux_asym 
equations = NonIdealQuasiCompressibleEulerEquations1D(eos) 
initial_condition = initial_condition_nozzle

solver = DGSEM(polydeg=3, volume_integral = VolumeIntegralFluxDifferencing(volume_flux),
                surface_flux=flux_lax_friedrichs) 

coordinates_min = 0.0
coordinates_max = 1.0 
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=8, 
                n_cells_max=30_000)
boundary_conditions = (x_neg = BoundaryConditionDirichlet(inlet_stagnation_nozzle),
                        x_pos = BoundaryConditionDirichlet(nozzle_outlet_subsonic));

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.3)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval = analysis_interval)

alive_callback = AliveCallback(analysis_interval = analysis_interval)

save_solution = SaveSolutionCallback(interval = 100,
                                     save_initial_solution = true,
                                     save_final_solution = true,
                                     solution_variables = cons2prim)

stepsize_callback = StepsizeCallback(cfl = 0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
            dt = stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            ode_default_options()..., callback = callbacks);