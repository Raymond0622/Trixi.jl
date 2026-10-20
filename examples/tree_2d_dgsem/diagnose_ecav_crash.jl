using OrdinaryDiffEqLowStorageRK
using Trixi

elixir = joinpath(@__DIR__, "elixir_ecav_2d_isentropic_vortex.jl")
code = read(elixir, String)
cut = first(findfirst("###############################################################################\n# run the simulation", code))
setup = code[1:cut - 1]

function with_overrides(src; checkerboard, mortar_type, mortar_nodes, level)
    s = src
    s = replace(s, r"mortar_type = \".*\"" => "mortar_type = $(repr(mortar_type))")
    s = replace(s, r"mortar_nodes = \".*\"" => "mortar_nodes = $(repr(mortar_nodes))")
    s = replace(s, r"checkerboard = (true|false);" => "checkerboard = $(checkerboard);")
    s = replace(s, r"initial_refinement_level = \d+" => "initial_refinement_level = $(level)")
    return s
end

function diagnose(label; checkerboard, mortar_type, mortar_nodes = "gauss_lobatto",
                  nsteps = 10_000)
    println("\n======== ", label, " ========")
    include_string(Main,
                   with_overrides(setup; checkerboard, mortar_type, mortar_nodes,
                                  level = 3))
    return Base.invokelatest() do
        semi = Main.semi
        mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
        u_ode = Trixi.compute_coefficients(0.0, semi)
        du_ode = similar(u_ode)
        Trixi.rhs!(du_ode, u_ode, semi, 0.0)
        coeffs = cache.artificial_viscosity.coefficients
        println("nmortars=", Trixi.nmortars(cache.mortars),
                " nelem=", Trixi.nelements(dg, cache))
        println("t=0  max|μ|=", maximum(abs, coeffs),
                "  min μ=", minimum(coeffs),
                "  max μ=", maximum(coeffs),
                "  nan du=", any(isnan, du_ode))

        tspan = (0.0, 0.2)
        ode = semidiscretize(semi, tspan)
        stepsize_callback = StepsizeCallback(cfl = 0.8)
        alive = AliveCallback(analysis_interval = 50)
        local ok = true
        try
            sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false);
                        dt = 1.0, save_everystep = false, adaptive = false,
                        maxiters = nsteps,
                        callback = CallbackSet(stepsize_callback, alive),
                        ode_default_options()...)
            uend = sol.u[end]
            println("retcode=", sol.retcode, "  t=", sol.t[end],
                    "  max|μ|=", maximum(abs, cache.artificial_viscosity.coefficients),
                    "  nan u=", any(isnan, uend),
                    "  inf u=", any(isinf, uend))
        catch e
            ok = false
            println("CRASH: ", sprint(showerror, e))
        end
        ok
    end
end

diagnose("conforming L2"; checkerboard = false, mortar_type = "l2")
diagnose("conforming entropy LGL"; checkerboard = false, mortar_type = "entropy")
diagnose("checkerboard entropy LGL"; checkerboard = true, mortar_type = "entropy")
diagnose("checkerboard entropy Gauss"; checkerboard = true, mortar_type = "entropy",
          mortar_nodes = "gauss")
