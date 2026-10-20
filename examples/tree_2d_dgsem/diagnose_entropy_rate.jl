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

function diagnose(label; checkerboard, mortar_type, mortar_nodes = "gauss_lobatto")
    println("\n======== ", label, " ========")
    include_string(Main,
                   with_overrides(setup; checkerboard, mortar_type, mortar_nodes,
                                  level = 3))
    return Base.invokelatest() do
        semi = Main.semi
        mesh, equations, dg, cache = Trixi.mesh_equations_solver_cache(semi)
        println("checkerboard=", Main.checkerboard,
                " mortar_type=", Main.mortar_type,
                " nmortars=", Trixi.nmortars(cache.mortars),
                " nelem=", Trixi.nelements(dg, cache))
        u_ode = Trixi.compute_coefficients(0.0, semi)
        du_ode = similar(u_ode)
        Trixi.rhs!(du_ode, u_ode, semi, 0.0)
        u = Trixi.wrap_array(u_ode, semi)
        du = Trixi.wrap_array(du_ode, semi)
        dsdt = Trixi.analyze(Trixi.entropy_timederivative, du, u, 0.0, mesh, equations, dg,
                             cache)
        println("∑∂S/∂U⋅Uₜ (normalized) = ", dsdt)
        coeffs = cache.artificial_viscosity.coefficients
        println("max |ECAV coeff| = ", maximum(abs, coeffs))
        println("nan du = ", any(isnan, du_ode))
        dsdt
    end
end

diagnose("conforming entropy LGL"; checkerboard = false, mortar_type = "entropy")
diagnose("checkerboard L2"; checkerboard = true, mortar_type = "l2")
diagnose("checkerboard entropy LGL"; checkerboard = true, mortar_type = "entropy")
