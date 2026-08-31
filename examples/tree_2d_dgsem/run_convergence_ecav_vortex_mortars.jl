using Printf
using Trixi

elixir = joinpath(@__DIR__, "elixir_ecav_2d_isentropic_vortex.jl")
iterations = 4

println("="^80)
println("Checkerboard TreeMesh: MortarL2")
println("="^80)
eocs_l2, errors_l2 = Trixi.convergence_test(elixir, iterations)

println("="^80)
println("Checkerboard TreeMesh: MortarEntropy")  
println("="^80)
eocs_ent, errors_ent = convergence_test(elixir, iterations; mortar_type = MortarEntropy)

mean_l2 = Trixi.calc_mean_convergence(eocs_l2)
mean_ent = Trixi.calc_mean_convergence(eocs_ent)

println("\n", "="^80)
println("Mean experimental orders of convergence (polydeg = 3, expected ≈ 4)")
println("="^80)
println("          rho     rho_v1  rho_v2  rho_e")
@printf("L2  L2    %.2f    %.2f    %.2f    %.2f\n", mean_l2[:l2]...)
@printf("L2  Ent   %.2f    %.2f    %.2f    %.2f\n", mean_ent[:l2]...)
@printf("L∞  L2    %.2f    %.2f    %.2f    %.2f\n", mean_l2[:linf]...)
@printf("L∞  Ent   %.2f    %.2f    %.2f    %.2f\n", mean_ent[:linf]...)
