using Printf
using Trixi

elixir = joinpath(@__DIR__, "elixir_ecav_2d_isentropic_vortex.jl")
iterations = 4

cases = (
    (:l2, MortarL2, "gauss_lobatto",
     "MortarL2, LGL SAT (Gauss quadrature sandwich reverse)"),
    (:entropy_lgl, MortarEntropy, "gauss_lobatto",
     "MortarEntropy, LGL nodes (forward/reverse on Gauss–Lobatto)"),
    (:entropy_gauss, MortarEntropy, "gauss",
     "MortarEntropy, Gauss nodes (GaussQuad SAT)"),
)

means = Dict{Symbol, Any}()
for (key, mortar_type, nodes, title) in cases
    println("="^80)
    println(title)
    println("="^80)
    eocs, _ = convergence_test(elixir, iterations;
                               mortar_type = mortar_type,
                               mortar_nodes = nodes)
    means[key] = Trixi.calc_mean_convergence(eocs)
end

println("\n", "="^80)
println("Mean experimental orders of convergence (polydeg = 3, expected ≈ 4)")
println("="^80)
println("                    rho     rho_v1  rho_v2  rho_e")
@printf("L2  MortarL2        %.2f    %.2f    %.2f    %.2f\n", means[:l2][:l2]...)
@printf("L2  Entropy LGL     %.2f    %.2f    %.2f    %.2f\n", means[:entropy_lgl][:l2]...)
@printf("L2  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:entropy_gauss][:l2]...)
@printf("L∞  MortarL2        %.2f    %.2f    %.2f    %.2f\n", means[:l2][:linf]...)
@printf("L∞  Entropy LGL     %.2f    %.2f    %.2f    %.2f\n", means[:entropy_lgl][:linf]...)
@printf("L∞  Entropy Gauss   %.2f    %.2f    %.2f    %.2f\n", means[:entropy_gauss][:linf]...)
