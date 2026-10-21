# Gauss-only EOC for the 3D density wave on a checkerboard TreeMesh.
#   {none, ecav} × polydeg {1, 2, 3, 4}
# MortarEntropy, levels 3→6, tspan (0, 0.1).
#
#   JULIA_NUM_THREADS=128 julia --project=run examples/tree_3d_dgsem/run_eoc_3d_density_wave.jl

using Trixi
using OrdinaryDiffEqLowStorageRK
using OrdinaryDiffEqSSPRK
using Printf

const CHECKERBOARD = true
const OUTDIR = joinpath(@__DIR__, "out_eoc_density_wave_checkerboard_L3_gauss128")
mkpath(OUTDIR)

const ELIXIR = joinpath(@__DIR__, "elixir_ecav_3d_density_wave.jl")
const NODES = (("gauss", "gauss"),)
const CAPTURINGS = ("none", "ecav")
const POLYDEGS = (1, 2, 3, 4)
const EOC_ITERS = 4
const EOC_LEVEL = 3
const EOC_TSPAN = (0.0, 0.1)

function mean_eoc(eocs, kind, var)
    col = eocs[kind][:, var]
    return sum(col) / length(col)
end

cases = collect(Iterators.product(NODES, CAPTURINGS, POLYDEGS))
n_cases = length(cases)
eoc_csv = joinpath(OUTDIR, "eoc.csv")
open(eoc_csv, "w") do io
    println(io,
            "case,nodes,capturing,polydeg,status,",
            "mean_l2_rho,mean_l2_rho_v1,mean_l2_rho_v2,mean_l2_rho_v3,mean_l2_rho_e,",
            "eoc_l2_rho_1,eoc_l2_rho_2,eoc_l2_rho_3,",
            "err_l2_rho_L3,err_l2_rho_L4,err_l2_rho_L5,err_l2_rho_L6,elapsed_s")
end

println("3D density-wave EOC: $n_cases cases, checkerboard=$CHECKERBOARD, levels $EOC_LEVEL to $(EOC_LEVEL + EOC_ITERS - 1), tspan $EOC_TSPAN")
flush(stdout)

for (icase, ((mortar_nodes, nodes_name), capturing, polydeg)) in enumerate(cases)
    case = "density_wave_$(capturing)_$(nodes_name)_N$(polydeg)"
    println()
    println("="^80)
    println("CASE $icase / $n_cases  $case")
    println("="^80)
    flush(stdout)

    t_case = time()
    status = "ok"
    mean_l2 = fill(NaN, 5)
    eoc_l2_rho = fill(NaN, 3)
    err_l2_rho = fill(NaN, 4)
    try
        eocs, errorsmatrix = convergence_test(ELIXIR, EOC_ITERS;
                                              polydeg = polydeg,
                                              mortar_nodes = mortar_nodes,
                                              shock_capturing = capturing,
                                              checkerboard = CHECKERBOARD,
                                              tspan = EOC_TSPAN,
                                              saveat_times = EOC_TSPAN[2],
                                              initial_refinement_level = EOC_LEVEL)
        for v in 1:5
            mean_l2[v] = mean_eoc(eocs, :l2, v)
        end
        eoc_l2_rho[1] = eocs[:l2][1, 1]
        eoc_l2_rho[2] = eocs[:l2][2, 1]
        eoc_l2_rho[3] = eocs[:l2][3, 1]
        err_l2_rho[1] = errorsmatrix[:l2][1, 1]
        err_l2_rho[2] = errorsmatrix[:l2][2, 1]
        err_l2_rho[3] = errorsmatrix[:l2][3, 1]
        err_l2_rho[4] = errorsmatrix[:l2][4, 1]
    catch err
        status = "FAIL: $(typeof(err)): $(err)"
        @error "case failed" case exception = (err, catch_backtrace())
    end
    elapsed = time() - t_case
    open(eoc_csv, "a") do io
        @printf(io, "%s,%s,%s,%d,%s,%.6e,%.6e,%.6e,%.6e,%.6e,%.4f,%.4f,%.4f,%.6e,%.6e,%.6e,%.6e,%.1f\n",
                case, nodes_name, capturing, polydeg, status,
                mean_l2[1], mean_l2[2], mean_l2[3], mean_l2[4], mean_l2[5],
                eoc_l2_rho[1], eoc_l2_rho[2], eoc_l2_rho[3],
                err_l2_rho[1], err_l2_rho[2], err_l2_rho[3], err_l2_rho[4], elapsed)
    end
    @printf("RESULT %s  mean L2 ρ EOC = %.2f   (%.2f, %.2f, %.2f)  errors %.3e %.3e %.3e %.3e  [%.0fs] %s\n",
            case, mean_l2[1], eoc_l2_rho[1], eoc_l2_rho[2], eoc_l2_rho[3],
            err_l2_rho[1], err_l2_rho[2], err_l2_rho[3], err_l2_rho[4], elapsed, status)
    flush(stdout)
end

println()
println("wrote ", eoc_csv)
