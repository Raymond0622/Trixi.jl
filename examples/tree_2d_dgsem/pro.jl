using StartUpDG
using LinearAlgebra

N = 3
rd = RefElemData(Line(), N; quad_rule_vol=gauss_quad(0,0,N))

# P = (N+1) x 2*(N+1) = [Proj_upper Proj_lower] * [upper_values; lower_values]
#                     = Proj_upper * upper_values + Proj_lower * lower_values

I_U = Trixi.mortar_interpolation_lgl_to_half_gauss(rd.N + 1, 1, Float64)
I_L = Trixi.mortar_interpolation_lgl_to_half_gauss(rd.N + 1, -1, Float64)


r, w = gauss_quad(0,0,N)
rq = [0.5 * (1 .+ r) .- 1; 0.5 * (1 .+ r)]
wq = 0.5 * [w; w]

V = vandermonde(Line(), N, r) / rd.VDM
Mf = V' * diagm(w) * V

# maps from Lobatto to composite Gauss quadrature
# V_modal(gauss) * rd.VDM ^{-1} * u_lgl 
# rd.VDM ^{-1} * u_lgl - maps u_lgl to legendre modal coefficient
# V_modal uses those modal coefficient to get to gauss nodes.
I_U = vandermonde(Line(), N, rq[1:N+1]) / rd.VDM
I_L = vandermonde(Line(), N, rq[N + 2:end])/rd.VDM

[I_U I_L] * [W;W]
I_U' * diagm(w/2) * I_U + I_L' * diagm(w/2) * I_L

Vq = vandermonde(Line(), N, rq) / rd.VDM
P = (Vq' * diagm(wq) * Vq) \ (Vq' * diagm(wq))
# P * rq.^3 - rd.r.^3
Mf - (Vq' * diagm(wq) * Vq)

# This is Trixi.jl's approach
lobatto2gauss = VDM_gauss / rd.VDM
gauss2lobatto = rd.VDM / VDM_gauss
VDM_gauss = vandermonde(Line(), N, r)
upper = vandermonde(Line(), N, rq[1:N+1]) / VDM_gauss
lower = vandermonde(Line(), N, rq[N+2:end]) / VDM_gauss

# inv(Diagonal(w)) * upper * Diagonal(w) ---> "operator"
gauss2lobatto * inv(Diagonal(w)) * [upper' * Diagonal(0.5 * w) lower' * Diagonal(0.5 * w)]