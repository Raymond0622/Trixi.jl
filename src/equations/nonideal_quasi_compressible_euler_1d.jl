# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    NonIdealQuasiCompressibleEulerEquations1D(equation_of_state)

The quasi compressible Euler equations
```math
\frac{\partial}{\partial t}
\begin{pmatrix}
    \rho \\ \rho v_1 \\ \rho e_{total}
\end{pmatrix}
+
\frac{\partial}{\partial x}
\begin{pmatrix}
    \rho v_1 \\ \rho v_1^2 + p \\ (\rho e_{total} + p) v_1
\end{pmatrix}
=
\begin{pmatrix}
    0 \\ 0 \\ 0
\end{pmatrix}
```
for a gas with pressure ``p`` specified by some equation of state in one space dimension.

Here, ``\rho`` is the density, ``v_1`` the velocity, ``e_{total}`` the specific total energy, 
and the pressure ``p`` is given in terms of specific volume ``V = 1/\rho`` and temperature ``T``
by some user-specified equation of state (EOS)
(see [`pressure(V, T, eos::IdealGas)`](@ref), [`pressure(V, T, eos::VanDerWaals)`](@ref)) as
```math
p = p(V, T)
```

Similarly, the internal energy is specified by `e = energy_internal(V, T, eos)`, see
[`energy_internal(V, T, eos::IdealGas)`](@ref), [`energy_internal(V, T, eos::VanDerWaals)`](@ref).

Because of this, the primitive variables are also defined to be `V, v1, T` (instead of 
`rho, v1, p` for `CompressibleEulerEquations1D`). The implementation also assumes 
mass basis unless otherwise specified.     
"""
struct NonIdealQuasiCompressibleEulerEquations1D{EoS <: AbstractEquationOfState} <:
       AbstractNonIdealCompressibleEulerEquations{1, 4}
    equation_of_state::EoS
end

function varnames(::typeof(cons2cons), ::NonIdealQuasiCompressibleEulerEquations1D)
    return ("a_rho", "a_rho_v1", "a_rho_e_total", "a")
end
varnames(::typeof(cons2prim), ::NonIdealQuasiCompressibleEulerEquations1D) = ("V", "v1", "T", "a")

# for plotting with PlotData1D(sol, solution_variables=density_velocity_pressure)
@inline function density_velocity_pressure(uA,
                                           equations::NonIdealQuasiCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    
    V, v1, T, a = cons2prim(u, equations)
    return SVector(rho, v1, pressure(V, T, eos))
end
varnames(::typeof(density_velocity_pressure), ::NonIdealQuasiCompressibleEulerEquations1D) = ("rho",
                                                                                         "v1",
                                                                                         "p", "a")

# Calculate 1D conservative flux for a single point
@inline function flux(uA, orientation::Integer,
                      equations::NonIdealQuasiCompressibleEulerEquations1D)
    eos = equations.equation_of_state

    rho, rho_v1, rho_e_total = A2cons(uA, equations)
    V, v1, T, a = cons2prim(uA, equations)
    p = pressure(V, T, eos)

    # Ignore orientation since it is always "1" in 1D
    f1 = a_rho_v1
    f2 = a_rho_v1 * v1
    f3 = a * (rho_e_total + p) * v1
    return SVector(f1, f2, f3)
end

@inline function flux_asym(uA_ll, uA_rr, orientation::Integer,
                      equations::NonIdealQuasiCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    u_ll = A2cons(uA_ll, equations)
    u_rr = A2cons(uA_rr, equations)

    p_ll = pressure(uA_ll, equations)
    p_rr = pressure(uA_rr, equations)
    
    rho_E_ll = u_ll[3]
    rho_E_rr = u_rr[3]
    V_ll, v1_ll, _, A_ll = cons2prim(uA_ll, equations)
    V_rr, v1_rr, _, A_rr = cons2prim(uA_rr, equations)

    rho_ll = inv(V_ll)
    rho_rr = inv(V_rr)
    
    p_avg  = 0.5 * (p_ll + p_rr)
        
    f1 = 0.5 * (A_ll * rho_ll * v1_ll + A_rr * rho_rr * v1_rr)
    f2 = 0.5 * (A_ll * rho_ll * v1_ll^2 + A_rr * rho_rr * v1_rr^2) + A_ll * p_avg
    f3 = 0.5 * (A_ll * (rho_E_ll + p_ll) * v1_ll + A_rr * (rho_E_rr + p_rr) * v1_rr)
    
    return SVector(f1, f2, f3, 0.0)
end

@inline function flux_lax_friedrichs(uA_ll, uA_rr, normal, equations::NonIdealQuasiCompressibleEulerEquations1D)
    model = equations.equation_of_state
    _, v_ll, _, _ = cons2prim(uA_ll, equations)
    _, v_rr, _, _ = cons2prim(uA_rr, equations)
    lambda = max_abs_speed(uA_ll, uA_rr, normal, equations)
    a = flux_nonsym(uA_ll, uA_rr, equations) * normal - 0.5 * lambda * (uA_ll - uA_rr)
    return SVector(a[1], a[2], a[3], 0.0);
end

# Calculate estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_naive(uA_ll, uA_rr, orientation::Integer,
                                     equations::NonIdealQuasiCompressibleEulerEquations1D)

    V_ll, v1_ll, T_ll, _ = cons2prim(uA_ll, equations)
    V_rr, v1_rr, T_rr, _ = cons2prim(uA_rr, equations)

    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)
    λ_min = v1_ll - c_ll
    λ_max = v1_rr + c_rr

    return λ_min, λ_max
end

# Less "cautious", i.e., less overestimating `λ_max` compared to `max_abs_speed_naive`
@inline function max_abs_speed(uA_ll, uA_rr, orientation::Integer,
                               equations::NonIdealQuasiCompressibleEulerEquations1D)

    V_ll, v1_ll, T_ll, _ = cons2prim(uA_ll, equations)
    V_rr, v1_rr, T_rr, _ = cons2prim(uA_rr, equations)

    v_mag_ll = abs(v1_ll)
    v_mag_rr = abs(v1_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll + c_ll, v_mag_rr + c_rr)
end

@inline function max_abs_speed_naive(uA_ll, uA_rr, orientation::Integer,
                                     equations::NonIdealQuasiCompressibleEulerEquations1D)

    V_ll, v1_ll, T_ll, _ = cons2prim(uA_ll, equations)
    V_rr, v1_rr, T_rr, _ = cons2prim(uA_rr, equations)

    v_mag_ll = abs(v1_ll)
    v_mag_rr = abs(v1_rr)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    return max(v_mag_ll, v_mag_rr) + max(c_ll, c_rr)
end

# More refined estimates for minimum and maximum wave speeds for HLL-type fluxes
@inline function min_max_speed_davis(uA_ll, uA_rr, orientation::Integer,
                                     equations::NonIdealQuasiCompressibleEulerEquations1D)

    V_ll, v1_ll, T_ll, _ = cons2prim(uA_ll, equations)
    V_rr, v1_rr, T_rr, _ = cons2prim(uA_rr, equations)

    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c_ll = speed_of_sound(V_ll, T_ll, eos)
    c_rr = speed_of_sound(V_rr, T_rr, eos)

    λ_min = min(v1_ll - c_ll, v1_rr - c_rr)
    λ_max = max(v1_ll + c_ll, v1_rr + c_rr)

    return λ_min, λ_max
end

@inline function max_abs_speeds(uA, equations::NonIdealQuasiCompressibleEulerEquations1D)
    u = A2cons(uA, equations)
    V, v1, T, _ = cons2prim(u, equations)
    # Calculate primitive variables and speed of sound
    eos = equations.equation_of_state
    c = speed_of_sound(V, T, eos)

    return (abs(v1) + c,)
end

@inline function A2cons(uA, equations::NonIdealQuasiCompressibleEulerEquations1D)
    A = uA[4]
    return SVector{3}(uA[1:3] ./ A)
end

# Convert conservative variables to primitive
@inline function cons2prim(uA, equations::NonIdealQuasiCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    rho, rho_v1, rho_e_total = A2cons(uA, equations)
    V = inv(rho)
    v1 = rho_v1 * V
    e = (rho_e_total - 0.5f0 * rho_v1 * v1) * V
    T = temperature(V, e, eos)
    return SVector(V, v1, T, a)
end

# Convert conservative variables to entropy
@inline function cons2entropy(uA, equations::NonIdealQuasiCompressibleEulerEquations1D)
    V, v1, T, _ = cons2prim(uA, equations)
    eos = equations.equation_of_state
    gibbs = gibbs_free_energy(V, T, eos)
    return inv(T) * SVector(gibbs - 0.5f0 * v1^2, v1, -1)
end

# Convert primitive to conservative variables
@inline function prim2cons(prim, equations::NonIdealQuasiCompressibleEulerEquations1D)
    eos = equations.equation_of_state
    V, v1, T, a = prim
    rho = inv(V)
    rho_v1 = rho * v1
    e = energy_internal(V, T, eos)
    rho_e_total = rho * e + 0.5f0 * rho_v1 * v1
    return a* SVector(rho, rho_v1, rho_e_total, 1.0)
end

@doc raw"""
    entropy_math(cons, equations::NonIdealCompressibleEulerEquations1D)

Calculate mathematical entropy for a conservative state `cons` as
```math
S = -\rho s
```
where `s` is the specific entropy determined by the equation of state.
"""
@inline function entropy_math(uA, equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u = A2cons(uA, equations)
    q = cons2prim(u, equations)
    V = first(q)
    T = last(q)
    a_rho = uA[1]
    S = -a_rho * entropy_specific(V, T, eos)
    return S
end

"""
    entropy(cons, equations::AbstractNonIdealEulerEquations)

Default entropy is the mathematical entropy
[`entropy_math(cons, equations::AbstractNonIdealEulerEquations)`](@ref).
"""
@inline function entropy(cons, equations::AbstractNonIdealCompressibleEulerEquations)
    return entropy_math(cons, equations)
end

@inline function density(uA, equations::AbstractNonIdealCompressibleEulerEquations)
    rho = uA[1] / uA[4]
    return rho
end

@inline function velocity(uA, orientation_or_normal,
                          equations::NonIdealQuasiCompressibleEulerEquations1D)
    return velocity(uA, equations)
end

@inline function velocity(uA, equations::NonIdealQuasiCompressibleEulerEquations1D)
    a_rho = uA[1]
    v1 = uA[2] / a_rho
    return v1
end

@inline function pressure(uA, equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u = A2cons(uA, equations)
    q = cons2prim(u, equations)
    V = q[1]
    T = q[3]
    p = pressure(V, T, eos)
    return p
end

@inline function density_pressure(uA,
                                  equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u = A2cons(uA, equations)
    rho = u[1]
    q = cons2prim(u, equations)
    V = q[1]
    T = q[3]
    p = pressure(V, T, eos)
    return rho * p
end

@inline function energy_internal(uA,
                                 equations::AbstractNonIdealCompressibleEulerEquations)
    eos = equations.equation_of_state
    u = A2cons(uA, equations)
    q = cons2prim(u, equations)
    V = q[1]
    T = q[3]
    e = energy_internal(V, T, eos)
    return e
end

@inline function internal_energy_density(uA,
                                         equations::NonIdealCompressibleEulerEquations1D)
    u = A2cons(uA, equations)
    rho, rho_v1, rho_e_total = u
    rho_e = rho_e_total - 0.5f0 * rho_v1^2 / rho
    return rho_e
end

# The default amplitude and frequency k are consistent with initial_condition_density_wave 
# for CompressibleEulerEquations1D. Note that this initial condition may not define admissible 
# solution states for all non-ideal equations of state!
function initial_condition_density_wave(x, t,
                                        equations::NonIdealCompressibleEulerEquations1D;
                                        amplitude = 0.98, k = 2)
    RealT = eltype(x)
    eos = equations.equation_of_state

    v1 = convert(RealT, 0.1)
    rho = 2e3*(1 + convert(RealT, amplitude) * sinpi(k * (x[1] - v1 * t)))
    p = 1e6

    V = inv(rho)

    # invert for temperature given p, V
    T = 1
    tol = 100 * eps(RealT)
    dp = pressure(V, T, eos) - p
    iter = 1
    while abs(dp) / abs(p) > tol && iter < 100
        dp = pressure(V, T, eos) - p
        dpdT_V = ForwardDiff.derivative(T -> pressure(V, T, eos), T)
        T = max(tol, T - dp / dpdT_V)
        iter += 1
    end
    if iter == 100
        println("Warning: solver for temperature(V, p) did not converge")
    end

    return prim2cons(SVector(V, v1, T, 1.0), equations)
end

#TODO documentation

function initial_condition_nozzle(x, t, equations::NonIdealQuasiCompressibleEulerEquations1D) 

    p_inflow() = 1e6;
    p_outflow() = 0.5e6;
    T0 = 453.0 # Kelvin
    if (x < 0.5) 
        p = 1e6
    else
        p = 0.5e6
    end
    p = (p_outflow() - p_inflow())*x + p_inflow()
    rho_inflow = density_pT(p, T0, equations)
    u_inflow = 0.0

    A = 1 + 0.5 * cos(2*pi*x)
    rho_e = rho_e_rhoP(rho_inflow, p, equations)
    rho_E = rho_e + 0.5 * rho_inflow * u_inflow^2;
    return A* SVector(rho_inflow, rho_inflow * u_inflow, rho_E, 1)

end
end # @muladd