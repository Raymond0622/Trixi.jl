
# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

@doc raw"""
    StiffenedGas{RealT <: Real} <: AbstractEquationOfState 

    This defines the stiffened gas equation of state. Similar to ideal gas law,
    this EOS adds stiffness paramter p_\infty and takes into account of 
    non-zero internal energy reference point q. 
    p + \gamma p_{\infty} = \rho (gamma - 1)(e - q).

    Setting p_{\infty}, q both to zero simplifies it down to ideal gas law.

# https://www.sciencedirect.com/science/article/pii/S0045793015001887
"""
struct StiffenedGas{RealT <: Real} <: AbstractEquationOfState
    pInf::RealT
    q::RealT # is the reference specific internal energy
    gamma::RealT
    cv::RealT
end

""" 

Default constructor is for Stiffened Gas for water, taken from p7 of
    https://www.sciencedirect.com/science/article/pii/S0045793015001887
    See also 


"""

function StiffenedGas(pInf = 1e9, q = -1167*1e3, gamma = 2.35, 
        cv = 1816.0)
    return StiffenedGas(promote(pInf, q, R, gamma, cv)...);
end

function pressure(V, T, eos::StiffenedGas)
    (; pInf, gamma, cv) = eos;
    return (gamma - 1) * cv * T/ V - pInf;
end

function energy_internal(V, T, eos::StiffenedGas)
    (; pInf, gamma, q) = eos;
    return (pressure(V, T, eos) + gamma * pInf)/((gamma - 1) /V) + q;
end

function entropy_specific(V, T, eos::StiffenedGas)
    (; cv, gamma, pInf) = model 
    p = pressure(V, T, eos)
    rho = inv(V)
    return cv*log((p + pInf)/rho^(gamma))
end

function speed_of_sound(V, T, eos::StiffenedGas)
    (; gamma, pInf) = eos
    p = pressure(V, T, eos)
    return sqrt(gamma * (p + pInf)*V)
end

function temperature(V, T, eos::StiffenedGas)
    (; pInf)
    p = pressure(V, T, eos)
    return (p + pInf) * V/((gamma - 1) * cv)
end 

end # @muladd

