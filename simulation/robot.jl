export Freeflyer

# Parameters from:
# asl_free_flyer/free_flyer_node/param/robots/enterprise.yaml 
# asl_free_flyer/free_flyer_control/src/waypoint_py_controller/parameters.py
# tribal knowledge

mutable struct Freeflyer{T<:AbstractFloat}
  mass_ff_min::T
  mass_ff_max::T
  mass_ff::T
  J_ff::T
  J_ff_inv::T
  J_w::T
  J_w_inv::T
  n_thrusters::Int
  r::T
  hard_limit_vel::T
  hard_limit_accel::T 
  hard_limit_ω::T
  hard_limit_α::T

  xb::Vector{T}
end

function Freeflyer{T}() where T
  n_thrusters = 8
  r = 0.157

	mass_ff_min = 15.36
	mass_ff_max = 18.08
  mass_ff = 0.5*(mass_ff_min+mass_ff_max)

  J_ff = 0.184 
  J_ff_inv = inv(J_ff)

  J_w = J_ff/6.43                   # Mark Mote calibration
  J_w_inv = inv(J_w)

  hard_limit_vel = 0.2              # set to match astrobee
  thrust_max = 2*0.185  # max thrust [N] from two thrusters 
  hard_limit_accel = thrust_max/mass_ff 

  hard_limit_ω = 20*π/180
  wheel_torque_limit = 0.593        # 84 oz-in (https://www.pololu.com/product/2822/specs)
  hard_limit_α = J_w_inv*wheel_torque_limit

  xb = [0.; 0.15; 0.]

  # new Freeflyer instance
  return Freeflyer{T}(mass_ff_min,mass_ff_max,mass_ff,J_ff,J_ff_inv,J_w,J_w_inv,n_thrusters,r,
    hard_limit_vel,hard_limit_accel,hard_limit_ω,hard_limit_α,xb)
end
Freeflyer(::Type{T} = Float64; kwargs...) where {T} = Freeflyer{T}(; kwargs...)
