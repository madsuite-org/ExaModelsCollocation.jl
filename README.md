# ExaModelsDynamic.jl

Automatically transcribes differential equations as algebraic constraints in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

[![Build Status](https://github.com/mit-shin-group/ExaModelsDynamic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mit-shin-group/ExaModelsDynamic.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Usage
Define differential and algebraic equations in the form of

$$
\frac{\mathrm{d}z}{\mathrm{d}t} = f(z,y,u,p,t),
\quad
z(t=0) = z_0(y,u,p),
\quad
g(z,y,u,p,t) = 0
$$

as numeric functions:
```julia
dzdt(z,y,u,p,t) = f(z,y,u,p,t)
z0(y,u,p)       = ... 
g(z,y,u,p,t)    = 0
```
where
* $z(t)$ is a differential state variable
* $y(t)$ is an algebraic state variable
* $u(t)$ is a control variable
* $p$ is a parameter
* $t$ is time.

Apply discretization and construct algebraic equations based on chosen method:
```julia
using ExaModels
using ExaModelsDynamic as EMD

core = ExaModels.ExaCore(; backend = CUDA.Backend(), concrete = Val(true))

K = 5 # Degree of interpolating polynomial
core = EMD.add_dae(core,
  dzdt(z,y,u,p,t), # Differential equations
  z0(y,u,p);       # Initial conditions
  g(z,y,u,p,t)                                        # Algebraic constraints
  nodes      = t_stops                                # Interval node placements, automatically chosen if not supplied
  basis      = EMD.Lagrange()                         # Basis representation for derivative {Lagrange(), RungeKutta()}
  polynomial = EMD.LagrangeInterpolation(degree = K), # Interpolating polynomial
  roots      = EMD.GaussRadau(),                      # Interpolation points {GaussRadau(), GaussLegendre(), GaussLobatto()}
)
```
