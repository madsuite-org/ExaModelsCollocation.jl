# ExaModelsDynamic.jl

Automatically transcribes differential equations as algebraic constraints in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mit-shin-group.github.io/ExaModelsDynamic.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mit-shin-group.github.io/ExaModelsDynamic.jl/dev/)
[![Build Status](https://github.com/mit-shin-group/ExaModelsDynamic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mit-shin-group/ExaModelsDynamic.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Usage
Define a differential-algebraic system

$$
\begin{aligned}
\frac{\mathrm{d}z}{\mathrm{d}t} = f(z,y,u,p,\theta,t), \quad z(t_0) = z_0(y,u,p,\theta) \\
g(z,y,u,p,\theta,t) = 0 \\
c(z,y,u,p,\theta,t) \le 0 \\
h_E(z(t_f),p,\theta) = 0 \\
t \in [t_0,t_f]
\end{aligned}
$$

as Julia functions, each **returning a vector**:
```julia
f(z,y,u,p,theta,t)  = ...   # right-hand side function (= dz/dt)
z0(y,u,p,theta)     = ...   # initial condition        (t = t₀)
g(z,y,u,p,theta,t)  = ...   # algebraic equality       (= 0)
c(z,y,u,p,theta,t)  = ...   # bounds/path inequality   (≤ 0)
hE(zf,p,theta)      = ...   # terminal equality        (= 0)
```
where
* $z(t)$ is a differential state variable
* $y(t)$ is an algebraic state variable
* $u(t)$ is a control variable
* $p$ is a free decision parameter
* $\theta$ is a mutable parameter
* $t$ is time.

Discretize and append the constraints with `add_dae`:
```julia
using ExaModels
using ExaModelsDynamic as EMD

# Initialize ExaCore
core = ExaModels.ExaCore(; concrete = Val(true))

# Time horizon
tspan = (t0, tf)

# Initial guess for optimizer 
init  = (
    u = ...,
    p = ...,
    theta = ...
) 

# Transcribes [dz/dt, z₀, g, c, hE] into algebraic constraints on 'core'
# Returns DAE metadata on 'dae'
core, dae = EMD.add_dae(core, f, z0, tspan, init;
  g          = g,                # Algebraic equality constraints, g(x) = 0
  c          = c,                # Path constraints, c(x) ≤ 0
  hE         = hE,               # Terminal constraints, hE(zf) = 0
  ufixed     = u,                # Fixed control profile, u(t) (defaults to a decision variable if omitted)
  bounds     = (z = ...,         # Variable bounds/simple path constraints
                y = ..., 
                u = ...,
                p = ...),        
  nodes      = tstops,           # Interval node placements (auto-chosen if not supplied)
  degree     = K,                # Degree of interpolating polynomial (defaults to 4)
  basis      = EMD.StateForm(),  # Basis representation for differential states {StateForm(), DerivativeForm()}
  polynomial = EMD.Lagrange(),   # Interpolating polynomial
  roots      = EMD.GaussRadau(), # Collocation points {GaussRadau(),0 GaussLegendre(), GaussLobatto()}
)
```
`add_dae` appends the collocation, continuity, initial-condition, algebraic, path, and terminal constraints to `core` and returns `dae`, which holds the variable handles (`dae.z`, `dae.zf`, `dae.p`, `dae.theta`, ...) and the collocation mesh.


## Example
[Van der Pol oscillator](https://mintoc.de/index.php/Van_der_Pol_Oscillator): drive the oscillator to rest with minimum control effort. The running cost $\int_0^{t_f}(z_1^2 + z_2^2 + u^2)\,\mathrm{d}t$ is transcribed as an augmented state $z_3$ with $\dot z_3 = z_1^2 + z_2^2 + u^2$, so the objective is simply its terminal value $z_3(t_f)$. The damping $\mu = \theta_1$ is a mutable parameter and the initial velocity $z_2(t_0) = p_1$ and the feed-forward drive amplitude $p_2$ are decision variables.

$$
\min_{u,~p}\; z_3(t_f)
\qquad \text{s.t.} \qquad
\begin{aligned}
\dot z_1 &= z_2 \\
\dot z_2 &= \theta_1 z_2 (1-z_1^2)  - z_1 + u + p_2 \cos(t) \\
\dot z_3 &= z_1^2 + z_2^2 + u^2
\end{aligned}
\qquad
z(t_0) = (0,~p_1,~0)
$$

```julia
using ExaModels
using ExaModelsDynamic as EMD

# Right-hand side function, returns [ż₁, ż₂, ż₃] evaluations
function f(z, y, u, p, theta, t)
    return [
        z[2],
        theta[1]*z[2]*(1 - z[1]^2) - z[1] + u[1] + p[2]*cos(t),  # theta₁ = μ, p₂ = drive amplitude, explicit t
        z[1]^2 + z[2]^2 + u[1]^2,                            # running cost
    ]
end

# Initial condition function
z0(y, u, p, theta) = [0.0, p[1], 0.0]

core = ExaModels.ExaCore(; concrete = Val(true))

tspan = (0.0, 5.0)
init  = (u = [0.0], p = [1.0, 0.0], theta = [1.0])      # initial guesses for u, p=[init velocity, drive amp]
                                                    # mutable parameter theta value: μ = 1

core, dae = EMD.add_dae(core, f, z0, tspan, init;
    nodes  = range(0, 5; length = 20),              # uniformly spaced mesh for t ∈ [0, 5]
    degree = 3,                                     # degree 3 interpolating polynomials
    bounds = (p = ([-2.0, -1.0], [2.0, 1.0]),),     # p₁ ∈ [-2, 2],  p₂ ∈ [-1, 1]
)

@add_obj(core, dae.zf[3])                           # objective = accumulated cost at t_f

model = ExaModels.ExaModel(core)

using MadNLP
result = madnlp(model)
```

Because `theta` is a mutable ExaModels parameter, the model can be re-solved without rebuilding:
```julia
for μ in (0.5, 1.0, 2.0)
    set_parameter!(core, dae.theta, [μ])
    madnlp(model)
end
```