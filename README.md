# ExaModelsDynamic.jl

Automatically transcribes differential equations as algebraic constraints in [ExaModels.jl](https://github.com/exanauts/ExaModels.jl).

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://mit-shin-group.github.io/ExaModelsDynamic.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://mit-shin-group.github.io/ExaModelsDynamic.jl/dev/)
[![Build Status](https://github.com/mit-shin-group/ExaModelsDynamic.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/mit-shin-group/ExaModelsDynamic.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Usage
Define a differential-algebraic system

$$
\frac{\mathrm{d}z}{\mathrm{d}t} = f(z,y,u,p,\theta,t),
\quad
z(t_0) = z_0(y,u,p,\theta),
\quad
g(z,y,u,p,\theta,t) = 0,
\quad
c(z,y,u,p,\theta,t) \le 0,
\quad
h_E(z(t_f),p,\theta) = 0
\quad
t \in [t_0,t_f]
$$

as Julia functions, each **returning a vector**:
```julia
f(z,y,u,p,θ,t)  = ...   # dz/dt
z0(y,u,p,θ)     = ...   # initial condition at t = t₀
g(z,y,u,p,θ,t)  = ...   # algebraic equality (= 0)
c(z,y,u,p,θ,t)  = ...   # path inequality    (≤ 0)
hE(zf,p,θ)      = ...   # terminal equality  (= 0)
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

core = ExaModels.ExaCore(; backend = CUDA.Backend(), concrete = Val(true))

tspan = (t0, tf) # Time horizon
init  = (u = ..., p = ..., θ = ...) # Initial guesses

# Transcribes [dz/dt, z₀, g, c, hE] into algebraic constraints on `core`
# Returns DAE metadata on 'dae'
core, dae = EMD.add_dae(core, f, z0, tspan, init;
  g          = g,                           # Algebraic equality constraints g(x) = 0
  c          = c,                           # Path constraints c(x) ≤ 0
  hE         = hE,                          # Terminal constraints hE(z_f) = 0
  u          = u,                           # Control profile u(t) (defaults to a decision variable if omitted)
  bounds     = (z = (zL, zU), y = ..., u = ..., p = ...), # Variable bounds/simple path constraints
  nodes      = tstops,                      # Interval node placements (auto-chosen if not supplied)
  degree     = K,                           # Degree of interpolating polynomial (defaults to 4)
  basis      = EMD.Lagrange(),              # Derivative representation {Lagrange(), RungeKutta()}
  polynomial = EMD.LagrangeInterpolation(), # Interpolating polynomial
  roots      = EMD.GaussRadau(),            # Collocation points {GaussRadau(), GaussLegendre(), GaussLobatto()}
)
```
`add_dae` appends the collocation, continuity, initial-condition, algebraic, path, and terminal constraints to `core` and returns `dae`, which holds the variable handles (`dae.z`, `dae.z_f`, `dae.p`, `dae.θ`, ...) and the collocation mesh.


## Example
[Van der Pol oscillator](https://mintoc.de/index.php/Van_der_Pol_Oscillator) optimal control: drive the oscillator to rest with minimum control effort. The running cost $\int_0^{t_f}(x_1^2 + x_2^2 + u^2)\\,\mathrm{d}t$ is transcribed as an augmented state $x_3$ with $\dot x_3 = x_1^2 + x_2^2 + u^2$, so the objective is simply its terminal value $x_3(t_f)$.

This variant puts every argument to work: the damping $\mu = \theta_1$ is a swept parameter; the initial velocity $x_2(0) = p_1$ and the feed-forward drive amplitude $p_2$ are decision variables; and time $t$ enters the drive explicitly.

$$
\min_{u,\,p}\; x_3(t_f)
\qquad \text{s.t.} \qquad
\begin{aligned}
\dot x_1 &= x_2 \\
\dot x_2 &= \theta_1 (1-x_1^2)\\,x_2 - x_1 + u + p_2 \cos t \\
\dot x_3 &= x_1^2 + x_2^2 + u^2
\end{aligned}
\qquad
x(0) = (0,\; p_1,\; 0)
$$

```julia
using ExaModels
using ExaModelsDynamic as EMD

# dz/dt — returns [ẋ₁, ẋ₂, ẋ₃]; ẋ₃ accumulates the running cost
function f(z, y, u, p, θ, t)
    return [
        z[2],
        θ[1]*(1 - z[1]^2)*z[2] - z[1] + u[1] + p[2]*cos(t),  # θ₁ = μ, p₂ = drive amplitude, explicit t
        z[1]^2 + z[2]^2 + u[1]^2,                            # running cost
    ]
end

# Initial condition — the initial velocity x₂(0) is the free decision variable p₁
z0(y, u, p, θ) = [0.0, p[1], 0.0]

core = ExaModels.ExaCore()

tspan = (0.0, 5.0)
init  = (u = [0.0], p = [1.0, 0.0], θ = [1.0])            # guesses: u, [init velocity, drive amp]; value: μ = 1

core, dae = EMD.add_dae(core, f, z0, tspan, init;
    nodes  = range(0, 5; length = 20),                    # uniform mesh (no forward solve → no OrdinaryDiffEq)
    degree = 4,
    bounds = (p = ([-2.0, -1.0], [2.0, 1.0]),),           # p₁ ∈ [-2, 2],  p₂ ∈ [-1, 1]
)

@add_obj(core, dae.z_f[3])                                # objective = accumulated cost at t_f

model = ExaModels.ExaModel(core)
# ... solve `model` with an NLP solver (e.g. MadNLP.jl or Ipopt via NLPModelsIpopt.jl) ...
```

Because the damping $\mu$ is an ExaModels parameter (`θ`), you can sweep it and re-solve **without rebuilding** the model:
```julia
for μ in (0.5, 1.0, 2.0)
    set_parameter!(core, dae.θ, [μ])
    # re-solve — the constraint structure is unchanged
end
```

See the [Examples](https://mit-shin-group.github.io/ExaModelsDynamic.jl/dev/examples/) page for more, including free/parametric initial conditions and parameter sweeps.
