# Examples

## Van der Pol optimal control

[Van der Pol oscillator](https://mintoc.de/index.php/Van_der_Pol_Oscillator) optimal
control: drive the oscillator to rest with minimum control effort. The running cost
``\int_0^{t_f}(z_1^2 + z_2^2 + u^2)\,\mathrm{d}t`` is transcribed as an augmented state
``z_3`` with ``\dot z_3 = z_1^2 + z_2^2 + u^2``, so the objective is simply its terminal
value ``z_3(t_f)`` — built from `dae.zf`, separately from `add_dae`.

This variant puts every argument to work: the damping ``\mu = \theta_1`` is a swept
parameter; the initial velocity ``z_2(0) = p_1`` and the feed-forward drive amplitude
``p_2`` are decision variables; and time ``t`` enters the drive explicitly.

```math
\min_{u,\,p}\; z_3(t_f)
\qquad \text{s.t.} \qquad
\begin{aligned}
\dot z_1 &= z_2 \\
\dot z_2 &= \theta_1 (1-z_1^2)\,z_2 - z_1 + u + p_2 \cos t \\
\dot z_3 &= z_1^2 + z_2^2 + u^2
\end{aligned}
\qquad
z(0) = (0,\; p_1,\; 0)
```

```julia
using ExaModels
using ExaModelsDynamic as EMD

# dz/dt — returns [ż₁, ż₂, ż₃]; ż₃ accumulates the running cost
function f(z, y, u, p, θ, t)
    return [
        z[2],
        θ[1]*(1 - z[1]^2)*z[2] - z[1] + u[1] + p[2]*cos(t),  # θ₁ = μ, p₂ = drive amplitude, explicit t
        z[1]^2 + z[2]^2 + u[1]^2,                            # running cost
    ]
end

# Initial condition — the initial velocity z₂(0) is the free decision variable p₁
z0(y, u, p, θ) = [0.0, p[1], 0.0]

core = ExaModels.ExaCore()

tspan = (0.0, 5.0)
init  = (u = [0.0], p = [1.0, 0.0], θ = [1.0])            # guesses: u, [init velocity, drive amp]; value: μ = 1

core, dae = EMD.add_dae(core, f, z0, tspan, init;
    nodes  = range(0, 5; length = 20),                    # uniform mesh (no forward solve → no OrdinaryDiffEq)
    degree = 4,
    bounds = (p = ([-2.0, -1.0], [2.0, 1.0]),),           # p₁ ∈ [-2, 2],  p₂ ∈ [-1, 1]
)

@add_obj(core, dae.zf[3])                                 # objective = accumulated cost at t_f

model = ExaModels.ExaModel(core)
# ... solve `model` with an NLP solver (e.g. MadNLP.jl or Ipopt via NLPModelsIpopt.jl) ...
```

### Free and parametric initial conditions

Because `z0` receives `p` and `θ`, an initial condition can be a decision variable or a
parameter. Above, `z₂(0) = p₁` makes the initial velocity a **free** decision variable
(free-initial-condition optimal control). To make a component a swept **parameter** instead,
reference `θ`, e.g. `z0(y,u,p,θ) = [θ[2], p[1], 0.0]`.

### Parameter sweeps

Because the damping ``\mu`` is an ExaModels parameter (`θ`), you can sweep it and re-solve
**without rebuilding** the model — the constraint structure is unchanged:

```julia
for μ in (0.5, 1.0, 2.0)
    set_parameter!(core, dae.θ, [μ])
    # re-solve ...
end
```
