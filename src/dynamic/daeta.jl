# Main DAEta struct
# TODO distinguish definite immutable values vs mutables
"""
    DAEta

Discretization metadata returned by [`add_dae`](@ref) (bound as `dae`). Carries the
ExaModels variable/parameter handles, the problem dimensions, the mesh and collocation
layout, and the handles of the appended structural constraints. `add_dae` transcribes the
DAE onto the `ExaCore`; the objective and any extra constraints are written separately by
the user against this object.

# Fields

Variable / parameter handles (`nothing` when the corresponding class is absent):
- `z`  — differential collocation states `z_{i,j}`
- `zb` — element-boundary states `zb_i` (`i = 0..N`); `zb[end] == zf`
- `y`  — algebraic collocation states `y_{i,j}`
- `u`  — controls `u_{i,j}`
- `p`  — decision parameters (ExaModels `Variable`)
- `theta` — mutable parameters (ExaModels `Parameter`)
- `zf` — terminal boundary state `zb_N`

Dimensions: `nz`, `ny`, `nu`, `np`, `ntheta`; `N` finite elements; `K` collocation points
per element (`= degree`).

Mesh & collocation layout:
- `weights` — reference-element [`BasisWeights`](@ref) (`A`, `b`, `tau`)
- `w`  — quadrature weights over the collocation roots (length `K`)
- `nodes` — element boundaries `τ̂₀..τ̂_N` (length `N+1`)
- `h`  — element lengths `hᵢ` (length `N`)
- `t`  — collocation times `t_{i,j}` (`N × K`)

Convenience:
- `method` — NamedTuple `(basis, polynomial, roots)` of the strategy objects used
- `con` — NamedTuple of appended constraint handles (collocation, continuity, initial,
  algebraic, path, terminal)
"""
struct DAEta{T,TZ,TZB,TY,TU,TP,TTheta,TZF,Th,Tt,TM,TC}
    # variable / parameter handles
    z::TZ
    zb::TZB
    y::TY
    u::TU
    p::TP
    theta::TTheta
    zf::TZF

    # dimensions
    nz::Int
    ny::Int
    nu::Int
    np::Int
    ntheta::Int
    N::Int
    K::Int

    # reference-element weights (Th/Tt: Vector{T}/Matrix{T}, or ExaModels Parameter under adaptive)
    weights::BasisWeights{T}
    w::Vector{T}

    # mesh & collocation layout
    nodes::Vector{T}
    h::Th
    t::Tt

    # convenience
    method::TM
    con::TC
end

function Base.show(io::IO, dae::DAEta)
    print(
        io,
        """
        DAEta

          states     nz = $(dae.nz),  ny = $(dae.ny),  controls nu = $(dae.nu)
          parameters np = $(dae.np),  ntheta = $(dae.ntheta)
          mesh       N  = $(dae.N) elements × K = $(dae.K) collocation points
          method     $(dae.method.basis), $(dae.method.polynomial), $(dae.method.roots)
        """,
    )
end
