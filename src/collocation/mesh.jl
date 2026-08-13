# Collocation mesh data structure
"""
    CollocationMesh{TN,TH,TT,PH,PT}

Contains collocation mesh details.

# Fields
- `nodes` : interval boundaries `nodes[i]`, i=1,…,N+1, for N intervals
- `h`     : interval lengths `h[i] = nodes[i+1] - nodes[i]`, for i=1,…,N
- `t`     : collocation times `t[i,j] = nodes[i] + h[i]*tau[j]`, for i=1,…,N, j=1,…,K

If `adaptive = false`, then `h` and `t` are numeric values.
If `adaptive = true`, then `h` and `t` are `ExaModels` parameters.
"""
struct CollocationMesh{TN,TH,TT,PH,PT}
    nodes::TN
    h::TH
    t::TT
    hpar::PH
    tpar::PT
end

# Helpers for keeping h,t name regardless of adaptive = true/false
Base.getproperty(m::CollocationMesh, name::Symbol) =
    name === :h ? _resolve(getfield(m, :hpar), getfield(m, :h)) :
    name === :t ? _resolve(getfield(m, :tpar), getfield(m, :t)) :
    getfield(m, name)
_resolve(par, arr) = par === nothing ? arr : par
Base.propertynames(::CollocationMesh) = (:nodes, :h, :t)

# Parse resolved boundaries and reference points into a CollocationMesh
function _get_mesh(nodes::AbstractVector, taus::AbstractVector)
    # Interval lengths h[i]
    h = diff(nodes)
    N = length(h)

    # Collocation points tau[j], j=1,...,K
    K = length(taus)

    # Collocation times t[i,j], i=1,...,N, j=1,...,K
    t = Matrix{eltype(h)}(undef, N, K)
    for i in 1:N, j in 1:K
        t[i,j] = nodes[i] + h[i] * taus[j]
    end

    return CollocationMesh(nodes, h, t, nothing, nothing)
end

# If adaptive = true, append parameter handles for h[i], t[i,j]
_with_parameters(mesh::CollocationMesh, hpar, tpar) =
    CollocationMesh(mesh.nodes, getfield(mesh, :h), getfield(mesh, :t), hpar, tpar)
