# Collocation mesh data structure
"""
    CollocationMesh{TN,TH,TT,PH,PT,TS}

Contains collocation mesh details.

# Fields
- `nodes` : interval boundaries `nodes[i]`, i=1,…,N+1, for N intervals
- `h`     : interval lengths `h[i] = nodes[i+1] - nodes[i]`, for i=1,…,N
- `t`     : collocation times `t[i,j] = nodes[i] + h[i]*tau[j]`, for i=1,…,N, j=1,…,K

If `M > 1` meshes, index by `nodes[m,i]`, `h[m,i]`, `t[m,i,j]` instead
If `adaptive = false`, then `h` and `t` are numeric values
If `adaptive = true`, then `h` and `t` are `ExaModels` parameters
If `unknown_horizon` = true, then tscale[m] is introduced to scale the time horizon per mesh
"""
struct CollocationMesh{TN,TH,TT,PH,PT,TS}
    nodes::TN
    h::TH
    t::TT
    hpar::PH
    tpar::PT
    horizon_scale::TS
end

# Helpers for keeping h,t name regardless of adaptive = true/false
Base.getproperty(m::CollocationMesh, name::Symbol) =
    name === :h ? _resolve(getfield(m, :hpar), getfield(m, :h)) :
    name === :t ? _resolve(getfield(m, :tpar), getfield(m, :t)) :
    getfield(m, name)
_resolve(par, arr) = par === nothing ? arr : par
Base.propertynames(::CollocationMesh) = (:nodes, :h, :t)

# How many meshes there are
_num_meshes(m::CollocationMesh) =
    getfield(m, :h) isa AbstractMatrix ? size(getfield(m, :h), 1) : 1

# Parse user node inputs as as usable form
_nodes_input(nodes::AbstractVector{<:Real}) = collect(float.(nodes))
_nodes_input(nodes::AbstractMatrix) = collect(float.(nodes))
function _nodes_input(nodes::AbstractVector)
    isempty(nodes) && throw(ArgumentError("nodes requires at least one mesh, got none"))
    n = length(first(nodes))
    all(m -> length(m) == n, nodes) || throw(ArgumentError(
        "nodes: every mesh needs the same $n boundaries. For meshes of different " *
        "interval counts, build a core per group with `ExaCore(c; tag = Collocation(…))`"
    ))
    return permutedims(reduce(hcat, [collect(float.(m)) for m in nodes]))
end

# Parse resolved boundaries and reference points into a CollocationMesh
function _get_mesh(nodes::AbstractVector, taus::AbstractVector)
    # Interval lengths h[i]
    h = diff(nodes)
    N = length(h)

    # Collocation points tau[j], j=1,...,K
    K = length(taus)

    t = _fill_t!(Matrix{eltype(h)}(undef, N, K), nodes, h, taus)

    return CollocationMesh(nodes, h, t, nothing, nothing, nothing)
end

function _get_mesh(nodes::AbstractMatrix, taus::AbstractVector)
    # Interval lengths h[m,i]
    h = diff(nodes; dims = 2)
    M, N = size(h)
    K = length(taus)

    t = _fill_t!(Array{eltype(h), 3}(undef, M, N, K), nodes, h, taus)

    return CollocationMesh(nodes, h, t, nothing, nothing, nothing)
end

# Collocation times t[i,j], i=1,...,N, j=1,...,K
function _fill_t!(t::AbstractMatrix, nodes::AbstractVector, h, taus)
    for i in axes(t, 1), j in axes(t, 2)
        t[i,j] = nodes[i] + h[i] * taus[j]
    end
    return t
end
function _fill_t!(t::AbstractArray{<:Any,3}, nodes::AbstractMatrix, h, taus)
    for m in axes(t, 1), i in axes(t, 2), j in axes(t, 3)
        t[m,i,j] = nodes[m,i] + h[m,i] * taus[j]
    end
    return t
end

# If adaptive = true, append parameter handles for h[i], t[i,j]
_with_parameters(mesh::CollocationMesh, hpar, tpar) = CollocationMesh(
    mesh.nodes, getfield(mesh, :h), getfield(mesh, :t),
    hpar, tpar, getfield(mesh, :horizon_scale),
)

# If unknown_horizon = true, store the tscale variable
_with_horizon_scale(mesh::CollocationMesh, scale) = CollocationMesh(
    mesh.nodes, getfield(mesh, :h), getfield(mesh, :t),
    getfield(mesh, :hpar), getfield(mesh, :tpar), scale,
)

# Read h/t at mesh mi, interval i (point k)
_hat(h::AbstractVector, mi, i) = h[i]
_hat(h::AbstractMatrix, mi, i) = h[mi, i]
_hat(h::ExaModels.Parameter{<:NTuple{1}}, mi, i) = h[i]
_hat(h::ExaModels.Parameter{<:NTuple{2}}, mi, i) = h[mi, i]

_tat(t::AbstractMatrix, mi, i, k) = t[i, k]
_tat(t::AbstractArray{<:Any,3}, mi, i, k) = t[mi, i, k]
_tat(t::ExaModels.Parameter{<:NTuple{2}}, mi, i, k) = t[i, k]
_tat(t::ExaModels.Parameter{<:NTuple{3}}, mi, i, k) = t[mi, i, k]

# Boundaries of mesh m
_mesh_nodes(mesh::CollocationMesh, m) = (n = getfield(mesh, :nodes);
    n isa AbstractVector ? n : view(n, m, :))

# Nominal start and horizon of mesh m
_t0(mesh::CollocationMesh, m) = first(_mesh_nodes(mesh, m))
_tnom(mesh::CollocationMesh, m) = last(_mesh_nodes(mesh, m)) - _t0(mesh, m)
_tnoms(mesh::CollocationMesh) = [_tnom(mesh, m) for m in 1:_num_meshes(mesh)]

# Interval widths of every mesh
_interval_widths(n::AbstractVector) = diff(n)
_interval_widths(n::AbstractMatrix) = diff(n; dims = 2)
