# access solution at any time t by evaluating interpolating polynomial at t
"""
    interpolate(model, result, z, t; mesh = nothing)

Evaluates the interpolating polynomial of a `CollocationVariable` at time `t`.

# Arguments
- `model`  : [`CollocationExaModel`](@ref)
- `result` : solved `ExaModel` result
- `z`      : `CollocationVariable` from [`add_var_collocation`](@ref)
- `t`      : vector of times within the mesh

# Keyword Argument
- `mesh` : interpolate profile against the chosen mesh

# Example
```julia
julia> zf = interpolate(model, result, z, last(model.nodes)) # z at terminal point

julia> zs = interpolate(model, result, z, range(0.0, 1.0; length = 101)) # zs across uniform mesh

julia> z2 = interpolate(model, result, z, 0.5; mesh = 2) # z(t = 0.5) on mesh 2
```
"""
function interpolate(c, zsol::AbstractArray, z::CollocationVariable, t::Real; mesh = nothing)
    m = _interp_mesh(c, z, mesh)
    xn, cols = _interp_basis(c, z)
    i, tau = _interp_locate(c, m, t)
    return _interp_slots(zsol, z, m, xn, _baryweights(xn), cols, i, tau)
end

function interpolate(
        c, zsol::AbstractArray, z::CollocationVariable, ts::AbstractVector; mesh = nothing,
    )
    m = _interp_mesh(c, z, mesh)
    xn, cols = _interp_basis(c, z)
    w = _baryweights(xn)
    return [
        _interp_slots(zsol, z, m, xn, w, cols, _interp_locate(c, m, t)...) for t in ts
    ]
end

interpolate(c, result, z::CollocationVariable, t; mesh = nothing) =
    interpolate(c, ExaModels.solution(result, z), z, t; mesh = mesh)

# The taus a block's coefficients sit at, and the solution() columns holding them. solution()
# is 1-based, so k = 0,…,K lands on 1,…,K+1.
function _interp_basis(c, z::CollocationVariable)
    taus = _weights(c).taus
    ks = collect(z.krange)
    cols(kk) = kk .- first(z.krange) .+ 1

    first(ks) == 0 || return (collect(taus), cols(ks))
    first(taus) == 0 && return (collect(taus), cols(ks[2:end]))
    return ([zero(eltype(taus)); taus], cols(ks))
end

# Which interval a time falls in on mesh m, and where in it
function _interp_locate(c, m::Integer, t::Real)
    nodes = _mesh_nodes(_mesh(c), m)
    first(nodes) <= t <= last(nodes) || throw(ArgumentError(
        "interpolate: t = $t is outside the mesh [$(first(nodes)), $(last(nodes))]"
    ))
    h = diff(nodes)
    i = clamp(searchsortedlast(nodes, t), 1, length(h))
    return i, (t - nodes[i]) / h[i]
end

# Which mesh to locate against: the one a pinned block names, or the one this call names, since
# which mesh to read is a property of the call and not of the block
function _interp_mesh(c, z::CollocationVariable, mesh)
    M = _num_meshes(c)
    z.mesh === nothing || mesh === nothing || mesh == z.mesh || throw(ArgumentError(
        "interpolate: $(z.name) is pinned to mesh $(z.mesh), so it carries no coefficients on " *
        "mesh $mesh"
    ))
    m = mesh === nothing ? z.mesh : mesh
    m === nothing && throw(ArgumentError(
        "interpolate: $(z.name) spans the $M meshes, so `mesh = m` says which of them to " *
        "locate t against"
    ))
    1 <= m <= M || throw(ArgumentError(
        "interpolate: `mesh` must name one of the $M meshes, got $m"
    ))
    return Int(m)
end

# One value per slot of the block, shaped as the declared dimensions. A block spanning every mesh
# carries the index as its last one, so that axis is sliced here rather than shaping the answer.
function _interp_slots(zsol, z::CollocationVariable, m, xn, w, cols, i, tau)
    dims, at = z.mesh === nothing ? (z.dims[1:(end - 1)], (m,)) : (z.dims, ())
    vals = [
        _evalpoly(view(zsol, s..., at..., i, cols), xn, w, tau)
        for s in Iterators.product(dims...)
    ]
    return isempty(dims) ? only(vals) : vals
end

_evalpoly(v, xn, w, tau) = sum(l * vi for (l, vi) in zip(_lagrange(xn, w, tau), v))
