# access solution at any time t by evaluating interpolating polynomial at t
"""
    interpolate(model, result, z, t)

Evaluates the interpolating polynomial of a `CollocationVariable` at time `t`.

# Arguments
- `model`  : [`CollocationExaModel`](@ref)
- `result` : solved ExaModels result
- `z`      : `CollocationVariable` from [`add_var_collocation`](@ref)
- `t`      : vector of times within the mesh

# Example
```julia
julia> zf = interpolate(model, result, z, last(model.nodes)) # z at terminal point

julia> zs = interpolate(model, result, z, range(0.0, 1.0; length = 101)) # zs across uniform mesh
```
"""
function interpolate(c, zsol::AbstractArray, z::CollocationVariable, t::Real)
    xn, cols = _interp_basis(c, z)
    i, tau = _interp_locate(c, t)
    return _interp_slots(zsol, z, xn, _baryweights(xn), cols, i, tau)
end

function interpolate(c, zsol::AbstractArray, z::CollocationVariable, ts::AbstractVector)
    xn, cols = _interp_basis(c, z)
    w = _baryweights(xn)
    return [
        _interp_slots(zsol, z, xn, w, cols, _interp_locate(c, t)...) for t in ts
    ]
end

interpolate(c, result, z::CollocationVariable, t) =
    interpolate(c, ExaModels.solution(result, z), z, t)

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

# Which interval a time falls in, and where in it
function _interp_locate(c, t::Real)
    nodes = c.nodes
    first(nodes) <= t <= last(nodes) || throw(ArgumentError(
        "interpolate: t = $t is outside the mesh [$(first(nodes)), $(last(nodes))]"
    ))
    h = diff(nodes)
    i = clamp(searchsortedlast(nodes, t), 1, length(h))
    return i, (t - nodes[i]) / h[i]
end

# One value per slot of the block, shaped as the declared dimensions
function _interp_slots(zsol, z::CollocationVariable, xn, w, cols, i, tau)
    vals = [
        _evalpoly(view(zsol, s..., i, cols), xn, w, tau)
        for s in Iterators.product(z.dims...)
    ]
    return isempty(z.dims) ? only(vals) : vals
end

_evalpoly(v, xn, w, tau) = sum(l * vi for (l, vi) in zip(_lagrange(xn, w, tau), v))
