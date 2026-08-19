# Move the mesh a core or model was built on, keeping the interval count
"""
    set_nodes!(model, nodes; mesh = nothing)

Relocates the placement of `nodes` for a `mesh` of a `CollocationExaModel`, given `adaptive = true`.
"""
function set_nodes!(c::Union{CollocationExaCore, CollocationExaModel}, nodes; mesh = nothing)
    m = _mesh(c)
    _isadaptive(m) || throw(ArgumentError(
        "set_nodes!: this mesh is not adaptive; build it with `adaptive = true`"
    ))
    old = getfield(m, :nodes)
    new = mesh === nothing ? _nodes_input(nodes) : _onemesh(old, mesh, nodes, _num_meshes(m))
    size(new) == size(old) || throw(DimensionMismatch(
        "set_nodes!: expected $(size(old)) boundaries, got $(size(new)); " *
        "changing the number of intervals needs a rebuild"
    ))
    all(>(0), _interval_widths(new)) ||
        throw(ArgumentError("set_nodes!: nodes must be strictly increasing along t"))

    taus = _weights(c).taus
    h, t = _hval(m), _tval(m)
    copyto!(old, new)
    h .= _interval_widths(old)
    _fill_t!(t, old, h, taus)

    ExaModels.set_value!(c, _hpar(m), h)
    ExaModels.set_value!(c, _tpar(m), t)
    return nothing
end

# Every mesh's boundaries with `mesh` replaced, so one of them moves down the same path
function _onemesh(old, mesh, nodes, M)
    old isa AbstractMatrix || throw(ArgumentError(
        "set_nodes!: this core has a single mesh, so there is nothing for `mesh = $mesh` to " *
        "name; call `set_nodes!(c, nodes)`"
    ))
    1 <= mesh <= M || throw(ArgumentError(
        "set_nodes!: `mesh` must name one of the $M meshes, got $mesh"
    ))
    row = _nodes_input(nodes)
    length(row) == size(old, 2) || throw(DimensionMismatch(
        "set_nodes!: expected $(size(old, 2)) boundaries, got $(length(row)); " *
        "changing the number of intervals needs a rebuild"
    ))
    every = copy(old)
    every[mesh, :] .= row
    return every
end
