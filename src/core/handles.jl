# CollocationVariable things
"""
    CollocationVariable

A variable block over the collocation mesh, from [`add_var_collocation`](@ref). Indexes like
an `ExaModels.Variable`: `z[v, c, i, k]`, or `z[i, k]` when no dimensions were declared.

# Fields
- `dims`   : declared dimensions, the shape at one collocation point
- `krange` : collocation index range, `0:K` with the interval-left boundary node, else `1:K`
- `mesh`   : the mesh the block is pinned to, or `nothing` when it spans every one
"""
struct CollocationVariable{S, O, T, D} <: ExaModels.AbstractVariable
    size::S
    length::O
    offset::O
    name::Symbol
    tag::T
    dims::D
    krange::UnitRange{Int}
    mesh::Union{Int, Nothing}
end

CollocationVariable(v::ExaModels.Variable, dims, krange::UnitRange{Int}, mesh) =
    CollocationVariable(v.size, v.length, v.offset, v.name, v.tag, dims, krange, mesh)

# Leading dimensions the caller declared; the mesh indices (i, k) are the rest.
_nleading(z::CollocationVariable) = length(z.dims)

# Where a row names its mesh: the last slot entry when the block carries the mesh axis, and
# the literal it was pinned to otherwise. One mesh pins every block, so nothing changes there.
_meshpos(z::CollocationVariable, slot) = z.mesh === nothing ? last(slot) : Fixed(z.mesh)

"""
    CollocationSlot

One slot of a [`CollocationVariable`](@ref), from indexing it by its declared dimensions alone.
[`add_con_collocation`](@ref) reads its target off one of these.

# Fields
- `var` : the block
- `idx` : the declared indices, as row entries or literals
"""
struct CollocationSlot{V, I}
    var::V
    idx::I
end

Base.show(io::IO, s::CollocationSlot) =
    print(io, "CollocationSlot ", s.var.name, "[", join(s.idx, ", "), "]")

# A slot names a target, so it is not a value. Reaching one through arithmetic means the
# macro's short spelling was written where add_con_collocation wants the mesh indices out.
_slot_misuse(s::CollocationSlot) = throw(ArgumentError(
    "$(s.var.name)[$(join(s.idx, ", "))] names a slot, not a value. Write the mesh indices " *
    "out as $(s.var.name)[$(join((s.idx..., "i", "k"), ", "))], or use @add_con_collocation, " *
    "which appends them."
))

for op in (:+, :-, :*, :/, :^)
    @eval Base.$op(s::CollocationSlot, ::Any) = _slot_misuse(s)
    @eval Base.$op(::Any, s::CollocationSlot) = _slot_misuse(s)
    @eval Base.$op(s::CollocationSlot, ::CollocationSlot) = _slot_misuse(s)
end
Base.:-(s::CollocationSlot) = _slot_misuse(s)
Base.:+(s::CollocationSlot) = _slot_misuse(s)

# Dimensions alone name a slot, dimensions plus the mesh indices a coefficient. The two arities
# never collide, a full index being `length(dims) + 2` long.
Base.getindex(z::CollocationVariable, idx...) = _index_collocation(z, idx)
Base.getindex(z::CollocationVariable, i) = _index_collocation(z, (i,))
Base.getindex(z::CollocationVariable, ::Colon) = _index_collocation(z, (:,))

function _index_collocation(z::CollocationVariable, idx)
    any(i -> i isa Colon, idx) && return _exaindex(z, idx)

    n, nd = length(idx), length(z.dims)
    n == nd && return CollocationSlot(z, idx)
    n == nd + 2 || throw(ArgumentError(
        "$(z.name): a collocation block takes $nd $(nd == 1 ? "index" : "indices") for one of " *
        "its slots or $(nd + 2) with the mesh, got $n"
    ))
    return _exaindex(z, idx)
end

_exaindex(z, idx) =
    invoke(Base.getindex, Tuple{ExaModels.AbstractVariable, Vararg{Any}}, z, idx...)

function Base.show(io::IO, z::CollocationVariable)
    print(
        io,
        """
        CollocationVariable

          $(z.name)[$(join((z.dims..., "i", "k"), ", "))] ∈ R^{$(join(ExaModels.size(z.size), " × "))}, k = $(z.krange)
        """,
    )
end

# ExaModels utilities for CollocationVariables
_var_range(z::CollocationVariable) = (z.offset + 1):(z.offset + z.length)

ExaModels.get_start(m::ExaModels.ExaModel, z::CollocationVariable) =
    view(m.meta.x0, _var_range(z))
ExaModels.get_lvar(m::ExaModels.ExaModel, z::CollocationVariable) =
    view(m.meta.lvar, _var_range(z))
ExaModels.get_uvar(m::ExaModels.ExaModel, z::CollocationVariable) =
    view(m.meta.uvar, _var_range(z))

ExaModels.set_start!(m::ExaModels.ExaModel, z::CollocationVariable, values) =
    _copy_into!(view(m.meta.x0, _var_range(z)), values, z, :set_start!)
ExaModels.set_lvar!(m::ExaModels.ExaModel, z::CollocationVariable, values) =
    _copy_into!(view(m.meta.lvar, _var_range(z)), values, z, :set_lvar!)
ExaModels.set_uvar!(m::ExaModels.ExaModel, z::CollocationVariable, values) =
    _copy_into!(view(m.meta.uvar, _var_range(z)), values, z, :set_uvar!)

function _copy_into!(dest, values, z::CollocationVariable, who::Symbol)
    length(values) == z.length || throw(DimensionMismatch(
        "$who: expected $(z.length) elements, got $(length(values))"
    ))
    return copyto!(dest, values)
end
