# Helper functions for building macros for API functions

# Split macro arguments into positional and keyword parts, accepting both
# `f(a, b; k = v)` and `f(a, b, k = v)` spellings.
function _split_collocation_args(macro_args)
    args = Any[]
    kwargs = Any[]
    for ex in macro_args
        if ex isa Expr && ex.head === :parameters
            append!(kwargs, ex.args)
        elseif ex isa Expr && (ex.head === :(=) || ex.head === :kw)
            push!(kwargs, Expr(:kw, ex.args[1], ex.args[2]))
        else
            push!(args, ex)
        end
    end
    return args, kwargs
end

# Appends the mesh indices to a block given exactly its declared dimensions, so `z[v]` and
# `z[v,i,k]` are one expression. Anything else indexes as written.
_colidx(x::CollocationVariable, i, k, idx...) =
    length(idx) == length(x.dims) ? x[idx..., i, k] : x[idx...]
_colidx(x, i, k, idx...) = x[idx...]
_colidx(x::CollocationVariable, i, k) = isempty(x.dims) ? x[i, k] : x
_colidx(x, i, k) = x

# Names an iteration pattern binds, in order: `v`, `(v, c)`, and `(v, (a, b))` all flatten
function _pattern_names(ex)
    ex isa Symbol && return Symbol[ex]
    ex isa Expr && ex.head === :tuple && return reduce(vcat, map(_pattern_names, ex.args))
    error("@add_con_collocation: cannot read the iteration variables out of `$ex`")
end

# Syntax that reads as a bare name but is not a value
const _NOT_OPERANDS = (:end, :begin)

# `end` and `begin` mean something only in index position, so a `ref` mentioning either is
# left as written rather than rewritten into a _colidx call
_indexonly(ex) = ex isa Symbol ? ex in _NOT_OPERANDS :
    ex isa Expr ? any(_indexonly, ex.args) : false

# Every operand of the body through _colidx; `skip` holds the names the body itself binds
function _complete(ex, skip)
    ex isa Symbol &&
        return ex in skip || ex in _NOT_OPERANDS ? ex : Expr(:call, _colidx, ex, :i, :k)
    ex isa Expr || return ex

    h = ex.head
    if h === :ref
        any(_indexonly, ex.args[2:end]) && return ex
        return Expr(:call, _colidx, ex.args[1], :i, :k, _completeall(ex.args[2:end], skip)...)
    elseif h === :. && length(ex.args) == 2 && ex.args[2] isa QuoteNode
        # a block reached as core.z completes like a bare one
        return Expr(:call, _colidx, ex, :i, :k)
    elseif h === :macrocall
        return ex
    elseif h === :call
        # the callee is not an operand
        return Expr(h, ex.args[1], _completeall(ex.args[2:end], skip)...)
    elseif h === :kw || h === :(=)
        return Expr(h, ex.args[1], _complete(ex.args[2], skip))
    elseif h === :-> || h === :function || h === :do
        return Expr(h, ex.args[1], _complete(ex.args[2], union(skip, _argnames(ex.args[1]))))
    elseif h === :generator || h === :comprehension || h === :let || h === :for
        return _complete_scoped(ex, skip)
    end
    return Expr(h, _completeall(ex.args, skip)...)
end

_completeall(exs, skip) = [_complete(e, skip) for e in exs]

# Bodies that bind their own names: the bindings are left alone and their bodies completed
# under them, so a lambda argument is never mistaken for a block.
function _complete_scoped(ex, skip)
    binds, rest = ex.head === :generator || ex.head === :comprehension ?
        (ex.args[2:end], ex.args[1:1]) : (ex.args[1:1], ex.args[2:end])
    inner = union(skip, reduce(vcat, map(_boundnames, binds); init = Symbol[]))
    parts = _completeall(rest, inner)
    return ex.head === :generator || ex.head === :comprehension ?
        Expr(ex.head, parts..., binds...) : Expr(ex.head, binds..., parts...)
end

_argnames(ex) = ex isa Symbol ? Symbol[ex] :
    ex isa Expr && ex.head === :tuple ? reduce(vcat, map(_argnames, ex.args); init = Symbol[]) :
    ex isa Expr && ex.head === :call ? reduce(vcat, map(_argnames, ex.args[2:end]); init = Symbol[]) :
    ex isa Expr && ex.head === :(::) ? _argnames(ex.args[1]) :
    Symbol[]

_boundnames(ex) = ex isa Expr && ex.head in (:(=), :in, :kw) ? _argnames(ex.args[1]) : Symbol[]

_name_val(name::Symbol) = Val(name)
_name_val(::Nothing) = nothing