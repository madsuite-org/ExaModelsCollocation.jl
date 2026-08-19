# toy example of r-refinement:
#   - solve NLP
#       - while (err > tol) or (mesh movement > movement_tol)
#           - estimate error
#           - relocate mesh
#           - check convergence criteria
#           - resolve NLP
#

ENV["GKSwstype"] = "100"

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Define problem -----

const A, T0, TEND = 100.0, 0.5, 1.0

# ODE right-hand side function
rhs(z, t) = A / (1 + A^2 * (t - T0)^2)

# Exact solution for comparison
zexact(t) = atan(A * (t - T0))

# ----- Create ExaModel -----

function example_model(nodes, K = 3)
    # Create CollocationExaCore with adaptive mesh (t is an ExaModels parameter)
    core = CollocationExaCore(nodes, K; adaptive = true)

    # Create CollocationVariable
    @add_var_collocation(core, z)

    # Create collocation constraints
    @add_con_collocation(core, coll, z[], rhs(z, t))

    # Create continuity constraints
    @add_con_continuity(core, cont, z)

    # Create objective function
    @add_con(core, ic, z[1, 0] - zexact(0.0) for _ in 1:1)

    return ExaModel(core)
end

function solve(model)
    result = madnlp(model; print_level = MadNLP.ERROR, tol = 1e-8)
    # Return the full primal too: the error estimate reads f out of it
    return result.solution, solution(result, model.z)
end

# ----- Right-hand side off the model -----

# Absolute x-index of one coefficient
_xidx(z, s, i, k) = z[s..., i, k].i

# Host copy, doctorable entry by entry after a GPU solve
_hostcopy(v) = copyto!(Vector{eltype(v)}(undef, length(v)), v)

# One block's coefficients off the raw primal, shaped as solution() shapes them
_blocksol(x, z) =
    reshape(view(x, (z.offset + 1):(z.offset + z.length)), ExaModels.size(z.size)...)

# The block's polynomial at t, one value per slot in the order Iterators.product(dims...) gives
_slotvals(model, zsol, z, t) =
    (v = interpolate(model, zsol, z, t); isempty(z.dims) ? (v,) : vec(v))

# Where a residual's rows keep the interval, collocation and slot indices. add_con_collocation
# recorded it, so nothing here has to reconstruct it.
_rowlayout(_, r) = r.fwhere

_slotof(d, L) = ExaModelsCollocation._slotof(d, L)

# An f reading neither a variable nor a parameter traces to a plain number rather than a node
_nodeval(v, xs, ts) = v isa Real ? v : v(nothing, xs, ts)

# The recorded f at one scratch point, as a function of (fine time, x, θ)
function _scratch_eval(model, r, row)
    if model.adaptive
        node = r.f(row)
        return (tf, xs, ts) -> _nodeval(node, xs, ts)
    end
    return (tf, xs, ts) -> _nodeval(r.f((row[1:(end - 1)]..., tf)), xs, ts)
end

# Every row of a residual keyed by the point it names, in one pass over its slots x N x K rows
_scratch_rows(r, L) = Dict((_slotof(d, L)..., d[L.i], d[L.k]) => d for d in r.fiter)

function _scratch_row(rows, r, s, i, k)
    haskey(rows, (s..., i, k)) || error(
        "estimate_error_phr: the residual over $(r.var.name)[$(join(s, ", "))] has no row " *
        "at (i = $i, k = $k). Pick another kscratch"
    )
    return rows[(s..., i, k)]
end

# ----- Error estimate -----

function estimate_error_phr(model, x; kscratch = 1)
    nodes, h, N = model.nodes, diff(model.nodes), model.N

    # The K+1 roots (non-collocation points to compare)
    fine = Collocation([0.0, 1.0], model.K + 1;
        roots = model.roots, 
        basis = DerivativeForm(),
        polynomial = model.polynomial
    )
    tau, Omega = fine.mode.weights.taus, fine.mode.weights.A

    # Every block is interpolated, not just the collocated ones: f reads controls there too
    blocks = [(z, _blocksol(x, z), collect(Iterators.product(z.dims...))) for z in model.block]
    lookup = [(_rowlayout(model, r), _scratch_rows(r, _rowlayout(model, r))) for r in model.resid]
    rslots = [(r, s, lk) for (r, lk) in zip(model.resid, lookup) for s in r.fwhich]
    ev = [[_scratch_eval(model, r, _scratch_row(rows, r, s, i, kscratch)) for i in 1:N]
          for (r, s, (_, rows)) in rslots]

    # Where a collocated slot's interpolated value belongs in znew
    at = Dict((r.var, s) => n for (n, (r, s, _)) in enumerate(rslots))

    xs, ts = _hostcopy(x), _hostcopy(model.θ)
    fnew = zeros(length(rslots), N, length(tau))
    znew = zeros(length(rslots), N, length(tau))

    for i in 1:N, (m, s) in enumerate(tau)
        tf = nodes[i] + h[i] * s

        # Move every block before any residual is evaluated, since f reads the whole state
        # vector at its point, and read from the pristine x, which the scratch write destroys
        for (z, zsol, slots) in blocks
            for (sl, v) in zip(slots, _slotvals(model, zsol, z, tf))
                xs[_xidx(z, sl, i, kscratch)] = v
                # znew = Interpolated states of the degree K Lagrange polynomial at the K+1 roots
                haskey(at, (z, sl)) && (znew[at[(z, sl)], i, m] = v)
            end
        end
        model.adaptive && (ts[model.mesh.t[i, kscratch].i] = tf)

        # the RHS function evaluated at the K+1 roots
        for n in eachindex(ev)
            fnew[n, i, m] = ev[n][i](tf, xs, ts)
        end
    end

    # Calculate error estimate
    return [
        # Compare:
        #   Interpolated states of the degree K Lagrange polynomial at the K+1 roots
        #   Integrated states to K+1 degree using K+1 interpolated state points
        maximum(
            let zi = view(_blocksol(x, r.var), sl..., i, :)
                maximum(
                    # (Integrated states to K+1 degree using K+1 interpolated state points) - znew
                    abs(zi[1] + h[i]*sum(Omega[j,m]*fnew[n,i,j] for j in eachindex(tau)) - znew[n,i,m])
                    for m in eachindex(tau)
                ) / (1 + maximum(abs, zi))
            end
            for (n, (r, sl, _)) in enumerate(rslots)
        )
        for i in 1:N
    ]
end
# NOTE: this error estimation is cheap because we already obtained the 
# polynomial coefficients (the discretized states) from solving the NLP

# ----- Convergence check -----

# Invert the collocation equations for f and compare against f evaluated there: says whether the
# NLP satisfied its collocation rows, not how accurate the discretization is
function residual_mismatch(model, x)
    h, N, K, A = diff(model.nodes), model.N, model.K, model.weights.A
    ts = _hostcopy(model.θ)
    worst = 0.0

    # DerivativeForm with tau_1 = 0 leaves that row reading 0 = 0, so f is not recoverable there
    !(model.basis isa StateForm) && first(model.weights.taus) == 0 && return NaN

    for r in model.resid
        L, z = _rowlayout(model, r), r.var
        rows = _scratch_rows(r, L)
        for s in r.fwhich, i in 1:N
            got = [_nodeval(r.f(_scratch_row(rows, r, s, i, k)), x, ts) for k in 1:K]
            zi = [x[_xidx(z, s, i, k)] for k in z.krange]
            want = model.basis isa StateForm ?
                [sum(A[j+1,k] * zi[j+1] for j in 0:K) / h[i] for k in 1:K] :
                transpose(A) \ [(zi[k+1] - zi[1]) / h[i] for k in 1:K]
            worst = max(worst, maximum(abs, got .- want) / (1 + maximum(abs, got)))
        end
    end
    return worst
end


# ----- Mesh update -----

function find_new_nodes(model, err; floor_frac = 0.1, passes = 2)
    nodes, h, N = model.nodes, diff(model.nodes), model.N

    # de Boor density (?)
    rho = err .^ (1 / (model.K + 1)) ./ h
    rho = rho .+ floor_frac * (sum(rho) / N)
    for _ in 1:passes
        rho = [(rho[max(i - 1, 1)] + 2rho[i] + rho[min(i + 1, N)]) / 4 for i in 1:N]
    end

    # invert cumulative mass to convert density to new node placement (?)
    W = cumsum([0.0; rho .* h])
    new = collect(float.(nodes))
    for m in 2:N
        target = W[end] * (m - 1) / N
        i = clamp(searchsortedlast(W, target), 1, N)
        new[m] = nodes[i] + (target - W[i]) / rho[i]
    end

    return new
end

# ----- r-refinement algorithm -----

# Keep track of how much the mesh moved for convergence criteria purposes
movement(new, old, h) = maximum(
    abs(new[j] - old[j]) / min(h[max(j - 1, 1)], h[min(j, length(h))])
    for j in eachindex(new)
)

# Re-interpolated states at interpolation points based on new node locations for better initial guess
function reinterpolate(model, zsol, new)
    # evaluate the degree K Lagrange polynomial through the solved coefficients
    z, taus = model.z, model.weights.taus
    tauof = first(z.krange) == 0 ? [zero(eltype(taus)); taus] : collect(taus)
    hnew = diff(new)
    return [
        interpolate(model, zsol, z, new[i] + hnew[i] * tau)
        for i in eachindex(hnew), tau in tauof
    ]
end

function solve_adaptively(
        model;
        tol = 1e-6,
        movetol = 1e-2,
        maxiters = 20,
        verbose = true,
    )
    history = []

    # Solve model
    x, zsol = solve(model)

    # while (max error < tol OR movement < movement tol)
    for it in 0:maxiters
        nodes = copy(model.nodes)

        # calculate error using Patterson-Hager-Rao
        err = estimate_error_phr(model, x)

        push!(history, (; nodes, zsol, err)) # tracking error histroy for plot

        # find new node placements based on error + de Boor equidistribution
        new_nodes = find_new_nodes(model, err)
        moved = movement(new_nodes, nodes, diff(model.nodes))
        verbose && println(
            "iteration $it: max error = $(maximum(err)), movement = $moved, " *
            "collocation mismatch = $(residual_mismatch(model, x))"
        )

        # convergence criteria
        (maximum(err) < tol || moved < movetol) && break

        # if criteria not satisfied, set new nodes and re-solve
        # NOTE: interpolate reads the mesh off the model, so the states move before the nodes do
        start = reinterpolate(model, zsol, new_nodes)
        set_nodes!(model, new_nodes) # <--- ExaModelsCollocation.jl feature with adaptive = true
        set_start!(model, model.z, start)
        x, zsol = solve(model)
    end

    return history
end


# ----- Solve -----

const N = 50
const K = 3

# Create ExaModel
model = example_model(range(0.0, TEND; length = N + 1), K)

# Solve with adaptive mesh refinement
history = solve_adaptively(model)


# ----- Plot -----

default(
    titlefontsize = 16, guidefontsize = 13, tickfontsize = 10,
    legendfontsize = 10, colorbar_titlefontsize = 12, plot_titlefontsize = 18,
)

atnodes(s) = [s.zsol[:, 1]; s.zsol[end, end]]

function error_color(history, exact; floor = 1.0e-9)
    err = [log10.(max.(abs.(atnodes(s) .- exact.(s.nodes)), floor)) for s in history]
    clims = (log10(floor), ceil(maximum(maximum, err)))

    return err, clims
end

function colorbar_strip(clims; rows = 256)
    decades = first(clims):last(clims)
    ramp = collect(range(first(clims), last(clims); length = rows))

    return heatmap(
        [0.0], ramp, reshape(ramp, :, 1);
        c = :jet, clims = clims, legend = false, colorbar = false, grid = false,
        xticks = false, ymirror = true, tickfontsize = 11,
        yticks = (decades, ["1e$(Int(d))" for d in decades]),
        title = "|z_exact - z|", titlefontsize = 12,
        right_margin = 4Plots.mm,
    )
end

function plot_mesh_history(history, exact)
    t = reduce(vcat, s.nodes for s in history)
    iteration = reduce(vcat, fill(i - 1, length(s.nodes)) for (i, s) in enumerate(history))
    err, clims = error_color(history, exact)

    return scatter(
        t, iteration;
        marker_z = reduce(vcat, err), c = :jet, clims = clims,
        markersize = 7, markerstrokewidth = 0, legend = false, colorbar = false,
        xlabel = "t", ylabel = "r-iter", guidefontsize = 16, yguidefontsize = 18,
        tickfontsize = 12,
        xlims = (0.0, TEND), ylims = (-0.5, length(history) - 0.5),
        yticks = 0:(length(history) - 1),
        title = "r-refinement: node placement per iteration", titlefontsize = 18,
        right_margin = 5Plots.mm,
    )
end

function plot_solution_3d(history, exact)
    zall = reduce(vcat, atnodes(s) for s in history)
    zfloor = minimum(zall) - 0.18 * (maximum(zall) - minimum(zall))
    errs, clims = error_color(history, exact)

    p = plot3d(;
        xlabel = "t", ylabel = "r-iter", zlabel = "z(t)",
        xlims = (0.0, TEND), ylims = (-0.5, length(history) - 0.5),
        zlims = (zfloor, maximum(zall)),
        yticks = 0:(length(history) - 1),
        legend = false, camera = (17.5, 15), colorbar = false,
        projection_type = :perspective,
        title = "r-refinement: state trajectory per iteration",
        left_margin = -14Plots.mm, right_margin = -6Plots.mm,
        top_margin = -14Plots.mm, bottom_margin = -14Plots.mm,
    )

    for (it, (s, err)) in enumerate(zip(history, errs))
        iteration = fill(it - 1, length(s.nodes))
        scatter3d!(
            p, s.nodes, iteration, fill(zfloor, length(s.nodes));
            marker_z = err, c = :jet, clims = clims,
            markersize = 3.5, markerstrokewidth = 0,
        )
        plot3d!(
            p, s.nodes, iteration, atnodes(s);
            line_z = err, c = :jet, clims = clims, linewidth = 2.5,
        )
    end
    return p
end

nodes = last(history).nodes
println("h ranges over [$(minimum(diff(nodes))), $(maximum(diff(nodes)))]")
for (it, s) in enumerate(history)
    znode = [s.zsol[:, 1]; s.zsol[end, end]]
    println("iteration $(it - 1): |z(T) - exact| = $(abs(znode[end] - zexact(TEND)))")
end
png = joinpath(@__DIR__, "mesh-refinement.png")
savefig(
    plot(
        plot_mesh_history(history, zexact),
        plot_solution_3d(history, zexact),
        colorbar_strip(last(error_color(history, zexact)));
        layout = grid(1, 3; widths = [0.48, 0.48, 0.04]), size = (1500, 720),
        left_margin = 8Plots.mm, bottom_margin = 5Plots.mm,
    ),
    png,
)
println("wrote $png")
