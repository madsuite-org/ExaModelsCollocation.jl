# r-refinement: solve, estimate the error per interval, move a fixed number of nodes to
# equidistribute it, repeat. set_nodes! is what makes the mesh move without a rebuild.
#
#   dz/dt = a / (1 + a^2 (t - t0)^2),   z(0) = atan(-a t0)   =>   z(t) = atan(a (t - t0))
#
# The rise spans ~2/a = 0.02, so N uniform intervals put the peak inside one of them and the
# quadrature barely sees it. Nothing in the loop knows zexact: it estimates its own error.
#
#   julia --project=examples examples/r_refinement.jl

ENV["GKSwstype"] = "100"                    # GR writes the png without a display

using ExaModels
using ExaModelsCollocation
using MadNLP
using Plots

# ----- Problem -----

const A, T0, TEND = 100.0, 0.5, 1.0

rhs(t) = A / (1 + A^2 * (t - T0)^2)         # traced into the model, called on numbers below
zexact(t) = atan(A * (t - T0))              # initial condition, and the plot's colour

# ----- Build model -----

function build(nodes; K = 3)
    core = CollocationExaCore(nodes, K; adaptive = true)
    @add_var_collocation(core, z)                       # no dimensions: z[i,k]

    # adaptive: t is a parameter block the rhs indexes, so the row ends at k
    tp = core.mesh.tpar
    itr = [(i, k) for i in 1:core.N, k in 1:core.K]
    @add_con_collocation(core, coll, z, rhs(tp[i, k]) for (i, k) in itr)
    @add_con_continuity(core, cont, z)
    ExaModels.@add_con(core, ic, z[1, 0] - zexact(0.0) for _ in 1:1)

    return ExaModels.ExaModel(core), z
end

# solution() is 1-based, so k = 0,...,K lands on 1,...,K+1
solve(model, z) =
    ExaModels.solution(madnlp(model; print_level = MadNLP.ERROR, tol = 1e-12), z)

# the N+1 boundary values: k = 0 of every interval, then z(TEND), which Radau puts at k = K
atnodes(zsol) = [zsol[:, 1]; zsol[end, end]]

# ----- Error estimate -----

# Lagrange basis j over `nodes`, and its derivative, at a tau that is not one of them (the
# package's delljk is the differentiation matrix, exact only at the nodes)
lagval(nodes, j, tau) =
    prod((tau - nodes[m]) / (nodes[j] - nodes[m]) for m in eachindex(nodes) if m != j)
dell(nodes, j, tau) =
    lagval(nodes, j, tau) * sum(1 / (tau - nodes[m]) for m in eachindex(nodes) if m != j)

# Local error per interval from the defect |dz_h/dt - f|, sampled between the collocation
# points -- at them it is zero by construction. No extra solve, no exact solution.
function localerr(zsol, nodes, taus)
    h = diff(nodes)
    x = [0.0; taus]                              # StateForm basis: the anchor, then taus
    at = [(x[j] + x[j + 1]) / 2 for j in 1:length(taus)]
    return [
        h[i] * sum(
            abs(
                sum(dell(x, j, s) * zsol[i, j] for j in eachindex(x)) / h[i] -
                    rhs(nodes[i] + h[i] * s)
            ) for s in at
        ) / length(at)
        for i in eachindex(h)
    ]
end

# ----- Mesh update -----

# Local error goes like h^(K+1), so equidistributing e^(1/(K+1)) equidistributes e. The root
# is load bearing: equidistributing e itself diverges.
monitor(est, K) = est .^ (1 / (K + 1))

# de Boor: give every interval an equal share of sum(w) by inverting the cumulative monitor,
# which is piecewise linear on the current mesh. N never changes -- that is the r in
# r-refinement, and what set_nodes! allows without a rebuild.
function equidistribute(nodes, w; floor_frac = 0.1, passes = 2)
    N = length(w)

    # floor keeps a flat interval from collapsing, smoothing keeps a spike from starving its
    # neighbours; together they are what make the loop settle
    w = w .+ floor_frac * (sum(w) / N)
    for _ in 1:passes
        w = [(w[max(i - 1, 1)] + 2w[i] + w[min(i + 1, N)]) / 4 for i in 1:N]
    end

    W = cumsum([0.0; w])                       # W[i]: monitor mass left of nodes[i]
    new = collect(float.(nodes))               # the horizon ends stay put
    for m in 2:N
        target = W[end] * (m - 1) / N
        i = clamp(searchsortedlast(W, target), 1, N)
        new[m] = nodes[i] + (target - W[i]) / w[i] * (nodes[i + 1] - nodes[i])
    end
    return new
end

# Node movement in units of the intervals it sits between. A mesh that reproduces itself is
# the fixed point, and this is the stopping test: with N fixed the error does not go to zero,
# it goes to whatever the equidistributed mesh gives.
movement(new, old, h) = maximum(
    abs(new[j] - old[j]) / min(h[max(j - 1, 1)], h[min(j, length(h))])
    for j in eachindex(new)
)

# Re-interpolate states at interpolation points based on moved nodes for 
# a better initial guess at next iteration
function reinterpolate(zsol, old, new, taus)
    x = [0.0; taus]
    hold, hnew = diff(old), diff(new)
    start = similar(zsol, size(zsol))
    for i in eachindex(hnew), (k, tau) in enumerate(x)
        t = new[i] + hnew[i] * tau
        ii = clamp(searchsortedlast(old, t), 1, length(hold))
        s = (t - old[ii]) / hold[ii]
        start[i, k] = sum(lagval(x, j, s) * zsol[ii, j] for j in eachindex(x))
    end
    return start
end

# ----- Refinement loop -----

function rrefine!(model, z; tol = 0.01, maxiters = 20)
    taus, history = model.weights.taus, []
    for it in 0:maxiters
        zsol = solve(model, z)
        nodes = copy(model.nodes)
        push!(history, (; nodes, znode = atnodes(zsol)))

        est = localerr(zsol, nodes, taus)
        new = equidistribute(nodes, monitor(est, length(taus)))
        moved = movement(new, nodes, model.mesh.h)
        println("iteration $it: movement = $moved, estimated error = $(sum(est))")

        moved < tol && break
        set_nodes!(model, new)              # writes h and t through; no rebuild
        set_start!(model, z, reinterpolate(zsol, nodes, new, taus))
    end
    return history
end

# ----- Plot solution -----

# One row per iteration, iteration 0 at the bottom, each node coloured by its own error.
function plot_history(history; floor = 1e-9)
    t = reduce(vcat, h.nodes for h in history)
    iteration = reduce(vcat, fill(i - 1, length(h.nodes)) for (i, h) in enumerate(history))

    # log10, floored: the span is eight decades, so a linear scale leaves every converged
    # node at 0 and only the starting mesh with any colour
    err = log10.(max.(reduce(vcat, [abs.(h.znode .- zexact.(h.nodes)) for h in history]), floor))
    err = (err .- minimum(err)) ./ (maximum(err) - minimum(err))

    scatter(
        t, iteration;
        marker_z = err, c = :jet, clims = (0, 1),
        markersize = 5, markerstrokewidth = 0, legend = false, colorbar = true,
        colorbar_title = "\nerror vs exact solution",
        colorbar_ticks = 0:0.2:1,
        xlabel = "t", ylabel = "refinement iteration",
        xlims = (0.0, TEND), ylims = (-0.5, length(history) - 0.5),
        yticks = 0:(length(history) - 1),
        title = "r-refinement: node placement per iteration",
        right_margin = 5Plots.mm,
    )
end

# ----- Run -----

N = 30
model, z = build(range(0.0, TEND; length = N + 1))
history = rrefine!(model, z)

nodes = last(history).nodes
println("h ranges over [$(minimum(diff(nodes))), $(maximum(diff(nodes)))]")

# against the exact solution, which the loop above never saw
for (it, h) in enumerate(history)
    println("iteration $(it - 1): |z(T) - exact| = $(abs(h.znode[end] - zexact(TEND)))")
end

png = joinpath(@__DIR__, "r_refinement.png")
savefig(plot_history(history), png)
println("wrote $png")
