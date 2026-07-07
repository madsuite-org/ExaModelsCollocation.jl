using ExaModelsDynamic
using Documenter

DocMeta.setdocmeta!(ExaModelsDynamic, :DocTestSetup, :(using ExaModelsDynamic); recursive = true)

makedocs(;
    modules = [ExaModelsDynamic],
    authors = "Joseph Choi <jsphchoi@mit.edu>",
    sitename = "ExaModelsDynamic.jl",
    format = Documenter.HTML(;
        canonical = "https://mit-shin-group.github.io/ExaModelsDynamic.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Usage" => "usage.md",
        "Examples" => "examples.md",
    ],
)

deploydocs(;
    repo = "github.com/mit-shin-group/ExaModelsDynamic.jl",
    devbranch = "main",
)
