using ExaModelsDAE
using Documenter

DocMeta.setdocmeta!(ExaModelsDAE, :DocTestSetup, :(using ExaModelsDAE); recursive = true)

makedocs(;
    modules = [ExaModelsDAE],
    authors = "Joseph Choi <jsphchoi@mit.edu>",
    sitename = "ExaModelsDAE.jl",
    format = Documenter.HTML(;
        canonical = "https://mit-shin-group.github.io/ExaModelsDAE.jl",
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
    repo = "github.com/mit-shin-group/ExaModelsDAE.jl",
    devbranch = "main",
)
