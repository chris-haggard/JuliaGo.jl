using Documenter
using JuliaGo

makedocs(
    sitename = "JuliaGo",
    format = Documenter.HTML(prettyurls = true),
    modules = [JuliaGo],
    pages = ["Home" => "index.md", "Documentation" => "Documentation.md"],
)

deploydocs(
    repo = "github.com/chris-haggard/JuliaGo.jl.git",
)
