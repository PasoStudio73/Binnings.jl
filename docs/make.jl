using Documenter
using Binnings

DocMeta.setdocmeta!(
    Binnings,
    :DocTestSetup,
    :(using Binnings);
    recursive = true
)

makedocs(;
    modules=[Binnings],
    authors="Michele Ghiotti, Federico Manzella, Riccardo Pasini",
    repo=Documenter.Remotes.GitHub("PasoStudio73", "Binnings.jl"),
    sitename="Binnings.jl",
    format=Documenter.HTML(;
        size_threshold=4000000,
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PasoStudio73.github.io/Binnings.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Binning" => "bin.md",
        "Algorithms" => "algos.md"
    ],
    warnonly=true,
)

deploydocs(;
    repo = "github.com/PasoStudio73/DataTreatments.jl",
    devbranch = "main",
    target = "build",
    branch = "gh-pages",
    versions = ["main" => "main", "stable" => "v^", "v#.#", "dev" => "dev"],
)
