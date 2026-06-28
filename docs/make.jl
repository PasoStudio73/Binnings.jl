using Binnings
using Documenter

DocMeta.setdocmeta!(Binnings, :DocTestSetup, :(using Binnings); recursive=true)

makedocs(;
    modules=[Binnings],
    authors="PasoStudio73 <paso.studio73@gmail.com> and contributors",
    sitename="Binnings.jl",
    format=Documenter.HTML(;
        canonical="https://PasoStudio73.github.io/Binnings.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/PasoStudio73/Binnings.jl",
    devbranch="main",
)
