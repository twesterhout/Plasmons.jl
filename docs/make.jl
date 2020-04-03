using Documenter, Plasmons

makedocs(;
    modules=[Plasmons],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/twesterhout/Plasmons.jl/blob/{commit}{path}#L{line}",
    sitename="Plasmons.jl",
    authors="Tom Westerhout",
    assets=String[],
)

deploydocs(;
    repo="github.com/twesterhout/Plasmons.jl",
)
