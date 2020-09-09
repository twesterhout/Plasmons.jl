using PackageCompiler

create_app(
    joinpath(dirname(@__FILE__), ".."),
    "plasmons-v1.0.0",
    precompile_statements_file = joinpath(dirname(@__FILE__), "..", "precompile_file.jl"),
    filter_stdlibs = true,
)
