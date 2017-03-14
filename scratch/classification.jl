using Distributions
using DataFrames
using Gadfly

srand(123)
d = DataFrame([rand(MvNormal([2,2], 1), 10)' fill("a", 10) ; rand(MvNormal([-2,-2], 1), 10)' fill("b", 10) ])

plot(d, x = "x1", y = "x2", color = "x3", Geom.point)
