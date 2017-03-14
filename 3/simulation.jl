using Base.Test
using DataFrames
using Gadfly

function generate_X(N, d = 3)
    rand(N, d)*2*2^(1/d) - 2^(1/d)
end

@test maximum(generate_X(100, 3)) <= 2^(1/3)
@test maximum(generate_X(100, 6)) <= 2^(1/6)
@test size(generate_X(100, 5)) == (100, 5)

function should_be_one(X::AbstractArray{Float64}, min = -1.0, max = 1.0)
    all([x <= max && x >= min for x in X])
end

function should_be_one(x::Float64, min = -1.0, max = 1.0)
    x <= max && x >= min
end

@test should_be_one(1.5) == false
@test should_be_one([-.5, .5]) == true
@test should_be_one([-.5, 1.5]) == false

function label_X(X, fn)
    [fn(X[i,:]) ? 1 : 0 for i in 1:size(X)[1]]
end

function get_edges(X)
    [(minimum(X[:,d]), maximum(X[:,d])) for d in 1:size(X)[2]]
end

function get_ones(X, Y)
    ones = [X[i,:]' for i in 1:length(Y) if Y[i] == 1]
    length(ones) > 0 ? reduce(vcat, ones) : ones
end

@test get_ones([1,2,3], [0,0,0]) == []
@test get_ones([1 2 3; 4 5 6], [1,1]) == [1 2 3; 4 5 6]

function get_corners(edges)
    m = [y[i] for y=edges, i=1:2]
    minimum(m), maximum(m)
end

function cube_classifier(X, Y)
    edges = get_edges(get_ones(X, Y))
    min, max = get_corners(edges)
    x -> should_be_one(x, min, max)
end

function rect_row(x, edges)
    all([should_be_one(x[i], edges[i][1], edges[i][2]) for i in 1:length(edges)])
end

@test rect_row([-1.0,0.0], [(0,1), (-1,1)]) == false
@test rect_row([0.9,0.0], [(0,1), (-1,1)]) == true

function rect_classifier(X, Y)
    edges = get_edges(get_ones(X, Y))
    x -> rect_row(x, edges)
end

#############################################
# Testing
#############################################

function test_classifier(X, Y, fn)
    predicted = label_X(X, fn)
    sum([i ? 0 : 1 for i in (predicted .== Y)])
end

function generate_and_test(N_test, N_train, d)
    # Train
    X_train = generate_X(N_train, d)
    Y_train = label_X(X_train, should_be_one)
    rect = rect_classifier(X_train, Y_train)
    cube = cube_classifier(X_train, Y_train)
    # Test
    X = generate_X(N_test, d)
    Y = label_X(X, should_be_one)
    [test_classifier(X, Y, cube) test_classifier(X, Y, rect)]
end

#############################################
# Plotting
#############################################

function format_results(results, variable)
    named = names!(DataFrame(results), [:Cube, :Rectangle, variable])
    df = stack(named, [:Cube, :Rectangle])
    names!(df, [:Classifier, :Errors, variable])
end

mean_trials(fn, K) = mean(reduce(vcat, [fn() for _ in 1:K]), 1)

function increasing_dimensions(N_test, N_train, max_d, K, step = 5)
    create_fn(d) = () -> generate_and_test(N_test, N_train, d)
    a = [[mean_trials(create_fn(d), K) d] for d in 1:step:max_d]
    format_results(reduce(vcat, a), :Dimensions)
end

function increasing_sample(N_test, max_N, d,  K, step = 10, start = 20)
    create_fn(n) = () -> generate_and_test(N_test, n, d)
    a = [[mean_trials(create_fn(n), K) n] for n in start:step:max_N]
    format_results(reduce(vcat, a), Symbol("Training Size"))
end

function plotter(frame, variable)
    plot(frame, x = variable, y = "Errors", color = "Classifier", Geom.line)
end


plotter(increasing_dimensions(500, 100, 200, 50, 5), "Dimensions")

plotter(increasing_sample(500, 1000, 10, 50, 50), "Training Size")
