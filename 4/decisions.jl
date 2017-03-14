using Distributions
using DataFrames
using Gadfly
using Base.Test

function tuples_to_data(a)
    if length(a) == 0
        return Dict(:X => [], :y => [])
    end
    X = reduce(vcat, [t[1]' for t in a])
    y  = [t[2] for t in a]
    Dict(:X => X, :y => y)
end

make_tuples(X, y) = [(X[i,:],y) for i in 1:size(X)[1]]

function make_ones(N, d)
    m = append!(ones(2), fill(0, d - 2))
    rand(MvNormal(m, 1), N)'
end

function make_zeros(N, d)
    m = zeros(d)
    rand(MvNormal(m, 1), N)'
end

function generate_distributions(N, d = 5)
    o = make_ones(Int(N/2), d)
    z = make_zeros(Int(N/2), d)
    vcat(make_tuples(o, 1), make_tuples(z,0))
end

function ent(targets)
    if length(targets) == 0
        return 0
    end
    probs = [count(t -> t == class, targets)/length(targets) for class in unique(targets)]
    sum([p*log(2, 1/p) for p in probs])
end

@testset "entropy!" begin
    @test ent([0,0,1,1]) == 1
    @test ent([0,0,0,0]) == 0
    @test ent([0,0,1,1,1]) < 1
end

function counter(a, b, class)
    prev = a[end]
    d = b[2] == class ?
        Dict(:a => prev[:a] - 1) :
        Dict(:b => prev[:b] + 1)
    vcat(a, [merge(prev, d)])
end

@test counter([Dict(:a => 5, :b => 0)], (3,1), 1)[end] == Dict(:a => 4, :b => 0)

make_counter(class) = (a,b) -> counter(a,b,class)

function find_threshold(x, y, class)
    vals = collect(zip(x,y))
    N_class = count(t -> t == class, y)
    a = reduce(make_counter(class), [Dict(:a => N_class, :b => 0)], vals)
    scores = [sum(values(d)) for d in a]
    sortperm(scores)[1], minimum(scores)
end

@testset "find threshold" begin
    @test find_threshold([1,2], [0,0], 1) == (1,0)
    @test find_threshold([1,1,6], [1,0,1], 0) == (1,1)
    @test find_threshold([1,1,6], [1,0,1], 1) == (2,1)
    @test find_threshold([1,2,3,4,5], [0,0,1,1,1], 1) == (6,2)
    @test find_threshold([1,2,3,4,5], [0,0,1,1,1], 0) == (3,0)
    @test find_threshold([1,2,3,4,5], [1,0,1,0,0], 1) == (2,1)
    @test find_threshold([1,2,3,4,5,6,7,8], [1,0,1,0,0,1,1,1], 1) == (9,3)
    @test find_threshold([1,2,3,4,5,6,7,8], [1,0,1,0,0,1,1,1], 0) == (6,2)
end

function info_gain(a, b)
    y = vcat(a,b)
    prob(s) = length(s)/length(y)
    new_ent = sum([prob(s) * ent(s) for s in [a,b]])
    ent(y) - new_ent
end

@testset "info gain" begin
    @test info_gain([1,1,0], [0,1]) <= .1
    @test info_gain([0,0,0], [1,1]) > .9
    @test info_gain([0,0,0,0,0,0,0], [1,0]) < .3
    @test info_gain([0,0,0,0,0,0,0], []) == 0
end

function sort_and_get_feature(data, i)
    x = data[:X][:,i]
    y = data[:y]
    vals = sort(collect(zip(x, y)), by = t -> t[1])
    [t[1] for t in vals], [t[2] for t in vals]
end

@test sort_and_get_feature(Dict(:X => [1 5 3; 4 2 6], :y => [0,1]), 1) == ([1,4], [0,1])
@test sort_and_get_feature(Dict(:X => [1 5 3; 4 2 6], :y => [0,1]), 2) == ([2,5], [1,0])

function thresh_value(x, y, i)
    # Check from left-right against both class types
    o = find_threshold(x, y, 1)
    z = find_threshold(x, y, 0)
    # ones are left, zeros is right! rename???
    if o[2] == z[2]
        thresh,a,b = o[1] < z[1] ? (o[1],:left, :right) : (z[1],:right,:left)
    else
        thresh,a,b = o[2] < z[2] ? (o[1],:left, :right) : (z[1],:right,:left)
    end
    val = thresh > 1 ? (x[thresh-1] + x[thresh])/2 : x[thresh]
    val, a, b
end

function make_fn(x, y, i)
    # function expects a single data point, checks the feature,
    # and returns left or right based on split
    val,a,b = thresh_value(x, y, i)
    z -> z[i] <= val ? a : b
end

@testset "make_fn" begin
    fn = make_fn([1,2,3,4,5,6,7,8], [1,0,1,0,0,1,1,1], 1)
    @test fn([1,2,3]) == :right
    @test fn([6,2,3]) == :left
    fn = make_fn([1,2,3,4,5,6,7,8], [1,1,0,0,0,0,0,0], 2)
    @test fn([1,2,3]) == :left
    @test fn([6,2,3]) == :left
    @test fn([6,3,3]) == :right
    fn = make_fn([1,1,6], [1,0,1], 1)
    @test fn(5) == :right
    @test fn(1) == :left
end

function splitter(data, i)
    x,y = sort_and_get_feature(data, i)
    make_fn(x,y,i)
end

function make_fns(data)
    d = size(data[:X])[2]
    [splitter(data, i) for i in 1:d]
end

@testset "make_fns" begin
    fns = make_fns(Dict(:X => [1 5 6; 6 3 1; 2 7 1], :y => [1,0,1]))
    @test length(fns) == 3
    @test fns[1]([1]) == :left
    @test fns[1]([6]) == :right
    @test fns[2]([6,3]) == :right
    @test fns[2]([6,4.5]) == :left
    @test fns[3]([1,1,5]) == :left
    @test fns[3]([1,1,0]) == :right
    srand(123)
    data = tuples_to_data(generate_distributions(50, 10))
    fns = make_fns(data)
    @test length(fns) == 10
end


function split_data_by_fn(data, fn)
    X = data[:X]
    y = data[:y]
    N = size(X)[1]
    dirs = [fn(X[i,:]) for i in 1:N]
    left = [(X[i,:],y[i]) for i in 1:N if dirs[i] == :left]
    right = [(X[i,:],y[i]) for i in 1:N if dirs[i] == :right]
    tuples_to_data(left), tuples_to_data(right)
end

@testset "split_data_by_fn" begin
    d = Dict(:X => [1 5 6; 6 3 1; 2 7 1], :y => [1,0,1])
    fn = x -> x[2] > 4 ? :left : :right
    @test split_data_by_fn(d, fn)[1] == Dict(:X => [1 5 6; 2 7 1], :y => [1, 1])
    @test split_data_by_fn(d, fn)[2] == Dict(:X => [6 3 1], :y => [0])
end


function find_next_split(data)
    fns = make_fns(data)
    splits = [split_data_by_fn(data, fn) for fn in fns]
    infos = [info_gain(a[:y], b[:y]) for (a,b) in splits]
    i = indmax(infos)
    fns[i], splits[i][1], splits[i][2]
end

@testset "find_next_split" begin
    d = Dict(:X => [1 5 6; 6 3 1; 2 7 1], :y => [1,0,1])
    fn, left, right = find_next_split(d)
    @test left == Dict(:X => [1 5 6; 2 7 1], :y => [1,1])
    @test right == Dict(:X => [6 3 1], :y => [0])
    @test fn([1]) == :left
    @test fn([8]) == :right
end


leaf(c) = Dict(:class => c)
make_leaf(y) = leaf(Int(round(mean(y))))

function build_tree(data, k)
    # Stop if we've reached max Depth.
    if k == 0
        return make_leaf(data[:y])
    end
    fn, left, right = find_next_split(data)
    # Stop also if we have a one-sided split
    if isempty(left[:y])
        return leaf(0)
    elseif isempty(right[:y])
        return leaf(1)
    end
    # Else recurse!
    Dict(:fn => fn,
         :left => build_tree(left, k-1),
         :right => build_tree(right, k-1))
end

function classifier(x, node)
    if haskey(node,:class)
        return node[:class]
    end
    dir = node[:fn](x)
    classifier(x, node[dir]) # recur
end

make_basic_classifier(tuples, k, S) = x -> classifier(x, build_tree(tuples_to_data(tuples), k))

@testset "stopping conditions" begin
    srand(100)
    data = tuples_to_data(generate_distributions(50, 4))
    @test build_tree(data, 0)[:class] == 0
    @test classifier([], Dict(:class => 0)) == 0
    @test classifier([], Dict(:fn => x -> :left, :left => Dict(:class => 1))) == 1
end

function bagged_trees(tuples, k, S)
    N = length(tuples)
    new_tuples = [getindex(tuples, rand(1:N, N)) for s in 1:S]
    sets = [tuples_to_data(s) for s in new_tuples]
    [build_tree(d, k) for d in sets]
end

vote(classifiers, x) = Int(round(mean([c(x) for c in classifiers])))
@test vote([x -> 1, x -> 0, x -> 0], [1,2,3]) == 0

function make_bagged_classifiers(tuples, k, S)
    trees = bagged_trees(tuples, k, S)
    classifiers = [x -> classifier(x, t) for t in trees]
    x -> vote(classifiers, x)
end

subsample(data, cols) = Dict(:X => data[:X][:,cols], :y => data[:y])

function subsampling_classifier(data, k, c = 2)
    dims = size(data[:X])[2]
    cols = rand(1:dims, 2)
    tree = build_tree(subsample(data, cols), k)
    x -> classifier(x[cols], tree)
end

function make_subsampling_classifiers(tuples, k, S)
    data = tuples_to_data(tuples)
    classifiers = [subsampling_classifier(data, k) for i in S]
    x -> vote(classifiers, x)
end

function test_classifiers(N, d, k, t, S, fn)
    tuples = generate_distributions(N, d)
    tests = generate_distributions(t, d)
    cl = fn(tuples, k, S)
    mean([y == cl(x) ? 0 : 1 for (x,y) in tests])
end

runner(N,d,k,t,S,fn,m) = mean([test_classifiers(N, d, k, t, S, fn) for i in 1:m])

function plot_standard(N, t, m, D = 2:4:18, K = 2:1:8)
    make_frame(v,d,k) = DataFrame(Error = v, Dimensions = string(d), Depth = k)
    d = vcat([make_frame(runner(N, d, k, t, 1, make_basic_classifier, m), d, k) for d in D for k in K])
    plot(d, x = :Depth, y = :Error, color = :Dimensions, Geom.line)
end

function plot_bagged(N, t, m, k = 2, D = 2:4:18, Q = 2:2:20)
    make_frame(v,d,q) = DataFrame(Error = v, Dimensions = string(d), Quorum = q)
    d = vcat([make_frame(runner(N, d, k, t, q, make_bagged_classifiers, m), d, q) for d in D for q in Q])
    plot(d, x = :Quorum, y = :Error, color = :Dimensions, Geom.line)
end

function plot_subsampling(N, t, m, k = 2, D = 2:4:18, Q = 2:2:20)
    make_frame(v,d,q) = DataFrame(Error = v, Dimensions = string(d), Quorum = q)
    d = vcat([make_frame(runner(N, d, k, t, q, make_subsampling_classifiers, m), d, q) for d in D for q in Q])
    plot(d, x = :Quorum, y = :Error, color = :Dimensions, Geom.line)
end

# n = 20, overfitting
plot_standard(20, 40, 1)

# n = 200, overfitting
plot_standard(200, 40, 1)

#n = 20
plot_bagged(20, 40, 1)

#n = 200
plot_bagged(200, 40, 1)

#n = 100
plot_subsampling(100, 40, 1)
