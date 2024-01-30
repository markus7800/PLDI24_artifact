using Distributions

# model and data adapted from https://github.com/mugamma/gmm/blob/master/pg.ipynb

const gt_k = 4
const gt_ys = [-7.87951290075215, -23.251364738213493, -5.34679518882793, -3.163770449770572,
10.524424782864525, 5.911987013277482, -19.228378698266436, 0.3898087330050574,
8.576922415766697, 7.727416085566447, -18.043123523482492, 9.108136117789305,
29.398734347901787, 2.8578485031858003, -20.716691460295685, -18.5075008084623,
-21.52338318392563, 10.062657028986715, -18.900545157827718, 3.339430437507262,
3.688098690412526, 4.209808727262307, 3.371091291010914, 30.376814419984456,
12.778653273596902, 28.063124205174137, 10.70527515161964, -18.99693615834304,
8.135342537554163, 29.720363913218446, 29.426043027354385, 28.40516772785764,
31.975585225366686, -20.642437143912638, 30.84807631345935, -21.46602061526647,
12.854676808303978, 30.685416799345685, 5.833520737134923, 7.602680172973942,
10.045516408942117, 28.62342173081479, -20.120184774438087, -18.80125468061715,
12.849708921404385, 31.342270731653656, 4.02761078481315, -19.953549865339976,
-2.574052170014683, -21.551814470820258, -2.8751904316333268,
13.159719198798443, 8.060416669497197, 12.933573330915458, 0.3325664001681059,
11.10817217269102, 28.12989207125211, 11.631846911966806, -15.90042467317705,
-0.8270272159702201, 11.535190070081708, 4.023136673956579,
-22.589713328053048, 28.378124912868305, -22.57083855780972,
29.373356677376297, 31.87675796607244, 2.14864533495531, 12.332798078071061,
8.434664672995181, 30.47732238916884, 11.199950328766784, 11.072188217008367,
29.536932243938097, 8.128833670186253, -16.33296115562885, 31.103677511944685,
-20.96644212192335, -20.280485886015406, 30.37107537844197, 10.581901339669418,
-4.6722903116912375, -20.320978011296315, 9.141857987635252, -18.6727012563551,
7.067728508554964, 5.664227155828871, 30.751158861494442, -20.198961378110013,
-4.689645356611053, 30.09552608716476, -19.31787364001907, -22.432589846769154,
-0.9580412415863696, 14.180597007125487, 4.052110659466889,
-18.978055134755582, 13.441194891615718, 7.983890038551439, 7.759003567480592]
const gt_zs = [2, 1, 2, 2, 3, 3, 1, 2, 3, 3, 1, 3, 4, 2, 1, 1, 1, 3, 1, 2, 2, 3, 2, 4, 3, 4,
      3, 1, 3, 4, 4, 4, 4, 1, 4, 1, 3, 4, 3, 3, 3, 4, 1, 1, 3, 4, 3, 1, 2, 1, 2,
      3, 3, 3, 2, 3, 4, 3, 1, 2, 3, 2, 1, 4, 1, 4, 4, 2, 3, 3, 4, 3, 3, 4, 3, 1,
      4, 1, 1, 4, 3, 2, 1, 3, 1, 3, 3, 4, 1, 2, 4, 1, 1, 2, 3, 2, 1, 3, 3, 3]
const gt_ws = [0.20096082191563705, 0.22119959941799663, 0.3382086364817468, 0.23963094218461967]
const gt_μs = [-20.0, 0.0, 10.0, 30.0]
const gt_σ²s = [3.0, 8.0, 7.0, 1.0]

struct TraceEntry{T}
    value::T
    logpdf::Float64
end
const Trace = Dict{Any,TraceEntry}

function distribution(variable::Symbol, tr::Trace)::Distribution
    if variable == :w
        K = gt_k
        δ = 5.0
        return Dirichlet(δ * ones(K))
    end
end
function distribution(variable::Pair{Symbol, Int64}, tr::Trace)::Distribution  
    if variable[1] == :z
        w = tr[:w].value
        return Categorical(w)
    elseif variable[1] == :y
        i = variable[2]
        z = tr[:z => i].value
        μ = tr[:μ => z].value
        σ² = tr[:σ² => z].value
        return Normal(μ, sqrt(σ²))
    elseif variable[1] == :μ
        ξ = 0.0
        κ = 0.01
        return Normal(ξ, 1/sqrt(κ))
    else
        @assert variable[1] == :σ²
        α = 2.0
        β = 10.0
        return InverseGamma(α, β)
    end
end

function propose(variable, tr::Trace)::TraceEntry
    d = distribution(variable, tr)
    value = rand(d)
    return TraceEntry(value, logpdf(d, value))
end

function score(variable, value, tr::Trace)::TraceEntry
    d = distribution(variable, tr)
    return TraceEntry(value, logpdf(d, value))
end

# const gt_ys = vcat([gt_ys for i in 1:10]...)

function get_latents(ys)
    latents = Any[]
    push!(latents, :w)
    append!(latents, [:μ => k for k in 1:gt_k])
    append!(latents, [:σ² => k for k in 1:gt_k])
    append!(latents, [:z => i for i in eachindex(ys)])
    return latents
end

function init_trace(latents, ys)
    trace = Trace()
    for variable in latents
        trace[variable] = propose(variable, trace)
    end

    for i in eachindex(ys)
        trace[:y => i] = score(:y => i, ys[i], trace)
    end
    return trace
end

function get_score(tr::Trace)
    weight = sum(entry.logpdf for (v, entry) in tr)
    return weight
end

function score_trace_naive(latents, ys, tr::Trace)
    # regardless of updated variable, we compute log p(x,y) by scoring every variable
    weight = 0.
    for variable in latents
        scored_entry = score(variable, tr[variable].value, tr)
        tr[variable] = scored_entry
        weight += scored_entry.logpdf
    end
    for i in eachindex(ys)
        scored_entry = score(:y => i, ys[i], tr)
        tr[:y => i] = scored_entry
        weight += scored_entry.logpdf
    end
    return weight
end

function score_trace_while(variable, ys, proposed_trace::Trace, current_weight::Float64, current_trace::Trace)
    # while loop static dependency optimisation:
    # in addition to rescoring the updated variable:
    # if we update variable w, we only need to rescore all z=>i i=1..N
    # if we update variables :μ=>k, :σ²=>k, :z=>i, we only need to rescore all y=>i i=1..N
    weight = current_weight

    # always rescore updated variable
    proposed_trace[variable] = score(variable, proposed_trace[variable].value, proposed_trace)
    weight += proposed_trace[variable].logpdf - current_trace[variable].logpdf

    if variable == :w
        # rescore z
        for i in eachindex(ys)
            dep = :z => i
            value = proposed_trace[dep].value
            scored_entry = score(dep, value, proposed_trace)
            weight += scored_entry.logpdf - current_trace[dep].logpdf
            proposed_trace[dep] = scored_entry
        end

    else # v[1] == :μ || v[1] == :σ² || v[1] == :z
        # rescore y
        for i in eachindex(ys)
            scored_entry = score(:y => i, ys[i], proposed_trace)
            proposed_trace[:y => i] = scored_entry
            weight += scored_entry.logpdf - current_trace[:y => i].logpdf
        end
    end

    return weight
end

function score_trace_forloop(variable, ys, proposed_trace::Trace, current_weight::Float64, current_trace::Trace)
    # (unrolled) for loop static dependency optimisation:
    # in addition to rescoring the updated variable:
    # if we update variable w, we only need to rescore all z=>i i=1..N
    # if we update variables :μ=>k, :σ²=>k, we only need to rescore all  y=>i i=1..N
    # if we :z=>i, we only need to rescore single y=>i
    weight = current_weight

    # always rescore updated variable
    proposed_trace[variable] = score(variable, proposed_trace[variable].value, proposed_trace)
    weight += proposed_trace[variable].logpdf - current_trace[variable].logpdf

    if variable == :w
        # rescore z
        for i in eachindex(ys)
            dep = :z => i
            value = proposed_trace[dep].value
            scored_entry = score(dep, value, proposed_trace)
            weight += scored_entry.logpdf - current_trace[dep].logpdf
            proposed_trace[dep] = scored_entry
        end

    elseif variable[1] == :μ || variable[1] == :σ²
        # rescore y
        for i in eachindex(ys)
            scored_entry = score(:y => i, ys[i], proposed_trace)
            proposed_trace[:y => i] = scored_entry
            weight += scored_entry.logpdf - current_trace[:y => i].logpdf
        end
    else
        # rescore y[i]
        @assert variable[1] == :z
        i = variable[2]
        scored_entry = score(:y => i, ys[i], proposed_trace)
        proposed_trace[:y => i] = scored_entry
        weight += scored_entry.logpdf - current_trace[:y => i].logpdf
    end

    return weight
end

function lmh(ys, n_iter, score_type::Symbol)
    latents = get_latents(ys)
    latent_names = copy(latents)
    traces = Vector{Trace}(undef, n_iter)

    # init
    current_trace = init_trace(latents, ys)
    current_weight = score_trace_naive(latents, ys, current_trace)

    for i in 1:n_iter
        Random.shuffle!(latent_names)

        for variable in latent_names
            proposed_trace = copy(current_trace)
            proposed_entry = propose(variable, current_trace)
            proposed_trace[variable] = proposed_entry

            if score_type == :naive
                proposed_weight = score_trace_naive(latents, ys, proposed_trace)
            elseif score_type == :for
                proposed_weight = score_trace_forloop(variable, ys, proposed_trace, current_weight, current_trace)
            else
                proposed_weight = score_trace_while(variable, ys, proposed_trace, current_weight, current_trace)
            end

            log_α = proposed_weight - current_weight + current_trace[variable].logpdf - proposed_trace[variable].logpdf
            if log(rand()) < log_α
                current_trace = proposed_trace
                current_weight = proposed_weight
            end
        end
        traces[i] = current_trace
    end

    return traces
end

import Random
begin
    naive_times = Float64[]
    while_times = Float64[]
    for_times = Float64[]
    data_sizes = 10:10:100
    n_iter = 3000
    for i in data_sizes
        println("LMH: data size: $i")
        ys = gt_ys[1:i]
        Random.seed!(0)
        stats = @timed lmh(ys, n_iter, :naive)
        push!(naive_times, stats.time)
        Random.seed!(0)
        stats = @timed lmh(ys, n_iter, :while)
        push!(while_times, stats.time)
        Random.seed!(0)
        stats = @timed lmh(ys, n_iter, :for)
        push!(for_times, stats.time)
    end

    println("Result:")
    println(naive_times)
    println(while_times)
    println(for_times)
end

import DelimitedFiles: writedlm

file_name = "gmm_result.txt"
open(file_name, "w") do io
    writedlm(io, [naive_times while_times for_times])
end