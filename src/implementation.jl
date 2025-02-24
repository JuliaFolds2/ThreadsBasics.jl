module Implementation

import OhMyThreads: treduce, tmapreduce, treducemap, tforeach, tmap, tmap!, tcollect
using OhMyThreads: @spawn, @spawnat, WithTaskLocals, promise_task_local, ChannelLike
using OhMyThreads.Tools: nthtid
using OhMyThreads: Scheduler,
                   DynamicScheduler, StaticScheduler, GreedyScheduler,
                   SerialScheduler
using OhMyThreads.Schedulers: chunking_enabled,
                              chunking_mode, ChunkingMode, NoChunking,
                              FixedSize, FixedCount, scheduler_from_symbol, NotGiven,
                              isgiven
using Base: @propagate_inbounds
using Base.Threads: nthreads, @threads
using BangBang: append!!
using ChunkSplitters: ChunkSplitters, index_chunks, Consecutive
using ChunkSplitters.Internals: AbstractChunks, IndexChunks

const MaybeScheduler = Union{NotGiven, Scheduler, Symbol}

include("macro_impl.jl")

function _index_chunks(sched, arg)
    C = chunking_mode(sched)
    @assert C != NoChunking
    if C == FixedCount
        index_chunks(arg;
            n = sched.nchunks,
            split = sched.split)::IndexChunks{
            typeof(arg), ChunkSplitters.Internals.FixedCount}
    elseif C == FixedSize
        index_chunks(arg;
            size = sched.chunksize,
            split = sched.split)::IndexChunks{
            typeof(arg), ChunkSplitters.Internals.FixedSize}
    end
end

function _scheduler_from_userinput(scheduler::MaybeScheduler; kwargs...)
    if scheduler isa Scheduler
        isempty(kwargs) || scheduler_and_kwargs_err(; kwargs...)
        _scheduler = scheduler
    elseif scheduler isa Symbol
        _scheduler = scheduler_from_symbol(scheduler; kwargs...)
    else # default fallback
        _scheduler = DynamicScheduler(; kwargs...)
    end
end

function _check_chunks_incompatible_kwargs(; kwargs...)
    ks = keys(kwargs)
    if :ntasks in ks || :nchunks in ks || :chunksize in ks || :split in ks
        error("You've provided `chunks` or `index_chunks` as input and, at the same time, "*
              "chunking related keyword arguments (e.g. `ntasks`, `chunksize`, or `split`). "*
              "This isn't supported. "*
              "Set the chunking options directly in the `chunks` or `index_chunks` call or drop the latter.")
    end
    if :chunking in ks
        for (k, v) in kwargs
            if k == :chunking && v == true
                error("You've provided `chunks` or `index_chunks` as input and, at the same time, " *
                      "have set chunking=true. This isn't supported.")
            end
        end
    end
    return nothing
end

function tmapreduce(f, op, Arrs...;
        scheduler::MaybeScheduler = NotGiven(),
        outputtype::Type = Any,
        init = NotGiven(),
        kwargs...)
    mapreduce_kwargs = isgiven(init) ? (; init) : (;)
    _scheduler = _scheduler_from_userinput(scheduler; kwargs...)

    A = first(Arrs)
    if A isa AbstractChunks || A isa ChunkSplitters.Internals.Enumerate
        _check_chunks_incompatible_kwargs(; kwargs...)
    end
    if _scheduler isa SerialScheduler || isempty(first(Arrs))
        # empty input collection → align with Base.mapreduce behavior
        mapreduce(f, op, Arrs...; mapreduce_kwargs...)
    else
        @noinline _tmapreduce(f, op, Arrs, outputtype, _scheduler, mapreduce_kwargs)
    end
end

@noinline function scheduler_and_kwargs_err(; kwargs...)
    kwargstr = join(string.(keys(kwargs)), ", ")
    throw(ArgumentError("Providing an explicit scheduler as well as direct keyword arguments (e.g. $(kwargstr)) is currently not supported."))
end

treducemap(op, f, A...; kwargs...) = tmapreduce(f, op, A...; kwargs...)

# DynamicScheduler: AbstractArray/Generic
function _tmapreduce(f,
        op,
        Arrs,
        ::Type{OutputType},
        scheduler::DynamicScheduler,
        mapreduce_kwargs)::OutputType where {OutputType}
    (; threadpool) = scheduler
    check_all_have_same_indices(Arrs)
    if chunking_enabled(scheduler)
        tasks = map(_index_chunks(scheduler, first(Arrs))) do inds
            args = map(A -> view(A, inds), Arrs)
            # Note, calling `promise_task_local` here is only safe because we're assuming that
            # Base.mapreduce isn't going to magically try to do multithreading on us...
            @spawn threadpool mapreduce(promise_task_local(f), promise_task_local(op),
                args...; $mapreduce_kwargs...)
        end
        mapreduce(fetch, promise_task_local(op), tasks)
    else
        tasks = map(eachindex(first(Arrs))) do i
            args = map(A -> @inbounds(A[i]), Arrs)
            @spawn threadpool promise_task_local(f)(args...)
        end
        mapreduce(fetch, promise_task_local(op), tasks; mapreduce_kwargs...)
    end
end

# DynamicScheduler: AbstractChunks
function _tmapreduce(f,
        op,
        Arrs::Union{Tuple{AbstractChunks{T}}, Tuple{ChunkSplitters.Internals.Enumerate{T}}},
        ::Type{OutputType},
        scheduler::DynamicScheduler,
        mapreduce_kwargs)::OutputType where {OutputType, T}
    (; threadpool) = scheduler
    tasks = map(only(Arrs)) do idcs
        @spawn threadpool promise_task_local(f)(idcs)
    end
    mapreduce(fetch, promise_task_local(op), tasks; mapreduce_kwargs...)
end

# StaticScheduler: AbstractArray/Generic
function _tmapreduce(f,
        op,
        Arrs,
        ::Type{OutputType},
        scheduler::StaticScheduler,
        mapreduce_kwargs)::OutputType where {OutputType}
    nt = nthreads()
    check_all_have_same_indices(Arrs)
    if chunking_enabled(scheduler)
        tasks = map(enumerate(_index_chunks(scheduler, first(Arrs)))) do (c, inds)
            tid = @inbounds nthtid(mod1(c, nt))
            args = map(A -> view(A, inds), Arrs)
            # Note, calling `promise_task_local` here is only safe because we're assuming that
            # Base.mapreduce isn't going to magically try to do multithreading on us...
            @spawnat tid mapreduce(promise_task_local(f), promise_task_local(op), args...;
                mapreduce_kwargs...)
        end
        # Note, calling `promise_task_local` here is only safe because we're assuming that
        # Base.mapreduce isn't going to magically try to do multithreading on us...
        mapreduce(fetch, promise_task_local(op), tasks)
    else
        tasks = map(enumerate(eachindex(first(Arrs)))) do (c, i)
            tid = @inbounds nthtid(mod1(c, nt))
            args = map(A -> @inbounds(A[i]), Arrs)
            @spawnat tid promise_task_local(f)(args...)
        end
        # Note, calling `promise_task_local` here is only safe because we're assuming that
        # Base.mapreduce isn't going to magically try to do multithreading on us...
        mapreduce(fetch, promise_task_local(op), tasks; mapreduce_kwargs...)
    end
end

# StaticScheduler: AbstractChunks
function _tmapreduce(f,
        op,
        Arrs::Tuple{AbstractChunks{T}}, # we don't support multiple chunks for now
        ::Type{OutputType},
        scheduler::StaticScheduler,
        mapreduce_kwargs)::OutputType where {OutputType, T}
    check_all_have_same_indices(Arrs)
    chnks = only(Arrs)
    nt = nthreads()
    tasks = map(enumerate(chnks)) do (c, idcs)
        tid = @inbounds nthtid(mod1(c, nt))
        # Note, calling `promise_task_local` here is only safe because we're assuming that
        # Base.mapreduce isn't going to magically try to do multithreading on us...
        @spawnat tid promise_task_local(f)(idcs)
    end
    # Note, calling `promise_task_local` here is only safe because we're assuming that
    # Base.mapreduce isn't going to magically try to do multithreading on us...
    mapreduce(fetch, promise_task_local(op), tasks; mapreduce_kwargs...)
end

# NOTE: once v1.12 releases we should switch this to wait(t; throw=false)
wait_nothrow(t) = Base._wait(t)


"""
    empty_collection_error(task)

Check if a task failed due to an empty collection error.
"""
function empty_collection_error end

@static if VERSION < v"1.11.0-"
    function empty_collection_error(task)
        task.result isa MethodError && task.result.f == Base.mapreduce_empty
    end
else
    function empty_collection_error(task)
        task.result isa ArgumentError &&
            task.result.msg ==
            "reducing over an empty collection is not allowed; consider supplying `init` to the reducer"
    end
end

# GreedyScheduler w/o chunking
function _tmapreduce(f,
        op,
        Arrs,
        ::Type{OutputType},
        scheduler::GreedyScheduler{NoChunking},
        mapreduce_kwargs)::OutputType where {OutputType}
    ntasks_desired = scheduler.ntasks
    if Base.IteratorSize(first(Arrs)) isa Base.SizeUnknown
        ntasks = ntasks_desired
        ch_len = 0
    else
        check_all_have_same_indices(Arrs)
        ntasks = min(length(first(Arrs)), ntasks_desired)
        ch_len = length(first(Arrs))
    end
    # TODO: Use ChannelLike for iterators that support it. Dispatch on IndexLinear?
    ch = Channel{Tuple{eltype.(Arrs)...}}(ch_len; spawn = true) do ch
        for args in zip(Arrs...)
            put!(ch, args)
        end
    end
    tasks = map(1:ntasks) do _
        # Note, calling `promise_task_local` here is only safe because we're assuming that
        # Base.mapreduce isn't going to magically try to do multithreading on us...
        @spawn mapreduce(promise_task_local(op), ch; mapreduce_kwargs...) do args
            promise_task_local(f)(args...)
        end
    end
    # Doing this because of https://github.com/JuliaFolds2/OhMyThreads.jl/issues/82
    # The idea is that if the channel gets fully consumed before a task gets started up,
    # then if the user does not supply an `init` kwarg, we'll get an error.
    # Current way of dealing with this is just filtering out `mapreduce_empty` method
    # errors. This may not be the most stable way of dealing with things, e.g. if the
    # name of the function throwing the error changes this could break, so long term
    # we may want to try a different design.
    filtered_tasks = filter(tasks) do stabletask
        task = stabletask.t
        istaskdone(task) || wait_nothrow(task)
        if empty_collection_error(task)
            false
        else
            true
        end
    end
    # Note, calling `promise_task_local` here is only safe because we're assuming that
    # Base.mapreduce isn't going to magically try to do multithreading on us...
    mapreduce(fetch, promise_task_local(op), filtered_tasks; mapreduce_kwargs...)
end

# GreedyScheduler w/ chunking
function _tmapreduce(f,
        op,
        Arrs,
        ::Type{OutputType},
        scheduler::GreedyScheduler,
        mapreduce_kwargs)::OutputType where {OutputType}
    if Base.IteratorSize(first(Arrs)) isa Base.SizeUnknown
        throw(ArgumentError("SizeUnkown iterators in combination with a greedy scheduler and chunking are currently not supported."))
    end
    check_all_have_same_indices(Arrs)
    chnks = _index_chunks(scheduler, first(Arrs))
    ntasks_desired = scheduler.ntasks
    ntasks = min(length(chnks), ntasks_desired)

    # ChunkSplitters.IndexChunks support everything needed for ChannelLike
    ch = ChannelLike(chnks)

    tasks = map(1:ntasks) do _
        # Note, calling `promise_task_local` here is only safe because we're assuming that
        # Base.mapreduce isn't going to magically try to do multithreading on us...
        @spawn mapreduce(promise_task_local(op), ch; mapreduce_kwargs...) do inds
            args = map(A -> view(A, inds), Arrs)
            mapreduce(promise_task_local(f), promise_task_local(op), args...)
        end
    end
    # Doing this because of https://github.com/JuliaFolds2/OhMyThreads.jl/issues/82
    # The idea is that if the channel gets fully consumed before a task gets started up,
    # then if the user does not supply an `init` kwarg, we'll get an error.
    # Current way of dealing with this is just filtering out `mapreduce_empty` method
    # errors. This may not be the most stable way of dealing with things, e.g. if the
    # name of the function throwing the error changes this could break, so long term
    # we may want to try a different design.
    filtered_tasks = filter(tasks) do stabletask
        task = stabletask.t
        istaskdone(task) || wait_nothrow(task)
        if empty_collection_error(task)
            false
        else
            true
        end
    end
    # Note, calling `promise_task_local` here is only safe because we're assuming that
    # Base.mapreduce isn't going to magically try to do multithreading on us...
    mapreduce(fetch, promise_task_local(op), filtered_tasks; mapreduce_kwargs...)
end

function check_all_have_same_indices(Arrs)
    let A = first(Arrs), Arrs = Arrs[2:end]
        if !all(B -> eachindex(A) == eachindex(B), Arrs)
            error("The indices of the input arrays must match the indices of the output array.")
        end
    end
end

#-------------------------------------------------------------

function treduce(op, A...; kwargs...)
    tmapreduce(identity, op, A...; kwargs...)
end

#-------------------------------------------------------------

function tforeach(f, A...; kwargs...)::Nothing
    tmapreduce(f, (l, r) -> l, A...; kwargs..., init = nothing, outputtype = Nothing)
end

#-------------------------------------------------------------

function maybe_rewrap(g::G, f::F) where {G, F}
    g(f)
end

"""
   maybe_rewrap(g, f)

takes a closure `g(f)` and if `f` is a `WithTaskLocals`, we're going
to unwrap `f` and delegate its `TaskLocalValues` to `g`.

This should always be equivalent to just calling `g(f)`.
"""
function maybe_rewrap(g::G, f::WithTaskLocals{F}) where {G, F}
    (; inner_func, tasklocals) = f
    WithTaskLocals(vals -> g(inner_func(vals)), tasklocals)
end

#------------------------------------------------------------

function tmap(f, ::Type{T}, A::AbstractArray, _Arrs::AbstractArray...; kwargs...) where {T}
    Arrs = (A, _Arrs...)
    tmap!(f, similar(A, T), Arrs...; kwargs...)
end

function tmap(f,
        A::Union{AbstractArray, AbstractChunks, ChunkSplitters.Internals.Enumerate},
        _Arrs::AbstractArray...;
        scheduler::MaybeScheduler = NotGiven(),
        kwargs...)
    _scheduler = _scheduler_from_userinput(scheduler; kwargs...)

    if _scheduler isa GreedyScheduler
        error("Greedy scheduler isn't supported with `tmap` unless you provide an `OutputElementType` argument, since the greedy schedule requires a commutative reducing operator.")
    end
    if chunking_enabled(_scheduler) && hasfield(typeof(_scheduler), :split) &&
       _scheduler.split != Consecutive()
        error("Only `split == Consecutive()` is supported because the parallel operation isn't commutative. (Scheduler: $_scheduler)")
    end
    if (A isa AbstractChunks || A isa ChunkSplitters.Internals.Enumerate)
        _check_chunks_incompatible_kwargs(; kwargs...)
        if chunking_enabled(_scheduler)
            if _scheduler isa DynamicScheduler
                _scheduler = DynamicScheduler(;
                    threadpool = _scheduler.threadpool,
                    chunking = false)
            elseif _scheduler isa StaticScheduler
                _scheduler = StaticScheduler(; chunking = false)
            else
                error("Can't disable chunking for this scheduler?! Shouldn't be reached.",
                    _scheduler)
            end
        end
    end

    Arrs = (A, _Arrs...)
    if _scheduler isa SerialScheduler || isempty(A)
        # empty input collection → align with Base.map behavior
        map(f, Arrs...; kwargs...)
    else
        check_all_have_same_indices(Arrs)
        @noinline _tmap(_scheduler, f, A, _Arrs...)
    end
end

# w/o chunking (DynamicScheduler{NoChunking}): AbstractArray
function _tmap(scheduler::DynamicScheduler{NoChunking},
        f,
        A::AbstractArray,
        _Arrs::AbstractArray...;)
    (; threadpool) = scheduler
    Arrs = (A, _Arrs...)
    tasks = map(eachindex(A)) do i
        @spawn threadpool begin
            args = map(A -> A[i], Arrs)
            promise_task_local(f)(args...)
        end
    end
    v = map(fetch, tasks)
    reshape(v, size(A)...)
end

# w/o chunking (DynamicScheduler{NoChunking}): AbstractChunks
function _tmap(scheduler::DynamicScheduler{NoChunking},
        f,
        A::Union{AbstractChunks, ChunkSplitters.Internals.Enumerate},
        _Arrs::AbstractArray...)
    (; threadpool) = scheduler
    tasks = map(A) do idcs
        @spawn threadpool promise_task_local(f)(idcs)
    end
    map(fetch, tasks)
end

# w/o chunking (StaticScheduler{NoChunking}): AbstractChunks
function _tmap(scheduler::StaticScheduler{NoChunking},
        f,
        A::AbstractChunks,
        _Arrs::AbstractArray...)
    nt = nthreads()
    tasks = map(enumerate(A)) do (c, idcs)
        tid = @inbounds nthtid(mod1(c, nt))
        @spawnat tid promise_task_local(f)(idcs)
    end
    map(fetch, tasks)
end

# w/o chunking (StaticScheduler{NoChunking}): AbstractArray
function _tmap(scheduler::StaticScheduler{NoChunking},
        f,
        A::AbstractArray,
        _Arrs::AbstractArray...;)
    Arrs = (A, _Arrs...)
    nt = nthreads()
    tasks = map(enumerate(A)) do (c, i)
        tid = @inbounds nthtid(mod1(c, nt))
        @spawnat tid begin
            args = map(A -> A[i], Arrs)
            promise_task_local(f)(args...)
        end
    end
    v = map(fetch, tasks)
    reshape(v, size(A)...)
end

# w/ chunking
function _tmap(scheduler::Scheduler,
        f,
        A::AbstractArray,
        _Arrs::AbstractArray...)
    Arrs = (A, _Arrs...)
    idcs = collect(_index_chunks(scheduler, A))
    reduction_f = append!!
    mapping_f = maybe_rewrap(f) do f
        (inds) -> begin
            args = map(A -> @view(A[inds]), Arrs)
            map(f, args...)
        end
    end
    v = tmapreduce(mapping_f, reduction_f, idcs; scheduler)
    reshape(v, size(A)...)
end

@propagate_inbounds function tmap!(f,
        out,
        A::AbstractArray,
        _Arrs::AbstractArray...;
        scheduler::MaybeScheduler = NotGiven(),
        kwargs...)
    _scheduler = _scheduler_from_userinput(scheduler; kwargs...)

    Arrs = (A, _Arrs...)
    if _scheduler isa SerialScheduler
        map!(f, out, Arrs...)
    else
        @boundscheck check_all_have_same_indices((out, Arrs...))
        mapping_f = maybe_rewrap(f) do f
            function mapping_function(i)
                args = map(A -> @inbounds(A[i]), Arrs)
                res = f(args...)
                out[i] = res
            end
        end
        @noinline tforeach(mapping_f, eachindex(out); scheduler = _scheduler)
        out
    end
end

#-------------------------------------------------------------

function tcollect(::Type{T}, gen::Base.Generator{<:AbstractArray}; kwargs...) where {T}
    tmap(gen.f, T, gen.iter; kwargs...)
end
tcollect(gen::Base.Generator{<:AbstractArray}; kwargs...) = tmap(gen.f, gen.iter; kwargs...)

tcollect(::Type{T}, A; kwargs...) where {T} = tmap(identity, T, A; kwargs...)
tcollect(A; kwargs...) = tmap(identity, A; kwargs...)

end # module Implementation
