using Test, OhMyThreads
using OhMyThreads: TaskLocalValue, WithTaskLocals, @fetch, promise_task_local
using OhMyThreads: Consecutive, RoundRobin
using OhMyThreads.Experimental: @barrier

include("Aqua.jl")

sets_to_test = [(~ = isapprox, f = sin ∘ *, op = +,
                    itrs = (rand(ComplexF64, 10, 10), rand(-10:10, 10, 10)),
                    init = complex(0.0))
                (~ = isapprox, f = cos, op = max, itrs = (1:100000,), init = 0.0)
                (~ = (==), f = round, op = vcat, itrs = (randn(1000),), init = Float64[])
                (~ = (==), f = last, op = *,
                    itrs = ([1 => "a", 2 => "b", 3 => "c", 4 => "d", 5 => "e"],),
                    init = "")]

ChunkedGreedy(; kwargs...) = GreedyScheduler(; kwargs...)

@testset "Basics" begin
    for (; ~, f, op, itrs, init) in sets_to_test
        @testset "f=$f, op=$op, itrs::$(typeof(itrs))" begin
            @testset for sched in (
                StaticScheduler, DynamicScheduler, GreedyScheduler,
                DynamicScheduler{OhMyThreads.Schedulers.NoChunking},
                SerialScheduler, ChunkedGreedy)
                @testset for split in (Consecutive(), RoundRobin(), :consecutive, :roundrobin)
                    for nchunks in (1, 2, 6)
                        if sched == GreedyScheduler
                            scheduler = sched(; ntasks = nchunks)
                        elseif sched == DynamicScheduler{OhMyThreads.Schedulers.NoChunking}
                            scheduler = DynamicScheduler(; chunking = false)
                        elseif sched == SerialScheduler
                            scheduler = SerialScheduler()
                        else
                            scheduler = sched(; nchunks, split)
                        end

                        kwargs = (; scheduler)
                        if (split in (RoundRobin(), :roundrobin) ||
                            sched ∈ (GreedyScheduler, ChunkedGreedy)) || op ∉ (vcat, *)
                            # scatter and greedy only works for commutative operators!
                        else
                            mapreduce_f_op_itr = mapreduce(f, op, itrs...)
                            @test tmapreduce(f, op, itrs...; init, kwargs...) ~ mapreduce_f_op_itr
                            @test treducemap(op, f, itrs...; init, kwargs...) ~ mapreduce_f_op_itr
                            @test treduce(op, f.(itrs...); init, kwargs...) ~ mapreduce_f_op_itr
                        end

                        split in (RoundRobin(), :roundrobin) && continue
                        map_f_itr = map(f, itrs...)
                        @test all(tmap(f, Any, itrs...; kwargs...) .~ map_f_itr)
                        @test all(tcollect(Any, (f(x...) for x in collect(zip(itrs...))); kwargs...) .~ map_f_itr)
                        @test all(tcollect(Any, f.(itrs...); kwargs...) .~ map_f_itr)

                        RT = Core.Compiler.return_type(f, Tuple{eltype.(itrs)...})

                        @test tmap(f, RT, itrs...; kwargs...) ~ map_f_itr
                        @test tcollect(RT, (f(x...) for x in collect(zip(itrs...))); kwargs...) ~ map_f_itr
                        @test tcollect(RT, f.(itrs...); kwargs...) ~ map_f_itr

                        if sched ∉ (GreedyScheduler, ChunkedGreedy)
                            @test tmap(f, itrs...; kwargs...) ~ map_f_itr
                            @test tcollect((f(x...) for x in collect(zip(itrs...))); kwargs...) ~ map_f_itr
                            @test tcollect(f.(itrs...); kwargs...) ~ map_f_itr
                        end
                    end
                end
            end
        end
    end
end;

@testset "ChunkSplitters.Chunk" begin
    x = rand(100)
    chnks = OhMyThreads.index_chunks(x; n = Threads.nthreads())
    for scheduler in (
        DynamicScheduler(; chunking = false), StaticScheduler(; chunking = false))
        @testset "$scheduler" begin
            @test tmap(x -> sin.(x), chnks; scheduler) ≈ map(x -> sin.(x), chnks)
            @test tmapreduce(x -> sin.(x), vcat, chnks; scheduler) ≈
                  mapreduce(x -> sin.(x), vcat, chnks)
            @test tcollect(chnks; scheduler) == collect(chnks)
            @test treduce(vcat, chnks; scheduler) == reduce(vcat, chnks)
            @test isnothing(tforeach(x -> sin.(x), chnks; scheduler))
        end
    end

    # enumerate(chunks)
    data = 1:100
    @test tmapreduce(+, enumerate(OhMyThreads.index_chunks(data; n=5)); chunking=false) do (i, idcs)
        [i, sum(@view(data[idcs]))]
    end == [sum(1:5), sum(data)]
    @test tmapreduce(+, enumerate(OhMyThreads.index_chunks(data; size=5)); chunking=false) do (i, idcs)
        [i, sum(@view(data[idcs]))]
    end == [sum(1:20), sum(data)]
    @test tmap(enumerate(OhMyThreads.index_chunks(data; n=5)); chunking=false) do (i, idcs)
        [i, idcs]
    end == [[1, 1:20], [2, 21:40], [3, 41:60], [4, 61:80], [5, 81:100]]
end;

@testset "macro API" begin
    # basic
    @test @tasks(for i in 1:3
        i
    end) |> isnothing

    # reduction
    @test @tasks(for i in 1:3
        @set reducer = (+)
        i
    end) == 6

    # scheduler settings
    for sched in (StaticScheduler(), DynamicScheduler(), GreedyScheduler())
        @test @tasks(for i in 1:3
            @set scheduler = sched
            i
        end) |> isnothing
    end
    # scheduler settings as symbols
    @test @tasks(for i in 1:3
        @set scheduler = :static
        i
    end) |> isnothing
    @test @tasks(for i in 1:3
        @set scheduler = :dynamic
        i
    end) |> isnothing
    @test @tasks(for i in 1:3
        @set scheduler = :greedy
        i
    end) |> isnothing

    # @set begin ... end
    @test @tasks(for i in 1:10
        @set begin
            scheduler = StaticScheduler()
            reducer = (+)
        end
        i
    end) == 55
    # multiple @set
    @test @tasks(for i in 1:10
        @set scheduler = StaticScheduler()
        i
        @set reducer = (+)
    end) == 55
    # @set init
    @test @tasks(for i in 1:10
        @set begin
            reducer = (+)
            init = 0.0
        end
        i
    end) === 55.0
    @test @tasks(for i in 1:10
        @set begin
            reducer = (+)
            init = 0.0 * im
        end
        i
    end) === (55.0 + 0.0im)

    # top-level "kwargs"
    @test @tasks(for i in 1:3
        @set scheduler = :static
        @set ntasks = 1
        i
    end) |> isnothing
    @test @tasks(for i in 1:3
        @set scheduler = :static
        @set nchunks = 2
        i
    end) |> isnothing
    @test @tasks(for i in 1:3
        @set scheduler = :dynamic
        @set chunksize = 2
        i
    end) |> isnothing
    @test @tasks(for i in 1:3
        @set scheduler = :dynamic
        @set chunking = false
        i
    end) |> isnothing
    @test_throws ArgumentError @tasks(for i in 1:3
        @set scheduler = DynamicScheduler()
        @set chunking = false
        i
    end)
    @test_throws MethodError @tasks(for i in 1:3
        @set scheduler = :serial
        @set chunking = false
        i
    end)
    @test_throws MethodError @tasks(for i in 1:3
        @set scheduler = :dynamic
        @set asd = 123
        i
    end)

    # TaskLocalValue
    ntd = 2 * Threads.nthreads()
    ptrs = Vector{Ptr{Nothing}}(undef, ntd)
    tids = Vector{UInt64}(undef, ntd)
    tid() = OhMyThreads.Tools.taskid()
    @test @tasks(for i in 1:ntd
        @local C::Vector{Float64} = rand(3)
        @set scheduler = :static
        ptrs[i] = pointer_from_objref(C)
        tids[i] = tid()
    end) |> isnothing
    # check that different iterations of a task
    # have access to the same C (same pointer)
    for t in unique(tids)
        @test allequal(ptrs[findall(==(t), tids)])
    end
    # TaskLocalValue (another fundamental check)
    @test @tasks(for i in 1:ntd
        @local x::Ref{Int64} = Ref(0)
        @set reducer = (+)
        @set scheduler = :static
        x[] += 1
        x[]
    end) == 1.5 * ntd # if a new x would be allocated per iteration, we'd get ntd here.
    # TaskLocalValue (begin ... end block), inferred TLV type
    @test @inferred (() -> @tasks for i in 1:10
        @local begin
            C = fill(4, 3, 3)
            x = fill(5.0, 3)
        end
        @set reducer = (+)
        sum(C * x)
    end)() == 1800

    # hygiene / escaping
    var = 3
    sched = StaticScheduler()
    sched_sym = :static
    data = rand(10)
    red = (a, b) -> a + b
    n = 2
    @test @tasks(for d in data
        @set scheduler = sched
        @set reducer = red
        var * d
    end) ≈ var * sum(data)
    @test @tasks(for d in data
        @set scheduler = sched_sym
        @set ntasks = n
        @set reducer = red
        var * d
    end) ≈ var * sum(data)

    struct SingleInt
        x::Int
    end
    @test @tasks(for _ in 1:10
        @local C = SingleInt(var)
        @set reducer = +
        C.x
    end) == 10 * var

    # enumerate(chunks)
    data = collect(1:100)
    @test @tasks(for (i, idcs) in enumerate(OhMyThreads.index_chunks(data; n=5))
        @set reducer = +
        @set chunking = false
        [i, sum(@view(data[idcs]))]
    end) == [sum(1:5), sum(data)]
    @test @tasks(for (i, idcs) in enumerate(OhMyThreads.index_chunks(data; size=5))
        @set reducer = +
        [i, sum(@view(data[idcs]))]
    end) == [sum(1:20), sum(data)]
    @test @tasks(for (i, idcs) in enumerate(OhMyThreads.index_chunks(1:100; n=5))
        @set chunking=false
        @set collect=true
        [i, idcs]
    end) == [[1, 1:20], [2, 21:40], [3, 41:60], [4, 61:80], [5, 81:100]]
end;

@testset "WithTaskLocals" begin
    let x = TaskLocalValue{Base.RefValue{Int}}(() -> Ref{Int}(0)),
        y = TaskLocalValue{Base.RefValue{Int}}(() -> Ref{Int}(0))
        # Equivalent to
        # function f()
        #    x[][] += 1
        #    x[][] += 1
        #    x[], y[]
        # end
        f = WithTaskLocals((x, y)) do (x, y)
            function ()
                x[] += 1
                y[] += 1
                x[], y[]
            end
        end
        # Make sure we can call `f` like a regular function
        @test f() == (1, 1)
        @test f() == (2, 2)
        @test @fetch(f()) == (1, 1)
        # Acceptable use of promise_task_local
        @test @fetch(promise_task_local(f)()) == (1, 1)
        # Acceptable use of promise_task_local
        @test promise_task_local(f)() == (3, 3)
        # Acceptable use of promise_task_local
        @test @fetch(promise_task_local(f)()) == (1, 1)
        # Acceptable use of promise_task_local
        g() = @fetch((promise_task_local(f)(); promise_task_local(f)(); f()))
        @test g() == (3, 3)
        @test g() == (3, 3)

        h = promise_task_local(f)
        # Unacceptable use of `promise_task_local`
        # This is essentially testing that if you use `promise_task_local`, then pass that to another task,
        # you could get data races, since we here have a different thread writing to another thread's value.
        @test @fetch(h()) == (4, 4)
        @test @fetch(h()) == (5, 5)
    end
end;

@testset "chunking mode + chunksize option" begin
    @test OhMyThreads.Schedulers.chunking_mode(SerialScheduler()) ==
          OhMyThreads.Schedulers.NoChunking
    for sched in (DynamicScheduler, StaticScheduler, GreedyScheduler)
        @test sched() isa sched
        @test sched(; chunksize = 2) isa sched

        @test OhMyThreads.Schedulers.chunking_mode(sched(; chunksize = 2)) ==
              OhMyThreads.Schedulers.FixedSize
        @test OhMyThreads.Schedulers.chunking_mode(sched(; nchunks = 2)) ==
              OhMyThreads.Schedulers.FixedCount
        @test OhMyThreads.Schedulers.chunking_mode(sched(; chunking = false)) ==
              OhMyThreads.Schedulers.NoChunking
        if sched != GreedyScheduler
            # For (Dynamic|Static)Scheduler `chunking = false` disables all chunking
            # arguments
            @test OhMyThreads.Schedulers.chunking_mode(sched(;
                nchunks = 2, chunksize = 4, chunking = false)) ==
                  OhMyThreads.Schedulers.NoChunking
            @test OhMyThreads.Schedulers.chunking_mode(sched(;
                nchunks = -2, chunksize = -4, split = :whatever, chunking = false)) ==
                  OhMyThreads.Schedulers.NoChunking
            @test OhMyThreads.Schedulers.chunking_enabled(sched(;
                nchunks = -2, chunksize = -4, chunking = false)) == false
            @test OhMyThreads.Schedulers.chunking_enabled(sched(;
                nchunks = 2, chunksize = 4, chunking = false)) == false
        else
            # For GreedyScheduler `nchunks` or `chunksize` overrides `chunking = false`
            @test OhMyThreads.Schedulers.chunking_mode(sched(;
                nchunks = 2, chunking = false)) ==
                  OhMyThreads.Schedulers.FixedCount
            @test OhMyThreads.Schedulers.chunking_mode(sched(;
                chunksize = 2, chunking = false)) ==
                  OhMyThreads.Schedulers.FixedSize
            @test OhMyThreads.Schedulers.chunking_enabled(sched(;
                nchunks = 2, chunking = false)) == true
            @test OhMyThreads.Schedulers.chunking_enabled(sched(;
                chunksize = 4, chunking = false)) == true
        end
        @test OhMyThreads.Schedulers.chunking_enabled(sched(; chunksize = 2)) == true
        @test OhMyThreads.Schedulers.chunking_enabled(sched(; nchunks = 2)) == true
        @test_throws ArgumentError sched(; nchunks = 2, chunksize = 3)
        @test_throws ArgumentError sched(; nchunks = 0, chunksize = 0)
        @test_throws ArgumentError sched(; nchunks = -2, chunksize = -3)
        @test_throws ArgumentError sched(; nchunks = 2, split = :whatever)

        let scheduler = sched(; chunksize = 2, split = :batch)
            @test tmapreduce(sin, +, 1:10; scheduler) ≈ mapreduce(sin, +, 1:10)
            @test tmap(sin, Float64, 1:10; scheduler) ≈ map(sin, 1:10)
            @test isnothing(tforeach(sin, 1:10; scheduler))
            @test treduce(+, 1:10; scheduler) ≈ reduce(+, 1:10)
        end
    end
end;

@testset "top-level kwargs" begin
    res_tmr = mapreduce(sin, +, 1:10000)

    # scheduler not given
    @test tmapreduce(sin, +, 1:10000; ntasks = 2) ≈ res_tmr
    @test tmapreduce(sin, +, 1:10000; nchunks = 2) ≈ res_tmr
    @test tmapreduce(sin, +, 1:10000; split = RoundRobin()) ≈ res_tmr
    @test tmapreduce(sin, +, 1:10000; chunksize = 2) ≈ res_tmr
    @test tmapreduce(sin, +, 1:10000; chunking = false) ≈ res_tmr

    # scheduler isa Scheduler
    @test tmapreduce(sin, +, 1:10000; scheduler = StaticScheduler()) ≈ res_tmr
    @test_throws ArgumentError tmapreduce(
        sin, +, 1:10000; ntasks = 2, scheduler = DynamicScheduler())
    @test_throws ArgumentError tmapreduce(
        sin, +, 1:10000; chunksize = 2, scheduler = DynamicScheduler())
    @test_throws ArgumentError tmapreduce(
        sin, +, 1:10000; split = RoundRobin(), scheduler = StaticScheduler())
    @test_throws ArgumentError tmapreduce(
        sin, +, 1:10000; ntasks = 3, scheduler = SerialScheduler())

    # scheduler isa Symbol
    for s in (:dynamic, :static, :serial, :greedy)
        @test tmapreduce(sin, +, 1:10000; scheduler = s, init = 0.0) ≈ res_tmr
    end
    for s in (:dynamic, :static, :greedy)
        @test tmapreduce(sin, +, 1:10000; ntasks = 2, scheduler = s, init = 0.0) ≈ res_tmr
    end
    for s in (:dynamic, :static)
        @test tmapreduce(sin, +, 1:10000; chunksize = 2, scheduler = s) ≈ res_tmr
        @test tmapreduce(sin, +, 1:10000; chunking = false, scheduler = s) ≈ res_tmr
        @test tmapreduce(sin, +, 1:10000; nchunks = 3, scheduler = s) ≈ res_tmr
        @test tmapreduce(sin, +, 1:10000; ntasks = 3, scheduler = s) ≈ res_tmr
        @test_throws ArgumentError tmapreduce(
            sin, +, 1:10000; ntasks = 3, nchunks = 2, scheduler = s)≈res_tmr
    end
    @test_throws ArgumentError tmapreduce(sin, +, 1:10000; scheduler = :whatever)
    @test_throws ArgumentError tmapreduce(
        sin, +, 1:10000; threadpool = :whatever, chunking = false)
end;

@testset "empty collections" begin
    @static if VERSION < v"1.11.0-"
        err = MethodError
    else
        err = ArgumentError
    end
    for empty_coll in (11:9, Float64[])
        for f in (sin, x -> im * x, identity)
            for op in (+, *, min)
                # mapreduce
                for init in (0.0, 0, 0.0 * im, 0.0f0)
                    @test tmapreduce(f, op, empty_coll; init) == init
                end
                # foreach
                @test tforeach(f, empty_coll) |> isnothing
                # reduce
                if op != min
                    @test treduce(op, empty_coll) == reduce(op, empty_coll)
                else
                    @test_throws err treduce(op, empty_coll)
                end
                # map
                @test tmap(f, empty_coll) == map(f, empty_coll)
                # collect
                @test tcollect(empty_coll) == collect(empty_coll)
            end
        end
    end
end;

# for testing @one_by_one region
mutable struct SingleAccessOnly
    in_use::Bool
    const lck::ReentrantLock
    SingleAccessOnly() = new(false, ReentrantLock())
end
function acquire(f, o::SingleAccessOnly)
    lock(o.lck) do
        o.in_use && throw(ErrorException("Already in use!"))
        o.in_use = true
    end
    try
        f()
    finally
        lock(o.lck) do
            !o.in_use && throw(ErrorException("Conflict!"))
            o.in_use = false
        end
    end
end

@testset "regions" begin
    @testset "@one_by_one" begin
        sao = SingleAccessOnly()
        try
            @tasks for i in 1:10
                @set ntasks = 10
                @one_by_one begin
                    acquire(sao) do
                        sleep(0.01)
                    end
                end
            end
        catch ErrorException
            @test false
        else
            @test true
        end

        # test escaping
        x = 0
        y = 0
        sao = SingleAccessOnly()
        try
            @tasks for i in 1:10
                @set ntasks = 10

                y += 1 # not safe (race condition)
                @one_by_one begin
                    x += 1 # parallel-safe because inside of one_by_one region
                    acquire(sao) do
                        sleep(0.01)
                    end
                end
            end
            @test x == 10
        catch ErrorException
            @test false
        end

        test_f = () -> begin
            x = 0
            y = 0
            @tasks for i in 1:10
                @set ntasks = 10

                y += 1 # not safe (race condition)
                @one_by_one begin
                    x += 1 # parallel-safe because inside of one_by_one region
                    acquire(sao) do
                        sleep(0.01)
                    end
                end
            end
            return x
        end
        @test test_f() == 10
    end

    @testset "@only_one" begin
        x = 0
        y = 0
        try
            @tasks for i in 1:10
                @set ntasks = 10

                y += 1 # not safe (race condition)
                @only_one begin
                    x += 1 # parallel-safe because only a single task will execute this
                end
            end
            @test x == 1 # only a single task should have incremented x
        catch ErrorException
            @test false
        end

        x = 0
        y = 0
        try
            @tasks for i in 1:10
                @set ntasks = 2

                y += 1 # not safe (race condition)
                @only_one begin
                    x += 1 # parallel-safe because only a single task will execute this
                end
            end
            @test x == 5 # a single task should have incremented x 5 times
        catch ErrorException
            @test false
        end

        test_f = () -> begin
            x = 0
            y = 0
            @tasks for i in 1:10
                @set ntasks = 2

                y += 1 # not safe (race condition)
                @only_one begin
                    x += 1 # parallel-safe because only a single task will execute this
                end
            end
            return x
        end
        @test test_f() == 5
    end

    @testset "@only_one + @one_by_one" begin
        x = 0
        y = 0
        try
            @tasks for i in 1:10
                @set ntasks = 10

                @only_one begin
                    x += 1 # parallel-safe
                end

                @one_by_one begin
                    y += 1 # parallel-safe
                end
            end
            @test x == 1 && y == 10
        catch ErrorException
            @test false
        end
    end
end;

@testset "@barrier" begin
    @test (@tasks for i in 1:20
        @set ntasks = 20
        @barrier
    end) |> isnothing

    @test try
        @macroexpand @tasks for i in 1:20
            @barrier
        end
        false
    catch
        true
    end

    @test try
        x = Threads.Atomic{Int64}(0)
        y = Threads.Atomic{Int64}(0)
        @tasks for i in 1:20
            @set ntasks = 20

            Threads.atomic_add!(x, 1)
            @barrier
            if x[] < 20 && y[] > 0 # x hasn't reached 20 yet and y is already > 0
                error("shouldn't happen")
            end
            Threads.atomic_add!(y, 1)
        end
        true
    catch ErrorException
        false
    end

    @test try
        x = Threads.Atomic{Int64}(0)
        y = Threads.Atomic{Int64}(0)
        @tasks for i in 1:20
            @set ntasks = 20

            Threads.atomic_add!(x, 1)
            @barrier
            Threads.atomic_add!(x, 1)
            @barrier
            if x[] < 40 && y[] > 0 # x hasn't reached 20 yet and y is already > 0
                error("shouldn't happen")
            end
            Threads.atomic_add!(y, 1)
        end
        true
    catch ErrorException
        false
    end
end

@testset "verbose special macro usage" begin
    # OhMyThreads.@set
    @test @tasks(for i in 1:3
        OhMyThreads.@set reducer = (+)
        i
    end) == 6
    @test @tasks(for i in 1:3
        OhMyThreads.@set begin
            reducer = (+)
        end
        i
    end) == 6
    # OhMyThreads.@local
    ntd = 2 * Threads.nthreads()
    @test @tasks(for i in 1:ntd
        OhMyThreads.@local x::Ref{Int64} = Ref(0)
        OhMyThreads.@set begin
            reducer = (+)
            scheduler = :static
        end
        x[] += 1
        x[]
    end) == @tasks(for i in 1:ntd
        @local x::Ref{Int64} = Ref(0)
        @set begin
            reducer = (+)
            scheduler = :static
        end
        x[] += 1
        x[]
    end)
    # OhMyThreads.@only_one
    x = 0
    y = 0
    try
        @tasks for i in 1:10
            OhMyThreads.@set ntasks = 10

            y += 1 # not safe (race condition)
            OhMyThreads.@only_one begin
                x += 1 # parallel-safe because only a single task will execute this
            end
        end
        @test x == 1 # only a single task should have incremented x
    catch ErrorException
        @test false
    end
    # OhMyThreads.@one_by_one
    test_f = () -> begin
        sao = SingleAccessOnly()
        x = 0
        y = 0
        @tasks for i in 1:10
            OhMyThreads.@set ntasks = 10

            y += 1 # not safe (race condition)
            OhMyThreads.@one_by_one begin
                x += 1 # parallel-safe because inside of one_by_one region
                acquire(sao) do
                    sleep(0.01)
                end
            end
        end
        return x
    end
    @test test_f() == 10
end

@testset "show schedulers" begin
    nt = Threads.nthreads(:default)

    @test repr("text/plain", DynamicScheduler()) ==
          """
          DynamicScheduler
          ├ Chunking: fixed count ($nt), split :consecutive
          └ Threadpool: default"""

    @test repr(
        "text/plain", DynamicScheduler(; chunking = false, threadpool = :interactive)) ==
          """
          DynamicScheduler
          ├ Chunking: none
          └ Threadpool: interactive"""

    @test repr("text/plain", StaticScheduler()) ==
          """StaticScheduler
          ├ Chunking: fixed count ($nt), split :consecutive
          └ Threadpool: default"""

    @test repr("text/plain", StaticScheduler(; chunksize = 2, split = :scatter)) ==
          """
          StaticScheduler
          ├ Chunking: fixed size (2), split :roundrobin
          └ Threadpool: default"""

    @test repr("text/plain", GreedyScheduler(; chunking = true)) ==
          """
         GreedyScheduler
         ├ Num. tasks: $nt
         ├ Chunking: fixed count ($(10 * nt)), split :roundrobin
         └ Threadpool: default"""
end

# Todo way more testing, and easier tests to deal with
