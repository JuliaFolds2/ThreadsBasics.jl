# # Parallel Monte Carlo
#
# Calculate the value of $\pi$ through parallel direct Monte Carlo.
#
# A unit circle is inscribed inside a unit square with side length 2 (from -1 to 1).
# The area of the circle is $\pi$, the area of the square is 4, and the ratio is $\pi/4$.
# This means that, if you throw $N$ darts randomly at the square, approximately $M=N\pi/4$
# of those darts will land inside the unit circle.
#
# Throw darts randomly at a unit square and count how many of them ($M$) landed inside of
# a unit circle. Approximate $\pi \approx 4M/N$.
#
# ## Sequential implementation:

function mc(N)
    M = 0 # number of darts that landed in the circle
    for i in 1:N
        if rand()^2 + rand()^2 < 1.0
            M += 1
        end
    end
    pi = 4 * M / N
    return pi
end

N = 100_000_000

mc(N)

# ## Parallelization with `tmapreduce`
#
# To parallelize the Monte Carlo simulation, we use [`tmapreduce`](@ref) with `+` as the reduction
# operator. For the map part, we take `1:N` as our input collection and "throw one dart" per
# element.

using OhMyThreads

function mc_parallel(N; kwargs...)
    M = tmapreduce(+, 1:N; kwargs...) do i
        rand()^2 + rand()^2 < 1.0
    end
    pi = 4 * M / N
    return pi
end

## or alternatively
##
## function mc_parallel(N)
##     M = @tasks for _ in 1:N
##         @set reducer = +
##         rand()^2 + rand()^2 < 1.0
##     end
##     pi = 4 * M / N
##     return pi
## end

mc_parallel(N)

# Let's run a quick benchmark.

using BenchmarkTools
using Base.Threads: nthreads

@assert nthreads() > 1 # make sure we have multiple Julia threads
@show nthreads()       # print out the number of threads

@btime mc($N) samples=10 evals=3;
@btime mc_parallel($N) samples=10 evals=3;

# ### Static scheduling
#
# Because the workload is highly uniform, it makes sense to also try the `StaticScheduler`
# and compare the performance of static and dynamic scheduling (with default parameters).

using OhMyThreads: StaticScheduler

@btime mc_parallel($N; scheduler=:dynamic) samples=10 evals=3; # default
@btime mc_parallel($N; scheduler=:static) samples=10 evals=3;

# ## Manual parallelization
#
# First, using the `chunks` function, we divide the iteration interval `1:N` into
# `nthreads()` parts. Then, we apply a regular (sequential) `map` to spawn a Julia task
# per chunk. Each task will locally and independently perform a sequential Monte Carlo
# simulation. Finally, we fetch the results and compute the average estimate for $\pi$.

using OhMyThreads: @spawn, chunks

function mc_parallel_manual(N; nchunks = nthreads())
    tasks = map(chunks(1:N; n = nchunks)) do chunk
        @spawn mc(length(chunk))
    end
    pi = sum(fetch, tasks) / nchunks
    return pi
end

mc_parallel_manual(N)

# And this is the performance:

@btime mc_parallel_manual($N) samples=10 evals=3;

# It is faster than `mc_parallel` above because the task-local computation
# `mc(length(chunk))` is faster than the implicit task-local computation within
# `tmapreduce` (which itself is a `mapreduce`).

chunk = first(chunks(1:N; n = nthreads()))

@btime mapreduce($+, $chunk) do i
    rand()^2 + rand()^2 < 1.0
end samples=10 evals=3;

@btime mc($(length(chunk))) samples=10 evals=3;
