import cProfile
import pstats
from io import StringIO

from mas import GridWorld, MonteCarloSweep

# Enable profiling
profiler = cProfile.Profile()
profiler.enable()

# =============== Code to profile here
terminals = {(6, 5): -50, (8, 8): 50}
walls = (
    [(1, i) for i in range(2, 7)]
    + [(i, 6) for i in range(2, 6)]
    + [(7, i) for i in range(1, 5)]
)

gw = GridWorld(terminals=terminals, walls=walls, gridsize=(9, 9))

mcs = MonteCarloSweep(gw)
mcs.evaluate(1000)
# ===============

# Disable profiling
profiler.disable()
stream = StringIO()
ps = pstats.Stats(profiler, stream=stream).sort_stats("tottime")
ps.print_stats()

# Write profiling report to file
with open(f"profile_report.txt", "w+") as f:
    f.write(stream.getvalue())
