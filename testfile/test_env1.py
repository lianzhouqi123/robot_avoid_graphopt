from developsuit.envs.env1 import Env
from testfile.plot_track import plot_track

# run_mode = "test"
run_mode = "plot_track"

show_mode = "show"
# show_mode = "no_show"
###########################################################
num_agents = 3

env = Env(num_agents, show_mode=show_mode)


if run_mode == "plot_track":
    track, track_board = env.test_env()

    plot_track(env, track, track_board)
