#!/usr/bin/env python3

"""
This is used to replay replays recorded during training.
"""

import os
import argparse
from random import choice
import vizdoom as vzd
from vizdoom_lib import vizdoom_settings
from vizdoom_lib.scenarios import scenarios


def run(args):
    scenario = scenarios[args.scenario_name]
    game = vzd.DoomGame()
    game.set_doom_scenario_path(
        os.path.join(vzd.scenarios_path, scenario['scenario_filename']))
    # game.load_config(
    #     os.path.join(vzd.scenarios_path, scenario['scenario_filename']))
    game.set_doom_map("map01")

    vizdoom_settings.setup_vizdoom(game)

    game.set_mode(vzd.Mode.PLAYER)
    game.init()

    print('')

    episode = args.start_episode
    while True:
        replay_filepath = args.replay_path_templ.format(
            episode=episode)
        print('\repisode', episode, flush=True, end='')
        game.replay_episode(replay_filepath)

        while not game.is_episode_finished():
            game.advance_action()
        episode += args.episode_stride

    game.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario-name', type=str, help='name of scenario')
    parser.add_argument(
        '--replay-path-templ', type=str,
        required=True, help='eg foo/blah_{episode}.lmp')
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--episode-stride', type=int, default=1000)
    args = parser.parse_args()
    assert args.scenario_name in scenarios
    run(args)
