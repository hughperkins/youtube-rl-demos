#!/usr/bin/env python3

"""
This is used to play some episodes for various models
stored during training
"""

import os
import argparse
from random import choice
import vizdoom as vzd
from vizdoom_lib import vizdoom_settings
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import run_model_lib


def run(args):
    model_runner = run_model_lib.ModelRunner(
        scenario_name=args.scenario_name
    )
    episode = args.start_episode
    while args.end_episode is None or episode <= args.end_episode:
        model_path = args.model_path_templ.format(
            episode=episode)
        model_runner.load_model(model_path)
        print('\repisode', episode, flush=True, end='')
        for i in range(args.num_games_per_model):
            model_runner.run_episode()
        episode += args.episode_stride


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario-name', type=str, required=True, help='name of scenario')
    parser.add_argument(
        '--model-path-templ', type=str,
        required=True, help='eg foo/blah_{episode}.pt')
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument('--end-episode', type=int)
    parser.add_argument('--episode-stride', type=int, default=1000)
    parser.add_argument(
        '--num-games-per-model', type=int, default=1,
        help='how many games to play for each loaded model')
    args = parser.parse_args()
    assert args.scenario_name in scenarios
    run(args)
