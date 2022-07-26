#!/usr/bin/env python3

"""
This is used to play some episodes for various models
stored during training
"""

import os
import time
import argparse
from random import choice
import vizdoom as vzd
from vizdoom_lib import vizdoom_settings
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import run_model_lib


def run(args):
    model_runner = run_model_lib.ModelRunner(
        scenario_name=args.scenario_name,
        relative_speed=args.relative_speed
    )
    if not args.sound:
        model_runner.game.close()
        model_runner.game.set_sound_enabled(False)
        model_runner.game.init()
    episode = args.start_episode
    while args.end_episode is None or episode <= args.end_episode:
        model_path = args.model_path_templ.format(
            episode=episode)
        model_runner.load_model(model_path)
        if episode == args.start_episode and args.warmup_games is not None:
            for i in range(args.warmup_games):
                print('warmup', i)
                model_runner.run_episode()
            print('DONE WARMUP')
            print('')
        print('episode', episode, flush=True, end='')
        if args.training_cost_per_episode_dollars is not None:
            total_training_cost = episode * args.training_cost_per_episode_dollars
            print(' cost $%.2f' % total_training_cost, end='')
        if args.training_time_per_episode_hours is not None:
            total_training_time = episode * args.training_time_per_episode_hours
            print(' time %.1f hours' % total_training_time, end='')
        print('')
        for i in range(args.num_games_per_model):
            res = model_runner.run_episode()
            print('     DEAD')
            print('     reward: %.1f' % res['reward'], 'steps: %.0f' % res['steps'])
            time.sleep(1)
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
    parser.add_argument('--sound', action='store_true')
    parser.add_argument(
        '--warmup-games', type=int,
        help='run some games without incrementing epsiode. Used to resize window')
    parser.add_argument('--episode-stride', type=int, default=1000)
    parser.add_argument('--relative-speed', type=float, default=4.0)
    parser.add_argument(
        '--num-games-per-model', type=int, default=1,
        help='how many games to play for each loaded model')
    parser.add_argument('--training-cost-per-episode-dollars', type=float)
    parser.add_argument('--training-time-per-episode-hours', type=float)
    args = parser.parse_args()
    assert args.scenario_name in scenarios
    run(args)
