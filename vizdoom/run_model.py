import os
from time import sleep
import vizdoom as vzd
import torch
from torch import nn, distributions
from vizdoom_lib.model import Net
import torch.nn.functional as F
import argparse
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import run_model_lib


def run(args):
    model_runner = run_model_lib.ModelRunner(
        scenario_name=args.scenario_name
    )
    model_runner.load_model(args.in_model_path)
    while True:
        res = model_runner.run_episode()
        print('reward', res['reward'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-model-path', type=str, required=True)
    parser.add_argument('--scenario-name', type=str, default='basic', help='name of scenario')
    args = parser.parse_args()
    assert args.scenario_name in scenarios
    run_model_lib.run(args)
