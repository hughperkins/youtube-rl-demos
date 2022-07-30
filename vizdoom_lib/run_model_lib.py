import os
from time import sleep
from numpy import int8
import vizdoom as vzd
import torch
from typing import Optional
from torch import distributions, int64
import torch.nn.functional as F
from vizdoom_lib.scenarios import scenarios
from vizdoom_lib import vizdoom_settings


class ModelRunner:
    def __init__(self, scenario_name: str, relative_speed: float = 1.0):
        game = vzd.DoomGame()

        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        scenario = scenarios[scenario_name]
        game.set_doom_scenario_path(os.path.join(
            vzd.scenarios_path, scenario['scenario_filename']))

        # Sets map to start (scenario .wad files can contain many maps).
        game.set_doom_map(scenario['map'])
        game.set_sound_enabled(True)

        vizdoom_settings.setup_vizdoom(game)

        game.set_available_buttons(scenario['available_buttons'])
        # Buttons that will be used can be also checked by:
        print("Available buttons:", [b.name for b in game.get_available_buttons()])

        game.set_available_game_variables([vzd.GameVariable.AMMO2])
        print("Available game variables:", [v.name for v in game.get_available_game_variables()])

        if scenario['episode_timeout'] is not None:
            game.set_episode_timeout(scenario['episode_timeout'])

        # Makes episodes start after 10 tics (~after raising the weapon)
        game.set_episode_start_time(10)

        game.set_window_visible(True)
        game.set_living_reward(scenario['living_reward'])
        game.set_mode(vzd.Mode.PLAYER)

        game.init()

        self.actions = [
            [True, False, False],
            [False, True, False],
            [False, False, True]
        ]

        # Sets time that will pause the engine after each action (in seconds)
        # Without this everything would go too fast for you to keep track of what's happening.
        self.sleep_time = 1.0 / vzd.DEFAULT_TICRATE / relative_speed  # = 0.028
        print('sleep time %.4f' % self.sleep_time)
        # self.sleep_time = 0.0

        self.game = game

    def load_model(self, model_path: str):
        self.model = torch.load(model_path)
    
    def run_episode(self):
        game = self.game
        game.new_episode()

        action_log_probs = []
        step = 0
        while not game.is_episode_finished():
            state = game.get_state()

            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels
            objects = state.objects
            sectors = state.sectors

            screen_buf_t = torch.from_numpy(screen_buf) / 255
            # [H][W][C]
            screen_buf_t = screen_buf_t.transpose(1, 2)
            screen_buf_t = screen_buf_t.transpose(0, 1)
            # [C][H][W]
            screen_buf_t = screen_buf_t.unsqueeze(0)
            # [N][C][H][W]
            action_logits = self.model(screen_buf_t)
            action_probs = F.softmax(action_logits, dim=-1)
            m = distributions.Categorical(action_probs)
            action = m.sample()
            action = action.item()

            # Makes an action (here random one) and returns a reward.
            r = game.make_action(self.actions[action])

            if self.sleep_time > 0:
                sleep(self.sleep_time)
            step += 1

        episode_reward = game.get_total_reward()
        return {'reward': game.get_total_reward(), 'steps': step}

    def close(self):
        self.game.close()
