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
        # Create DoomGame instance. It will run the game and communicate with you.
        game = vzd.DoomGame()

        # Now it's time for configuration!
        # load_config could be used to load configuration instead of doing it here with code.
        # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
        # game.load_config("../../scenarios/basic.cfg")

        # Sets path to additional resources wad file which is basically your scenario wad.
        # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
        scenario = scenarios[scenario_name]
        game.set_doom_scenario_path(os.path.join(
            vzd.scenarios_path, scenario['scenario_filename']))

        # Sets map to start (scenario .wad files can contain many maps).
        game.set_doom_map("map01")

        vizdoom_settings.setup_vizdoom(game)

        # Adds buttons that will be allowed to use.
        # This can be done by adding buttons one by one:
        # game.clear_available_buttons()
        # game.add_available_button(vzd.Button.MOVE_LEFT)
        # game.add_available_button(vzd.Button.MOVE_RIGHT)
        # game.add_available_button(vzd.Button.ATTACK)
        # Or by setting them all at once:
        game.set_available_buttons(scenario['buttons'])
        # Buttons that will be used can be also checked by:
        print("Available buttons:", [b.name for b in game.get_available_buttons()])

        # Adds game variables that will be included in state.
        # Similarly to buttons, they can be added one by one:
        # game.clear_available_game_variables()
        # game.add_available_game_variable(vzd.GameVariable.AMMO2)
        # Or:
        game.set_available_game_variables([vzd.GameVariable.AMMO2])
        print("Available game variables:", [v.name for v in game.get_available_game_variables()])

        if scenario['episode_timeout'] is not None:
            game.set_episode_timeout(scenario['episode_timeout'])

        # Makes episodes start after 10 tics (~after raising the weapon)
        game.set_episode_start_time(10)

        # Makes the window appear (turned on by default)
        game.set_window_visible(True)

        # Turns on the sound. (turned off by default)
        # game.set_sound_enabled(True)
        # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
        # the sound is only useful for humans watching the game.

        # Sets the living reward (for each move) to -1
        game.set_living_reward(scenario['living_reward'])

        # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
        game.set_mode(vzd.Mode.PLAYER)

        # Enables engine output to console, in case of a problem this might provide additional information.
        #game.set_console_enabled(True)

        # Initialize the game. Further configuration won't take any effect from now on.
        game.init()

        # Define some actions. Each list entry corresponds to declared buttons:
        # MOVE_LEFT, MOVE_RIGHT, ATTACK
        # game.get_available_buttons_size() can be used to check the number of available buttons.
        # 5 more combinations are naturally possible but only 3 are included for transparency when watching.
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

            # Gets the state
            state = game.get_state()

            # Which consists of:
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

        # Check how the episode went.
        episode_reward = game.get_total_reward()
        # print('total reward', game.get_total_reward())
        return {'reward': game.get_total_reward(), 'steps': step}

    def close(self):
        self.game.close()
