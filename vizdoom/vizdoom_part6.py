#!/usr/bin/env python3

#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

import os
from random import choice
from time import sleep
import vizdoom as vzd
import torch
import json
from torch import nn, optim, distributions
from vizdoom_lib.model import Net
import torch.nn.functional as F


lr = 0.0001
save_path = 'vizdoom/model.pt'


if __name__ == "__main__":
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path(os.path.join(vzd.scenarios_path, "basic.wad"))

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables depth buffer.
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(True)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed to use.
    # This can be done by adding buttons one by one:
    # game.clear_available_buttons()
    # game.add_available_button(vzd.Button.MOVE_LEFT)
    # game.add_available_button(vzd.Button.MOVE_RIGHT)
    # game.add_available_button(vzd.Button.ATTACK)
    # Or by setting them all at once:
    game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    # Buttons that will be used can be also checked by:
    print("Available buttons:", [b.name for b in game.get_available_buttons()])

    # Adds game variables that will be included in state.
    # Similarly to buttons, they can be added one by one:
    # game.clear_available_game_variables()
    # game.add_available_game_variable(vzd.GameVariable.AMMO2)
    # Or:
    game.set_available_game_variables([vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(200)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    game.set_window_visible(False)

    # Turns on the sound. (turned off by default)
    # game.set_sound_enabled(True)
    # Because of some problems with OpenAL on Ubuntu 20.04, we keep this line commented,
    # the sound is only useful for humans watching the game.

    # Sets the living reward (for each move) to -1
    game.set_living_reward(-1)

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
    actions = [
        [True, False, False],
        [False, True, False],
        [False, False, True]
    ]

    # Run this many episodes
    # episodes = 1000000

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    # sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
    sleep_time = 0.0

    model = Net(image_height=120, image_width=160, num_actions=len(actions))
    opt = optim.RMSprop(lr=lr, params=model.parameters())
    out_f = open('log.txt', 'w')

    i = 0
    while True:

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        action_log_probs = []
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
            action_logits = model(screen_buf_t)
            action_probs = F.softmax(action_logits)
            # print('action_probs', action_probs)
            m = distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            # print('log_prob', log_prob)
            action_log_probs.append(log_prob)
            action = action.item()

            # Games variables can be also accessed via
            # (including the ones that were not added as available to a game state):
            #game.get_game_variable(GameVariable.AMMO2)

            # Makes an action (here random one) and returns a reward.
            r = game.make_action(actions[action])

            # Makes a "prolonged" action and skip frames:
            # skiprate = 4
            # r = game.make_action(choice(actions), skiprate)

            # The same could be achieved with:
            # game.set_action(choice(actions))
            # game.advance_action(skiprate)
            # r = game.get_last_reward()

            if sleep_time > 0:
                sleep(sleep_time)

        # Check how the episode went.
        episode_reward = game.get_total_reward()
        episode_reward = episode_reward / 100
        per_timestep_losses = [- log_prob * episode_reward for log_prob in action_log_probs]
        per_timestep_losses_t = torch.stack(per_timestep_losses)
        # print('per_timestep_losses_t', per_timestep_losses_t)
        loss = per_timestep_losses_t.sum()
        print('episode', i, 'total reward', game.get_total_reward(), 'loss %.4f' % loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()
        out_f.write(json.dumps({
            'episode': i,
            'loss': loss.item(),
            'reward': game.get_total_reward()
        }) + '\n')
        out_f.flush()
        if i % 10000 == 0:
            torch.save(model, save_path)
            print('saved model')
        i += 1

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()