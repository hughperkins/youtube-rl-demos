#!/usr/bin/env python3
import os
from random import choice
from time import sleep
import vizdoom as vzd
import torch
import json
from torch import nn, optim, distributions
from vizdoom_lib.model import Net
import torch.nn.functional as F
import argparse
from vizdoom_lib.scenarios import scenarios


def run(args):
    scenario = scenarios[args.scenario_name]

    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path(
        os.path.join(vzd.scenarios_path, scenario['scenario_filename']))

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
    # game.set_available_buttons([vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK])
    game.set_available_buttons(scenario['buttons'])
    # Buttons that will be used can be also checked by:
    print("Available buttons:", [b.name for b in game.get_available_buttons()])

    # Adds game variables that will be included in state.
    # Similarly to buttons, they can be added one by one:
    # game.clear_available_game_variables()
    # game.add_available_game_variable(vzd.GameVariable.AMMO2)
    # Or:
    # game.set_available_game_variables([vzd.GameVariable.AMMO2])
    game.set_available_game_variables([
        vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    # Causes episodes to finish after 200 tics (actions)
    if scenario['episode_timeout'] is not None:
        game.set_episode_timeout(scenario['episode_timeout'])

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    game.set_window_visible(args.visible)

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
    actions = [
        [True, False, False],
        [False, True, False],
        [False, False, True]
    ]

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    # sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028
    sleep_time = 0.0

    model = Net(image_height=120, image_width=160, num_actions=len(actions))
    opt = optim.RMSprop(lr=args.lr, params=model.parameters())
    out_f = open(args.log_path, 'w')

    i = 0
    batch_loss = 0.0
    batch_reward = 0.0
    batch_argmax_action_prop = 0.0
    while True:

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()

        action_log_probs = []
        last_var_values_str = ''
        if args.visible:
            print('=== new episode === ')
        episode_entropy = 0.0
        episode_steps = 0
        episode_argmax_action_taken = 0
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

            var_values_str = ' '.join([
                f'{v:.3f}' for v in vars])

            if var_values_str != last_var_values_str:
                if args.visible:
                    print(var_values_str)
                last_var_values_str = var_values_str
            # print('vars', vars, type(vars))
            # health = vars[0].item()
            # if health != last_health:
            #     print('health', health)
            #     last_health = health
            # health = varsvzd.GameVariable.HEALTH

            screen_buf_t = torch.from_numpy(screen_buf) / 255
            # [H][W][C]
            screen_buf_t = screen_buf_t.transpose(1, 2)
            screen_buf_t = screen_buf_t.transpose(0, 1)
            # [C][H][W]
            screen_buf_t = screen_buf_t.unsqueeze(0)
            # [N][C][H][W]
            action_logits = model(screen_buf_t)
            action_probs = F.softmax(action_logits)
            # 0.0 0.0 1.0 => low entropy
            # argmax=2
            #         2
            # % of time the sample matches argmax = 100%
            # 0.1 0.1 0.8 => higher entropy
            #   0  1   2
            # % of time the sample matches argmax = 80%
            action_log_probs_product = action_probs * action_probs.log()
            step_entropy = (- action_log_probs_product).sum(1).sum()

            episode_entropy += step_entropy
            m = distributions.Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            action_log_probs.append(log_prob)
            action = action.item()
            _, argmax_action = action_probs.max(dim=-1)
            argmax_action = argmax_action.item()
            if argmax_action == action:
                episode_argmax_action_taken += 1

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

            episode_steps += 1

        reward_scaling = scenario['reward_scaling']
        reward_baseline = scenario['reward_baseline']
        episode_reward = (
            game.get_total_reward() * reward_scaling - reward_baseline)
        if args.visible:
            print('episode reward', game.get_total_reward())

        per_timestep_losses = [- log_prob * episode_reward for log_prob in action_log_probs]
        per_timestep_losses_t = torch.stack(per_timestep_losses)
        reward_loss = per_timestep_losses_t.sum()
        entropy_loss = - args.ent_reg * episode_entropy
        episode_argmax_action_prop = episode_argmax_action_taken / episode_steps
        total_loss = reward_loss + entropy_loss
        total_loss.backward()
        batch_loss += total_loss.item()
        batch_reward += game.get_total_reward()
        batch_argmax_action_prop += episode_argmax_action_prop
        if (i + 1) % args.accumulate_episodes == 0:
            b = i // args.accumulate_episodes
            batch_avg_reward = batch_reward / args.accumulate_episodes
            batch_avg_loss = batch_loss / args.accumulate_episodes
            batch_avg_argmax_action_prop = batch_argmax_action_prop / args.accumulate_episodes
            print(
                'batch', b, 'reward %.1f' % batch_avg_reward, 'loss %.4f' % batch_avg_loss,
                'argmax_prop %.3f' % batch_avg_argmax_action_prop)
            opt.step()
            opt.zero_grad()
            out_f.write(json.dumps({
                'batch': b,
                'loss': batch_avg_loss,
                'argmax_action_prop': batch_avg_argmax_action_prop,
                'reward': batch_avg_reward
            }) + '\n')
            out_f.flush()
            batch_loss = 0.0
            batch_reward = 0.0
            batch_argmax_action_prop = 0.0
        if i % 512 == 0:
            torch.save(model, args.model_path)
            print('saved model')
        i += 1

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--accumulate-episodes', type=int, default=16,
        help='how many episodes to accumulate gradients over before opt step')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--model-path', type=str, default='vizdoom/models/model.pt')
    parser.add_argument('--log-path', type=str, default='vizdoom/logs/log.txt')
    parser.add_argument('--visible', action='store_true')
    parser.add_argument(
        '--ent-reg', type=float, default=0.001,
        help='entropy regularization, encourages exploration')
    parser.add_argument('--scenario-name', type=str, help='name of scenario')
    args = parser.parse_args()
    assert args.scenario_name in scenarios
    run(args)
