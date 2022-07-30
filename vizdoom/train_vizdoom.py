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
from vizdoom_lib import vizdoom_settings


def run(args):
    scenario = scenarios[args.scenario_name]

    game = vzd.DoomGame()

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path(
        os.path.join(vzd.scenarios_path, scenario['scenario_filename']))

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map(scenario['map'])

    vizdoom_settings.setup_vizdoom(game)

    game.set_available_buttons(scenario['available_buttons'])
    print("Available buttons:", [b.name for b in game.get_available_buttons()])

    game.set_available_game_variables([
        vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2])
    print("Available game variables:", [v.name for v in game.get_available_game_variables()])

    if scenario['episode_timeout'] is not None:
        game.set_episode_timeout(scenario['episode_timeout'])

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)
    game.set_window_visible(args.visible)
    game.set_living_reward(scenario['living_reward'])
    game.set_mode(vzd.Mode.PLAYER)

    game.init()

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

    episode = 0
    batch_loss = 0.0
    batch_reward = 0.0
    batch_argmax_action_prop = 0.0
    recorded_last_episode = False
    while True:
        record_this_episode = (
            args.record_every is not None and
            args.replay_path_templ is not None and
            episode % args.record_every == 0
        )
        if recorded_last_episode != record_this_episode:
            game.close()
            game.init()
        
        record_filepath = ''
        if record_this_episode:
            print('episode', episode)
            record_filepath = args.replay_path_templ.format(episode=episode)
            print('    recording to ' + record_filepath)

        game.new_episode(record_filepath)

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
            action_probs = F.softmax(action_logits, dim=-1)
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

            if sleep_time > 0:
                sleep(sleep_time)

            if record_this_episode:
                game.send_game_command('stop')
            episode_steps += 1
        recorded_last_episode = record_this_episode

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
        if (episode + 1) % args.accumulate_episodes == 0:
            b = episode // args.accumulate_episodes
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
        if episode % args.save_model_every == 0:
            save_path = args.model_path_templ.format(episode=episode)
            torch.save(model, save_path)
            print(f'saved model to {save_path}')
        episode += 1

    game.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--accumulate-episodes', type=int, default=16,
        help='how many episodes to accumulate gradients over before opt step')
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument(
        '--model-path-templ', type=str,
        default='vizdoom/models/model_{episode}.pt',
        help='can use {episode} which will be replaced by episode')
    parser.add_argument(
        '--save-model-every', type=int, default=1000,
        help='how often to save model, number of episodes')
    parser.add_argument('--log-path', type=str, default='vizdoom/logs/log.txt')
    parser.add_argument('--visible', action='store_true')
    parser.add_argument(
        '--record-every', type=int,
        help='record replay every this many episodes')
    parser.add_argument(
        '--replay-path-templ', type=str,
        help='eg vizdoom/replays_foo{episode}.lmp')
    parser.add_argument(
        '--ent-reg', type=float, default=0.001,
        help='entropy regularization, encourages exploration')
    parser.add_argument(
        '--scenario-name', type=str, choices=scenarios.keys(), help='name of scenario')
    args = parser.parse_args()
    run(args)
