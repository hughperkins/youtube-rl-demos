"""
Try using policy gradients for mountain car, somehow...
"""
from email.charset import QP
import gym
import argparse
import itertools
import random
import torch
import json
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque


class QPredictor(nn.Module):
    def __init__(self, state_size: int, num_actions: int):
        super().__init__()
        self.h1 = nn.Linear(state_size, 16)
        self.h2 = nn.Linear(16, 16)
        self.h3 = nn.Linear(16, num_actions)

    def forward(self, state):
        # print('forward state.size()', state.size())
        x = self.h1(state)
        x = torch.tanh(x)
        x = self.h2(x)
        x = torch.tanh(x)
        x = self.h3(x)
        # print('forward x.size()', x.size())
        return x


def save_float_image(filepath: str, tensor: torch.Tensor):
    # print('tensor.size()', tensor.size(), tensor.dtype)
    tensor = (tensor * 255).byte()
    # tensor = tensor.transpose(0, 1)
    print('tensor.size()', tensor.size(), tensor.dtype)
    im = Image.fromarray(tensor.numpy())
    im.save(filepath)


def run(args):
    env = gym.make('MountainCar-v0', new_step_api=True)
    target_q_predictor = QPredictor(2, 3)
    main_q_predictor = QPredictor(2, 3)
    opt = optim.RMSprop(lr=args.lr, params=main_q_predictor.parameters())
    target_q_predictor.load_state_dict(main_q_predictor.state_dict())
    global_step = 0
    f_logfile = open(args.logfile, 'w')
    batch_reward_sum = 0.0
    batch_loss_sum = 0.0
    batch_actions_sums = [0] * 3
    replay_buffer = deque()

    def train_batch():
        if args.replay_samples >= len(replay_buffer):
            return 0.0
        samples = random.sample(replay_buffer, args.replay_samples)
        batch_loss = 0.0
        for sample in samples:
            # print('sample', sample)
            state_t, state_t1, action_t, reward_t, done = map(sample.__getitem__, [
                'state', 'new_state', 'action', 'reward', 'done'
            ])
            pred_qv = main_q_predictor(state_t)
            if done:
                target_q_t = reward_t
            else:
                with torch.no_grad():
                    target_qv_t1 = target_q_predictor(state_t1)
                    _, target_a_t1 = target_qv_t1.max(dim=-1)
                    target_q_t = args.reward_decay * target_qv_t1[0, target_a_t1] + reward_t
            loss = ((target_q_t - pred_qv[0, action_t]) * (target_q_t - pred_qv[0, action_t])).sqrt()
            batch_loss += loss.item()
            loss.backward()
        opt.step()
        opt.zero_grad()
        return batch_loss

    for episode in itertools.count():
        state = env.reset()
        print('episode', episode)
        render = (episode % args.render_every) == 0
        while True:
            # print('state', state)
            state_t = torch.from_numpy(state).unsqueeze(0)
            # print('state_t.size()', state_t.size())
            pred_qv = main_q_predictor(state_t)
            if random.random() < args.eps:
                action_t = random.randint(0, 2)
            else:
                _, action_t = pred_qv.max(dim=-1)
                action_t = action_t.item()
            batch_actions_sums[action_t] += 1
            res = env.step(action_t)
            if render:
                env.render()
            # print('len', len(res), res)
            new_state, reward_t, done, truncated, info = res
            new_state_t = torch.from_numpy(new_state).unsqueeze(0)
            # print('new_state_t.size()', new_state_t.size())
            batch_reward_sum += reward_t

            replay_buffer.append({
                'state': state_t,
                'action': action_t,
                'new_state': new_state_t,
                'reward': reward_t,
                'done': done
            })
            if len(replay_buffer) > args.replay_buffer_size:
                replay_buffer.popleft()

            loss = train_batch()

            with torch.no_grad():
                target_qv_t1 = target_q_predictor(new_state_t)
            _, target_a_t1 = target_qv_t1.max(dim=-1)
            target_q_t = args.reward_decay * target_qv_t1[0, target_a_t1] + reward_t

            # print('pred_av', pred_qv)
            # loss = target_q_t * pred_qv[0, action_t]
            # loss = ((target_q_t - pred_qv[0, action_t]) * (target_q_t - pred_qv[0, action_t])).sqrt()
            batch_loss_sum += loss
            # print('loss', loss)
            # loss.backward()
            if ((global_step + 1) % args.accum_grad) == 0:
                b = global_step // args.accum_grad
                # print('batch', b, 'reward %.3f' % (batch_reward_sum / args.accum_grad))
                # opt.step()
                # opt.zero_grad()
                if ((b + 1) % args.copy_weights_every_batch) == 0:
                    print('copying weights to target')
                    target_q_predictor.load_state_dict(main_q_predictor.state_dict())
                if (b % args.draw_q_every_batches) == 0:
                    # lets plot the predicted action for now
                    actions = torch.zeros((11, 11, 3))
                    # we can also plot the maximum q value at each state
                    values = torch.zeros((11, 11))
                    for xi, x in enumerate(torch.arange(-1.0, 1.2, 0.2)):
                        for vi, v in enumerate(torch.arange(-1.0, 1.2, 0.2)):
                            with torch.no_grad():
                                q = target_q_predictor(torch.tensor([x, v]))
                            _value, _action = q.max(dim=-1)
                            _action = _action.item()
                            actions[xi, vi, _action] = 1.0
                            values[xi, vi] = _value
                    # print('actions', actions)
                    print(values)
                    min_v = values.min().item()
                    max_v = values.max().item()
                    save_float_image(args.q_graphs_dir + '/actions.png', actions)
                    save_float_image(args.q_graphs_dir + '/values.png', (values - min_v) / (max_v - min_v))
                    plt.cla()
                    plt.plot(values[10])
                    plt.savefig(args.q_graphs_dir + '/q_v.png')
                    plt.cla()
                    plt.plot(values[:, 10])
                    plt.savefig(args.q_graphs_dir + '/q_x.png')
                    print('plotted to ', args.q_graphs_dir)
                f_logfile.write(json.dumps({
                    'batch': b,
                    'reward': batch_reward_sum / args.accum_grad,
                    'loss': batch_loss_sum / args.accum_grad
                }) + '\n')
                f_logfile.flush()
                # print('batch_actions_sums', batch_actions_sums)
                batch_actions_sums = [0] * 3
                batch_reward_sum = 0.0
                batch_loss_sum = 0.0

            # print('reward', reward)
            state = new_state
            global_step += 1
            if done:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--render-every', type=int, default=32)
    parser.add_argument(
        '--copy-weights-every-batch', type=int, default=32,
        help='how often to copy weightsfrom main to target')
    parser.add_argument('--logfile', type=str, required=True)
    parser.add_argument('--replay-buffer-size', type=int, default=10000)
    parser.add_argument('--replay-samples', type=int, default=16)
    parser.add_argument('--reward-decay', type=float, default=0.95)
    parser.add_argument('--draw-q-every-batches', type=int, default=256)
    parser.add_argument('--q-graphs-dir', type=str, default='tmp/q')
    # parser.add_argument('--q-values-imagefile', type=str, default='tmp/q_values.png')
    parser.add_argument('--accum-grad', type=int, default=1, help='Accumulate gradients over how many steps')
    parser.add_argument(
        '--eps', type=float, default=0.05, help='probability of random exploration action')
    args = parser.parse_args()
    run(args)
