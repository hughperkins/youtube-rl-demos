import vizdoom as vzd


scenarios = {
    'basic': {
        'buttons': [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK],
        'living_reward': -1,
        'scenario_filename': 'basic.wad',
        'episode_timeout': 200,
        'reward_scaling': 0.01,
        'reward_baseline': 0.0
    },
    'defend_the_center': {
        'buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK],
        'living_reward': 0,
        'scenario_filename': 'defend_the_center.wad',
        'episode_timeout': None,
        'reward_scaling': 1.0,
        'reward_baseline': 2.0
    }
}
