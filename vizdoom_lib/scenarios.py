import vizdoom as vzd


scenarios = {
    'basic': {
        'buttons': [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK],
        'living_reward': -1,
        'scenario_filename': 'basic.wad'
    },
    'defend_the_center': {
        'buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK],
        'living_reward': 0,
        'scenario_filename': 'defend_the_center.wad'
    }
}
