import vizdoom as vzd


scenarios = {
    'basic': {
        'available_buttons': [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT, vzd.Button.ATTACK],
        'map': 'map01',
        'living_reward': -1,
        'scenario_filename': 'basic.wad',
        'episode_timeout': 200,
        'reward_scaling': 0.01,
        'reward_baseline': 0.0,
    },
    'defend_the_center': {
        'map': 'map01',
        'available_buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK],
        'living_reward': 0,
        'scenario_filename': 'defend_the_center.wad',
        'episode_timeout': None,
        'reward_scaling': 1.0,
        'reward_baseline': 2.0
    },
    'deadly_corridor': {
        'scenario_filename': 'deadly_corridor.wad',
        'map': 'map01',
        'living_reward': 0,
        'doom_skill': 5,
        'episode_timeout': 4200,
        'available_buttons': [
            vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK,
            vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT],
        'keys': [pygame.K_h, pygame.K_l, pygame.K_SPACE,
            pygame.K_a, pygame.K_f],
    },
    'defend_the_line': {
        'scenario_filename': 'defend_the_line.wad',
        'map': 'map01',
        'living_reward': 0,
        'episode_timeout': None,
        'available_buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK],
        'keys': [pygame.K_h, pygame.K_l, pygame.K_SPACE],
    },
    'health_gathering': {
        'scenario_filename': 'health_gathering.wad',
        'map': 'map01',
        'living_reward': 1,
        'episode_timeout': None,
        'available_buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD],
        'keys': [pygame.K_h, pygame.K_l, pygame.K_SPACE],
    },
    'my_way_home': {
        'scenario_filename': 'my_way_home.wad',
        'map': 'map01',
        'living_reward': -0.0001,
        'episode_timeout': 2100,
        'available_buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.MOVE_FORWARD],
        'keys': [pygame.K_h, pygame.K_l, pygame.K_SPACE],
    },
    'predict_position': {
        'scenario_filename': 'predict_position.wad',
        'map': 'map01',
        'living_reward': -0.0001,
        'episode_timeout': 300,
        'available_buttons': [vzd.Button.TURN_LEFT, vzd.Button.TURN_RIGHT, vzd.Button.ATTACK],
        'keys': [pygame.K_h, pygame.K_l, pygame.K_SPACE],
    },
    'take_cover': {
        'scenario_filename': 'take_cover.wad',
        'map': 'map01',
        'living_reward': 1,
        'episode_timeout': None,
        'available_buttons': [vzd.Button.MOVE_LEFT, vzd.Button.MOVE_RIGHT],
        'keys': [pygame.K_a, pygame.K_f],
    },
}
