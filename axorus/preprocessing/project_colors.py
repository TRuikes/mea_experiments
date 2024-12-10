import colormaps as cmaps
import numpy as np


class ProjectColors:
    def __init__(self):
        return

    @staticmethod
    def laser_level(laser_level, alpha=1):

        min_laser_level = 50
        max_laser_level = 100

        laser_level = int((laser_level - min_laser_level) / (max_laser_level - min_laser_level) * 100)

        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[laser_level, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def n_turns(n_turns, alpha=1):

        min_n_turns = 25
        max_n_turns = 36

        laser_level = int((n_turns - min_n_turns) / (max_n_turns - min_n_turns) * 100)

        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[laser_level, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def train_period(train_period, alpha=1):
        t_periods = [1000, 500, 250, 100, 50, 25]
        n_periods = len(t_periods)
        i = np.where(t_periods == train_period)[0][0]

        r, g, b = cmaps.lavender.discrete(n_periods).colors[i, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def random_color(clr_i):
        r, g, b = cmaps.sinebow_dark.discrete(10).colors[clr_i, :]
        return f'rgba({r}, {g}, {b}, 1)'

    @staticmethod
    def duty_cycle(duty_cycle):
        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.torch.cut(0.2, 'left').cut(0.2, 'right').discrete(100).colors[int(duty_cycle), :]
        return f'rgba({r}, {g}, {b}, 1)'

    @staticmethod
    def min_max_map(val, min_val, max_val):
        rel_i = int((val - min_val) / (max_val - min_val) * 100)
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[rel_i, :]
        return f'rgba({r}, {g}, {b}, {1})'


if __name__ == '__main__':
    # p = ProjectColors()
    cmaps.show_cmaps_all()