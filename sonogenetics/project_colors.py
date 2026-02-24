import colormaps as cmaps
import numpy as np


class ProjectColors:
    def __init__(self):
        self._animal_map = cmaps.BlueDarkOrange18.colors
        return

    @staticmethod
    def laser_level(laser_level, alpha=1):

        min_laser_level = 3000
        max_laser_level = 8000

        laser_level_rel = int((laser_level - min_laser_level) / (max_laser_level - min_laser_level) * 100)

        # cmaps.cet_l_bmy.discrete(100).colors
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[laser_level_rel, :]
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
    def duty_cycle(duty_cycle, *, alpha=None, dc_min=None, dc_max=None):
        if dc_min is not None:
            duty_cycle = int((duty_cycle - dc_min) / (dc_max - dc_min) * 100)

        # default opacity if not specified
        if alpha is None:
            alpha = 1.0

        r, g, b = (
            cmaps.torch
            .cut(0.2, 'left')
            .cut(0.2, 'right')
            .discrete(100)
            .colors[int(duty_cycle), :]
        )
        return f'rgba({r}, {g}, {b}, {alpha})'

    # def duty_cycle(duty_cycle):
    #     # cmaps.cet_l_bmy.discrete(100).colors
    #     r, g, b = cmaps.torch.cut(0.2, 'left').cut(0.2, 'right').discrete(100).colors[int(duty_cycle), :]
    #     return f'rgba({r}, {g}, {b}, 1)'

    @staticmethod
    def padmd_stim(duty_cycle, *, alpha=None):
        # default opacity if not specified
        if alpha is None:
            alpha = 1.0

        r, g, b = (
            cmaps.greens
            .cut(0.2, 'left')
            .cut(0.2, 'right')
            .discrete(100)
            .colors[int(duty_cycle), :]
        )
        return f'rgba({r}, {g}, {b}, {alpha})'
    
    @staticmethod
    def repetition_frequency(prf, alpha=1):
        rel_i = int((prf - 1000) / (8000 - 1000) * 100)
        r, g, b = cmaps.torch.cut(0.2, 'left').cut(0.2, 'right').discrete(100).colors[rel_i, :]
        return f'rgba({r}, {g}, {b}, {alpha})'

    @staticmethod
    def min_max_map(val, min_val, max_val):
        rel_i = int((val - min_val) / (max_val - min_val) * 100)
        r, g, b = cmaps.cet_l_bmy.cut(0.1, 'left').cut(0.1, 'right').discrete(100).colors[rel_i, :]
        return f'rgba({r}, {g}, {b}, {1})'

    def animal_color(self, animal, alpha=1, idx=3):
        if animal == 'P23H':
            idx = -idx
        clrraw = self._animal_map[idx, :]
        return f'rgba({clrraw[0]}, {clrraw[1]}, {clrraw[2]}, {alpha})'


    @staticmethod
    def blocker_color(blocker, alpha):
        if blocker == 'noblocker' or blocker == 'none':
            idx = 5
        elif blocker == 'lap4':
            idx = 11
        elif blocker == 'lap4acet':
            idx = 1
        elif blocker == 'washout':
            idx = 3

        clrraw = cmaps.grads_rainbow.colors[idx, :]
        return f'rgba({clrraw[0]}, {clrraw[1]}, {clrraw[2]}, {alpha})'


if __name__ == '__main__':
    # p = ProjectColors()
    # cmaps.show_cmaps_all()

    clrs = ProjectColors()
    x = clrs.blocker_color('noblocker', 1)