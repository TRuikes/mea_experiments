import colormaps as cmaps


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


if __name__ == '__main__':
    p = ProjectColors()
    p.laser_level(50)