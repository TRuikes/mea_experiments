import numpy as np
from scipy.optimize import curve_fit
import math

def gaussian2D_flat(x, amp, x0, y0, rx, ry, rot):
    return gaussian2D(x, amp, x0, y0, rx, ry, rot).flatten()


def reduced_gaussian2D(x, amp, rx, ry, rot):
    # "Reduced" 2D gaussian used for the first fit stage of double_gaussian_fit: the
    # center is held FIXED (passed inside x) and only amplitude/radii/rotation are fit.
    # x packs [ny, nx, x0, y0] — the spatial STA shape and the peak location.
    ny, nx, x0, y0 = x
    return gaussian2D((ny, nx), amp, x0, y0, rx, ry, rot)


def reduced_gaussian2D_flat(x, amp, rx, ry, rot):
    return reduced_gaussian2D(x, amp, rx, ry, rot).flatten()


def gaussian2D(
    shape,
    amp,
    x0,
    y0,
    sigma_x,
    sigma_y,
    angle,
):
    if sigma_x == 0:
        sigma_x = 0.001

    if sigma_y == 0:
        sigma_y = 0.001
    shape = (int(shape[0]), int(shape[1]))
    x = np.linspace(0, shape[1], shape[1])
    y = np.linspace(0, shape[0], shape[0])
    X, Y = np.meshgrid(x, y)

    theta = 3.14 * angle / 180
    a = (math.cos(theta) ** 2) / (2 * sigma_x**2) + (math.sin(theta) ** 2) / (
        2 * sigma_y**2
    )
    b = -(math.sin(2 * theta)) / (4 * sigma_x**2) + (math.sin(2 * theta)) / (
        4 * sigma_y**2
    )
    c = (math.sin(theta) ** 2) / (2 * sigma_x**2) + (math.cos(theta) ** 2) / (
        2 * sigma_y**2
    )

    return amp * np.exp(
        -(
            a * np.power((X - x0), 2)
            + 2 * b * np.multiply((X - x0), (Y - y0))
            + c * np.power((Y - y0), 2)
        )
    )


def double_gaussian_fit(
    spatial: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center = np.unravel_index(np.argmax(np.abs(spatial), axis=None), spatial.shape)
    ydata = spatial.flatten()

    # First fit without center variability
    first_guess = [spatial[center[0], center[1]], 1, 1, 0]
    xdata = [spatial.shape[0], spatial.shape[1], center[1], center[0]]

    ellipse_params_bounds = (
        (-2, 0.1, 0.1, 0),
        (2, spatial.shape[0], spatial.shape[0], 180),
    )

    opt, cov = curve_fit(
        reduced_gaussian2D_flat,
        xdata,
        ydata,
        p0=first_guess,
        bounds=ellipse_params_bounds,
    )

    # Second fit with center variability
    xdata = spatial.shape
    second_guess = [opt[0], center[1], center[0], opt[1], opt[2], opt[3]]

    ellipse_params_bounds = (
        (-2, 0, 0, 0.1, 0.1, 0),
        (
            2,
            spatial.shape[0],
            spatial.shape[0],
            spatial.shape[0],
            spatial.shape[0],
            180,
        ),
    )
    return curve_fit(
        gaussian2D_flat, xdata, ydata, p0=second_guess, bounds=ellipse_params_bounds
    )




