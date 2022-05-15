from typing import Optional

import numpy as np
from matplotlib.axes import Axes


def draw_circle(
    ax: Axes,
    c: np.ndarray,
    r: float,
    n: Optional[np.ndarray] = None,
    color: str = "black",
) -> None:
    """Draw a circle on a specified Axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the circle.
    c : np.ndarray, shape=(3,)
        Center of circle.
    r : float
        Radius of circle.
    n : np.ndarray, shape=(3,)
        Normal vector of plane in which circle lies.
    color : str, default='black'
        Color of the circle.
    """
    if ax.name == "3d":
        assert c.shape == (3,)
        assert n is not None and n.shape == (3,)

        # parameterize circle by planar spanning vectors
        a = np.random.rand(3)
        a = np.cross(n, a)
        b = np.cross(n, a)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)

        # draw the circle
        thetas = np.linspace(0, 2 * np.pi, 361)
        xs = c[0] + r * np.cos(thetas) * a[0] + r * np.sin(thetas) * b[0]
        ys = c[1] + r * np.cos(thetas) * a[1] + r * np.sin(thetas) * b[1]
        zs = c[2] + r * np.cos(thetas) * a[2] + r * np.sin(thetas) * b[2]

        ax.plot(xs, ys, zs, color=color)

    else:
        assert c.shape == (2,)
        assert n is None

        a = np.array([1, 0])
        b = np.array([0, 1])

        # draw the circle
        thetas = np.linspace(0, 2 * np.pi, 361)
        xs = c[0] + r * np.cos(thetas) * a[0] + r * np.sin(thetas) * b[0]
        ys = c[1] + r * np.cos(thetas) * a[1] + r * np.sin(thetas) * b[1]

        ax.plot(xs, ys, color=color)
