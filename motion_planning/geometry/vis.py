import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .scene import Scene

def plot_scene(scene: Scene, ax=None, color=(0.2, 0.6, 0.8, 0.3),
               start=None, via=None, goal=None, spline=None, show_legend=True):
    """Render blocks, and optionally start (blue), via (green), and goal (red) points.

    Args:
        scene: Scene instance with blocks.
        ax: Optional matplotlib 3D axis.
        color: RGBA tuple for block faces.
        start: Optional 3D iterable (x, y, z) for start point (blue).
        via: Optional 3D iterable (x, y, z) for via point (green).
        goal: Optional 3D iterable (x, y, z) for goal point (red).
        spline: Optional callable S(u) that returns (n,3) points for plotting.
        show_legend: Whether to add a legend when any markers are present.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    if scene.blocks:
        all_pts = []
        for b in scene.blocks:
            v = b.vertices_world()
            all_pts.append(v)
            faces = [
                [v[0], v[1], v[2], v[3]],
                [v[4], v[5], v[6], v[7]],
                [v[0], v[1], v[5], v[4]],
                [v[2], v[3], v[7], v[6]],
                [v[1], v[2], v[6], v[5]],
                [v[4], v[7], v[3], v[0]],
            ]
            pc = Poly3DCollection(faces, alpha=color[3], facecolor=color[:3], edgecolor='k', linewidths=0.5)
            ax.add_collection3d(pc)

        all_pts = np.vstack(all_pts)
        mins = all_pts.min(axis=0) - 0.1
        maxs = all_pts.max(axis=0) + 0.1
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[1], maxs[1])
        ax.set_zlim(mins[2], maxs[2])

    # Plot markers
    handles = []
    if start is not None:
        start = np.asarray(start, dtype=float).ravel()
        h = ax.scatter([start[0]], [start[1]], [start[2]], s=70, c='blue', depthshade=True, label='Start')
        handles.append(h)
    if via is not None:
        via = np.asarray(via, dtype=float).ravel()
        h = ax.scatter([via[0]], [via[1]], [via[2]], s=70, c='green', depthshade=True, label='Via')
        handles.append(h)
    if goal is not None:
        goal = np.asarray(goal, dtype=float).ravel()
        h = ax.scatter([goal[0]], [goal[1]], [goal[2]], s=70, c='red', depthshade=True, label='Goal')
        handles.append(h)

    # plot spline if provided
    if hasattr(scene, 'S') and scene.S is not None:
        u = np.linspace(0, 1, 200)
        curve = scene.S(u)
        ax.plot(curve[:,0], curve[:,1], curve[:,2], 'k-', lw=2, label='C2 B-spline')
        if show_legend:
            handles.append(ax.lines[-1])  # add the curve to the legend

    if show_legend and handles:
        ax.legend(loc='upper right')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    return ax


def plot_start_via_goal(ax, start=None, via=None, goal=None, show_legend=True):
    """Convenience helper to plot start (blue), via (green), goal (red) on an existing axis."""
    handles = []
    if start is not None:
        start = np.asarray(start, dtype=float).ravel()
        handles.append(ax.scatter([start[0]], [start[1]], [start[2]], s=70, c='blue', depthshade=True, label='Start'))
    if via is not None:
        via = np.asarray(via, dtype=float).ravel()
        handles.append(ax.scatter([via[0]], [via[1]], [via[2]], s=70, c='green', depthshade=True, label='Via'))
    if goal is not None:
        goal = np.asarray(goal, dtype=float).ravel()
        handles.append(ax.scatter([goal[0]], [goal[1]], [goal[2]], s=70, c='red', depthshade=True, label='Goal'))
    if show_legend and handles:
        ax.legend(loc='upper right')
    return ax