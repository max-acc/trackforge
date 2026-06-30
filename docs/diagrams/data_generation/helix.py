"""
Module for plotting helix track paths.
Used for visualization and visual debugging.
"""

import matplotlib.pyplot as plt


def plot_rphi_plane(x, y, z_positions, helix_center_x, helix_center_y):
    """
    Plot a single track in the r-phi plane.

    :param x:               The x coordinates of the track.
    :param y:               The y coordinates of the track.
    :param z_positions:     The z positions of the detector.
    :param helix_center_x:  The x coordinate of the helix center.
    :param helix_center_y:  The y coordinate of the helix center.
    """
    plt.plot(x, y, "blue")
    plt.plot(helix_center_x, helix_center_y, "go-", label='Helix Center')
    plt.plot(x[0], y[0], "ro-", label='Collision Point')
    plt.plot(x[z_positions[1]], y[z_positions[1]], "bo-", label='Detector Hits')
    for i in z_positions:
        plt.plot(x[i], y[i], "bo-")
    plt.xlim(-100, 100)
    plt.ylim(-100, 100)
    plt.grid(True)
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.title("Track Path in r-phi plane")
    plt.show()

def plot_track(x, y, z, z_positions, helix_center_x, helix_center_y):
    """
    Plot a single track in 3D.

    :param x:               The x coordinates of the track.
    :param y:               The y coordinates of the track.
    :param z:               The z positions of the track.
    :param z_positions:     The z positions of the detector.
    :param helix_center_x:  The x coordinate of the helix center.
    :param helix_center_y:  The y coordinate of the helix center.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the hits (scatter)
    #ax.scatter(x, y, z, c=z, cmap='viridis', s=80, label='Detector Hits')

    # Plot the track line connecting the hits
    ax.plot3D(helix_center_x, helix_center_y, z, 'go-', linewidth=2, alpha=0.8, label='Helix Center')
    ax.plot3D(x[0], y[0], z[0], "ro-", label='Collision Point')
    ax.plot3D(x, y, z, 'blue', linewidth=2, alpha=0.8)
    ax.plot3D(
        x[z_positions[1]], y[z_positions[1]],
        z_positions[1] if z[1] >= 0 else -z_positions[1],
        "bo-", label='Detector Hits')
    for i in z_positions:
        ax.plot3D(x[i], y[i], i if z[i] >= 0 else -i, "bo-")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Helix Track")
    ax.legend()
    plt.tight_layout()
    plt.show()
