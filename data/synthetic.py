import numpy as np
import diagrams.data_generation.helix as plt

def add3Dnoise(x, y, z, sd_x, sd_y, sd_z):
    """
    Adds random noise to 3D points.

    :param x:
    :param y:
    :param z:
    :param sd_x:
    :param sd_y:
    :param sd_z:
    :return:
    """
    x = x + np.random.normal(0, sd_x, x.shape)
    y = y + np.random.normal(0, sd_y, y.shape)
    z = z + np.random.normal(0, sd_z, z.shape)
    return x, y, z

def generate_helix_track(z_positions=(10, 25, 50, 80, 130, 200), r=(50.0, 25.0), noise_sd=(1.0, 1.0, 1.0), plot=False):
    """
    Generate hits for one helical track respecting discrete detector layers.
    Coordinates can be only detected at the coordinates of the z_positions, so there are only
    abs(z_positions) detection points.

    TODO
    model missing hits

    :param z_positions: The position of each detector relative to the collision point.
    :param r:           Tuple containing mean and standard deviation of the helix radius.
    :param noise_sd:    Tuple containing standard deviation for each detection point. Z coordinate can be corrected to its actual value, as this is known.
    :param plot:        Flag whether the helix should be plotted
    :return: Stack of detection points of shape (num detection points, 3)
    """

    theta0  = np.random.uniform(0, 2 * np.pi)   # initial angle in z direction
    phi0    = np.random.uniform(0, 2 * np.pi)   # initial angle in phi-r plane
    charge_sign = np.random.choice([-1, 1])         # rotation direction based on charge

    # random radius, representing different momentum of particle
    radius = max(np.abs(np.random.normal(r[0], r[1])), 15.0)
    # TODO: note that this is not a normal distribution anymore -> however tight radii are highly unlikely -> maybe switch

    helix_center_x = radius * np.cos(phi0)
    helix_center_y = radius * np.sin(phi0)


    # initial angle in z direction and charge sign
    #phi0 = np.random.uniform(-np.pi, np.pi)

    # helix angular frequency   TODO this arbitrary, and depending on the strength of the magnetic field
    omega = charge_sign * (1.0 / radius) * 10

    # z hit positions; the z direction is dependant on theta
    z = np.sign(np.cos(theta0)) * (np.arange(250) if plot else np.array(z_positions))

    # plane hit phases
    phi =  3 * np.pi + phi0 + omega * z

    # plane hits
    x = helix_center_x + radius * np.cos(phi)
    y = helix_center_y + radius * np.sin(phi)

    if plot:
        plt.plot_rphi_plane(x, y, z_positions, helix_center_x, helix_center_y)
        plt.plot_track(x, y, z, z_positions, helix_center_x, helix_center_y)

    return np.stack(
        [add3Dnoise(x, y, z, noise_sd[0], noise_sd[1], noise_sd[2])],
        axis=1)  # (num_hits, 3)

def create_synthetic_event(num_tracks=10, hits_per_track=10, noise_hits= 800):
    all_hits = []
    track_ids = []

    # generate real tracks
    for track_id in range(num_tracks):
        hits = generate_helix_track()
        all_hits.append(hits)
        track_ids.extend([track_id] * 5)

    # add noise hits
    noise = np.random.uniform(-150, 150, (noise_hits, 3))
    all_hits.append(noise)
    track_ids.extend([-1] * noise_hits)

    hits = np.vstack(all_hits)
    track_ids = np.array(track_ids)

    # convert coordinates
    r = np.sqrt(hits[:, 0]**2 + hits[:, 1]**2)
    phi = np.arctan2(hits[:, 1], hits[:, 0])
    z = hits[:, 2]
    hit_features = np.stack([r, np.cos(phi), np.sin(phi), z], axis=1)

    return hit_features, track_ids

if __name__ == '__main__':
    create_synthetic_event()