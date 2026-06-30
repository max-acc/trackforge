"""
Module specifying the synthetic data/helix generation.
"""

import numpy as np


def add_2d_noise(
        x,
        y,
        sd_x,
        sd_y
):
    """
    Adds random noise to 3D points.

    The z component does not need noise, as we can assume we know the position of the detector, and can clean up this
    data.

    :param x:       The x coordinate of the point.
    :param y:       The y coordinate of the point.
    :param sd_x:    The standard deviation in x direction.
    :param sd_y:    The standard deviation in y direction.
    :return:    A tuple containing the new x and y coordinates with noise.
    """
    x = x + np.random.normal(0, sd_x, x.shape)
    y = y + np.random.normal(0, sd_y, y.shape)
    return x, y

def generate_helix_track(
        general_config,
        plot=False
):
    """
    Generate hits for one helical track respecting discrete detector layers.
    Coordinates can be only detected at the coordinates of the z_positions.

    :return: Stack of detection x,y,z points of shape (num detectors, 3)
    """
    config = general_config.get_helix_config()

    theta0  = np.random.uniform(0, 2 * np.pi)   # initial angle in z direction
    phi0    = np.random.uniform(0, 2 * np.pi)   # initial angle in phi-r plane
    charge_sign = np.random.choice([-1, 1])         # rotation direction based on charge

    # random radius, representing different momentum of particle
    radius = np.maximum(np.abs(
        np.random.normal(config['radius'], config['radius_sd'])
    ), config['min_radius'])
    # TODO: note that this is not a normal distribution anymore -> however tight radii are highly unlikely
    #  -> maybe switch -> can also be way smaller, as this regularizes the pitch in z direction

    helix_center_x = radius * np.cos(phi0)
    helix_center_y = radius * np.sin(phi0)

    # z hit positions
    if np.abs(np.cos(theta0)) < 0.1:     # Particle is almost perpendicular
        return None
    z = (np.arange(250) if plot else np.array(config['z_positions']))   # more points makes plotting easier to evaluate

    # rate of phase change per unit z -> absorbs magnetic field and unit system
    omega = charge_sign * np.tan(theta0) / radius * config['scale']

    phi = np.pi + phi0 + omega * z
    x = helix_center_x + radius * np.cos(phi)
    y = helix_center_y + radius * np.sin(phi)

    if plot:
        # import only if helix track should be plottet
        import docs.diagrams.data_generation.helix as helix_plots
        helix_plots.plot_rphi_plane(x, y, config['z_positions'], helix_center_x, helix_center_y)
        helix_plots.plot_track(x, y, z, config['z_positions'], helix_center_x, helix_center_y)

    x_noise, y_noise = add_2d_noise(x, y, config['noise_sd'][0], config['noise_sd'][1])
    return np.stack([x_noise, y_noise, z], axis=1)

def generate_noise_hits(noise_hits, general_config):
    """
    Generate noise hits for detector layers.

    :noise_hits:        The number of nosise hits to generate.
    :general_config:    The config data required for generating noise hits.
    :return:    The x,y,z coordinates of the noise hits of shape (noise_hits, 3).
    """
    config = general_config.get_noise_config()

    noise_layer_idx = np.random.randint(0, len(config['z_positions']), noise_hits)

    z   = np.array(config['z_positions'])[noise_layer_idx]
    r   = np.sqrt(np.random.uniform(config['disk_r_range'][0]**2, config['disk_r_range'][1]**2, noise_hits))
    phi = np.random.uniform(0, 2 * np.pi, noise_hits)

    x   = r * np.cos(phi)
    y   = r * np.sin(phi)

    return np.stack([x, y, z], axis=1)

def create_synthetic_event(general_config):
    """
    Creates a synthetic event of particle collisions.

    :general_config:    The config data required for generating synthetic events.
    :return:    The detector hits of particles of shape ((num_tracks * 6 + 800, 4), track number).
    """
    config = general_config.get_event_config()

    all_hits = []
    track_ids = []

    # --- generate real tracks ---
    for track_id in range(
            config['num_tracks'] # TODO revert
            # + np.random.randint(config['num_tracks_dev'][0], config['num_tracks_dev'][1])
    ):
        hits = None
        while hits is None: # retry on near perpendicular track
            hits = generate_helix_track(general_config)
        all_hits.append(hits)
        track_ids.extend([track_id] * len(hits))

    # --- noise hits on actual disk surface ---
    noise_hits = config['noise_hits'] # TODO revert
    # + abs(np.random.randint(config['noise_hits_dev'][0], config['noise_hits_dev'][1]))

    all_hits.append(generate_noise_hits(noise_hits, general_config))
    track_ids.extend([-1] * noise_hits)

    hits = np.vstack(all_hits)
    track_ids = np.array(track_ids)

    # translate hits to detector geometry
    r = np.sqrt(hits[:, 0]**2 + hits[:, 1]**2)
    phi = np.arctan2(hits[:, 1], hits[:, 0])
    z = hits[:, 2]
    hit_features = np.stack([r, np.cos(phi), np.sin(phi), z], axis=1)

    return hit_features, track_ids
