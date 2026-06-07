import numpy as np
import diagrams.data_generation.helix as helix_plots

# ----------------------------------------------------------------------------------------------------------------------
# --- TODO -------------------------------------------------------------------------------------------------------------
# - Missing hits are common -> currently this is not modeled
# - Detector geometry in real experiment is different ->
# - Hit readouts are a cluster of pixels rather than a point -> hit has a shape and charge
# - Particle tracks are only modeled into positive z direction
# ----------------------------------------------------------------------------------------------------------------------

def add3Dnoise(
        x,
        y,
        sd_x,
        sd_y
):
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
    return x, y

def generate_helix_track(
        z_positions=(10, 25, 50, 80, 130, 200),
        r=(50.0, 25.0),
        noise_sd=(1.0, 1.0),
        plot=False
):
    """
    Generate hits for one helical track respecting discrete detector layers.
    Coordinates can be only detected at the coordinates of the z_positions, so there are only
    abs(z_positions) detection points.

    :param z_positions: The position of each detector relative to the collision point.
    :param r:           Tuple containing mean and standard deviation of the helix radius.
    :param noise_sd:    Tuple containing standard deviation for each detection point.
    :param plot:        Flag whether the helix should be plotted.
    :return: Stack of detection x,y,z points of shape (num detectors, 3)
    """

    theta0  = np.random.uniform(0, 2 * np.pi)   # initial angle in z direction
    phi0    = np.random.uniform(0, 2 * np.pi)   # initial angle in phi-r plane
    charge_sign = np.random.choice([-1, 1])         # rotation direction based on charge

    # random radius, representing different momentum of particle
    radius = np.maximum(np.abs(np.random.normal(r[0], r[1])), 5.0)
    # TODO: note that this is not a normal distribution anymore -> however tight radii are highly unlikely -> maybe switch -> can also be way smaller, as this regularizes the pitch in z direction

    helix_center_x = radius * np.cos(phi0)
    helix_center_y = radius * np.sin(phi0)

    # z hit positions
    if np.abs(np.cos(theta0)) < 0.1:     # Particle is almost perpendicular
        return None
    z = (np.arange(250) if plot else np.array(z_positions))

    # Rate of phase change per unit z -> absorbs magnetic field and unit system
    SCALE = 10.0
    omega = charge_sign * np.tan(theta0) / radius * SCALE

    phi = np.pi + phi0 + omega * z
    x = helix_center_x + radius * np.cos(phi)
    y = helix_center_y + radius * np.sin(phi)

    if plot:
        helix_plots.plot_rphi_plane(x, y, z_positions, helix_center_x, helix_center_y)
        helix_plots.plot_track(x, y, z, z_positions, helix_center_x, helix_center_y)

    x_noise, y_noise = add3Dnoise(x, y, noise_sd[0], noise_sd[1])
    return np.stack([x_noise, y_noise, z], axis=1)

def generate_noise_hits(
    noise_hits,
    z_positions,
    disk_r_range
):
    """
    Generate noise hits for detector layers.

    :param noise_hits:      The number of noise hits to generate.
    :param z_positions:     The position of each detector relative to the collision point.
    :param disk_r_range:    Tuple containing the minimum and maximum distance where noise hits should be generated.
    :return:    The x,y,z coordinates of the noise hits of shape (noise_hits, 3).
    """
    noise_layer_idx = np.random.randint(0, len(z_positions), noise_hits)

    z   = np.array(z_positions)[noise_layer_idx]
    r   = np.sqrt(np.random.uniform(disk_r_range[0]**2, disk_r_range[1]**2, noise_hits))
    phi = np.random.uniform(0, 2 * np.pi, noise_hits)

    x   = r * np.cos(phi)
    y   = r * np.sin(phi)

    return np.stack([x, y, z], axis=1)

def create_synthetic_event(
        num_tracks=10,
        noise_hits=800,
        z_positions=(10, 25, 50, 80, 130, 200),
        disk_r_range=(10.0, 300.0)
):
    """
    Creates a synthetic event of particle collisions.

    :param num_tracks:      The number of tracks to generate.
    :param noise_hits:      The number of noise hits to generate.
    :param z_positions:     The position of each detector relative to the collision point.
    :param disk_r_range:    The range of the detector radii, where noise hits should be generated.
    :return:    The detector hits of particles of shape ((num_tracks * 6 + 800, 4), track number).
    """
    all_hits = []
    track_ids = []

    # --- generate real tracks ---
    for track_id in range(num_tracks):
        hits = None
        while hits is None: # retry on near perpendicular track
            hits = generate_helix_track(z_positions=z_positions)
        all_hits.append(hits)
        track_ids.extend([track_id] * len(hits))

    # --- noise hits on actual disk surface ---
    all_hits.append(generate_noise_hits(
        noise_hits,
        z_positions,
        disk_r_range
    ))
    track_ids.extend([-1] * noise_hits)

    hits = np.vstack(all_hits)
    track_ids = np.array(track_ids)

    # translate hits to detector geometry
    r = np.sqrt(hits[:, 0]**2 + hits[:, 1]**2)
    phi = np.arctan2(hits[:, 1], hits[:, 0])
    z = hits[:, 2]
    hit_features = np.stack([r, np.cos(phi), np.sin(phi), z], axis=1)

    return hit_features, track_ids

if __name__ == '__main__':
    create_synthetic_event()