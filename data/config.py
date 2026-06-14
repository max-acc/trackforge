import yaml
class Config:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def get_num_events(self) -> int:
        """
        Get the number of events.

        :return:    The number of events as `int`.
        """
        return self.config['event']['num_events']

    def get_event_config(self):
        return {
            'num_tracks':       self.config['event']['num_tracks'],
            'num_tracks_dev':   self.config['event']['num_tracks_dev'],
            'noise_hits':       self.config['event']['noise_hits'],
            'noise_hits_dev':   self.config['event']['noise_hits_dev'],
        }

    def get_helix_config(self):
        return {
            'z_positions':      self.config['detector']['z_positions'],
            'radius':           self.config['track']['radius_mean'],
            'radius_sd':        self.config['track']['radius_std'],
            'noise_sd':         self.config['track']['noise_sd'],
            'scale':            self.config['helix']['scale'],
            'min_radius':       self.config['helix']['min_radius'],
        }

    def get_noise_config(self) -> dict:
        """
        Getter for returning noise specifig configuartion data.

        :return:    A dictionairy containing the `z_positions` of the detector, and the `disk_r_range` of the detector.
        """
        return {
            'z_positions':      self.config['detector']['z_positions'],
            'disk_r_range':     self.config['detector']['disk_r_range'],
        }

