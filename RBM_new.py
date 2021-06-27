import numpy as np

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.debug_print = True

        np_rng = np.random.RandomState(1234)

        self.weights = np.asarray(np_rng.uniform(
                    low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                    high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                    size=(num_visible, num_hidden)))
        self.v_bias = np.zeros((num_visible, 1))
        self.h_bias = np.zeros((num_hidden, 1))
        self.errors = []

    def free_energy(self, vis, hid):
        energy = - np.dot(vis.T, np.dot(self.weights, hid)) - np.dot(self.v_bias.T, vis) - np.dot(self.h_bias.T, hid)
        return energy

    def hid_given_vis_prob(self, vis):
        hid_prob = self.sigmoid(self.h_bias + np.dot(self.weights.T, vis))
        return hid_prob

    def vis_given_hid_prob(self, hid):
        vis_prob = self.sigmoid(self.v_bias + np.dot(self.weights, hid))
        return vis_prob

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def probs_to_spins(probs):
