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

    def hid_given_vis(self, vis):
        hid_prob = self.hid_given_vis_prob(vis)
        return self.probs_to_spins(hid_prob)

    def vis_given_hid(self, hid):
        vis_prob = self.vis_given_hid_prob(hid)
        return self.probs_to_spins(vis_prob)

    def vis_to_vis(self, vis, gibbs_steps):
        for step in range(gibbs_steps):
            hid = self.hid_given_vis(vis)
            vis = self.vis_given_hid(hid)
        return vis

    def hid_to_hid(self, hid, gibbs_steps):
        for step in range(gibbs_steps):
            vis = self.vis_given_hid(hid)
            hid = self.hid_given_vis(vis)
        return hid

    def gradient(self, training_vis, gibbs_steps):
        pos_hid_probs = self.hid_given_vis_prob(training_vis)
        pos_hid_states = [self.probs_to_spins(probs) for probs in pos_hid_probs]
        pos_associations = np.dot(training_vis.T, pos_hid_probs)

        neg_hid_states = self.hid_to_hid(pos_hid_states, gibbs_steps)
        neg_vis_probs = self.vis_given_hid_prob(neg_hid_states)
        neg_vis_states = [self.probs_to_spins(probs) for probs in neg_vis_probs]
        neg_hid_probs = slef.hid_given_vis_prob(neg_vis_states)
        neg_associations = np.dot(neg_vis_probs.T, neg_hid_probs)

        gradient = - pos_associations + neg_associations
        return gradient

    def train(self, training_vis, learning_rate = 0.01, test_size = 0.2, batch_size = 100): # train for 1 step
        split data


    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def probs_to_spins(probs):
        spins = []
        for prob in probs:
            if prob > np.random.rand():
                spins.append(1)
            else:
                spins.append(0)
        return spins
