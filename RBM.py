import numpy as np
import copy

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

        neg_hid_states = self.hid_to_hid(pos_hid_states, gibbs_steps - 1)
        neg_vis_probs = self.vis_given_hid_prob(neg_hid_states)
        neg_vis_states = [self.probs_to_spins(probs) for probs in neg_vis_probs]
        neg_hid_probs = slef.hid_given_vis_prob(neg_vis_states)
        neg_associations = np.dot(neg_vis_states.T, neg_hid_probs)

        gradient = pos_associations - neg_associations

        error = np.sum((training_vis - neg_visible_states) ** 2)

        return gradient, error

    def train(self, training_vis, max_epochs, learning_rate, batch_size, gibbs_steps):
        num_examples = training_vis.shape[0]
        for epochh in range(max_epochs):
            error = 0
            np.random.shuffle(training_vis)
            batch_number = int(num_examples / batch_size)
            data_batches = np.split(data_new, batch_number)

            for batch in range(batch_number):
                data_batch = data_batches[batch]
                weight_gradient, batch_error = self.gradient(data_batch, gibbs_steps)
                self.weights -= learning_rate * weight_gradient / batch_size
                error += batch_error / batch_size
            self.errors.append(error)
            if self.debug_print:
              print("Epoch %s: error is %s" % (epoch, error))

    def daydream(self, num_samples, gibbs_steps):
        samples = []
        init_vis = np.random.rand(self.num_visible)
        sample = self.vis_to_vis(init_vis, gibbs_steps)
        samples.append(copy.deepcopy(sample))
        for i in range(num_samples - 1):
            sample = self.vis_to_vis(sample, gibbs_steps)
            samples.append(copy.deepcopy(sample))
        return samples

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
