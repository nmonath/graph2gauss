import scipy.sparse as sp
from g2g.model import Graph2Gauss
from g2g.utils import load_dataset, score_link_prediction, score_node_classification
import numpy as np


from absl import flags
from absl import app
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_string('input', '', 'data path')
flags.DEFINE_string('labels', '', 'data path')
flags.DEFINE_string('output', '', 'data path')
flags.DEFINE_integer('dim', 2, 'data path')
flags.DEFINE_integer('tol', 50, 'data path')

logging.set_verbosity(logging.INFO)


def cluster(nn_graph, dim=2):
    g2g = Graph2Gauss(A=nn_graph, X=nn_graph + sp.eye(nn_graph.shape[0]), tolerance=FLAGS.tol, L=dim, verbose=True, p_val=0.0, p_test=0.00)
    sess = g2g.train()
    mu, sigma = sess.run([g2g.mu, g2g.sigma])
    return mu,sigma

def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N

def kl(pm, pv, qm, qv):
    log_dpv = np.log(pv).sum(axis=1, keepdims=True)
    log_dqv = np.log(qv).sum(axis=1, keepdims=True)
    iqv  =1/ (qv)
    diffs = np.expand_dims(qm,1) - pm
    t1 = (log_dqv - log_dpv.T)
    t2 = (np.expand_dims(iqv,1) * pv).sum(axis=2)
    t3 = diffs * np.expand_dims(iqv,1) * diffs
    t3sum = t3.sum(axis=2)
    N = pm.shape[1]
    res = 0.5 * (t1 + t2 + t3sum - N)
    return res.T

def predict(mu, sigma):
    r = kl(mu, sigma, mu, sigma)
    np.fill_diagonal(r, np.inf)
    var_norms = np.linalg.norm(sigma, axis=1)
    sorted_norms = np.argsort(var_norms)
    rs = r[sorted_norms, :][:, sorted_norms]
    rs[np.tril_indices_from(rs)] = np.inf
    p = np.argmin(rs, 1)
    p[rs[np.arange(p.shape[0]), p] == np.inf] = -1
    return p,sorted_norms

def tree_from_parent(p, lbls, filename):
    """

    p[i] gives the parent of i
    also add the data point i as a child of i

    :param p:
    :param lbls:
    :param filename:
    :return:
    """
    with open(filename, 'w') as fout:
        for i in range(p.shape[0]):
            fout.write("%s\ti%s\t%s\n" % (i,i,lbls[i]))
            fout.write("i%s\ti%s\tNone\n" % (i, p[i]))

        # now make sure that we have a root node.
        fout.write("i-1\tNone\tNone\n")


def load_graph(filename):
    init_knn_dist = sp.load_npz(filename)
    return init_knn_dist

def main(argv):
    graph = load_graph(FLAGS.input)
    labels = np.loadtxt(FLAGS.labels).astype(np.int32)
    mu, sigma = cluster(graph,FLAGS.dim)
    logging.info('mu.shape %s', str(mu.shape))
    p, sorted_norms = predict(mu, sigma)
    tree_from_parent(p, labels[sorted_norms], FLAGS.output)

if __name__ == "__main__":
    app.run(main)