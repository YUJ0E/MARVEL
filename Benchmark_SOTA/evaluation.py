import numpy as np
import func
from scipy.stats import norm

def calc_path_prob(path, mymap, T, samples=None, S=1000):    
    if mymap.model == 'G':
        # path = np.flatnonzero(x)
        mu_sum = np.sum(mymap.mu[path])
        cov_sum = np.sum(mymap.cov[path][:, path])
        return norm.cdf(T, mu_sum, np.sqrt(cov_sum))
    else:
        if samples is None:
            samples = func.generate_samples(mymap, S)
        x = np.zeros(mymap.n_link)
        x[path] = 1
        return np.sum(np.where(np.dot(samples.T, x) <= T, 1, 0)) / samples.shape[1]


def calc_post_prob(path, mymap, T, N, S):
    ''' 
    evaluate the performance of a path in terms of its posterior probability (the same way as how GP4 is evaluated).
    '''

    v_hat = 0

    x = np.zeros(mymap.n_link)
    x[path] = 1
    x = x.reshape(-1, 1)
    x = np.delete(x,path[0],0)

    rng = np.random.default_rng()

    if mymap.model == 'G':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path[0])
        path_con = [i if i < path[0] else i-1 for i in path[1:]]
        for i in range(N):
            sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            T_temp = T - sample
            if T_temp > 0:
                mu_con = func.update_mu(mu_sub, cov_sub, sample)
                mu_sum = np.sum(mu_con[path_con])
                cov_sum = np.sum(cov_con[path_con][:, path_con])
                v_hat += norm.cdf(T_temp, mu_sum, np.sqrt(cov_sum))
                # samples = rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=mymap.decom)
                # v_hat += np.sum(np.where(np.dot(samples, x) <= T_temp, 1, 0)) / samples.shape[1]

    elif mymap.model == 'log':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path[0])
        for i in range(N):
            sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            T_temp = T - np.exp(sample)
            if T_temp > 0:
                mu_con = func.update_mu(mu_sub, cov_sub, sample)
                samples = np.exp(rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=mymap.decom))
                v_hat += np.sum(np.where(np.dot(samples, x) <= T_temp, 1, 0)) / samples.shape[1]

    elif mymap.model == 'bi':
        mu1_sub, cov1_sub, cov1_con = func.update_param(mymap.mu, mymap.cov, path[0])
        mu2_sub, cov2_sub, cov2_con = func.update_param(mymap.mu2, mymap.cov2, path[0])
        for i in range(N):
            sample = func.generate_biGP_samples(mymap.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
            T_temp = T - sample
            if T_temp > 0:
                mu1_con = func.update_mu(mu1_sub, cov1_sub, sample)
                mu2_con = func.update_mu(mu2_sub, cov2_sub, sample)
                samples = func.generate_biGP_samples(mymap.phi_bi, mu1_con, mu2_con, cov1_con, cov2_con, S, method=mymap.decom)
                v_hat += np.sum(np.where(np.dot(samples, x) <= T_temp, 1, 0)) / samples.shape[1]

    return v_hat / N

def calc_post_prob_DOT(J, U, mymap, N, S, delta):
    ''' 
    evaluate the performance of a DOT calculated routing policy in terms of its posterior probability (the same way as how GP4 is evaluated).
    '''

    path_0 = U[mymap.r_0,0]
    max_time = U.shape[1] - 1

    node_1 = func.find_next_node(mymap, mymap.r_0, path_0)

    if node_1 == mymap.r_s:
        return J[mymap.r_0,0]

    rng = np.random.default_rng()
    v_hat = 0

    if mymap.model == 'G':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path_0)
        for i in range(N):
            sample_i = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            time_i = np.ceil(sample_i/delta).astype(int)
            if max_time > time_i:
                mu_con = func.update_mu(mu_sub, cov_sub, sample_i)
                samples = rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=mymap.decom)
                v_temp = 0
                for j in range(S):
                    curr_node = node_1
                    sample = sample_i
                    curr_time = time_i
                    while True:
                        next_link = U[curr_node, curr_time]
                        if next_link == -1:
                            break
                        curr_node = func.find_next_node(mymap, curr_node, next_link)
                        if next_link > path_0:
                            next_link -= 1
                        sample += samples[j, next_link]
                        curr_time = np.ceil(sample/delta).astype(int)
                        if max_time < curr_time:
                            break
                        elif max_time == curr_time and curr_node != mymap.r_s:
                            break
                        elif curr_node == mymap.r_s:
                            v_temp += 1
                            break
                v_hat += v_temp / S

    if mymap.model == 'log':
        mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, path_0)
        for i in range(N):
            sample_i = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
            time_i = np.ceil(np.exp(sample_i)/delta).astype(int)
            if max_time > time_i:
                mu_con = func.update_mu(mu_sub, cov_sub, sample_i)
                samples = np.exp(rng.multivariate_normal(mu_con.reshape(-1), cov_con, S, method=mymap.decom))
                v_temp = 0
                for j in range(S):
                    curr_node = node_1
                    sample = np.exp(sample_i)
                    curr_time = time_i
                    while True:
                        next_link = U[curr_node, curr_time]
                        if next_link == -1:
                            break
                        curr_node = func.find_next_node(mymap, curr_node, next_link)
                        if next_link > path_0:
                            next_link -= 1
                        sample += samples[j, next_link]
                        curr_time = np.ceil(sample/delta).astype(int)
                        if max_time < curr_time:
                            break
                        elif max_time == curr_time and curr_node != mymap.r_s:
                            break
                        elif curr_node == mymap.r_s:
                            v_temp += 1
                            break
                v_hat += v_temp / S

    if mymap.model == 'bi':
        mu1_sub, cov1_sub, cov1_con = func.update_param(mymap.mu, mymap.cov, path_0)
        mu2_sub, cov2_sub, cov2_con = func.update_param(mymap.mu2, mymap.cov2, path_0)
        for i in range(N):
            sample_i = func.generate_biGP_samples(mymap.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
            time_i = np.ceil(sample_i/delta).astype(int)
            if max_time > time_i:
                mu1_con = func.update_mu(mu1_sub, cov1_sub, sample_i)
                mu2_con = func.update_mu(mu2_sub, cov2_sub, sample_i)
                samples = func.generate_biGP_samples(mymap.phi_bi, mu1_con, mu2_con, cov1_con, cov2_con, S, method=mymap.decom)
                v_temp = 0
                for j in range(S):
                    curr_node = node_1
                    sample = sample_i
                    curr_time = time_i
                    while True:
                        next_link = U[curr_node, curr_time]
                        if next_link == -1:
                            break
                        curr_node = func.find_next_node(mymap, curr_node, next_link)
                        if next_link > path_0:
                            next_link -= 1
                        sample += samples[j, next_link]
                        curr_time = np.ceil(sample/delta).astype(int)
                        if max_time < curr_time:
                            break
                        elif max_time == curr_time and curr_node != mymap.r_s:
                            break
                        elif curr_node == mymap.r_s:
                            v_temp += 1
                            break
                v_hat += v_temp / S

    return v_hat / N