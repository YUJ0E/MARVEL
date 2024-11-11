import numpy as np
import func
import time
from func import Map
from benchmark import PLM
from scipy.stats import norm

def GP4(mymap, T, N, S):
    print('current node: {}'.format(mymap.r_0 + 1))
    value_max = 0
    map_temp = Map(mymap.model, mymap.decom)

    for _, next_node, d in mymap.G.out_edges(mymap.r_0, data=True):
        if func.dijkstra(mymap.G, next_node, mymap.r_s)[0] == -1:
            continue

        link = d['index']
        print('current link: {}'.format(link + 1))

        if next_node == mymap.r_s:
            value = norm.cdf(T, mymap.mu[link].item(), np.sqrt(mymap.cov[link, link]))
            if value >= value_max:
                value_max = value
                selected_link = link

        else:
            v_hat = 0
            G_temp = func.remove_graph_edge(mymap.G, link)
            mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, link)

            for i in range(N):
                sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
                T_temp = T - sample
                if T_temp > 0:
                    mu_con = func.update_mu(mu_sub, cov_sub, sample)
                    map_temp.make_map_with_G(mu=mu_con, cov=cov_con, OD_true=[next_node, mymap.r_s], G=G_temp)
                    v_hat += PLM(mymap=map_temp, T=T_temp, S=S)[0]

            value = v_hat / N

            if value >= value_max:
                value_max = value
                selected_link = link

    return selected_link, value_max

def logGP4(mymap, T, N, S):
    print('current node: {}'.format(mymap.r_0 + 1))
    value_max = 0
    map_temp = Map(mymap.model, mymap.decom)

    for _, next_node, d in mymap.G.out_edges(mymap.r_0, data=True):
        if func.dijkstra(mymap.G, next_node, mymap.r_s)[0] == -1:
            continue

        link = d['index']
        print('current link: {}'.format(link + 1))

        if next_node == mymap.r_s:
            value = norm.cdf(np.log(T), mymap.mu[link].item(), np.sqrt(mymap.cov[link, link]))
            if value >= value_max:
                value_max = value
                selected_link = link

        else:
            v_hat = 0
            G_temp = func.remove_graph_edge(mymap.G, link)
            mu_sub, cov_sub, cov_con = func.update_param(mymap.mu, mymap.cov, link)

            for i in range(N):
                sample = np.random.normal(mu_sub[2], np.sqrt(cov_sub[22]))
                T_temp = T - np.exp(sample)
                if T_temp > 0:
                    mu_con = func.update_mu(mu_sub, cov_sub, sample)
                    map_temp.make_map_with_G(mu=mu_con, cov=cov_con, OD_true=[next_node, mymap.r_s], G=G_temp, )
                    v_hat += PLM(mymap=map_temp, T=T_temp, S=S)[0]

            value = v_hat / N

            if value >= value_max:
                value_max = value
                selected_link = link

    return selected_link, value_max

def biGP4(mymap, T, N, S):
    print('current node: {}'.format(mymap.r_0 + 1))
    value_max = 0
    map_temp = Map(mymap.model, mymap.decom)

    for _, next_node, d in mymap.G.out_edges(mymap.r_0, data=True):
        if func.dijkstra(mymap.G, next_node, mymap.r_s)[0] == -1:
            continue

        link = d['index']
        print('current link: {}'.format(link + 1))

        if next_node == mymap.r_s:
            value1 = norm.cdf(T, mymap.mu[link].item(), np.sqrt(mymap.cov[link, link]))
            value2 = norm.cdf(T, mymap.mu2[link].item(), np.sqrt(mymap.cov2[link, link]))
            value = func.calc_bi_gauss(mymap.phi_bi, value1, value2)
            if value >= value_max:
                value_max = value
                selected_link = link

        else:
            v_hat = 0
            G_temp = func.remove_graph_edge(mymap.G, link)
            mu1_sub, cov1_sub, cov1_con = func.update_param(mymap.mu, mymap.cov, link)
            mu2_sub, cov2_sub, cov2_con = func.update_param(mymap.mu2, mymap.cov2, link)

            for i in range(N):
                sample = func.generate_biGP_samples(mymap.phi_bi, mu1_sub[2], mu2_sub[2], cov1_sub[22], cov2_sub[22], 1).item()
                T_temp = T - sample
                if T_temp > 0:
                    mu1_con = func.update_mu(mu1_sub, cov1_sub, sample)
                    mu2_con = func.update_mu(mu2_sub, cov2_sub, sample)
                    map_temp.make_map_with_G(mu=mu1_con, cov=cov1_con, OD_true=[next_node, mymap.r_s], G=G_temp, mu2=mu2_con, cov2=cov2_con, phi_bi=mymap.phi_bi)
                    v_hat += PLM(mymap=map_temp, T=T_temp, S=S, model="bi")[0]

            value = v_hat / N

            if value >= value_max:
                value_max = value
                selected_link = link

    return selected_link, value_max

def GP4_iterations(mymap, T, N, S, MaxIter):
    '''
    run a variant of GP4 for MaxIter times and return the statistics
    '''

    pro = []
    t_delta = []

    for ite in range(MaxIter):
        print('GP4 iteration #{}'.format(ite))
        t1 = time.perf_counter()

        if mymap.model == 'G':
            selected_link, prob = GP4(mymap, T, N, S)
        elif mymap.model == 'log':
            selected_link, prob = logGP4(mymap, T, N, S)
        elif mymap.model == 'bi':
            selected_link, prob = biGP4(mymap, T, N, S)

        t_delta.append(time.perf_counter() - t1)
        print('probability: {}, selected link: {}\n'.format(prob, selected_link+1))
        pro.append(prob)

    return np.mean(pro), np.std(pro, ddof=1), np.mean(t_delta), np.max(t_delta)

