import scipy
import numpy as np
import os
import numpy.random as nprand
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
import random


def Expectation_Maximization(i):
    g = i

    def func():
        return [np.random.random(),np.random.random(),1/g]

    params = []
    for x in range(g):
        params.append(func())
    params = np.array(params)

    b = np.ones(shape=(g,3))

    LL_old = np.zeros(shape=(1,g))

    new_params = np.zeros(shape=(g,3))

    prob_mat = np.zeros(shape=(500,g))

    clusters = [[] for x in range(g)]

    LL = np.zeros(shape=(1,g))

    def Gaussian_func(x,mu,sig):
        a = (1 / (math.sqrt(2 * math.pi)* sig)) * math.exp( -(math.pow(x - mu,2) / (2 * sig * sig)) )
        return a

    # Find prob of each point to every cluster ....
    for c in range(g):
        i = 0
        for point in data1:
            gauss = Gaussian_func(point,params[c][0],params[c][1])
            p_x_c = (gauss * params[c][2])
            prob_mat[i][c] = p_x_c
            i=i+1

    np.savetxt("prob_mat.csv", prob_mat, delimiter=",")

    for r in range(500):
        sm = np.sum(prob_mat[r])
        prob_mat[r] = prob_mat[r] / sm

    np.savetxt("prob_sum.csv", prob_mat, delimiter=",")

    N_i = np.sum(prob_mat,axis=0)

    data_sq = np.multiply(np.array(data1),np.array(data1))
    data_norm = np.array(data1)

    for c in range(g):
        new_mu = np.sum(np.multiply(prob_mat[:,c],data1)) / N_i[c]

        t1 = np.sum(np.multiply(prob_mat[:,c],data_sq)) / N_i[c]
        t2 = math.pow((np.sum(np.multiply(prob_mat[:,c],data_norm)) / N_i[c]),2)
        new_sig = math.sqrt(t1 - t2)

        new_wt = N_i[c] / N_i.sum()

        new_params[c][0] = new_mu
        new_params[c][1] = new_sig
        new_params[c][2] = new_wt


    # Assign the points to the cluster
    # assign_old = np.zeros(shape=(500,1))
    assign = np.argmax(prob_mat,axis = 1)
    # assign = assign.reshape(500,1)


    i=0
    for items in assign:
        clusters[items].append(data1[i])
        i=i+1

    for j in range(g):
        clusters[j] = np.array(clusters[j])

    # Calculate the log-Likelihood...
    LL_list = []
    LL_sum_list = []
    LL_sum_old = 0

    for c in range(g):
        total = 0
        for items in clusters[c]:
            total = total + math.log(Gaussian_func(items,new_params[c][0],new_params[c][1]),10)
        LL[0][c] = total

    LL_sum =  round(LL.sum(),3)
    # print(0,LL,LL_sum)
    LL_list.append(LL)
    LL_sum_list.append(LL_sum)

    loop = 1
    # while (LL.sum() - LL_old.sum() < 10):
    while(LL_sum_old != LL_sum):

        params = new_params
        LL_sum_old = LL_sum
        # Find prob of each point to every cluster ....
        for c in range(g):
            i = 0
            for point in data1:
                gauss = Gaussian_func(point, params[c][0], params[c][1])
                p_x_c = (gauss * params[c][2])
                prob_mat[i][c] = p_x_c
                i = i + 1

        np.savetxt("prob_mat.csv", prob_mat, delimiter=",")

        for r in range(500):
            sm = np.sum(prob_mat[r])
            prob_mat[r] = prob_mat[r] / sm

        np.savetxt("prob_sum.csv", prob_mat, delimiter=",")

        N_i = np.sum(prob_mat, axis=0)

        # Calculate the New parameters ....
        # Start with empty gaussians .....
        new_params = np.zeros(shape=(g, 3))

        data_sq = np.multiply(np.array(data1), np.array(data1))
        data_norm = np.array(data1)

        for c in range(g):
            new_mu = np.sum(np.multiply(prob_mat[:, c], data1)) / N_i[c]

            t1 = np.sum(np.multiply(prob_mat[:, c], data_sq)) / N_i[c]
            t2 = math.pow((np.sum(np.multiply(prob_mat[:, c], data_norm)) / N_i[c]), 2)
            new_sig = math.sqrt(t1 - t2)

            new_wt = N_i[c] / N_i.sum()

            new_params[c][0] = new_mu
            new_params[c][1] = new_sig
            new_params[c][2] = new_wt

        # Assign the points to the cluster
        assign = np.argmax(prob_mat, axis=1)
        assign = assign.reshape(500, 1)

        clusters = [[] for x in range(g)]

        i = 0
        for items in assign:
            clusters[items].append(data1[i])
            i = i + 1

        for j in range(g):
            clusters[j] = np.array(clusters[j])

        # Calculate the log-Likelihood...
        LL = np.zeros(shape=(1, g))

        for c in range(g):
            total = 0
            for items in clusters[c]:
                total = total + math.log(Gaussian_func(items, new_params[c][0], new_params[c][1]),10)
            LL[0][c] = total

        LL_sum = round(LL.sum(),3)
        # print(loop, LL, LL.sum())
        LL_list.append(LL)
        LL_sum_list.append(LL_sum)
        # print(LL_list)
        loop=loop+1

    # print('The Gaussian Parameters when number of Gaussians are 4:')
    # print('        mu       sigma       p(C_i)')
    # print(new_params)
    return [new_params,[LL,LL.sum()],LL_list,LL_sum_list]


if __name__ == '__main__':

    distributions = [2,4]
    f = open('data1.txt', 'r')
    data1 = []
    for line in f:
        data1.append(float(line))

    for i in distributions:
        parameters = Expectation_Maximization(i);

        print('The Gaussian Parameters when number of Gaussians are '+str(i)+' for data1 are: ')
        print('        mu       sigma       p(C_i)')
        print(parameters[0])

        print('The Log Likelihood of the distributions are:')
        print(parameters[1])

        print('List of all Log Likelihood:')
        X = np.concatenate(parameters[2], axis=0)
        # print(X.shape)

        t = np.arange(X.shape[0])
        # for dis in range(i):
        #     s0 = X[:, dis]
        #     line, = plt.plot(t, s0, c=np.random.rand(3, 1), lw=2)

        s_total = np.array(parameters[3])
        # print(s_total.shape)
        line, = plt.plot(t, s_total, 'k', lw=2)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Log Likelihood')
        plt.title('Number of Iterations Versus Log Likelihood')
        plt.savefig('LL for Data1 for Gaussian - ' + str(i) + '.png')
        plt.clf()

