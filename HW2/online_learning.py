import numpy as np
import matplotlib.pyplot as plt
import os
import shutil


def factorial(n):
    """
    factorial: calculate factorial n!
    :param n: input number
    :return: n!
    """
    fact=1
    for i in range(1,n+1):
        fact*=i
    return fact


def draw_beta(X,a,b):
    beta=np.zeros(len(X))
    for i,x in enumerate(X):
        beta[i]=x ** (a - 1) * (1 - x) ** (b - 1) * factorial(a + b - 1) / (factorial(a - 1) * factorial(b - 1))
    return beta


if __name__=='__main__':
    cases=1
    while True:
        print("Case #" + str(cases) + ':')
        a=input("Enter a: ")
        b=input("Enter b: ")
        f_case=1
        f=open("testfile.txt","r")
        a_prior=int(a)
        b_prior=int(b)

        # check if the folder exist (for saving distribution imags)
        folder="results/"+str(cases)
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        os.chmod(folder,0o777)

        for cnt,bin_outcome in enumerate(f.readlines()):
            bin_outcome=bin_outcome.strip()
            print("case "+ str(cnt+1)+ ": "+bin_outcome)
            N=len(bin_outcome)
            m=0
            for i in bin_outcome:
                if i=='1':
                    m+=1
            p=float(m)/N

            # posterior_a=m+a-1, posterior_b=N-m+b-1
            a_posterior=m+a_prior
            b_posterior=N-m+b_prior

            # calculate C^N_m*p^m*(1-p)^(N-m)
            likelihood=factorial(N)/(factorial(m)*factorial(N-m))*p**m*(1-p)**(N-m)
            print("Likelihood: "+str(likelihood))
            s="Beta prior:     a={} b={}".format(a_prior,b_prior)
            print(s)
            s="Beta posterior: a={} b={}".format(a_posterior,b_posterior)
            print(s)


            X=np.linspace(0,1,100)

            # draw prior
            prior=draw_beta(X,a_prior,b_prior)
            plt.subplot(1,3,1)
            plt.plot(X,prior,color="red")
            plt.xlabel(r"$\mu$")
            plt.title("prior")

            # draw likelihood
            binomial=np.zeros(100)
            for i,x in enumerate(X):
                binomial[i]=factorial(N)/(factorial(m)*factorial(N-m))*x**m*(1-x)**(N-m)
            plt.subplot(1, 3, 2)
            plt.plot(X, binomial, color="blue")
            plt.xlabel(r"$\mu$")
            plt.title("likelihood function")

            # draw posterior
            posterior=draw_beta(X,a_posterior,b_posterior)
            plt.subplot(1,3,3)
            plt.plot(X,posterior,color="red")
            plt.xlabel(r"$\mu$")
            plt.title("posterior")

            plt.savefig(folder+"/"+str(cnt+1).zfill(2)+".jpg")

            # prior=posterior
            a_prior=a_posterior
            b_prior=b_posterior
            break
            print("\n")

        break
        print("\n")
        cases+=1

