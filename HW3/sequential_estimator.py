from utils import Gaussian_data_gen


if __name__=='__main__':
    m=float(input("Input mean: "))
    s=float(input("Input variance: "))
    print(f'Data point source function: N({m}, {s})')
    eps=1e-4
    mean_diff=10
    var_diff=10
    n=0
    old_mean,old_var,old_M2n=0,0,0
    while mean_diff>eps or var_diff>eps: # converge condition
        new_data=Gaussian_data_gen(m,s) # generate data from Gaussian distribution
        print(f'Add data point: {new_data}')
        n+=1

        # Welford online algorithm
        new_mean=old_mean+(new_data-old_mean)/n
        new_M2n=old_M2n+(new_data-old_mean)*(new_data-new_mean) # stability
        new_var=new_M2n/n

        print(f'Mean = {new_mean}\t Variance = {new_var}')
        mean_diff=abs(new_mean-old_mean)
        var_diff=abs(new_var-old_var)
        old_mean,old_M2n,old_var=new_mean,new_M2n,new_var
