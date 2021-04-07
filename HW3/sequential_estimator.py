from utils import Gaussian_data_gen


if __name__=='__main__':
    m=float(input("Input mean: "))
    s=float(input("Input variance: "))
    print(f'Data point source function: N({m}, {s})')
    new_data=Gaussian_data_gen(m,s)
    print(f'Add data point: {new_data}')
