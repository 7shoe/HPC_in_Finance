import numpy as np
from scipy import stats
import numexpr as ne
import time

# Monte Carlo-based simulation
def call_MC(S, K=100, t=1.0, r=0.03, sigma=0.3, N=int(1e6), seed=778):
    '''Monte Carlo-based simulation of a European Call Price based on N simulations'''
    np.random.seed(seed)
    PTs = S*np.random.lognormal(mean=(r - 0.5*sigma**2)*t, sigma=sigma, size=N) - K
    return np.mean(np.where(PTs > 0, PTs, 0))*np.exp(-r*t)


def main():
    t1 = time.time()
    c1 = call_MC(100.0, seed=8737)
    c2 = call_MC(110.0, seed=1436)
    c3 = call_MC(90.0,  seed=5357)

    t2 = time.time()
    print('Duration:', round(1000*(t2-t1), 1), 'ms')

if __name__ == '__main__':
    main()