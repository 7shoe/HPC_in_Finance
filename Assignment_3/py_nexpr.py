import numpy as np
import numexpr as ne
import time


def main():

    ne.set_num_threads(5)

    t1 = time.time()

    N = int(1e6)
    sigma = 0.3
    S1 = 100
    S2 = 110
    S3 = 100
    K = 100
    r = 0.03
    t = 1.0
    KS1 = K / S1
    KS2 = K / S2
    KS3 = K / S3

    disc = np.exp(-r*t) / N

    x1 = np.random.lognormal(mean=(r - 0.5*sigma**2)*t, sigma=sigma, size=N)
    x2 = np.random.lognormal(mean=(r - 0.5*sigma**2)*t, sigma=sigma, size=N)
    x3 = np.random.lognormal(mean=(r - 0.5*sigma**2)*t, sigma=sigma, size=N)

    #x1 = x[0::3]
    #x2 = x[1::3]
    #x3 = x[2::3]

    c1 = disc * ne.evaluate('sum(where(x1 > KS1, (S1*x1 - K), 0))')
    c2 = disc * ne.evaluate('sum(where(x2 > KS2, (S2*x2 - K), 0))')
    c3 = disc * ne.evaluate('sum(where(x3 > KS3, (S3*x3 - K), 0))')

    t2 = time.time()
    #print('c1: ', c1)
    #print('c2: ', c2)
    #print('c3: ', c3)

    print('Duration: ', round(1000*(t2-t1), 2), 'ms')
if __name__ == '__main__':
    main()