import numpy as np 
import matplotlib.pyplot as plt 

T = 5  # Time horizon
N = 5000  # 100, 10000

def brownian_mot(N,T):  # Function to simulate a Brownian motion
    mu, sigma = 0, np.sqrt(1/N) # Mean and standard deviation 
    s = np.random.normal(mu, sigma, N*T) # N*T normal distributed random numbers 
    B = np.zeros(N*T+1)  # Vector creation
    for i in range(N*T):
        B[i+1]=B[i]+s[i] # Increments of BM on each time step
    return B   # Return the Brownian motion

# Approximate first stochastic integral
B = brownian_mot(N,T) # Brownian motion realization
Y = np.zeros(N*T+1) 
Y_calculated = np.zeros(N*T+1)
for i in range(N*T):
    Y[i+1] = Y[i]+B[i]*(B[i+1]-B[i])  # Stochastic integral approximation formula
    Y_calculated[i+1] = 0.5*(B[i+1]**2-(i+1)/N) # Exact calculation of the right hand side

# Approximate second stochastic integral
Y2 = np.zeros(N*T+1) 
Y2_approx = np.zeros(N*T+1) # To approximatethe pathwise Stieltjes integral on the right hand side
Y2_calculated = np.zeros(N*T+1)
for i in range(N*T):
    Y2[i+1] = Y2[i]+(B[i]**2)*(B[i+1]-B[i])  # Stochastic integral approximation formula
    Y2_approx[i+1] = Y2_approx[i] + B[i]/N  # We approximate the pathwise Stieltjes integral
    Y2_calculated[i+1] = (B[i+1]**3)/3 - Y2_approx[i+1] # And calculate the right hand side 

# Approximate third stochastic integral
Y3 = np.zeros(N*T+1)
Y3_approx = np.zeros(N*T+1) 
Y3_calculated = np.zeros(N*T+1)
for i in range(N*T):
    Y3[i+1] = Y3[i]+((i/N)**2)*(B[i+1]-B[i])  # Stochastic integral
    Y3_approx[i+1] = Y3_approx[i] + 2*(i/N*B[i])/N  # Stieltjes integral
    Y3_calculated[i+1] = (((i+1)/N)**2)*B[i+1] - Y3_approx[i+1]  # Right hand side

x = np.linspace(0, T, N*T+1)   # X-axis discretization
plt.plot(x,Y,label=r'$\int_0^t B_s dB_s$')  # The notation r'$...$' is the way to use a LaTex math environment in matplotlib
plt.plot(x,Y_calculated,label=r'$\frac{1}{2}(B_t^2-t)$')
plt.plot(x,Y2,label=r'$\int_0^t B_s^2dB_s$')
plt.plot(x,Y2_calculated,label=r'$\frac{1}{3}B_t^3-\int_0^t B_sds$')
plt.plot(x,Y3,label=r'$\int_0^t s^2dB_s$')
plt.plot(x,Y3_calculated,label=r'$t^2B_t-2\int_0^t sB_sds$')
plt.title("Approximating stochastic integral processes")
plt.xlabel('t') # Labeling x-axis
plt.legend(loc='best') # Legend at the position where Python finds it the best
plt.show()