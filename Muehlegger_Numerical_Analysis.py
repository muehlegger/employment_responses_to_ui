import numpy as np
from numpy.random import randn
import numpy.polynomial.chebyshev as np_cheb 

import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, vectorize

import scipy.optimize

import scipy.stats as stats

from fredapi import Fred
import statsmodels.api as sm

#-----------------------------------------------------
# SETTINGS 

# run code using calibrated parameters as described in thesis - table 1
calibrated = True

# if "calibrated == False", run the full calibration as described in thesis,
# or manually set a value for "l" and calibrate remaining parameters. 
# for full calibration set "True", or for manual calibration set "False"
default_calibration = True

# if "default_calibration == False", choose the value for "l":
l = 1.25

# URLs of folders for figures and tables
output_folder = "C:/Users/NAME/Documents/OUTPUT/"

figure_folder = output_folder
table_folder = output_folder
#-----------------------------------------------------
# DEFINE STYLE FOR PLOTS

plt.style.use('seaborn-dark')
plt.style.use('seaborn-dark')
styleparams = {'font.family': "serif",
               'legend.fontsize': 13,
               'legend.handlelength': 2,
               'axes.titlesize': 16,
               'axes.labelsize': 14,
               'xtick.labelsize': 13,
               'ytick.labelsize': 13}

plt.rcParams.update(styleparams)

colors = ["darkcyan", "gold", "tomato"]
colors_vars = ["darkcyan", "black", "tomato", "yellowgreen", "maroon", "peru", "steelblue"]#["darkcyan", "gold", "tomato"]

# set sizes of figures
figsize = (9,6)
figsize2= (16,5)

#-----------------------------------------------------
# GENERAL FUNCTIONS
    
def get_params_df(params, params_names):
    """
    transform numpy array of parameters into Dataframe
    for better visualisation
    """
    dfp = pd.DataFrame(params, index=params_names).rename(columns={"index": 'variable', 0: 'value'})
    return dfp

def get_stst_df(steady_state_values):
    """
    transform numpy array of steady-state values into Dataframe
    for better visualisation
    """
    stst_keys = np.array(["U", "V", "B", "Φ", "h", "w*"])
    return pd.Series(steady_state_values, index=stst_keys)

#-----------------------------------------------------
# METHODS FOR FUNCTION APPROXIMATION
# the functions scale_up(), scale_down(), plot_state_function_2D() and approximate_state_function_2D() 
# have been taken from QuantEcon - Notes:
# Mohammed Aït Lahcen (2020) - A brief note on Chebyshev approximation (https://notes.quantecon.org/submission/5f6a23677312a0001658ee16)

@njit
def scale_up(y, x_min, x_max):
    """
    Scales up z \in [-1,1] to x \in [x_min,x_max]
    where y = (2 * (x - x_min) / (x_max - x_min)) - 1
    """
    
    return x_min + (y + 1) * (x_max - x_min) / 2

@njit
def scale_down(x, x_min, x_max):
    """
    Scales down x \in [x_min,x_max] to z \in [-1,1]
    where z = f(x) = (2 * (x - x_min) / (x_max - x_min)) - 1
    """    
    
    return (2 * (x - x_min) / (x_max - x_min)) - 1

def get_cheb_grid(x_min, x_max, m):
    """ 
    get m Chebyshev nodes on the interval [x_min, x_max].
    """
    y = np_cheb.chebpts1(m) # get nodes on [-1, 1] first
    return x_min + (x_max - x_min) * 0.5 * (y + 1) 

def approximate_state_function_1D(x_grid, y, m):
    """
    """
    return np_cheb.chebfit(x_grid, y, m-1)

def approximate_state_function_2D(f, x_min, x_max, y_min, y_max, m_x, m_y):
    """
    """
    # order of approximating polynomial
    n_x = m_x - 1
    n_y = m_y - 1

    # generate chebyshev nodes (the roots of Chebyshev polynomials, a Chebyshev polynomial of degree m-1 has m roots)
    r_x = -np.cos((2*np.arange(1,m_x+1) - 1) * np.pi / (2*m_x))
    r_y = -np.cos((2*np.arange(1,m_y+1) - 1) * np.pi / (2*m_y))

    # scale up nodes to function domain
    x = scale_up(r_x,x_min,x_max)
    y = scale_up(r_y,y_min,y_max)

    # build the Chebyshev polynomials for each variable separately
    Tx = np.zeros((m_x,n_x+1))

    Tx[:,0] = np.ones((m_x,1)).T

    Tx[:,1] = r_x.T

    for i in range(1,n_x):
        Tx[:,i+1] = 2 * r_x * Tx[:,i] - Tx[:,i-1]


    Ty = np.zeros((m_y,n_y+1))

    Ty[:,0] = np.ones((m_y,1)).T

    Ty[:,1] = r_y.T

    for i in range(1,n_y):
        Ty[:,i+1] = 2 * r_y * Ty[:,i] - Ty[:,i-1]


    # Build matrix of f(x,y)
    F = np.zeros((m_x,m_y))

    for j_x in range(m_x):
        for j_y in range(m_y):
            F[j_x,j_y] = f[j_x, j_y]

    κ = n_x

    α = np.zeros((n_x+1,n_y+1))

    for i_x in range(n_x+1):

        for i_y in range(n_y+1):

            if i_x + i_y <= κ:

                α[i_x,i_y] = (Tx[:,i_x].T @ F @ Ty[:,i_y]) / np.outer(np.diag(Tx.T @ Tx)[i_x], np.diag(Ty.T @ Ty).T[i_y])

    # Compute Chebyshev coefficients
    α = (Tx.T @ F @ Ty) / np.outer(np.diag(Tx.T @ Tx), np.diag(Ty.T @ Ty).T ) # use Matrix outer product for the denominator

    # Build approximation of f(x,y)
    G = Tx @ α @ Ty.T
    print("Maximum approx. Error:", np.max(np.abs(G - F)))
    
    return α, G, F

def get_approximation_value(α, X_choice, X_min, X_max, Y_choice=None, Y_min=None, Y_max=None, dim=1):
    """
    return the value of an approximated function at the desired location on the grid state space
    will return an array when a choice variable is given as an array
    """
    if dim==1:
        X = np_cheb.chebval(X_choice, α)
    elif dim==2:
        X = np_cheb.chebval2d(scale_down(X_choice, X_min, X_max), scale_down(Y_choice, Y_min, Y_max), α)
    return X

def plot_state_function_2D(x_grid, y_grid, F, z_label, text):
    # Generate meshgrid coordinates for 3d plot
    xg, yg = np.meshgrid(x_grid, y_grid)

    # Plot approximation
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(20, 230)
    ax.plot_surface(xg,
                    yg,
                    F.T,
                    rstride=2, cstride=2,
                    cmap='Blues', #cm.jet
                    alpha=1,
                    linewidth=0.25)
    ax.set_facecolor('white')
    ax.set_xlabel('$b$', fontsize=14)
    ax.set_ylabel('$\phi$', fontsize=14)
    ax.set_zlabel(z_label, fontsize=14)
    ax.w_xaxis.line.set_color("red")

    plt.show() 
    fig.savefig(figure_folder+'figure_3d_{}.png'.format(text),bbox_inches='tight', dpi=300)
    fig.savefig(figure_folder+'figure_3d_{}.tif'.format(text),bbox_inches='tight')

#-----------------------------------------------------
# FUNCTIONS FOR NUMERICAL CALCULATIONS

@njit
def util(x, c=0):
    """
    utility function
    log of wages or benefits less linear costs
    x, c=0 -> np.log(x) - c
    """
    return np.log(x) - c

@njit
def perform_VFI_u(u_0, w_draws, B_grid, Φ_grid, params):
    """
    iterate two-dimensional, one-state value function for unemployment
    given initial guess, array of wage-draws, grid on both dimensions and parameters
    u_0, w_draws, B_grid, Φ_grid, params -> u
    """
    beta = params[0]
    c = params[3]
    u0 = u_0.copy()
    
    #create new arrays
    u0_new = np.zeros_like(u0)
    W = np.zeros_like(w_draws)

    it = 0
    tol = 1e-8
    eps = 1
    while eps>tol:
        it+=1
        print("\tVFI # iteration:", it, "| eps:", eps)
        for i in range(len(B_grid)):
            for j in range(len(Φ_grid)):
                u = u0[i,j]
                W = get_W_0(w_draws, params, u)
                v = get_v(W, u)
                u0_new[i,j] = util(B_grid[i], c) + beta * (Φ_grid[j] * v + (1-Φ_grid[j]) * u)
        eps = np.linalg.norm(u0_new-u0)
        u0 = u0_new.copy()
    print("within VFI c =", c)   
    return u0

@njit
def perform_VFI_u1(U0, w_draws, benfits, Φ_grid, params):
    """
    iterate one-dimensional, two-state value function for unemployment in the high state
    given one dimensional array of value of u in the low state, an array of wage-draws, 
    array of benefits (low, high - state) grid on offer-rate dimension and parameters
    U0, w_draws, benfits, Φ_grid, params -> u1
    """
    beta = params[0]
    lamb = params[2]
    c = params[3]
    
    b_L, b_H = benfits
    
    U1 = U0.copy()
    
    #create new arrays
    U1_new = np.zeros_like(U1)
    W = np.zeros_like(w_draws)

    it = 0
    tol = 1e-8
    eps = 1
    while eps>tol:
        it+=1
        print("\tVFI # iteration:", it, "| eps:", eps)
        for j in range(len(Φ_grid)):
            u1 = U1[j]
            u0 = U0[j]
            W = get_W_1(w_draws, params, u1, u0)
            v = get_v(W, u1)
            U1_new[j] = util(b_H, c) + beta * (lamb*(Φ_grid[j] * v + (1-Φ_grid[j]) * u1)+(1-lamb)*((u0-util(b_L, c))/beta))
            
        eps = np.linalg.norm(U1_new-U1)
        U1 = U1_new.copy()
        
    return U1

@njit
def perform_VFI_u0(u_0, b, w_draws, Φ_grid, params):
    """
    iterate one-dimensional, one-state value function for unemployment in the low state
    given initial guess, benefit (int), an array of wage-draws, 
    grid on offer-rate dimension and parameters
    u_0, b, w_draws, Φ_grid, params -> u0
    """
    beta = params[0]
    c = params[3]
    #create new arrays
    U0 = u_0.copy()
    U0_new = np.zeros_like(u_0)
    W = np.zeros_like(w_draws)

    it = 0
    tol = 1e-8
    eps = 1
    while eps>tol:
        it+=1
        for j in range(len(Φ_grid)):
            u0 = U0[j]
            W = get_W_0(w_draws, params, u0)
            v = get_v(W, u0)
            U0_new[j] = util(b, c) + beta * (Φ_grid[j] * v + (1-Φ_grid[j]) * u0)
            
        eps = np.linalg.norm(U0_new-U0)
        U0 = U0_new.copy()
        
    return U0


@njit
def get_W_0(w, params, u):
    """
    get the value of being employed in state "L"
    w, params, u -> W
    """
    beta = params[0]
    delta = params[1]

    return (1/(1-beta*(1-delta))) * (util(w) + beta * delta * u)

@njit
def get_W_1(w, params, u1, u0):
    """
    get the value of being employed in state "H"
    w, params, u1, u0 -> W
    """
    beta = params[0]
    delta = params[1]
    lamb = params[2]

    return (1/(1-beta*(1-delta))) * util(w) + (1/(1-beta*lamb*(1-delta))) * ( delta*beta*lamb * u1 + ((delta*beta*(1-lamb))/(1-beta*(1-delta))) * u0)

@njit    
def get_v(W, u):
    """
    get the value of receiving a wage offer
    return a scalar 
    W, u -> v
    """
    return np.mean(np.maximum(W, u))


@njit
def get_Wres(w_grid, u, params):
    """
    return reservation wages on the grid points of benefits and offer rates (2D)
    in the general case (low state)
    w_grid, u, params -> Wres
    """
    W = np.zeros_like(w_grid)
    Wres = np.zeros_like(u)
    
    for i in range(u.shape[0]):
        for j in range(u.shape[1]):
            u_ = u[i,j]
            W = get_W_0(w_grid, params, u_)
            Wres[i,j] = w_grid[np.min(np.argwhere(W-u_>0))]

    return Wres

@njit
def get_Wres_1(w_grid, u1, u0, params):
    """
    return reservation wages on the grid points of offer rates (1D)
    in the high state given 1D value function of unemployment u1 and u0
    w_grid, u1, u0, params -> Wres1
    """
    W = np.zeros_like(w_grid)
    Wres1 = np.zeros_like(u1)

    for j in range(u1.shape[0]):
        u1_ = u1[j]
        u0_ = u0[j]
        W = get_W_1(w_grid, params, u1_, u0_)
        Wres1[j] = w_grid[np.min(np.argwhere(W-u1_>0))]
        
    return Wres1

@njit
def get_Wres_0(w_grid, u0, params):
    """
    return reservation wages on the grid points of offer rates (1D)
    in the low state given 1D value function of unemployment u0
    w_grid, u0, params -> Wres0
    """
    W = np.zeros_like(w_grid)
    Wres0 = np.zeros_like(u0)

    for j in range(u0.shape[0]):
        u0_ = u0[j]
        W = get_W_0(w_grid, params, u0_)
        Wres0[j] = w_grid[np.min(np.argwhere(W-u0_>0))]
        
    return Wres0

def get_2stage_vfkt_coeffs(α_u, α_w, benfits, w_draws, B_min, B_max, Φ_grid, Φ_min, Φ_max, params_i):
    """
    calculate two-stage value function approximation coefficients given 2D-Value function, and benefits in low & high state
    return all approximation coefficients of reservation wages
    (α_w, α_w0, α_w1) == (2D reswages, 1D low state, 1D high state)
    """
    c = params[3]
    u0_i = get_approximation_value(α_u, benfits[0], B_min, B_max, Φ_grid, Φ_min, Φ_max, dim=2)
    Wres0 = get_approximation_value(α_w, benfits[0], B_min, B_max, Φ_grid, Φ_min, Φ_max, dim=2)
    α_w0 = approximate_state_function_1D(Φ_grid, Wres0, m_Φ)
    
    u1 = perform_VFI_u1(u0_i, w_draws, benfits, Φ_grid, params_i)
    Wres1 = get_Wres_1(w_grid, u1, u0_i, params_i)
    α_w1 = approximate_state_function_1D(Φ_grid, Wres1, m_Φ)

    return α_w, α_w0, α_w1

#-----------------------------------------------------
    #DYNAMIC SYSTEM

@njit
def M(U, V, params):
    """
    return the value of the matching function according to den Haan
    given the unemployment and vacancy rates, and elasticity parameter of unemployment
    U, V, params -> M
    """
    l = params[4]
    return U*V/(U**l+V**l)**(1/l)

@njit
def get_offer_rate(U, V, params):
    """
    return the offer rate phi 
    given U, V, and matching function parameters
    U, V, params -> Φ
    """
    q = M(U, V, params) / V
    theta = V/U
    return theta * q

def get_hazard_rate(u, v, b, B_min, B_max, Φ_min, Φ_max, α_w, params):
    """
    get hazard-rate given unemployment, vacancy rate, and benefits
    u, v, b, B_min, B_max, Φ_min, Φ_max, α_w, params -> h
    """
    p = get_offer_rate(u, v, params)
    wres = get_approximation_value(α_w, b, B_min, B_max, p, Φ_min, Φ_max, dim=2)
    return p * (1 - get_lognorm_cdf(wres, params))

@njit
def get_cobb_douglas_job_offer_rate(u, v, m0, alpha):
    """
    return the wage-offer rate calculated using cobb-doublas matching function
    u, v, m0, alpha -> Φ
    """
    return m0 * (v/u)**(1-alpha)

@njit
def iterate_V(V, Vbar, params, shock_v=0):
    """
    iterate vacancy rate, given previous rate V, steady state value Vbar, parameters, and shock value
    V_(t-1), Vbar, params, shock_v=0 -> V_(t)
    """
    rho_v = params[9]
    Vln, Vbarln = np.log(V), np.log(Vbar)
    shock = (Vln - Vbarln) * rho_v + shock_v
    return np.e**(Vbarln + shock)


def iterate_U(U, V, w_res, params):
    """
    iterate through the law of motion for unemployment U_t = f(U_{t-1}).
    return new unemployment rate given U_{t-1}, V{t-1}, wres{t-1}
    """
    delta = params[1]

    F_w = get_lognorm_cdf(w_res, params)
    phi = get_offer_rate(U, V, params)
    
    return U + delta * (1 - U) - phi * (1 - F_w) * U

def get_benefit_policy(U, V, α, w_res_stst, B_min, B_max, Φ_min, Φ_max, params):
    """
    calculate benefits for constant reservation wage policy, given
    U, V and steady-state w_res
    V, U, P, B, W_res "=" stst_values
    """
    
    Φ = get_offer_rate(U, V, params)

    B = scipy.optimize.bisect(iterate_benefits, B_min, B_max, args=(Φ, w_res_stst, B_min, B_max, Φ_min, Φ_max, α))
    return B

def iterate_benefits(b, Φ, w_res_stst, B_min, B_max, Φ_min, Φ_max, α):
    return get_approximation_value(α, b, B_min, B_max, Φ, Φ_min, Φ_max, dim=2) - w_res_stst

#-----------------------------------------------------

#Fit lognormal distribution to data moments, namely: mean, P10, P25, P50, P75, P90

def get_lognorm_cdf(x, params):
    """
    return CDF of lognormal-distribution
    """
    mu = params[7]
    sigma = params[8]
    return stats.lognorm.cdf(x, s=sigma, scale=np.exp(mu))

@njit
def get_wage_params(mean, SD):
    """
    get parameters of the lognormal distribution
    given mean and variance of observed wages
    """
    var = SD**2
    σ = np.sqrt(np.log(var/mean**2 + 1))
    μ = np.log(mean) - σ**2/2
    return μ, σ

def get_lognorm_moments(μ, σ):
    """
    get parameters of the lognormal distribution
    given mean and variance of observed wages
    """
    mean = np.e**(μ + (σ**2)/2)
    var = np.e**(2*μ + σ**2) * (np.e**(σ**2) - 1)
    SD = np.sqrt(var)
    return mean, SD

def get_frictional_dist_params(mean, SD, factor, log=False):
    """
    calculate frictional component of lognormal distribution according to factor and return parameters
    if log==True: input variables mean, SD equal to parameters μ, σ of lognormal distribution
    """
    if log==False:
        mean_out = mean
        SD_out = np.sqrt(SD**2 * factor)
    else:
        mean_in, SD_in = get_lognorm_moments(mean, SD)
        SD_out = np.sqrt(SD_in**2 * factor)
        mean_out, SD_out = get_wage_params(mean_in, SD_out)
        
    return mean_out, SD_out

def get_lognormal_draws(μ, σ, n=25000, seed=1234):
    """
    return array containing n draws of lognormally distributed wages given parameters
    """
    np.random.seed(seed)
    z = np.random.randn(n)
    w_draws = np.exp(μ + σ * z)
    return w_draws

def get_uniform_draws(params, n=25000, seed=1234):
    """
    return array containing n draws of uniformly distributed wages given parameters
    """
    a, b = params[5], params[6]
    w_draws = np.random.rand(n) * (b-a) + a
    return w_draws

def plot_wage_draws(w_draws):
    fig, ax = plt.subplots()
    print(np.min(w_draws), np.max(w_draws), np.mean(w_draws))
    (counts, bins) = np.histogram(w_draws, bins=range(0, int(np.mean(w_draws)*2.5), 50))
    factor = 1
    ax.hist(bins[:-1], bins, weights=factor*counts)
    ax.set_title("Histogram of draws of lognormal wages")
    ax.axvline(w_draws.mean(), color="black", label="mean wage")
    ax.axvline(np.median(w_draws), color="brown", label="wage of median")

def fit_lognormal(p0, N, dmoments):
    """
    fit lognormal distribution with parameters mu and omega
    to mean, and percentiles 10, 25, 50, 75, and 90
    return parameters mu and omega of lognormal distribution
    """
    mu, omega = p0
    W = np.eye(len(dmoments))
    bnds = ((0, 10), (0, 2))
    results = scipy.optimize.minimize(iterate_distribution_estimation, p0, args=(dmoments, W, N),
                          method='L-BFGS-B',
                          bounds=(bnds)) #, options={'eps': 1e-1}
    print(results)
    return results.x

"""def iterate_distribution_estimation(p, dmoments, W, N):
    mu, sigma = p
    w_draw = get_lognormal_draws(mu, sigma, N)
    mmoments = np.array([w_draw.mean(), np.percentile(w_draw, 10), np.percentile(w_draw, 25), np.percentile(w_draw, 50), np.percentile(w_draw, 75), np.percentile(w_draw, 90)])
    e = (mmoments - dmoments)/dmoments

    return e.T @ W @ e"""

def iterate_distribution_estimation(p, dmoments, W, N):
    mu, sigma = p

    mmoments = np.zeros(6)
    mmoments[0] = stats.lognorm.mean(s=sigma, scale=np.exp(mu))
    mmoments[1:] = stats.lognorm.ppf(np.array([0.1, 0.25, 0.5, 0.75, 0.9]), s=sigma, scale=np.exp(mu))
    e = (mmoments - dmoments)/dmoments

    return e.T @ W @ e

#-----------------------------------------------------
# FUNCTION FOR SIMULATING THE MODEL
    
def simulate_model(T, benfits, stst_values, α, B_min, B_max, Φ_min, Φ_max, params, v_shock_sn, policies=False):
    """
    return V, U, P, B, W
    V, U, P, B, W "=" stst_values
    """
    
    #pass values
    b_L, b_H = benfits
    
    α_w, α_w0, α_w1 = α
    μ_ε, σ_ε = params[10], params[11]
    
    #get steady-state values
    U0, V0, B0, Φ0, h0, wres0 = stst_values
    
    #scale given array of shocks
    v_shock = v_shock_sn * σ_ε + μ_ε
  
    #ititiate arrays
    V = np.zeros(T+1)           # Vacancies
    U = np.zeros([T+1, 3])      # Unemployment
    P = np.zeros([T+1, 3])      # Wage-offer rate
    B = np.zeros([T+1, 3])      # Benefits
    W = np.zeros([T+1, 3])      # Reservation Wages
    A = np.zeros([T+1, 3])      # Job Acceptance Rate
    H = np.zeros([T+1, 3])      # Hazard Rate
    S = np.zeros([T+1, 3])      #
    Uma = np.zeros([T+1, 3])    # U - Moving Average
    
    #setting initial value
    V[0] = V0
    U[0,:] = U0
    P[0,:] = Φ0
    B[0,:] = B0
    W[0,:] = wres0
    H[0,:] = h0
    B[:,0] = B0
    B[0,2] = benfits[0]
    A[0,:] = h0/Φ0
    
    #print("Initial Values:", V[0], U[0,0], P[0,0], B[0,0], W[0,0])
    #-----------simulation with constant low benefits--------------------------
    for t in range(1, T+1):
        #print("General; Period:", t)
        V[t] = iterate_V(V[t-1], V0, params, v_shock[t])
        U[t,0] = iterate_U(U[t-1,0], V[t-1], W[t-1, 0], params)       
        P[t,0] = get_offer_rate(U[t,0], V[t], params)
        W[t,0] = get_approximation_value(α_w, B0, B_min, B_max, P[t,0], Φ_min, Φ_max, dim=2)
        A[t,0] = (1 - get_lognorm_cdf(W[t,0], params))
        H[t,0] = P[t,0] * A[t,0]
        
        if t>13:
            Uma[t,0] = (np.sum(U[t-13:t,0]))/13
        else:
            Uma[t,0] = np.sum(U[:t,0])/t

        if (t>52)and(Uma[t,0]>np.min(Uma[t-52:t,0])+0.005):   #52 weeks == 12 months
            S[t,0]=1
        else:
            S[t,0]=0
            
    if policies==True:
        #----------constant reservation-wage policy-------------------------------
        for t in range(1, T+1):
            U[t,1] = iterate_U(U[t-1,1], V[t-1], W[t-1,1], params)
            B[t,1] = get_benefit_policy(U[t,1], V[t], α_w, wres0, B_min, B_max, Φ_min, Φ_max, params)
            P[t,1] = get_offer_rate(U[t,1], V[t], params)
            W[t,1] = get_approximation_value(α_w, B[t,1], B_min, B_max, P[t,1], Φ_min, Φ_max, dim=2)
            A[t,1] = (1 - get_lognorm_cdf(W[t,1], params))
            H[t,1] = P[t,1] * A[t,1]
        
            if t>13:
                Uma[t,1] = (np.sum(U[t-13:t,1]))/13
            else:
                Uma[t,1] = np.sum(U[:t,1])/t
                
            if (t>52)and(Uma[t,1]>np.min(Uma[t-52:t,1])+0.005):   #52 weeks == 12 months
                S[t,1]=1
            else:
                S[t,1]=0
                
        #----------2-stage benefit policy-----------------------------------------
        for t in range(1, T+1):
            
            if t>13:
                Uma[t,2] = (np.sum(U[t-13:t,2]))/13
            else:
                Uma[t,2] = np.sum(U[:t,2])/t
            
            U[t,2] = iterate_U(U[t-1,2], V[t-1], W[t-1,2], params)

            P[t,2] = get_offer_rate(U[t,2], V[t], params)
            
            if (t>52)and(Uma[t,2]>np.min(Uma[t-52:t,2])+0.005):   #52 weeks == 12 months
                S[t,2]=1
                B[t,2] = benfits[1]
                W[t,2] = get_approximation_value(α_w1, P[t,2], Φ_min, Φ_max, dim=1)
            else: 
                S[t,2]=0
                B[t,2] = benfits[0]
                W[t,2] = get_approximation_value(α_w0, P[t,2], Φ_min, Φ_max, dim=1)
            A[t,2] = (1 - get_lognorm_cdf(W[t,2], params))
            H[t,2] = P[t,2] * A[t,2]
            
    return V, U, P, B, W, A, H, S, Uma


#--------------------------------------------------------------------------------
#-----------------------------------------------------
# SETTING PARAMETER VALUES

# ANNUAL DISCOUNT RATE
r = 0.05

wpa = 365.25/7                      # weeks per year
wpm = 365.25/(12*7)                 # weeks per month
beta_m = 1-((1+r)**(1/12)-1)        # monthly
beta_w = 1-((1+r)**(1/wpa)-1)       # weekly


#-----------------------------------------------
# Shimer (2005) parameters
#- - - - - - - - - - - - - - - - - - - - - - - -
#JOB SEPARATION RATE
delta_m = 0.034                     # monthly
delta_w = 1-(1-delta_m)**(1/wpm)    # weekly
#- - - - - - - - - - - - - - - - - - - - - - - -
#ELASTICITY OF JOB-FINDING RATE WRT. UNEMPLOYMENT IN DMP MATCHING MODEL
alpha = 0.72

#-----------------------------------------------
#WAGE OFFER DISTRIBUTION
# fit lognormal distribution to data moments from https://www.bls.gov/oes/tables.htm
# moments: mean, P10, P25, P50, P75, P90 (P## -> PERCENTILE)
dmoments = np.array([36520, 14750, 19060, 28400, 44500, 67330])

# frictional component of variance of wage offer distribution
factor = 0.1  

p0 = 6, 0.2                    # initial guess
Nw_est = 10**7                 # wage draws for estimation
dmoments = dmoments/(12*wpm)   # calculate weekly equivalence of annual data moments

# get parameters of lognormal distribution to fit data moments 
mu, sigma = fit_lognormal(p0, Nw_est, dmoments)

#calculate frictional wage distribution 
μ_w, σ_w = get_frictional_dist_params(mu, sigma, factor, log=True)

# get wage draws (all variance, frictional component only)
N_w = 40000
w_draws_all_var = get_lognormal_draws(mu, sigma, N_w)
w_draws_frictional = get_lognormal_draws(μ_w, σ_w, N_w)

# create figures (histograms) of both distributions (pdf)
x1 = np.linspace(w_draws_all_var.min(), np.percentile(w_draws_all_var, 97), 1000)
x2 = np.linspace(w_draws_frictional.min(),np.percentile(w_draws_frictional, 99.9), 1000)

fig, ax = plt.subplots(1, 2, figsize=figsize2)
wfactor = 1
ax[0].set_title("All Variance")
(counts, bins) = np.histogram(w_draws_all_var, bins=range(0, int(np.mean(w_draws_all_var)*2.5), 50))
ax[0].hist(bins[:-1], bins, density=True, weights=wfactor*counts, label="wage draws")
ax[0].plot(x1, stats.lognorm.pdf(x1, s=sigma, scale=np.exp(mu)), color="black", label="Model pdf")
ax[0].axvline(w_draws_all_var.mean(), color="brown", label="mean wage")
ax[0].axvline(np.percentile(w_draws_all_var, 10), ls=':', color="darkgrey", label="percentiles (10, 25, 50, 75, 90)")
ax[0].axvline(np.percentile(w_draws_all_var, 25), ls=':', color="darkgrey")
ax[0].axvline(np.percentile(w_draws_all_var, 50), ls=':', color="darkgrey")
ax[0].axvline(np.percentile(w_draws_all_var, 75), ls=':', color="darkgrey")
ax[0].axvline(np.percentile(w_draws_all_var, 90), ls=':', color="darkgrey")
ax[0].set_xlabel("$w$")
ax[0].set_ylabel("$\mathrm{f}(w)$")


ax[1].set_title("Frictional Component - {}% of all variance".format(str(factor*100)))
(counts, bins) = np.histogram(w_draws_frictional, bins=range(0, int(np.mean(w_draws_frictional)*2.5), 50))
ax[1].hist(bins[:-1], bins, density=True, weights=wfactor*counts)
ax[1].plot(x2, stats.lognorm.pdf(x2, s=σ_w, scale=np.exp(μ_w)), color="black") 
ax[1].axvline(w_draws_frictional.mean(), color="brown")
ax[1].axvline(np.percentile(w_draws_frictional, 10), ls=':', color="darkgrey")
ax[1].axvline(np.percentile(w_draws_frictional, 25), ls=':', color="darkgrey")
ax[1].axvline(np.percentile(w_draws_frictional, 50), ls=':', color="darkgrey")
ax[1].axvline(np.percentile(w_draws_frictional, 75), ls=':', color="darkgrey")
ax[1].axvline(np.percentile(w_draws_frictional, 90), ls=':', color="darkgrey")
ax[1].set_xlabel(ax[0].get_xlabel())
ax[1].set_ylabel(ax[0].get_ylabel())
fig.legend(bbox_to_anchor=(0.48,0.888))
fig.savefig(figure_folder+'figure_hist_distributions.png', bbox_inches='tight', dpi=300)
fig.savefig(figure_folder+'figure_hist_distributions.tif', bbox_inches='tight')

# create figure of model distributions and data moments
moments_y = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
mmoments_x = np.array([np.percentile(w_draws_all_var, 10), np.percentile(w_draws_all_var, 25), np.percentile(w_draws_all_var, 50), np.percentile(w_draws_all_var, 75), np.percentile(w_draws_all_var, 90)])
bs = np.linspace(0, 1300, 1300) 
fig2, ax2 = plt.subplots(figsize=(9,6))
ax2.plot(stats.lognorm.cdf(bs, s=sigma, scale=np.exp(mu)), label="estimated distr. - all variance", color="grey")
ax2.axvline(get_lognorm_moments(mu, sigma)[0], color="black", ls="-.", label="mean of model distr.")
ax2.axvline(dmoments[0], color="red", ls="-.", label="mean of data")
ax2.plot(mmoments_x, moments_y, ".", color="black", label="model percentiles")
ax2.plot(dmoments[1:], moments_y, '.', color="red", label="data percentiles")
ax2.plot(stats.lognorm.cdf(bs, s=σ_w, scale=np.exp(μ_w)), label="model wage offer distr.", color="grey", ls=":")
ax2.set_xlabel("$w$")
ax2.set_ylabel("$\mathrm{F}(w)$")
ax2.legend()
fig2.savefig(figure_folder+'figure_cdf_distributions.png', bbox_inches='tight', dpi=300)
fig2.savefig(figure_folder+'figure_cdf_distributions.tif', bbox_inches='tight')
#--------------------------------------------------------------------------------
# SET UP PARAMETER VECTOR - MODEL PERIOD IS ONE WEEK

params_dict = {"β": beta_w,
          "δ": delta_w,
          "λ": 0,               # periods probability that UI decreases
          "c": 0,           # scaling parameter for matching function
          "l": alpha,           # concavity of matching function
          "a": 300,             # lower bound for uniform wages
          "b": 1500,            # upper bound for uniform wages
          "μ": μ_w,             # mean of lognormal wages
          "σ": σ_w,             # standard deviation of lognormal wages
          "ρ_v": 0,             # AR1 autocorrelation of vacancy shock term
          "μ_ε": 0,             # mean of normally distributed shock to vacancies
          "σ_ε": 0}             # standard deviation of normally distributed shock to vacancies

params = np.array(list(params_dict.values())) 
params_names = np.array(list(params_dict.keys()))

dfp = get_params_df(params, params_names)
print()
print("Parameter Values (so far):")
print(dfp)
print()

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
# Final parameter values after correct calibration

params_final_dict = {
    "β": 0.9990645013682375,
    "δ": 0.007923762171326287,
    "λ": 0.9867986798679867,
    "c": 1.8570186594557,
    "l": 0.6298698845546233,
    "a": 300,
    "b": 1500,
    "μ_w": 6.512406449367585,
    "σ_w": 0.2077413029765497,
    "ρ_v": 0.9521847602170885,
    "μ_ε": 0,
    "σ_ε": 0.11432727896698923}

params_final = np.array(list(params_final_dict.values()))
params_final_df = get_params_df(params_final, params_names)

steady_state_values_final_dict = {
    "U": 0.05772549019607839,
    "V": 0.029,
    "B": 309.60959981692207,
    "Φ": 0.2272543186906766,
    "h": 0.1293424982694485,
    "w*": 649.5074138629934}

steady_state_values_final = np.array(list(steady_state_values_final_dict.values()))
steady_state_final_df = get_stst_df(steady_state_values_final)

#--------------------------------------------------------------------------------
# CHEBYSHEV GRID POINTS FOR VALUE FUNCTION
m_B = 15
B_min, B_max = 0.05, 3000
B_grid = get_cheb_grid(B_min, B_max, m_B)
m_Φ = 15
Φ_min, Φ_max = 0, 1
Φ_grid = get_cheb_grid(Φ_min, Φ_max, m_Φ)
params_cheb = np.array([m_B, B_grid, B_min, B_max, m_Φ, Φ_grid, Φ_min, Φ_max])

#grid of wages for reservation wage calculations
w_grid = np.linspace(1, np.max(w_draws_frictional)+2000, 2000000)
#--------------------------------------------------------------------------------
#STEADY STATE PARAMETERS

#replacement rate and benefits
replacement_rate = 0.45
mean_wage = get_lognorm_moments(μ_w, σ_w)[0]
Bbar = mean_wage * replacement_rate # replacement rate on mean wage

# set benefits for two-state policy model to equal steady-state benefits in low-, and mean-wage in high state
benfits = np.array([Bbar, mean_wage]) 

#get steady state unemployment
"""
Hodrick-Prescott smoothing parameter. 
A value of 1600 is suggested for quarterly data. 
Ravn and Uhlig suggest using a value of 6.25 (1600/4**4) for annual data 
and 129600 (1600*3**4) for monthly data.
"""
fred = Fred(api_key='e79d0f51323915f0c9a8bd9cace5ef1b')
Udata = fred.get_series('UNRATE')/100 # get US Unemployment Rate data
dd1, dd2 = '01.01.1953', '31.12.2003' # choose timeframe equal to the one from Shimer (2005)
Udata = Udata.loc[dd1:dd2]            # confine data to chosen period
UHP0 = sm.tsa.filters.hpfilter(Udata, lamb=129600) # get Hodruck-Prescott filtered data

fig3, ax3 = plt.subplots(1, 2, figsize=figsize2)
ax3[0].set_title("I")
ax3[0].plot(Udata*100, label="Original", color="darkcyan")
ax3[0].plot(UHP0[1]*100, label="Trend", ls="-.", color="black")
ax3[0].set_xlabel("Year")
ax3[0].set_ylabel("Percent")
ax3[0].legend()

ax3[1].set_title("II")
ax3[1].plot(UHP0[0]*100, color="darkcyan", label="Deviation from trend")
ax3[1].axhline(0, ls="-.", color="black")
ax3[1].set_xlabel(ax3[0].get_xlabel())
ax3[1].set_ylabel("p.P. deviation from trend")
ax3[1].legend()

fig3.savefig(figure_folder+'figure_urate.png', bbox_inches='tight', dpi=300)
fig3.savefig(figure_folder+'figure_urate.tif', bbox_inches='tight')

# calculate unemployment data moments
Ubar = Udata.mean()
U_SD = np.sqrt(UHP0[0].var())
U_auto = UHP0[0].autocorr()
print("Moments of imported Unemployment Data:")
print("E[U] =\t\t", Ubar)
print("SD[U] =\t\t", U_SD)
print("Autocorr[U] =\t", U_auto)

#vacancies
Vbar = 0.029  # Valetta (2005) data from 1960 - 2005: help-wanted index adjusted using Job Openings and Labor Turnover Survey “JOLTS”

#get JOLTS Series and use HP-filter to get SD and Autocorr. of the Time-Series for comparison
Vdata = fred.get_series('JTSJOR')/100 # get US JOLTS Vacancy Rate data
VHP0 = sm.tsa.filters.hpfilter(Vdata, lamb=129600) # get Hodruck-Prescott filtered data

steady_state_values = np.ones(6)
steady_state_values[0] = Ubar
steady_state_values[1] = Vbar
steady_state_values[2] = Bbar
steady_state_values[4] = params[1]*(1-Ubar) / Ubar

print()
print("Steady-State Values (so far):")
print(get_stst_df(steady_state_values))
#-----------------------------------------------------
#-----------------------------------------------------
# CALIBRATE ELASTICITY PARAMETER OF MATCHING FUNCTION, AND COSTS OF BEING UNEMPLOYED (l and c)

def iterate_c_accurate(c_i, Φ_hat, steady_state_values, w_grid_i, w_draws_i, B_min, B_max, m_B, Φ_grid, Φ_min, Φ_max, m_Φ, params_i):
    global jst
    jst += 1
    print("\t\t'c' iteration:", jst, " \tc =", c_i)
    beta = params_i[0]
    h = steady_state_values[4]
    params_i[3] = c_i
    
    #initial guess
    u_0 = util(np.outer(B_grid, np.ones(m_Φ)))/(1-beta)
    u_i = perform_VFI_u(u_0, w_draws_i, B_grid, Φ_grid, params_i)
    Wres_i = get_Wres(w_grid_i, u_i, params_i)
    α_w_i = approximate_state_function_2D(Wres_i, B_min, B_max, Φ_min, Φ_max, m_B, m_Φ)[0]
    w_res_i = get_approximation_value(α_w_i, steady_state_values[2], B_min, B_max, Φ_hat, Φ_min, Φ_max, dim=2)
    
    h_i = Φ_hat * (1 - get_lognorm_cdf(w_res_i, params_i))
    
    return h_i - h

def iterate_c(c_i, Φ_hat, steady_state_values, w_grid_i, w_draws_i, Φ_grid, Φ_min, Φ_max, m_Φ, params_i):
    global jst
    jst += 1
    print("\t\t'c' iteration:", jst, " \tc =", c_i)
    h = steady_state_values[4]
    params_i[3] = c_i
    u_0_1d = util(steady_state_values[2] * np.ones(m_Φ))
    u0_i = perform_VFI_u0(u_0_1d, steady_state_values[2], w_draws_i, Φ_grid, params_i)
    Wres_i = get_Wres_0(w_grid_i, u0_i, params_i)
    α_w_i = approximate_state_function_1D(Φ_grid, Wres_i, m_Φ)
    w_res_i = get_approximation_value(α_w_i, Φ_hat, Φ_min, Φ_max, dim=1)
    
    h_i = Φ_hat * (1 - get_lognorm_cdf(w_res_i, params_i))
    
    return h_i - h

def iterate_l(l, steady_state_values, w_grid_i, w_draws_i, Φ_grid, Φ_min, Φ_max, m_Φ, params_i):
    global ist
    global jst
    ist += 1
    jst = 0
    print("\t'l' iteration:", ist, "\t L =", l)
    params_i[4] = l
    Ubar, Vbar, Bbar = steady_state_values[:3]
    d = 1e-6

    Φ_hat = get_offer_rate(Ubar, Vbar, params_i)
    Φ_hat_d = get_offer_rate(Ubar+d, Vbar, params_i)
    
    c_i = scipy.optimize.bisect(iterate_c, 0, np.log(Bbar), rtol = 1e-6, args=(Φ_hat, steady_state_values, w_grid_i, w_draws_i, Φ_grid, Φ_min, Φ_max, m_Φ, params_i))
    u_0_1d = util(steady_state_values[2] * np.ones(m_Φ))
    u0_i = perform_VFI_u0(u_0_1d, steady_state_values[2], w_draws_i, Φ_grid, params_i)
    Wres_i = get_Wres_0(w_grid_i, u0_i, params_i)
    α_w_i = approximate_state_function_1D(Φ_grid, Wres_i, m_Φ)
    
    w_res_i_d = get_approximation_value(α_w_i, Φ_hat_d, Φ_min, Φ_max, dim=1)
    w_res_i = get_approximation_value(α_w_i, Φ_hat, Φ_min, Φ_max, dim=1)
    
    h_d = Φ_hat_d * (1 - get_lognorm_cdf(w_res_i_d, params_i))
    h_ = Φ_hat * (1 - get_lognorm_cdf(w_res_i, params_i))
    e_h = (h_d-h_)/d * Ubar / h_
    print()
    print("e_h:", e_h)
    print("--------------------------------")
    print()
    return abs(e_h - (0.72-1))

def get_parameters(steady_state_values, w_grid_i, w_draws_i, B_min, B_max, m_B, Φ_grid, Φ_min, Φ_max, m_Φ, default_calibration, params):
    global ist, jst
    params_i = params.copy()
    if default_calibration == True:
        ist = 0
        jst = 0
        arguments = (steady_state_values, w_grid_i, w_draws_i, Φ_grid, Φ_min, Φ_max, m_Φ, params_i)
        bnds = (0.1,1) 
        bracket = (0.1,1)
        α_w_i = np.zeros(15)
        results = scipy.optimize.minimize_scalar(iterate_l, bounds=bnds, tol=1e-2, args=(arguments), method='bounded')
        print(results)
        l = results.x
        params_i[4] = l
    else:
        l = params_i[4]
    Φ_hat = get_offer_rate(steady_state_values[0], steady_state_values[1], params_i)
    #a, b = 1.4, 2
    a, b = 0, np.log(steady_state_values[2])
    jst = 0
    c = scipy.optimize.bisect(iterate_c_accurate, a, b, rtol = 1e-10, args=(Φ_hat, steady_state_values, w_grid_i, w_draws_i, B_min, B_max, m_B, Φ_grid, Φ_min, Φ_max, m_Φ, params_i))
    return c, l 


if calibrated == True:
    params = params_final.copy()
    steady_state_values = steady_state_values_final.copy()
else:
    if default_calibration == False:
        params[4] = l
    # get parameter for matching function "l" and fit costs of being unemployed to match steady-state hazard rates
    c, l = get_parameters(steady_state_values, w_grid, w_draws_frictional, B_min, B_max, m_B, Φ_grid, Φ_min, Φ_max, m_Φ, default_calibration, params)

        
    params[3] = c
    params[4] = l
    
    # calculate steady-state value of wage-offer rate
    steady_state_values[3] = get_offer_rate(steady_state_values[0], steady_state_values[1], params)


# show parameter, and steady-state values
print("parameter values:")
print(get_params_df(params, params_names))
print()
print("steady-state values:")
print(get_stst_df(steady_state_values))
print()
print("----------------------------------")
#-----------------------------------------------------
#-----------------------------------------------------
#CALCULATE 2D-VALUE FUNCTION OF UNEMPLOYMENT 
# given parameters (β, δ, c, μ, σ) and grid points for B and Φ

print("-> approximate 2D value function")
print()

#initial guess
u_0 = util(np.outer(B_grid, np.ones(m_Φ)))
#define wage draws
w_draws = w_draws_frictional.copy()

#iterate Value Function
u = perform_VFI_u(u_0, w_draws, B_grid, Φ_grid, params)
plot_state_function_2D(B_grid, Φ_grid, u, "u", "uvalue")
#approximate value function
α_u = approximate_state_function_2D(u, B_min, B_max, Φ_min, Φ_max, m_B, m_Φ)[0]

#get reswage-value function
Wres = get_Wres(w_grid, u, params)
plot_state_function_2D(B_grid, Φ_grid, Wres, "w*", "wres")
α_w = approximate_state_function_2D(Wres, B_min, B_max, Φ_min, Φ_max, m_B, m_Φ)[0]

#test: plot smoothness at steady-state value of offer-rate as a function benefits
Bgrid = np.linspace(B_min, B_max, 10000)
Phigrid = np.linspace(Φ_min, Φ_max, 1000)

fig, ax = plt.subplots(1,2, figsize=(16,6))
fig.suptitle("Reservation Wages")
ax[0].set_title("$w^*$ = $f(b)$ with $\phi = \hat{\phi}$")
ax[0].plot(Bgrid, get_approximation_value(α_w, Bgrid, B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2))
ax[0].set_xlabel("b")
ax[0].set_ylabel("$w^*$")

ax[1].set_title("$w^*$ = $f(\phi)$ with $b = \hat{b}$")
ax[1].plot(Phigrid, get_approximation_value(α_w, steady_state_values[2], B_min, B_max, Phigrid, Φ_min, Φ_max, dim=2))
ax[1].set_xlabel("$\phi$")
ax[1].set_ylabel("$w^*$")

# pass steady-state value for reservation wages to parameter array
steady_state_values[5] = get_approximation_value(α_w, steady_state_values[2], B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2)

#get SD of v integral: sqrt(β * Φ * var(v)/N) = 6.602 see Judd 1998 p. 292
W_stst = get_W_0(w_draws_frictional, params, get_approximation_value(α_u, steady_state_values[2], B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2))
v_mean = get_v(W_stst, get_approximation_value(α_u, steady_state_values[2], B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2))
v_var = np.var(np.maximum(get_W_0(w_draws_frictional, params, get_approximation_value(α_u, steady_state_values[2], B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2)), get_approximation_value(α_u, steady_state_values[2], B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2)))
v_SD = np.sqrt(v_var/len(w_draws_frictional))
u_SD = steady_state_values[3] * params[0] / (1-params[0]*(1-steady_state_values[3])) * np.sqrt(v_var)
u_stst = get_approximation_value(α_u, steady_state_values[2], B_min, B_max, steady_state_values[3], Φ_min, Φ_max, dim=2)

#-----------------------------------------------------
#-----------------------------------------------------
#CREATE MATCHING FUNCTION FIGURE

# shimer values
m0 = steady_state_values[4] * (steady_state_values[1]/steady_state_values[0])**(0.72-1)
alpha = 0.72

x = np.linspace(0, 1, 1000)

#get derivative of h wrt. U
d = 1e-6
h_du = get_hazard_rate(steady_state_values[0]+d, steady_state_values[1], steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params)
h_ = get_hazard_rate(steady_state_values[0], steady_state_values[1], steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params)
e_hu = (h_du-h_)/d

h_dv = get_hazard_rate(steady_state_values[0], steady_state_values[1]+d, steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params)
h_ = get_hazard_rate(steady_state_values[0], steady_state_values[1], steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params)
e_hv = (h_dv-h_)/d

print("Model Elasticity of hazard rate wrt. U: \t{:.3f}".format(e_hu * steady_state_values[0] / h_))
print("Shimer Elasticity of job finding rate: \t\t", 0.72-1)
print()

#define line for derivative
dx = 0.1
yl = np.zeros([2,2])
yl[0,0] = h_ - e_hu * dx
yl[0,1] = h_ + e_hu * dx
yl[1,0] = h_ - e_hv * dx
yl[1,1] = h_ + e_hv * dx
xl = np.array([[steady_state_values[0]-dx, steady_state_values[0]+dx],[steady_state_values[1]-dx, steady_state_values[1]+dx]])

# plot figures
fig, ax = plt.subplots(1,2, figsize=(16,6))

ax[0].set_title("I")
ax[0].plot(x, get_hazard_rate(x, steady_state_values[1], steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params), color=colors_vars[4], label="Model")
ax[0].axhline(h_, ls=":", color="grey")
ax[0].plot(x, get_cobb_douglas_job_offer_rate(x, steady_state_values[1], m0, alpha), color="black", label="Shimer (2005)")
ax[0].axvline(steady_state_values[0], ls=":", color="grey", label="Steady-State Values")
ax[0].plot(xl[0,:], yl[0,:], ls="--", color="grey", label="$\partial h / \partial x$ in the Steady-State")
ax[0].set_xlim(0, 0.2)
ax[0].set_xlabel("$U$")
ax[0].set_ylabel("$h$")

ax[1].set_title("II")
ax[1].axhline(h_, ls=":", color="grey")
ax[1].axvline(steady_state_values[1], ls=":", color="grey")
ax[1].plot(x, get_hazard_rate(steady_state_values[0], x, steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params), color=colors_vars[4])
ax[1].plot(x, get_cobb_douglas_job_offer_rate(steady_state_values[0], x, m0, alpha), color="black")
ax[1].plot(xl[1,:], yl[1,:], ls="--", color="grey")
ax[1].set_xlim(0, 0.2)
ax[1].set_xlabel("$V$")
ax[1].set_ylabel(ax[0].get_ylabel())
fig.legend(bbox_to_anchor=(0.48,0.888))

fig.savefig(figure_folder+'figure_uexit_rates.png', bbox_inches='tight', dpi=300)
fig.savefig(figure_folder+'figure_uexit_rates.tif', bbox_inches='tight')

#-----------------------------------------------------
#CREATE MATCHING FUNCTION FIGURE - WAGE-OFFER RATE

x = np.linspace(0, 1, 1000)

#get derivative of h wrt. U
d = 1e-6
p_d = get_offer_rate(steady_state_values[0]+d, steady_state_values[1], params)
p_ = get_offer_rate(steady_state_values[0], steady_state_values[1], params)
e_p = (p_d-p_)/d

print("Model Elasticity of wage-offer rate wrt. U: \t", e_p * steady_state_values[0] / p_)

# plot figures
fig, ax = plt.subplots(1,2, figsize=(16,6))

ax[0].set_title("I")
ax[0].plot(x, get_offer_rate(x, steady_state_values[1], params), color=colors_vars[3], label="Wage-Offer Rates")
ax[0].plot(x, get_hazard_rate(x, steady_state_values[1], steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params), color=colors_vars[4], label="Hazard Rates")
ax[0].plot(x, get_hazard_rate(x, steady_state_values[1], steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params)/get_offer_rate(x, steady_state_values[1], params), color=colors_vars[6], label="Offer Acceptance Rates")
#ax[0].axhline(p_, ls=":", color="grey")
ax[0].axvline(steady_state_values[0], ls=":", color="grey", label="Steady-State Value")
#ax[0].plot(xl, yl, ls="--", color="grey", label="Elasticity")
ax[0].set_xlim(0, 0.3)
ax[0].set_xlabel("$U$")
ax[0].set_ylabel("Rate")

ax[1].set_title("II")
#ax[1].axhline(p_, ls=":", color="grey")
ax[1].axvline(steady_state_values[1], ls=":", color="grey")
ax[1].plot(x, get_offer_rate(steady_state_values[0], x, params), color=colors_vars[3])
ax[1].plot(x, get_hazard_rate(steady_state_values[0], x, steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params), color=colors_vars[4])
ax[1].plot(x, get_hazard_rate(steady_state_values[0], x, steady_state_values[2], B_min, B_max, Φ_min, Φ_max, α_w, params)/get_offer_rate(steady_state_values[0], x, params), color=colors_vars[6])
ax[1].set_xlim(ax[0].get_xlim())
ax[1].set_xlabel("$V$")
ax[1].set_ylabel(ax[0].get_ylabel())

fig.legend(bbox_to_anchor=(0.9,0.88))#loc='upper right'

fig.savefig(figure_folder+'figure_model_rates.png', bbox_inches='tight', dpi=300)
fig.savefig(figure_folder+'figure_model_rates.tif', bbox_inches='tight')
#-----------------------------------------------------
# CALIBRATE PARAMETERS OF THE SHOCK PROCESS

def iteration_SSM(x, T, w, dm, benfits, steady_state_values, skip, α, B_min, B_max, Φ_min, Φ_max, v_shock_s, paramsi):
    global j
    j+=1
    print(j, "\trho, epsilon:\t", x)
    print()
    rho, epsilon = x
    paramsi[9]  = rho
    paramsi[11] = epsilon
    
    U = simulate_model(T, benfits, steady_state_values, α, B_min, B_max, Φ_min, Φ_max, paramsi, v_shock_s, policies=False)[1]
    dfu = pd.Series(U[skip:,0], index=pd.date_range('01/01/2000', periods=len(U[skip:,0]), freq='W'))
    dfum = dfu.resample('M').mean()
    
    # get model moment conditions
    mm[0] = np.sqrt(dfum.var())
    mm[1] = dfum.autocorr()
    # calculate percent deviation of model from data moment
    m = (mm - dm)/dm
    print("\tmm :\t\t", mm)
    print("\tdm :\t\t", dm)
    print("\tPercent Deviation:\t", m)
    print()
    Q = m.T @ w @ m
    print("\tQ =\t\t", Q)
    print("\t-------------------------------------")
    print()
    return Q

#get approximation coefficients
params[2] = 1
α = get_2stage_vfkt_coeffs(α_u, α_w, benfits, w_draws_frictional, B_min, B_max, Φ_grid, Φ_min, Φ_max, params)

# ititial guess
rho=0.93
epsilon=0.1
x0 = np.array([rho, epsilon])

# use a copy of the parameter array for calculation
paramsi = params.copy()

# periods used for simulation
T = int(1e4)

# create a standard-normal shock process of lenght T for the simulation
np.random.seed(123456)
v_shock_s = np.random.normal(0, 1, T+1)

# get data moments
dm = np.array([np.sqrt(UHP0[0].var()), UHP0[0].autocorr()])
#initialize arrays 
mm = np.zeros(2)
m = np.zeros(2)

j = 0 # counter (global variable)
w = np.eye(2) # identity matrix for weighting
skip = 100 # skip the first 100 periods to forget steady-state

print("estimate shock process parameters: rho, sigma")
arguments = (T, w, dm, benfits, steady_state_values, skip, α, B_min, B_max, Φ_min, Φ_max, v_shock_s, paramsi)
bnds = ((0.5,0.99), (0.005, 0.3)) # bounds on the estimated parameters
results = scipy.optimize.minimize(iteration_SSM, x0, args=(arguments),
                          method='L-BFGS-B',
                          bounds=(bnds))

# pass results to global array of parameters 
params[9], params[11] = results.x
print()
print("rho =", params[9], "| sigma =", params[11])

#-----------------------------------------------------
# SIMULATE THE MODEL
T = 10000
np.random.seed(1872)
v_shock_s = np.random.normal(0, 1, T+1)

params_i = params.copy()
params_i[2] = 1
α2 = get_2stage_vfkt_coeffs(α_u, α_w, benfits, w_draws, B_min, B_max, Φ_grid, Φ_min, Φ_max, params_i)
V2, U2, P2, B2, W2, A2, h2, S2, Uma2 = simulate_model(T, benfits, steady_state_values, α2, B_min, B_max, Φ_min, Φ_max, params, v_shock_s, policies=True)

change = np.where(S2[:-1,2] != S2[1:,2])[0]
ST = []
for i in range(0, len(change)-2,2):
    ST.append(change[i+1]+1-change[i]+1)
STa = np.array(ST)

params[2] = 1-1/STa.mean()
α = get_2stage_vfkt_coeffs(α_u, α_w, benfits, w_draws, B_min, B_max, Φ_grid, Φ_min, Φ_max, params)
V, U, P, B, W, A, h, S, Uma = simulate_model(T, benfits, steady_state_values, α, B_min, B_max, Φ_min, Φ_max, params, v_shock_s, policies=True)

dfw = pd.DataFrame(np.stack((V, U[:,0], U[:,1], U[:,2], B[:,0], B[:,1], B[:,2], P[:,0], P[:,1], P[:,2], A[:,0], A[:,1], A[:,2], h[:,0], h[:,1], h[:,2]), axis=0).T, columns=["$V$", "$U_0$", "$U_1$", "$U_2$", "$B_0$", "$B_1$", "$B_2$", "$\phi_0$", "$\phi_1$", "$\phi_2$", "$a_0$", "$a_1$", "$a_2$", "$h_0$", "$h_1$", "$h_2$"], index=pd.date_range('01/01/2000', periods=len(U[:,0]), freq='W'))
dfm = dfw.resample('M').mean()

#------------------------------------------------------
#GENERATE PARAMETERS TABLE

params_string = (r"\begin{tabular}[h]{lr}""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"\multicolumn{2}{c}{General} \\""\n"+ 
                 "\t"r"\hline""\n"+
                 "\t"r"$\beta$ &  ${:,.4f}$ \\".format(params[0])+"\n"+
                 "\t"r"$\delta$ & ${:,.4f}$ \\".format(params[1])+"\n"+
                 "\t"r"$\lambda$ & ${:,.4f}$ \\".format(params[2])+"\n"+
                 "\t"r"\\"+"\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"\multicolumn{2}{c}{Matching function and Cost Parameter}\\"+"\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"$l$ & ${:,.4f}$ \\".format(params[4])+"\n"+
                 "\t"r"$c$ & ${:,.4f}$ \\".format(params[3])+"\n"+
                 "\t"r"\\""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"\multicolumn{2}{c}{Wage-Offer Distribution} \\""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"\textit{{$\mu_w$}} & ${:,.4f}$ \\".format(params[7])+"\n"+
                 "\t"r"\textit{{$\sigma_w$}} & ${:,.4f}$ \\".format(params[8])+"\n"+
                 "\t"r"\\""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"\multicolumn{2}{c}{Shock Process to Vacancies} \\""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"$\sigma_{{\epsilon}}$ & ${:,.4f}$ \\".format(params[11])+"\n"+
                 "\t"r"$\rho_v$ & ${:,.4f}$ \\".format(params[9])+"\n"+
                 "\t"r"\\""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"\multicolumn{2}{c}{Steady-State Values} \\""\n"+
                 "\t"r"\hline""\n"+
                 "\t"r"$\hat{{b}} = b^L$ & ${:,.2f}$ \\".format(benfits[0])+"\n"+
                 "\t"r"$b^H$ & ${:,.2f}$ \\".format(benfits[1])+"\n"+
                 "\t"r"$\hat{{V}}$ & ${:,.4f}$ \\".format(steady_state_values[1])+"\n"+
                 "\t"r"$\hat{{U}}$ & ${:,.4f}$ \\".format(steady_state_values[0])+"\n"+
                 "\t"r"$\hat{{\phi}}$ & ${:,.4f}$ \\".format(steady_state_values[3])+"\n"+
                 "\t"r"$\hat{{a}}$ & ${:,.4f}$ \\".format(1-get_lognorm_cdf(steady_state_values[5], params))+"\n"+
                 "\t"r"$\hat{{h}}$ & ${:,.4f}$ \\".format(steady_state_values[4])+"\n"+
                 "\t"r"$\hat{{w}}^*$ & ${:,.2f}$ \\".format(steady_state_values[5])+"\n"+
                 "\t"r"\hline""\n"+
                 r"\end{tabular}""\n"+
                 r"\\")

params_file = open(table_folder+"table_parameters.tex", "w") 
params_file.write(params_string)
params_file.close()
#------------------------------------------------------
# CREATE CORRELATION TABLES

columns = list([["$V$", "$U_0$", "$B_0$", "$\phi_0$", "$a_0$", "$h_0$"],["$V$", "$U_1$", "$B_1$", "$\phi_1$", "$a_1$", "$h_1$"],["$V$", "$U_2$", "$B_2$", "$\phi_2$", "$a_2$", "$h_2$"]])

dfw0 = dfw.loc[:,columns[0]]
dfw1 = dfw.loc[:,columns[1]]
dfw2 = dfw.loc[:,columns[2]]

collection = [dfw0, dfw1, dfw2]
corr_collection = []
for item in collection:
    dfw_corr = item.corr()
    dfw_corr = dfw_corr.applymap("{:,.3f}".format)
    for li in range(len(dfw_corr)):
        for lj in range(li):
            dfw_corr.iloc[li,lj] = ""
    corr_collection.append(dfw_corr)
corr_collection[0]

latex = []
toprule = '\\toprule\n'
bottomrule = '\\\\\n\\bottomrule\n'
Bautocorr = [1.0, collection[1].loc[:, "$B_1$"].autocorr(), collection[2].loc[:, "$B_2$"].autocorr()]
for i in range(3):
    latex_list = list(corr_collection[i].to_latex(index=True).replace(str(toprule), "").replace(str(bottomrule), "").replace("\\$V\\$", "$V$").replace("\\$U\\_{}\\$".format(i), "$U_{}$".format(i)).replace("\\$B\\_{}\\$".format(i), "$B_{}$".format(i)).replace("\\$\\textbackslash phi\\_{}\\$".format(i), "$\phi_{}$".format(i)).replace("\\$a\\_{}\\$".format(i), "$a_{}$".format(i)).replace("\\$h\\_{}\\$".format(i), "$h_{}$".format(i)).replace("lllllll", "c|rrrrrr"))
    latex_list.insert(529, "\\\\ \n\hline \nSD & {:,.4f} & {:,.4f} & {:06.2f} & {:,.4f} & {:,.4f} & {:,.4f} \\\\\n".format(np.sqrt(collection[i].loc[:, "$V$".format(i)].var()), np.sqrt(collection[i].loc[:, "$U_{}$".format(i)].var()), np.sqrt(collection[i].loc[:, "$B_{}$".format(i)].var()), np.sqrt(collection[i].loc[:, "$\phi_{}$".format(i)].var()), np.sqrt(collection[i].loc[:, "$a_{}$".format(i)].var()), np.sqrt(collection[i].loc[:, "$h_{}$".format(i)].var())))
    latex_list.insert(530, "Autocorr. & {:,.4f} & {:,.4f} & {:,.4f} & {:,.4f} & {:,.4f} & {:,.4f} \\\\\n".format(collection[i].loc[:, "$V$"].autocorr(), collection[i].loc[:, "$U_{}$".format(i)].autocorr(), Bautocorr[i], collection[i].loc[:, "$\phi_{}$".format(i)].autocorr(), collection[i].loc[:, "$a_{}$".format(i)].autocorr(), collection[i].loc[:, "$h_{}$".format(i)].autocorr()))
    if i == 1:
        section = latex_list[523:529]
        del latex_list[523:529]
        latex_list.insert(525, ''.join(section))
    latex.append(''.join(latex_list))
    corr_file = open(table_folder+"table_correlations_{}.tex".format(i), "w") 
    corr_file.write(latex[i])
    corr_file.close()

#-------------------------------------------------  
# WRITE LATEX VACANCY MOMENTS TABLE (MONTHLY VALUES)

corr0_string = (r"\begin{tabular}[h]{c | c c c}""\n"+
                 "\t"r"& \textbf{Model} & \multicolumn{2}{c}{\textbf{JOLTS}} \\""\n"+ 
                 "\t"r"Periods & \textit{all} & \textit{2000-2005} & \textit{2000-2019} \\""\n"+ 
                 "\t"r"\hline"+"\n"
                 "\t"r"$\sigma_v$ & {:,.4f} & {:,.4f} & {:,.4f}\\".format(np.sqrt(dfm.loc[:,"$V$"].var()), np.sqrt(VHP0[0].iloc[:61].var()), np.sqrt(VHP0[0].iloc[:-14].var()))+"\n"+
                 "\t"r"$\gamma_v$ & {:,.4f} & {:,.4f} & {:,.4f}\\".format(dfm.loc[:,"$V$"].autocorr(), VHP0[0].iloc[:61].autocorr(), VHP0[0].iloc[:-14].autocorr())+"\n"+
                 r"\end{tabular}")

corr0_file = open(table_folder+"table_vacancy_moments.tex", "w") 
corr0_file.write(corr0_string)
corr0_file.close()

#-------------------------------------------------
    
start = 800
t1, t2 = 0, 1000
#t1, t2 = 0, T
fig, ax = plt.subplots(3,2, figsize=(16,17))
fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)
fig2, ax2 = plt.subplots(2,2, figsize=(14,10))
fig3, ax3 = plt.subplots(1,2, figsize=figsize2)
fig4, ax4 = plt.subplots(2,2, figsize=(16,9))
fig4.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.35)
fig5, ax5 = plt.subplots(1,2, figsize=figsize2)

#plt.suptitle("Simulation of different Policies",fontsize=16)
change2 = np.where(S[:-1,2] != S[1:,2])[0]-start
change1 = np.where(S[:-1,1] != S[1:,1])[0]-start
change0 = np.where(S[:-1,0] != S[1:,0])[0]-start
ST0 = []
ST1 = []
ST2 = []
for i in range(0, len(change)-2,2):
    ax[0,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax[0,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax[1,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax[1,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax[2,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax[2,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax2[0,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax2[0,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax2[1,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax2[1,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax3[0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax3[1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax4[0,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax4[0,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax4[1,0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax4[1,1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax5[0].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ax5[1].axvspan(change2[i]+1, change2[i+1]+1, alpha=0.2, color='grey')
    ST2.append(change2[i+1]+1-change2[i]+1)
    ST0.append(change0[i+1]+1-change0[i]+1)
    ST1.append(change1[i+1]+1-change1[i]+1)

STa0 = np.array(ST0)
STa1 = np.array(ST1)
STa2 = np.array(ST2)
print("Mean duration of high-state in Baseline:", (STa0.mean()/wpm), "months")
print("Mean duration of high-state in Constant Benefit:", (STa1.mean()/wpm), "months")
print("Mean duration of high-state in Binary-State:", (STa2.mean()/wpm), "months")

X = np.arange(0-start, len(V)-start)

ax[0,0].set_title("Vacancies")
ax[0,0].plot(X, V*100, color="black", label="V", zorder=3)
ax[0,0].axhline(steady_state_values[1]*100, color="black", ls=':', label="Steady-State Value")
ax[0,0].set_xlim(t1,t2)
minim, maxim = np.min(V[t1:t2+1]*100), np.max(V[t1:t2+1]*100)
ax[0,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
#ax[0,0].legend(loc='upper left')
ax[0,0].set_ylabel("$\%$")
ax[0,0].set_xlabel("Weeks")

ax[0,1].set_title("Unemployment Rates")
ax[0,1].plot(X, U[:,0]*100, color=colors[0], label="no Policy", zorder=3)
ax[0,1].plot(X, U[:,1]*100, color=colors[1], label="constant $w^*$")
ax[0,1].plot(X, U[:,2]*100, color=colors[2], label="Binary-State")
ax[0,1].axhline(steady_state_values[0]*100, color="black", ls=':', label="Steady-State Value")
#ax[0,1].plot(np.roll(Uma, -1), color="grey", label="12 months MA", ls='dotted')
ax[0,1].set_xlim(t1,t2)
minim, maxim = np.min(U[t1:t2+1,:]*100), np.max(U[t1:t2+1,:]*100)
ax[0,1].set_ylim(minim-(maxim-minim)/4+0.01, maxim+(maxim-minim)/4+0.01)
ax[0,1].set_ylabel("$\%$")
ax[0,1].set_xlabel(ax[0,0].get_xlabel())
ax[0,1].legend(loc="upper right")

ax[1,0].set_title("Benefits")
ax[1,0].plot(X, B[:,0]/mean_wage, color=colors[0], label="no policy", zorder=3)
ax[1,0].plot(X, B[:,1]/mean_wage, color=colors[1], label="constant $w^*$ Policy")
ax[1,0].plot(X, B[:,2]/mean_wage, color=colors[2], label="Binary-State Policy")
ax[1,0].set_xlim(t1,t2)
minim, maxim = np.min(B[t1:t2+1,:]/mean_wage), np.max(B[t1:t2+1,:]/mean_wage)
ax[1,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax[1,0].set_ylabel("Factor of mean wages")
ax[1,0].set_xlabel(ax[0,0].get_xlabel())


ax[1,1].set_title("Reservation Wages")
ax[1,1].plot(X, A[:,0]*100, color=colors[0], label="constant $b$", zorder=3)
ax[1,1].plot(X, A[:,1]*100, color=colors[1], label="endogenous $b$")
ax[1,1].plot(X, A[:,2]*100, color=colors[2], label="binary-state")
#ax[1,1].axhline(W[:,0].mean(), color="grey", label="mean Value")
ax[1,1].set_xlim(t1,t2)
minim, maxim = np.min(A[t1:t2+1,:]*100), np.max(A[t1:t2+1,:]*100)
ax[1,1].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax[1,1].set_ylabel("$\%$")
ax[1,1].set_xlabel(ax[0,0].get_xlabel())
#ax[1,1].legend()

ax[2,0].set_title("Hazard Rates")
ax[2,0].plot(X, h[:,0]*100, color=colors[0], label="no policy", zorder=3)
ax[2,0].plot(X, h[:,1]*100, color=colors[1], label="constant $w^*$")
ax[2,0].plot(X, h[:,2]*100, color=colors[2], label="Binary-State")
ax[2,0].axhline(steady_state_values[4]*100, color="black", ls=':', label="Steady-State Value")
ax[2,0].set_xlim(t1,t2)
minim, maxim = np.min(h[t1:t2+1,:]*100), np.max(h[t1:t2+1,:]*100)
ax[2,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax[2,0].set_ylabel("$\%$")
ax[2,0].set_xlabel(ax[0,0].get_xlabel())

ax[2,1].set_title("Job-Offer Rates")
ax[2,1].plot(X, P[:,0]*100, color=colors[0], label="no policy", zorder=3)
ax[2,1].plot(X, P[:,1]*100, color=colors[1], label="constant $w^*$")
ax[2,1].plot(X, P[:,2]*100, color=colors[2], label="Binary-State")
ax[2,1].axhline(steady_state_values[3]*100, color="black", ls=':', label="Steady-State Value")
ax[2,1].set_xlim(t1,t2)
minim, maxim = np.min(P[t1:t2+1,:]*100), np.max(P[t1:t2+1,:]*100)
ax[2,1].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax[2,1].set_ylabel("$\%$")
ax[2,1].set_xlabel(ax[0,0].get_xlabel())

handles, labels = ax[2,0].get_legend_handles_labels()
#fig.legend(handles, labels, bbox_to_anchor=(0.905,0.884))#loc='lower center'
fig.savefig(figure_folder+'figure_simulation.png', bbox_inches='tight', dpi=300)
fig.savefig(figure_folder+'figure_simulation.tif', bbox_inches='tight')


ax3[0].set_title("I")
a = (A-(steady_state_values[4]/steady_state_values[3]))*100
ax3[0].plot(X, a[:,0], color= colors_vars[6], label="$a$")
ax3[0].plot(X, (P[:,0]-steady_state_values[3])*100, color= colors_vars[3], label="$\phi$")
ax3[0].plot(X, (h[:,0]-steady_state_values[4])*100, color= colors_vars[4], label="$h$")
ax3[0].axhline(0, ls=":", color="black", label="Steady-State")
ax3[0].set_xlim(t1, t2)
ax3[0].set_ylabel('p.P. deviation from Steady-State')
ax3[0].set_xlabel(ax[0,0].get_xlabel())
minim, maxim = np.min(a[t1:t2,:]), np.max(a[t1:t2,:])
ax3[0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
#ax3[0].legend(loc="upper left")

ax3[1].set_title("II")
a = (A-(steady_state_values[4]/steady_state_values[3]))*100
ax3[1].plot(X, a[:,1], color= colors_vars[6], label="$a$")
ax3[1].plot(X, (P[:,1]-steady_state_values[3])*100, color= colors_vars[3], label="$\phi$")
ax3[1].plot(X, (h[:,1]-steady_state_values[4])*100, color= colors_vars[4], label="$h$")
ax3[1].axhline(0, ls=":", color="black", label="Steady-State")
ax3[1].set_xlim(t1, t2)
ax3[1].set_ylabel('p.P. deviation from Steady-State')
ax3[1].set_xlabel(ax[0,0].get_xlabel())
minim, maxim = np.min(a[t1:t2,:]), np.max(a[t1:t2,:])
ax3[1].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax3[1].legend(loc="upper left")

fig3.savefig(figure_folder+'figure_sim_rates.png', bbox_inches='tight', dpi=300)
fig3.savefig(figure_folder+'figure_sim_rates.tif', bbox_inches='tight')

#--------------------------------------------------------------------------------------
ax2[0,0].set_title("Unemployment Rates")
ax2[0,0].axhline(0, color=colors[0])
ax2[0,0].plot(X, (U[:,1]-U[:,0])*100, color=colors[1], label="constant $w^*$")
ax2[0,0].plot(X, (U[:,2]-U[:,0])*100, color=colors[2], label="Binary-State with adaptive Expectations")
ax2[0,0].plot(X, (U2[:,2]-U[:,0])*100, color=colors[2], ls=":", label="Binary-State without Expectations")
minim, maxim = np.min((U[t1:t2,1]-U[t1:t2,0])*100), np.max((U[t1:t2,1]-U[t1:t2,0])*100)
ax2[0,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax2[0,0].set_xlim(t1,t2)
ax2[0,0].set_ylabel("p. P. Deviation from $U_0$")
ax2[0,0].set_xlabel(ax[0,0].get_xlabel())
ax2[0,0].legend()

#ax2[0,0].set_xlim(t1,t2)
minim, maxim = np.min(U[t1:t2,:]-np.outer(U[t1:t2,0],np.ones(3))), np.max(U[t1:t2,:]-np.outer(U[t1:t2,0],np.ones(3)))
#ax2[0,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax2[0,1].set_title("Offer-Acceptance Rates")
ax2[0,1].axhline(0, color=colors[0], label="no policy")
ax2[0,1].plot(X, (A[:,1]-A[:,0])*100, color=colors[1], label="constant $w^*$ Policy")
ax2[0,1].plot(X, (A[:,2]-A[:,0])*100, color=colors[2], label="Binary-State with adaptive Expectations")
ax2[0,1].plot(X, (A2[:,2]-A[:,0])*100, color=colors[2], ls=":", label="Binary-State without Expectations")
ax2[0,1].set_xlim(t1,t2)
minim, maxim = np.min((A[t1:t2,:]-np.outer(A[t1:t2,0],np.ones(3)))*100), np.max((A[t1:t2,:]-np.outer(A[t1:t2,0],np.ones(3)))*100)
ax2[0,1].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax2[0,1].set_xlabel(ax[0,0].get_xlabel())
ax2[0,1].set_ylabel("p. P. Deviation from $U_0$")

ax2[1,0].set_title("Hazard Rates")
ax2[1,0].axhline(0, color=colors[0], label="no policy")
ax2[1,0].plot(X, (h[:,1]-h[:,0])*100, color=colors[1], label="constant $w^*$")
ax2[1,0].plot(X, (h[:,2]-h[:,0])*100, color=colors[2], label="Binary-State - adaptive $\lambda$")
ax2[1,0].plot(X, (h2[:,2]-h[:,0])*100, color=colors[2], ls=":", label="Binary-State - $\lambda=1$")
ax2[1,0].set_xlim(t1,t2)
minim, maxim = np.min((h[t1:t2,:]-np.outer(h[t1:t2,0],np.ones(3)))*100), np.max((h[t1:t2,:]-np.outer(h[t1:t2,0],np.ones(3)))*100)
ax2[1,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax2[1,0].set_xlabel(ax[0,0].get_xlabel())
ax2[1,0].set_ylabel("p. P. Deviation from $U_0$")

ax2[1,1].set_title("Wage-Offer Rates")
ax2[1,1].axhline(0, color=colors[0], label="no policy")
ax2[1,1].plot(X, (P[:,1]-P[:,0])*100, color=colors[1], label="constant $w^*$")
ax2[1,1].plot(X, (P[:,2]-P[:,0])*100, color=colors[2], label="Binary-State - adaptive $\lambda$")
ax2[1,1].plot(X, (P2[:,2]-P[:,0])*100, color=colors[2], ls=":", label="Binary-State - $\lambda=1$")
ax2[1,1].set_xlim(t1,t2)
ax2[1,1].set_xlabel(ax[0,0].get_xlabel())
minim, maxim = np.min((P[t1:t2,:]-np.outer(P[t1:t2,0],np.ones(3)))*100), np.max((P[t1:t2,:]-np.outer(P[t1:t2,0],np.ones(3)))*100)
#ax2[1,1].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax2[1,1].set_ylim(ax2[1,0].get_ylim())
ax2[1,1].set_ylabel("p. P. Deviation from $U_0$")

fig2.savefig(figure_folder+'figure_expectations.png', bbox_inches='tight', dpi=300)
fig2.savefig(figure_folder+'figure_expectations_rates.tif')
#---------------------------------------------------------------------------------------------------------------------

ax5[0].set_title("Constant $w^*$-Policy")
ax5[0].axhline(0, color="black", ls="-.", label="no policy")
ax5[0].plot(X, (h[:,1]-h[:,0])*100, color=colors_vars[4], label="$\Delta h$")
ax5[0].plot(X, (P[:,1]-P[:,0])*100, color=colors_vars[3], label="$\Delta \phi$")
ax5[0].plot(X, (A[:,1]-A[:,0])*100, color=colors_vars[6], label="$\Delta a$")
ax5[0].set_xlim(t1,t2)
minim, maxim = np.min((A[t1:t2,1]-A[t1:t2,0])*100), np.max((A[t1:t2,1]-A[t1:t2,0])*100)
ax5[0].set_ylim(minim-(maxim-minim)/8, maxim+(maxim-minim)/8)
ax5[0].set_xlabel(ax[0,0].get_xlabel())
ax5[0].set_ylabel("p.P. Deviation from no-Policy Rate")

ax5[1].set_title("Binary-State Benefits")
ax5[1].axhline(0, color="black", ls="-.")
ax5[1].plot(X, (h[:,2]-h[:,0])*100, color=colors_vars[4], label="$\Delta h$")
ax5[1].plot(X, (P[:,2]-P[:,0])*100, color=colors_vars[3], label="$\Delta \phi$")
ax5[1].plot(X, (A[:,2]-A[:,0])*100, color=colors_vars[6], label="$\Delta a$")
ax5[1].set_xlim(t1,t2)
minim, maxim = np.min((A[t1:t2,2]-A[t1:t2,0])*100), np.max((A[t1:t2,2]-A[t1:t2,0])*100)
ax5[1].set_ylim(ax5[0].get_ylim())
ax5[1].set_xlabel(ax[0,0].get_xlabel())
ax5[1].set_ylabel(ax5[0].get_ylabel())
ax5[1].legend()
fig5.savefig(figure_folder+'figure_deviation_rates.png', bbox_inches='tight', dpi=300)
fig5.savefig(figure_folder+'figure_deviation_rates.tif')

#----------------------------------------------------------------------
ax4[0,0].set_title("no Policy")
a = (A-(steady_state_values[4]/steady_state_values[3]))*100
ax4[0,0].plot(X, a[:,0], color= colors_vars[6], label="$a$")
ax4[0,0].plot(X, (P[:,0]-steady_state_values[3])*100, color= colors_vars[3], label="$\phi$")
ax4[0,0].plot(X, (h[:,0]-steady_state_values[4])*100, color= colors_vars[4], label="$h$")
ax4[0,0].axhline(0, ls=":", color="black", label="Steady-State")
ax4[0,0].set_xlim(t1, t2)
ax4[0,0].set_ylabel('p.P. deviation from Steady-State')
ax4[0,0].set_xlabel(ax[0,0].get_xlabel())
minim, maxim = np.min(a[t1:t2,:]), np.max(a[t1:t2,:])
ax4[0,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax4[0,0].legend(loc="upper left")

ax4[0,1].set_title("Binary-State Benefits")
ax4[0,1].plot(X, a[:,2], color = colors_vars[6], label="$a$")
ax4[0,1].plot(X, (P[:,2]-steady_state_values[3])*100, color= colors_vars[3], label="$\phi$")
ax4[0,1].plot(X, (h[:,2]-steady_state_values[4])*100, color= colors_vars[4], label="$h$")
ax4[0,1].axhline(0, ls=":", color="black", label="Steady-State")
ax4[0,1].set_xlim(t1, t2)
ax4[0,1].set_ylabel('p.P. deviation from Steady-State')
ax4[0,1].set_xlabel(ax[0,0].get_xlabel())
minim, maxim = np.min(a[t1:t2,:]), np.max(a[t1:t2,:])
ax4[0,1].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)

ax4[1,0].set_title("Constant $w^*$-Policy")
ax4[1,0].plot(X, a[:,1], color= colors_vars[6], label="$a$")
ax4[1,0].plot(X, (P[:,1]-steady_state_values[3])*100, color= colors_vars[3], label="$\phi$")
ax4[1,0].plot(X, (h[:,1]-steady_state_values[4])*100, color= colors_vars[4], label="$h$")
ax4[1,0].axhline(0, ls=":", color="black", label="Steady-State")
ax4[1,0].set_xlim(t1, t2)
ax4[1,0].set_ylabel('p.P. deviation from Steady-State')
ax4[1,0].set_xlabel(ax[0,0].get_xlabel())
minim, maxim = np.min(a[t1:t2,:]), np.max(a[t1:t2,:])
ax4[1,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax4[1,0].legend(loc="upper left")

ax4[1,1].set_title("Binary-State vs. no Policy")
ax4[1,1].plot(X, a[:,2]-a[:,0], color=colors_vars[6])
ax4[1,1].plot(X, (P[:,2]-steady_state_values[3])*100-(P[:,0]-steady_state_values[3])*100, color=colors_vars[3])
ax4[1,1].plot(X, (h[:,2]-steady_state_values[4])*100-(h[:,0]-steady_state_values[4])*100, color=colors_vars[4])
minim, maxim = np.min(a[t1+start:t2+start,2]-a[t1+start:t2+start,0]), np.max(a[t1+start:t2+start,2]-a[t1:t2,0])
ax4[1,0].set_ylim(minim-(maxim-minim)/4, maxim+(maxim-minim)/4)
ax4[1,1].set_xlim(t1,t2)
ax4[1,1].set_ylabel("p.P. Difference in Rates")
ax4[1,1].set_xlabel(ax[0,0].get_xlabel())

fig4.savefig(figure_folder+'figure_sim_rates_policy.png', bbox_inches='tight', dpi=300)
fig4.savefig(figure_folder+'figure_sim_rates_policy.tif')

#---------------------------------------------------------------------
#BEVERIDGE CURVE SCATTERPLOT
tt=10000
fig, ax = plt.subplots()
ax.plot(U[start:start+tt,0], V[start:start+tt], '.')
ax.set_ylim(0,0.11)
ax.set_xlim(0.042,0.105)
ax.set_xlabel("$U$")
ax.set_ylabel("$V$")
ax.set_title("Beveridge Curve")
fig.savefig(figure_folder+"figure_beveridge_curve.png")

#---------------------------------------------------------------------
#IMPULSE RESPONSE FUNCTIONS
def get_impulse_responses(T, stst_values, α_u, α_w, v_shock, b_shock, t_hs, params_i):
    """
    """
    V = np.zeros(T)
    U = np.zeros([T,3])
    P = np.zeros([T,3])
    W = np.zeros([T,3])
    A = np.zeros([T,3])
    H = np.zeros([T,3])
    B = np.zeros([T,3])
    
    params_i[2] = 1-(1/t_hs)
    print("lambda =", params_i[2])
    benfits_i = np.array([steady_state_values[2], steady_state_values[2]*(1+b_shock/100)])
    α = get_2stage_vfkt_coeffs(α_u, α_w, benfits_i, w_draws, B_min, B_max, Φ_grid, Φ_min, Φ_max, params_i)
    α_w0, α_w1 = α[1:]
    
    U0, V0, B0, Φ0, h0, wres0 = stst_values
    A0 = 1-get_lognorm_cdf(wres0, params_i)
    V[0] = V0 * (1+v_shock/100)
    U[0,:] = U0
    B[:,0] = B0
    B[:t_hs,2] = benfits_i[1]
    B[t_hs:,2] = benfits_i[0]
    
    for t in range(T):
        P[t,0] = get_offer_rate(U[t,0], V[t], params_i)
        W[t,0] = get_approximation_value(α_w, B0, B_min, B_max, P[t,0], Φ_min, Φ_max, dim=2)
        A[t,0] = 1 - get_lognorm_cdf(W[t,0], params_i)
        H[t,0] = P[t,0] * A[t,0]
        
        B[t,1] = get_benefit_policy(U[t,1], V[t], α_w, wres0, B_min, B_max, Φ_min, Φ_max, params_i)
        P[t,1] = get_offer_rate(U[t,1], V[t], params_i)
        W[t,1] = get_approximation_value(α_w, B[t,1], B_min, B_max, P[t,1], Φ_min, Φ_max, dim=2)
        A[t,1] = 1 - get_lognorm_cdf(W[t,1], params_i)
        H[t,1] = P[t,1] * A[t,1]
        
        P[t,2] = get_offer_rate(U[t,2], V[t], params_i)
        if t<t_hs:
            W[t,2] = get_approximation_value(α_w1, P[t,2], Φ_min, Φ_max, dim=1)
        else:
            W[t,2] = get_approximation_value(α_w0, P[t,2], Φ_min, Φ_max, dim=1)
            
        A[t,2] = 1 - get_lognorm_cdf(W[t,2], params_i)
        H[t,2] = P[t,2] * A[t,2]
        
        if t < T-1:
            V[t+1] = iterate_V(V[t], V0, params_i)
            U[t+1,0] = iterate_U(U[t,0], V[t], W[t,0], params_i)
            U[t+1,1] = iterate_U(U[t,1], V[t], W[t,1], params_i)
            U[t+1,2] = iterate_U(U[t,2], V[t], W[t,2], params_i)   
        
        Vr = (V-V0)/V0 * 100
        Ur = (U-U0)/U0 * 100
        Pr = (P-Φ0)/Φ0 * 100
        Br = (B-B0)/B0 * 100
        Wr = (W-wres0)/wres0 * 100
        Ar = (A-A0)/(A0) * 100
        Hr = (H-h0)/h0 * 100
    return Vr, Ur, Pr, Br, Wr, Ar, Hr

t_st = 16
params_i = params.copy()
Vir, Uir, Pir, Bir, Wir, Air, Hir = get_impulse_responses(120, steady_state_values, α_u, α_w, -10, benfits[1]/benfits[0]*100, t_st, params_i)

fig, ax = plt.subplots(1, 2, figsize=(16,6))

ax[1].set_title("constant $w^*$ Policy")
ax[1].plot(Vir, label="$V$", color=colors_vars[1])
ax[1].plot(Uir[:,1], label="$U$", color=colors_vars[0])
ax[1].plot(Bir[:,1], label="$b$", color=colors_vars[2])
#ax[1].plot(Wir[:,1], label="$w^*=\hat{w}^*$", color=colors_vars[5])
ax[1].plot(Pir[:,1], ls="-.", label="$\phi$", color=colors_vars[3],zorder=10)
ax[1].plot(Air[:,1], ls="-.", label="$a = \hat{a}$", color=colors_vars[5])
ax[1].plot(Hir[:,1], ls="-.", label="$h \equiv \phi$", color=colors_vars[4])
ax[1].set_ylabel("Percent Deviation")
ax[1].set_xlabel("Weeks")
ax[1].legend()

ax[0].set_title("no Policy")
ax[0].plot(Vir, label="$V$", color=colors_vars[1])
ax[0].plot(Uir[:,0], label="$U$", color=colors_vars[0])
ax[0].plot(Bir[:,0], label="$b=\hat{b}$", color=colors_vars[2])
#ax[0].plot(Wir[:,0], label="$w^*$", color=colors_vars[5])
ax[0].plot(Pir[:,0], ls="-.", label="$\phi$", color=colors_vars[3])
ax[0].plot(Air[:,0], ls="-.", label="$a$", color=colors_vars[5])
ax[0].plot(Hir[:,0], ls="-.", label="$h$", color=colors_vars[4])
ax[0].set_ylim(ax[1].get_ylim())
ax[0].set_ylabel(ax[1].get_ylabel())
ax[0].set_xlabel(ax[1].get_xlabel())
ax[0].legend()

fig.savefig(figure_folder+'figure_irf_1.png', bbox_inches='tight', dpi=300)
fig.savefig(figure_folder+'figure_irf_1.tif', bbox_inches='tight')


#--------------------------------------------------------------------
# WRITE TEXT FILE WITH VALUES FOR TEXT

corr0_string = ("Unemployment Rates""\n"+ 
                 "E(U) \t\t\t {:,.4f}".format(dfm.loc[:,"$U_0$"].mean())+"\n"+
                 "SD(U) \t\t\t {:,.4f}".format(np.sqrt(dfm.loc[:,"$U_0$"].var()))+"\n"+
                 "Autocorr(U) \t\t {:,.4f}".format(dfm.loc[:,"$U_0$"].autocorr())+"\n"+
                 "\n"+
                 "JOLTS Vacancy Rates (first five years)""\n"+ 
                 "E(V) \t\t\t {:,.4f}".format(Vdata.iloc[:61].mean())+"\n"+
                 "SD(V) \t\t\t {:,.4f}".format(np.sqrt(VHP0[0].iloc[:61].var()))+"\n"+
                 "Autocorr(V) \t\t {:,.4f}".format(VHP0[0].iloc[:61].autocorr())+"\n"+
                 "JOLTSVacancy Rates (all)""\n"+
                 "E(V) \t\t\t {:,.4f}".format(Vdata.iloc[:-14].mean())+"\n"+
                 "SD(V) \t\t\t {:,.4f}".format(np.sqrt(VHP0[0].iloc[:-14].var()))+"\n"+
                 "Autocorr(V) \t\t {:,.4f}".format(VHP0[0].iloc[:-14].autocorr())+"\n"+
                 "\n"+
                 "Monte Carlo Integration Statistics""\n"+ 
                 "E(v) \t\t\t {:,.4f}".format(v_mean)+"\n"+ 
                 "SE(v) \t\t\t {:,.4f}".format(v_SD)+"\n"+ 
                 "E(u) \t\t\t {:,.4f}".format(u_stst)+"\n"+ 
                 "SD(v) \t\t\t {:,.4f}".format(u_SD)+"\n"+ 
                 "\n"+
                 "Elasticity of Hazard Rates""\n"+ 
                 "e_h(u) \t\t\t {:,.4f}".format(e_hu * steady_state_values[0] / h_)+"\n"+                  
                 "e_h(v) \t\t\t {:,.4f}".format(e_hv * steady_state_values[1] / h_)+"\n"+   
                 "\n"+
                 "Duration of a Recession in months""\n"+ 
                 "no pol. \t\t {:,.4f}".format((STa0.mean()/wpm))+"\n"+  
                 "const. w* \t\t {:,.4f}".format((STa1.mean()/wpm))+"\n"+  
                 "binary \t\t\t {:,.4f}".format((STa2.mean()/wpm))+"\n"+  
                 "\n"+
		 "Difference in U wrt. Expectations""\n"+ 
                 "dU(all) \t\t {:,.4f}".format(((U2[:,2]-U[:,2])*100).mean())+"\n"+                 
                 "dU(rec.) \t\t {:,.4f}".format(((U2[np.where(S[:,2]==1),2][0]-U[np.where(S[:,2]==1),2][0])*100).mean())+"\n"+ 
                 "\n"+
                 "Steady State Values for ""\n"+ 
                 "phi \t\t\t {:,.4f}".format(steady_state_values[3])+"\n"+ 
                 "a \t\t\t {:,.4f}".format(1-get_lognorm_cdf(steady_state_values[5], params))+"\n")
 
corr0_file = open(table_folder+"values_for_text.txt", "w") 
corr0_file.write(corr0_string)
corr0_file.close()
