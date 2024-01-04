import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.special import kv
import torch
from torch_interpolations import RegularGridInterpolator as interpolate



e = 1.602176634e-19  # Elementary charge in Coulombs
m_e = 9.10938356e-31  # Electron mass in kg
c = 299792458  # Speed of light in m/s
hbar = 1.054571817e-34  # Reduced Planck constant in J*s

miu0 = np.pi * 4.e-7 # Vacuum permeability in H/m
eps0 = 1.0 / miu0 / c / c # Vacuum permittivity in F/m
Es = 1.3*1e18 # Electric field strength in V/m
# print(Es)

def adaptive_simpsons(f, a, b, tol=1E-9, imax=9000):
    order = 4
    i1 = 0

    def _work(a1, a2, f1, f2, _tol):
        nonlocal i1
        i1 += 1       

        m1 = (a1 + a2) / 2
        fm = f(m1)
        estim = (a2 - a1) / 6 * (f1 + 4 * fm + f2)

        left_estim = ((m1 - a1) / 6) * (f1 + 4 * f(m1) + fm)
        right_estim = ((a2 - m1) / 6) * (fm + 4 * f(m1) + f2)
        
        err = left_estim + right_estim - estim
              
        if abs(err) <= (2 ** order - 1) * _tol or i1 > imax:
            return left_estim + right_estim + err / (2 ** order - 1)
        else:
            return _work(a1, m1, f1, fm, 0.5 * _tol) + _work(m1, a2, fm, f2, 0.5 * _tol)

    return _work(a, b, f(a), f(b), tol)

Classical_limit = False
CUDAQ = True


def F1(z_q):
    if z_q > 2.8e-6:
        integral = quad(lambda x: kv(5/3, x), z_q, np.inf)[0]
        return z_q*integral
    else:
        return 2.15* z_q**(1/3)

    integral = 0
    for i in range(1000):
        integral += kv(5/3, z_q + i*0.01)
    return  z_q*integral*0.001

def F2(z_q):
    return  z_q*kv(2/3, z_q)

def test_F1F2():
    # plot F1(z_q)
    z_q = np.linspace(0, 1, 1000)
    plt.plot(z_q, [F1(z) for z in z_q])
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.close()
    # exit()
    # plot F2(z_q)
    z_q = np.linspace(0, 1, 1000)
    plt.plot(z_q, [F2(z) for z in z_q])
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.close()

# test_F1F2()
# exit()

if CUDAQ:
    print("Creating lookup tables for F1 and F2...")
    # cashed F2
    x = np.logspace(-30, 2, 200) 
    valuesF2 = [F2(z) for z in x]
    valuesF2 = torch.tensor(valuesF2, device='cuda', dtype=torch.float64)
    x = [torch.tensor(x, device='cuda', dtype=torch.float64)]
    cashedF2 = interpolate(x, valuesF2)
    
    # cashed F1
    x = np.logspace(-30, 2, 800) # valid to delta = 1e-7
    valuesF1 = [F1(z) if z > 2.8e-6 else 2.15* z**(1/3) for z in x]
    for i in range(len(valuesF1)-1,0,-1):
        if valuesF1[i] < 0:
            # print("x[i] =", x[i])
            # F1_integral = adaptive_simpsons(lambda y: kv(5/3, y), x[i], x[i]+10,imax=80)
            # valuesF1[i] = x[i] * F1_integral
            valuesF1[i] = 2.15* x[i]**(1/3) 
        
        
    valuesF1 = torch.tensor(valuesF1, device='cuda', dtype=torch.float64)
    x = [torch.tensor(x, device='cuda', dtype=torch.float64)]
    cashedF1 = interpolate(x, valuesF1)
    
    print("Done.")
    
# expects numpy array
def interpolateF(x, cashedF):
    x = [torch.tensor(x, device='cuda', dtype=torch.float64)]
    return cashedF(x)

# expect cuda tensors
def interpolateF_(x, cashedF):
    return cashedF([x])

def test_interpolateF():
    # plot interpolated F1(z_q) and F2(z_q)
    z_q = np.logspace(-9, 0, 1000)
    plt.plot(z_q, interpolateF(z_q, cashedF1).cpu().numpy(), label="interpolated F1")
    #doted line
    plt.plot(z_q, [F1(z) for z in z_q], label="original F1", linestyle = '--')
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()
    plt.plot(z_q, interpolateF(z_q, cashedF2).cpu().numpy(), label="interpolated F2")
    plt.plot(z_q, [F2(z) for z in z_q], label="original F2")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()
    plt.close()
# test_interpolateF()
# exit()

# particle in magnetic field perpendicular to velocity
def Heff_CUDA_B_only(v, Bmag):
    return v*Bmag
    

def dotTensor(x,y):
    return (x*y).sum(-1)
# general formula for Heff -> v, momentum, B, E in SI are tensors of 3-tensors
def Heff_CUDA(v,vmag, B, E ):
    l1 = ((torch.cross(v,B)+E)**2).sum(-1)
    l2 = dotTensor(v/vmag,E)**2
    # if l1 < l2: -> very important
    mask = l1 > l2   
    
    return torch.sqrt( (l1 - l2)*mask )
   

def P(delta, gamma, Heff, dt, cutoff = 100, scheme = "adaptive_simpsons"):
    global Classical_limit
    cutoff = delta + cutoff
    
    # Calculating chi and z_q
    chi = gamma * Heff / Es
    
    if Classical_limit:
        z_q = 2 * delta / (3 * chi )  # Classical limit
    else:
        z_q = 2 * delta / (3 * chi * (1 - delta))  # Quantum limit
    
    
    # Calculating F1(z_q)
    if scheme == "quad":
        F1_integral = quad(lambda x: kv(5/3, x), z_q, cutoff)[0]
    
    # integrate by hand
    elif scheme == "stupid":
        integral = 0
        dx = 0.1
        for i in range(int((cutoff)/dx)):
            x = z_q + i*dx
            integral += kv(5/3, x)
        F1_integral = integral*dx
    
    elif scheme == "adaptive_simpsons":
        F1_integral = adaptive_simpsons(lambda x: kv(5/3, x), z_q, cutoff,imax=80)
        # print("F1_integral =", F1_integral)
    # return F1_integral
    F1_result = z_q * F1_integral
    # return F1_result
    
    # Inlining F2(z_q)
    if Classical_limit:
        F2_result = 0
    else:
        F2_result = z_q * kv(2/3, z_q)
    
    
    
    # Calculating the numerator
    if Classical_limit:
        numerator = dt * e**2*m_e*c * np.sqrt(3) *chi * (F1_result ) #classical limit
    else:
        numerator = dt * e**2*m_e*c * np.sqrt(3) *chi* (1 - delta) * (F1_result + 3 * delta * z_q * chi / 2 * F2_result )
    
    denumerator = 2 * np.pi * gamma*delta *hbar**2 * eps0 * 4*np.pi
    
    return numerator/denumerator

# delta, gamma, Heff, are vectors
def P_CUDA(delta, gamma, Heff, dt):
    global Classical_limit
    
    # Calculating chi and z_q. Es - in SI, Heff - in SI, gamma - unitless
    chi = gamma * Heff / Es
    
    # print("chi = ", chi)
    # print("Heff = ", Heff)
    # print("delta = ", delta)
    # print("gamma = ", gamma)
    
    
    if Classical_limit:
        z_q = 2 * delta / (3 * chi )  # Classical limit
    else:
        z_q = 2 * delta / (3 * chi * (1 - delta))  # Quantum limit
    # print("z_q = ", z_q)
    F1_result = interpolateF_(z_q, cashedF1)
    F2_result = interpolateF_(z_q, cashedF2)
    
    # print("F1_result = ", F1_result)
    # print("F2_result = ", F2_result)
    
    # Calculating the numerator
    numericFactor = dt * (e**2 * m_e * c /( hbar**2 * eps0 * 4 * np.pi)) # <- lot of numerical noise propably

    requirement1 = numericFactor * 1.5*chi**(2/3) / gamma
    requirement2 = numericFactor * 0.5*chi**( 1 ) / gamma
        
    # print("requirement1 = ", requirement1.max())
    # print("requirement2 = ", requirement2.max())
    
    numericFactor *= np.sqrt(3)/(2 * np.pi)
    # print("numericFactor = ", numericFactor)
    
    if Classical_limit:
        numerator1 = chi * (F1_result ) #classical limit
        numerator2 = 1
    else:
        numerator1 = (1 - delta)*chi 
        numerator2 = (F1_result + 3 * delta * z_q * chi / 2 * F2_result )
        
    denumerator = gamma*delta 
    
    # print("numerator1 = ", numerator1)
    # print("numerator2 = ", numerator2)
    # print("denumerator = ", denumerator)
    # print("numericFactor = ", numericFactor)
    
    return numericFactor * (numerator1/denumerator*numerator2)

def test_P_CUDA():
    # ch = 0.22 # 1e-7T ~ 60as
    # plot W(delta) for different gamma
    gamma = 100
    delta = np.logspace(-4, 0, 1000)

    Heff = Es/gamma # chi = 1
    dt = 1e-17
    

                                                                                            #  CHECK IN MATHEMATICA -> quad agrees
    delta3 = delta**3
    P_values = [3*d**2*P(d3, gamma, Heff, dt,scheme = "quad" ) for d3,d in zip(delta3,delta)]
    # P_values = [P(d, gamma, Heff, dt,scheme = "quad" ) for d3,d in zip(delta3,delta)]

    gamma = torch.ones(delta.shape, device='cuda', dtype=torch.float64)*gamma
    Heff = torch.ones(delta.shape, device='cuda', dtype=torch.float64)*Heff
    dt = torch.tensor(dt, device='cuda', dtype=torch.float64)
    delta = torch.tensor(delta, device='cuda', dtype=torch.float64)
    delta3 = delta**3
    P_CUDA_values = 3*delta**2*P_CUDA(delta3, gamma, Heff, dt )
    # P_CUDA_values = P_CUDA(delta, gamma, Heff, dt )
    P_CUDA_values = P_CUDA_values.cpu().numpy()

    
    
    # print(P_values)
    # print(P_CUDA_values)
    delta = delta.cpu().numpy()

    plt.plot(delta, P_values, label="quad")
    plt.plot(delta, P_CUDA_values, label="CUDA")
    plt.legend()
    plt.xlabel("delta")
    plt.ylabel("P")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()
    plt.close()
    
    
    # delta = torch.tensor(0.01, device='cpu', dtype=torch.float64)
    # val =  3*delta**2*P_CUDA(delta**3, gamma, Heff*0.01, dt )
    # print("val = ", val.cpu().numpy())
    
# test_P_CUDA()
# exit()

def Generate_photon(Heff, gamma, dt):   
    r1, r2 = np.random.rand(2)
    if r2 < 3*r1**2*P(r1**3, gamma, Heff, dt):    
        return r1
    return 0

def Generate_photon_CUDA(Heff, gamma, dt):   
    r1 = torch.rand(gamma.shape, device='cuda', dtype=torch.float64)
    # mask r1 less than 1e-7
    mask = r1 > 1e-7
    
    P = 3*r1**2*P_CUDA(r1**3, gamma, Heff, dt)
    P = P*mask
    
    mask = P < 1
    P = P*mask
    # print("Propability: ",P)
    r2 = torch.rand(gamma.shape, device='cuda', dtype=torch.float64)
    mask = r2 < P
    return r1*mask




def test_Generate_photon_CUDA():
    gamma = 100
    Heff = Es/gamma*0.229 # chi = 0.229
    delta = np.logspace(-7, 0, 20_000_000)
    
    gamma = torch.ones(delta.shape, device='cuda', dtype=torch.float64)*gamma
    Heff = torch.ones(delta.shape, device='cuda', dtype=torch.float64)*Heff
    dt = torch.tensor(1e-18, device='cuda', dtype=torch.float64)
    energyAll = []
    for i in range(10):
        energy = Generate_photon_CUDA(Heff, gamma, dt)
        
        energy = energy[energy != 0]
        energy = energy.cpu().numpy()
        energyAll.append(energy)
        print("i = ", i, "len(energy) = ", len(energy))
        
    energy = np.concatenate(energyAll)
    # histogram of the generated photons
    bins = np.logspace(-3, 0, 1000)
    plt.hist(energy, bins=bins)
    # title
    plt.title("Histogram of generated photons. Number of photons = " + str(len(energy)))
    plt.xlabel("r1")
    plt.ylabel("count")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()
    
# test_Generate_photon_CUDA()
# exit(0)
    
def Generate_photons(Heff, gamma, dt, N):
    photons = []
    i=1
    while len(photons) < N:
        photon = Generate_photon(Heff, gamma, dt)
        if photon is not None:
            photons.append(photon)
            # print(photon)
        if i % 10000 == 0:
            print(i/1000,"k,\t photons:", len(photons)/1000,"k",sep = "")
        i+=1
    return np.array(photons), dt*i

def test_Generate_photon():
    N = 10_000
    Heff = Es/100 # chi = 1
    gamma = 100
    dt = 1e-19
    photons, time = Generate_photons(Heff, gamma, dt, N)
    # print(photons)
    # remove nans and non numbers from the list
    # photons = np.array([p for p in photons if not np.isnan(p) and np.isfinite(p)])

    energy = photons * gamma * m_e * c**2
    print("\n\nTotal energy =",np.sum(energy))
    print("time =", time)
    print("power = ", np.sum(energy)/time, "J/s")


    bins = np.logspace(-3, 0, 100)
    plt.hist(photons, bins=bins)
    plt.xlabel("delta")
    plt.ylabel("Energy?")

    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()


# usage
# r1s = Generate_photon_CUDA(Heff_CUDA, gammas[i], dt)
 


 
    



