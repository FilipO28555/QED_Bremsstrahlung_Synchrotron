import cv2
import numpy as np
import torch
from torch.cuda import Stream #test
import time as timeit
import matplotlib.pyplot as plt

SynchrotronQ = True #enable Synchrotron radiation calculation?
if SynchrotronQ:
    from Synchrotron_lib import Generate_photon_CUDA, Heff_CUDA


q = 1.602176634e-19  # Elementary charge in Coulombs
m_e = 9.10938356e-31  # Electron mass in kg
qm = q / m_e  # Charge-to-mass ratio for electron in C/kg
c = 299792458  # Speed of light in m/s
hbar = 1.054571817e-34  # Reduced Planck constant in J*s

miu0 = np.pi * 4.e-7 # Vacuum permeability in H/m
eps0 = 1.0 / miu0 / c / c # Vacuum permittivity in F/m
Es = 1.3*1e18 # Electric field strength in V/m

def Tens(x):
    return torch.tensor(x, dtype=torch.float64, device='cuda')

def dotTensor(x,y):
    return (x*y).sum(-1)

                    #Laser parameters:
time = 0

w = 2*np.pi*c/800e-9 # Angular frequency for 800nm laser
k = Tens([0,w/c,0]) # Wave vector for 800nm laser
a0 = 90           #Normalized vector potential of an electromagnetic wave
init_gamma = 100  #Initial gamma value for the particles

print("k = {:e}".format(w/c))
print("w = {:e}".format(w))

El = a0*m_e*w*c/q #Electric field strength in V/m

                    #Time and space
Tperiod = 2*np.pi/w
substeps = 6
dt = Tperiod/2 # 10 steps per period
dt /= substeps #change dt

kappa = w/c/2/np.pi # wave number
scale = 1/kappa # SIZE pixels = scale*meters
scaleX = scale*100
scaleY = scale*100

print("scale=",scale, "meters")
print("dt = {:e}".format(dt))


def envelope(x):
    global time
    # parameters:
    sig = scaleY/12
    # sig = 1e-6
    x0 = time*c
    y0 = scaleX/2
    return torch.exp(-(x[:,1]-x0)**2/2/sig**2) * torch.exp(-(x[:,0]-y0)**2/sig**2)

    
def getEandB(x):
    global time
    
    phase = w*time - dotTensor(x,k)
    magnitude = torch.cos(phase)*El
    magnitude = magnitude * envelope(x)
    
    vectorE = torch.zeros_like(x)
    vectorE[:,0] = magnitude
    vectorB = torch.zeros_like(x)
    vectorB[:,2] = -magnitude/c
    
    return vectorE, vectorB

energyAll = []
def plotEnergy():            
    energy = np.concatenate(energyAll)
    # histogram of the generated photons
    # bins = np.logspace(-3, 0, int(len(energy)/500))
    # bins = np.linspace(0, 1, int(len(energy)))
    plt.hist(energy)
    # title
    plt.title("Histogram of generated photons. Number of photons = " + str(len(energy)))
    plt.xlabel("energy")
    plt.ylabel("count")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    plt.close()

def getGamma(velocities):
    vmag = torch.norm(velocities,dim=-1)
    return 1/torch.sqrt(1-(vmag/c)**2)

def update_Boris(positions,velocities,dt):
    # stream1 = Stream(device=torch.device('cuda'))
    # stream2 = Stream(device=torch.device('cuda'))
    
    
    E,B = getEandB(positions)

    #relativistic correction
    vmag = torch.norm(velocities,dim=-1)
    gamma = 1/torch.sqrt(1-(vmag/c)**2)
    # print(gamma.mean().cpu().numpy())
    
    momentum = m_e*velocities*gamma.unsqueeze(-1)
    
    if SynchrotronQ:
        heff = Heff_CUDA(velocities,vmag.unsqueeze(-1), B, E ) 
        #check for nan
        # mask = torch.isnan(heff)
        # print("heff nan: ", torch.sum(mask).item())
        r1s = Generate_photon_CUDA(heff, gamma, dt)
        momentum = momentum - momentum/torch.norm(momentum,dim=1).unsqueeze(-1) * (r1s*(gamma*m_e*c)).unsqueeze(-1)
        mask = r1s > 0
        
        r1s = r1s[mask]
        energy = r1s*gamma[mask]*(m_e*c**2)
        energy = energy.cpu().numpy()
        energyAll.append(energy)
        # print("time = ", time, "len(energy) = ", len(energy))
        
    
    # velMinus = velocities + qm*E*dt/2
    momMinus = momentum + q*E*dt/2
    mommag = torch.norm(momMinus,dim=-1)
    gamma = 1/torch.sqrt(1+(mommag/(m_e*c))**2)
    gamma = gamma.unsqueeze(-1)
    t = qm*B*(dt/2)*gamma
    # Vprime = Vminus + torch.cross(Vminus,t)
    momPrime = momMinus + torch.cross(momMinus,t)
    
    tmag2 = torch.norm(t,dim=-1)**2 
    tmag2 = tmag2.unsqueeze(-1)
    s = 2*t/(1+tmag2)
    
    # velocities = Vminus + torch.cross(Vprime,s)
    momentum = momMinus + torch.cross(momPrime,s)
    
    momentum = momentum + q*E*dt/2
    gamma = torch.sqrt(1+(torch.norm(momentum,dim=-1)/(m_e*c))**2)
    gamma = gamma.unsqueeze(-1)

    velocities = momentum/(m_e*gamma)
    
    # print(velocities)
    return positions + velocities*dt, velocities
    
# Create some balls
# n_balls = 1_000_000
n_balls = 10_000
maxBalls = 10_000
displayBalls = n_balls if n_balls<maxBalls else maxBalls


def reset(gamma):
    positions = ((torch.rand(n_balls,3,dtype=torch.float64, device='cuda')/2 + 0.25)+Tens([0,-0.15,0]))*Tens([scaleX,scaleY*3,0])
    # positions = (torch.zeros(n_balls,3,dtype=torch.float64, device='cuda')/2 + 0.5)*Tens([scaleX,scaleY*0.5,0])
    velocities = torch.zeros_like(positions)
    #initial velocity
    init_vel = c * np.sqrt(1 - 1 / gamma**2)
    velocities[:,1] = -init_vel
    positions[:,1] += init_vel*dt*100

    return positions, velocities

positions, velocities = reset(init_gamma)

# Simulation loop

SIZE = 900

start = timeit.time()+1
iter = 0
fps = 1

def speed_test(minBalls,maxBalls,steps_num,averadge_of = 100):
    global positions, velocities, n_balls, time, iter, fps
    
    time_for_balls = []
    
    balls_num = np.linspace(minBalls,maxBalls,steps_num,dtype=int)
    
    # get it up and running
    torch.cuda.empty_cache()
    n_balls = minBalls
    positions, velocities = reset(init_gamma)
    start = timeit.time()
    for _ in range(averadge_of):
        positions, velocities =  update_Boris(positions,velocities,dt)
        time += dt
    
    # start the real test
    for b in balls_num:
        # clear gpu memory
        torch.cuda.empty_cache()
        
        n_balls = b
        positions, velocities = reset(init_gamma)
        start = timeit.time()
        for _ in range(averadge_of):
            positions, velocities =  update_Boris(positions,velocities,dt)
            time += dt
        
        time_for_balls.append(timeit.time()-start)
    
        print("n_balls = ", n_balls, "time = ", time_for_balls[-1]/averadge_of)
    
    time_for_balls = np.array(time_for_balls)/averadge_of*1000
    # plot
    plt.plot(balls_num,time_for_balls)
    plt.title("Time for different number of balls")
    plt.xlabel("number of balls")
    plt.ylabel("time [ms]")
    plt.show()
    
# speed_test(1_000_000,15_000_000,15)

# exit()

    
while True:
    # FPS counter
    if iter%10==0:
        cv2.setWindowTitle('image', "fps = {:.0f} \t\t".format(fps/10)+"mean time from last frame = {:.0f} ms".format(1/fps*10000)+"  simulation time = {:.3e} ms".format(time))
        # print("fps = {:.0f} \t\t".format(fps/10)+"mean time from last frame = {:.0f} ms".format(1/fps*10000))
        fps = 0
    else:
        fps += 1/(timeit.time()-start)
    start = timeit.time()
    iter+=1
    
    # SIMULATION
    for _ in range(substeps):
        positions, velocities =  update_Boris(positions,velocities,dt)
        time += dt
        if time > 3.8e-13:
            time = 0
            gammas = getGamma(velocities)
            print("time = ", time, "mean error gamma = ", (gammas-init_gamma).mean().cpu().numpy())
            positions, velocities = reset(init_gamma)
            # exit()
            
    
    # Visualization using OpenCV
    alpha = 1
    img = np.zeros((SIZE,SIZE,4), np.uint8) # Create a black image
    pos = (positions[:displayBalls] * SIZE / Tens([scaleX,scaleY,1])).type(torch.int32).cpu().numpy()
    
        # show Electric field
    E,B = getEandB(positions[:displayBalls])
    # print(E)
    E = (E*SIZE/El/20).type(torch.int32).cpu().numpy()
    B = (B*c*SIZE/El/40).type(torch.int32).cpu().numpy()
    for e,b,p in zip(E,B,pos):
        cv2.line(img, (p[0], p[1]), (p[0]+e[0], p[1]+e[1]), (255,0,0, 255*alpha), 1) # Ex
        cv2.line(img, (p[0], p[1]), (p[0]+b[0], p[1]+b[2]), (0,255,0, 255*alpha), 1) # Bz
     
    
    for position in pos:
        cv2.circle(img, (position[0], position[1]), 1, (0,0,255,255*alpha), -1)
   
    # Convert the original image to RGBA
    img_bgr = np.zeros((SIZE, SIZE, 3), np.uint8)  # Your original BGR image
    img_original_rgba = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2BGRA)
    # Blend the RGBA image with the original image
    img = cv2.addWeighted(img_original_rgba, 1, img, alpha, 0)
    
    cv2.imshow('image', img)
    # print("hello world")
    
    
    # cv2.waitKey(1)
    # if R key pressed - reset
    if cv2.waitKey(1) & 0xFF == ord('r'):
        time = 0
        positions, velocities = reset(init_gamma)
    if cv2.waitKey(1) & 0xFF == ord('p'):
        plotEnergy()
    

cv2.destroyAllWindows() # Destroy all the windows