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
substeps = 1
dt = Tperiod/100 # 10 steps per period
dt /= substeps #change dt

kappa = w/c/2/np.pi # wave number
scale = 1/kappa # SIZE pixels = scale*meters
scaleX = scale*10
scaleY = scale*10

print("scale=",scale, "meters")
print("dt = {:e}".format(dt))
print("envelope sigma = {:e}".format(scaleY/12))

def envelope(x):
    global time
    # parameters:
    sig = scaleY/12
    # sig = 1e-6
    x0 = time*c
    y0 = scaleX/2
    return torch.exp(-(x[:,1]-x0)**2/2/sig**2) # * torch.exp(-(x[:,0]-y0)**2/sig**2)

    
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
def plotEnergy(save=False,returnImg=False):                    
    energy = np.concatenate(energyAll)
    if len(energyAll) == 0:
        energy = [0]
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
    
    if returnImg:
        fig = plt.gcf()
        # Convert the Matplotlib figure to an RGB image
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return data
    
    if save:
        plt.savefig("Energy/Energy_"+str(time)+".png")
    else:
        plt.show()
    plt.close()

def getGamma(velocities):
    vmag = torch.norm(velocities,dim=-1)
    return 1/torch.sqrt(1-(vmag/c)**2)

def update_Boris(positions,velocities,dt):
    
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
n_balls = 50_000
maxBalls = 50_000
displayBalls = n_balls if n_balls<maxBalls else maxBalls


def reset(gamma):
    positions = ((torch.rand(n_balls,3,dtype=torch.float64, device='cuda')/2 + 0.25)+Tens([0,-0.15,0]))*Tens([scaleX,scaleY*3,0])
    # positions = (torch.zeros(n_balls,3,dtype=torch.float64, device='cuda') + 1)*Tens([scaleX*0.5,scaleY*0.9,0])
    velocities = torch.zeros_like(positions)
    #initial velocity
    init_vel = c * np.sqrt(1 - 1 / gamma**2)
    velocities[:,1] = -init_vel
    positions[:,1] += init_vel*dt*100

    return positions, velocities

positions, velocities = reset(init_gamma)

print("Particle initially at: ", positions[0].cpu().numpy())
print("E field at initial coordinates: ", getEandB(positions)[0][0].cpu().numpy())
print("B field at initial coordinates: ", getEandB(positions)[1][0].cpu().numpy())
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

def draw_points(pos,img,color):
    # Apply mask
    mask_x = pos[:, 0].ge(0) & pos[:, 0].lt(SIZE)
    mask_y = pos[:, 1].ge(0) & pos[:, 1].lt(SIZE)
    pos = pos[mask_x & mask_y]
    # Prepare indices for advanced indexing
    rows = pos[:, 1]
    cols = pos[:, 0]
    # Advanced indexing for batched addition
    img[rows, cols] += color
    return img, pos, (mask_x & mask_y)

def make_gif(images, fname,duration = 0.02,repetitions = 0):
    import imageio
    # duration is the duration of each frame in seconds
    imageio.mimsave(fname, images, duration=duration, loop=repetitions)
    print("gif saved to: ", fname)

gifQ = True
gif_images = []
while True:
    # FPS counter
    if iter%10==0:
        cv2.setWindowTitle('image', "fps = {:.0f} \t\t".format(fps/10)+"mean time from last frame = {:.0f} ms".format(1/fps*10000)+"  simulation time = {:.3e} s".format(time))
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
        if time > 3.8e-14:
            time = 0
            gammas = getGamma(velocities)
            print("time = ", time, "mean error gamma = ", (gammas-init_gamma).mean().cpu().numpy())
            positions, velocities = reset(init_gamma)
            # plotEnergy()
            if gifQ:
                make_gif(gif_images, "boris_scattered_init.gif")
                exit()
            
    # Visualization but drawing on a tensor in pytorch
    # Create the image tensor on GPU
    img = torch.zeros((SIZE, SIZE, 3), dtype=torch.uint8, device='cuda')
    pos = (positions[:maxBalls, :2] * SIZE / torch.tensor([scaleX, scaleY], device='cuda')).type(torch.int32)
    col = torch.tensor([[0, 0, 200]], dtype=torch.uint8, device='cuda')
    img, pos, mask = draw_points(pos,img,col)

    #     # show Electric field
    E,B = getEandB(positions[:maxBalls][mask])
    # print(E)
    E = (E/El).type(torch.float32)
    B = (B*c/El).type(torch.float32)
    EB = torch.cat((E,B),dim=0)
    
    #Draw E and B
    lines = torch.zeros((pos.shape[0]*2,40,2), dtype=torch.float32, device='cuda')
    lines[:,0,:] = torch.cat((pos[:],pos[:]),dim=0)
    
    for i in range(1,40):
        lines[:,i,0] = lines[:,i-1,0] + EB[:,0]
        lines[:,i,1] = lines[:,i-1,1] + EB[:,2]
    lines = lines.type(torch.int32)
    # reshape lines to be a vector of 2d points
    lines = lines.reshape((-1, 2))
    # Draw E
    col = torch.tensor([[0, 200, 0]], dtype=torch.uint8, device='cuda')
    img, _, _ = draw_points(lines[:40*pos.shape[0]],img,col)
    # Draw B
    col = torch.tensor([[200, 0, 0]], dtype=torch.uint8, device='cuda')
    img, _, _ = draw_points(lines[40*pos.shape[0]:],img,col)
    
    img = img.cpu().numpy()
    
    
    if gifQ:
        # img = plotEnergy(returnImg=True)
        gif_images.append(img)    
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('image', img)
    
    # if R key pressed - reset
    if cv2.waitKey(1) & 0xFF == ord('r'):
        time = 0
        positions, velocities = reset(init_gamma)
    if cv2.waitKey(1) & 0xFF == ord('p') and SynchrotronQ:
        plotEnergy()
    

cv2.destroyAllWindows() # Destroy all the windows