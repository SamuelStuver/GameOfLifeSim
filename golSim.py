import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit

ON = 255
OFF = 0
vals = [ON, OFF]

gridTotalList = []

def randomGrid(N):
    # returns a grid of NxN random values
    N = int(N)
    gridArray = np.random.choice(vals, N*N, p=[0.2, 0.8]).reshape(N, N)
    return gridArray

def addGlider(i, j, grid):
    # Adds a glider with top left cell at i, j
    glider = np.array(
            [[0, 0, 255],
             [255, 0, 255], 
             [0, 255, 255]]
            )
    grid [i:i+3, j:j+3] = glider

def update(frameNum, img, grid, N):
    # copy grid since we require 8 neighbors for calculation
    # and we go line by line
    newGrid = grid.copy()
    gridTotal = 0
    for i in range(N):
        for j in range(N):
            gridTotal += grid[i,j]
            #print(i,j, N)
            # compute 8-neighbor sum using toroidal boundary conditions
            # x and y wrap around so that the simulation takes place on a toroidal surface
            total = int((grid[i, (j-1)%N] + grid[i, (j+1)%N] + 
                         grid[(i-1)%N, j] + grid[(i+1)%N, j]  +
                         grid[(i-1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N] +
                         grid[(i+1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N])/255)
            # Apply Conway's rules
            if grid[i, j] == ON:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = OFF
            else:
                if total == 3:
                    newGrid[i, j] = ON

    # update data
    img.set_data(newGrid)
    grid[:] = newGrid[:]

    gridTotal = int(gridTotal/255)
    print("cells on: {}".format(gridTotal))
    gridTotalList.append(gridTotal)

    return img,

def exponential_func(x, a, b, c):
    return a*np.exp(-b*x)+c

def main():
    parser = argparse.ArgumentParser(description="Runs Conway's Game of Life simulation")

    parser.add_argument('--grid-size', dest='N', required=False)
    parser.add_argument('--mov-file', dest='movfile', required=False)
    parser.add_argument('--interval', dest='interval', required=False)
    parser.add_argument('--glider', action='store_true', required=False)
    args = parser.parse_args()

    # set grid size
    N = 100
    if args.N and int(args.N) > 8:
        N = int(args.N)

    # set animation update interval
    updateInterval = 50
    if args.interval:
        updateInterval = int(args.interval)

    # declare grid
    grid = np.array([])
    # check if "glider" demo flag is specified
    if args.glider:
        grid = np.zeros(N*N).reshape(N,N)
        addGlider(1,1,grid)
    else:
        # populate grid with random on/off - more off than on
        grid = randomGrid(N)
    
    # Calculate starting active cells:
    N_0 = 0
    for i in range(N):
        for j in range(N):
            N_0 += grid[i, j]
    
    N_0 = int(N_0 / 255)
    gridTotalList.append(N_0)

    # set up the animation,
    fig, ax = plt.subplots()
    img = ax.imshow(grid, interpolation='nearest')
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N, ),
                                  frames=100,
                                  interval=updateInterval,
                                  save_count=sys.maxsize)

    # number of frames?
    # set the output file
    if args.movfile:
        ani.save(args.movfile, fps=30, extra_args=['-vcodec', 'libx264'])
    
    plt.show()
    
    x = np.arange(0, len(gridTotalList))
   
    #popt, pcov = curve_fit(exponential_func, x, gridTotalList, p0=(8000, -0.01, 0))
    
    #y = exponential_func(x, *popt)

    plt.plot(gridTotalList)
    #plt.plot(y, color='red')

    plt.xlabel('Step')
    plt.ylabel('# Active Cells')
    plt.show()

if __name__ == "__main__":
    main()
