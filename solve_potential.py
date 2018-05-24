# http://mpi4py.scipy.org/docs/usrman/index.html
# http://mpi4py.scipy.org/docs/usrman/tutorial.html
# http://mpi4py.scipy.org/docs/usrman/overview.html#point-to-point-communications
# https://info.gwdg.de/~ceulig/docs-dev/doku.php
# domain boundaries are always 0
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

def parse_parameters(filename):
    parameters = {}
    with open(filename, 'r') as f:
        for line in f:
            # ignore comments and empty lines
            if not line[0].isspace() and line[0] != '#':
                param, val = line.split()[:2] # also ignore comments after entries
                try: # if it makes sense, convert to int
                    parameters[param] = int(val)
                except ValueError:
                    parameters[param] = val

    return parameters

class Grid():
    def __init__(self, rank, grid_dim, grid_left, grid_right,
                 rank_left, rank_right, rank_up, rank_down):
        self.rank = int(rank)
        self.grid_dim = grid_dim
        self.grid_left = grid_left
        self.grid_right = grid_right # up to but not including
        self.rank_left = rank_left
        self.rank_right = rank_right
        self.rank_up = rank_up
        self.rank_down = rank_down

        # Include ghost zones in data, but not x or y
        self.cell_edges_x = np.linspace(grid_left[0], grid_right[0], grid_dim[0]+1)
        self.cell_edges_y = np.linspace(grid_left[1], grid_right[1], grid_dim[1]+1)
        self.data = np.zeros((self.cell_edges_x.size+1, self.cell_edges_y.size+1))

    def _create_boundary_arrays(self): 
        # Copy the edges of the active zone  
        self.l_edge  = self.data[:,1].copy() # x is second index  
        self.r_edge  = self.data[:,-2].copy()
        self.u_edge  = self.data[-2,:].copy() # numpy flips y-axis
        self.d_edge  = self.data[1,:].copy()

        # Create empty arrays to hold edges from other active zones
        self.l_ghost = np.empty(self.data.shape[1]) 
        self.r_ghost = np.empty(self.data.shape[1])
        self.u_ghost = np.empty(self.data.shape[0])
        self.d_ghost = np.empty(self.data.shape[0])


    def _share_boundaries(self):
        req = []
        if self.rank_left is not None:
            # Send edges to their ghost zones; tag is my own rank;
            # use Python objects (these array views aren't always contiguous)
            comm.isend(self.l_edge, dest=self.rank_left, tag=self.rank)
            #print(self.l_edge)
            # Recieve ghost zones from their edges; tag is their rank
            req.append( comm.irecv(buf=self.l_ghost, source=self.rank_left,
                        tag=self.rank_left))
            #req = comm.irecv(source=self.rank_left, tag=self.rank_left)
            #self.l_ghost = req.wait()

        if self.rank_right is not None:
            comm.isend(self.r_edge, dest=self.rank_right, tag=self.rank)
            
            req.append( comm.irecv(buf=self.r_ghost, source=self.rank_right,
                        tag=self.rank_right))
            #req = comm.irecv(source=self.rank_right, tag=self.rank_right)
            #self.r_ghost = req.wait()

        if self.rank_up is not None:
            comm.isend(self.u_edge, dest=self.rank_up, tag=self.rank)
            
            req.append( comm.irecv(buf=self.u_ghost, source=self.rank_up,
                        tag=self.rank_up))
            #req = comm.irecv(source=self.rank_up, tag=self.rank_up)
            #self.u_ghost = req.wait()

        if self.rank_down is not None:
            comm.isend(self.d_edge, dest=self.rank_down, tag=self.rank)
            
            req.append( comm.irecv(buf=self.d_ghost, source=self.rank_down,
                        tag=self.rank_down))
            #req = comm.irecv(source=self.rank_down, tag=self.rank_down)
            #self.d_ghost = req.wait()

        MPI.Request.Waitall(req)

    def update_boundaries(self):
        self._create_boundary_arrays()
        self._share_boundaries()
        self.data[:,0] = self.l_ghost
        self.data[:,-1] = self.r_ghost
        self.data[-1,:] = self.u_ghost
        self.data[0,:] = self.d_ghost

if __name__ == "__main__":

    # Process command line arguments
    if len(sys.argv) != 2:
        raise RuntimeError("Must supply parameter file")

    # no MPI init!!
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    # Read parameters
    if my_rank == 0: # only one process should read the file
        parameters = parse_parameters(sys.argv[1])
    else:
        parameters = None # everyone else needs something to hold the data

    parameters = comm.bcast(parameters, root=0)

    # Process parameters; set defaults
    xlen = parameters.get('domain_x')
    ylen = parameters.get('domain_y')
    xdim = parameters.get('cells_x') # returns None if not in parameters
    ydim = parameters.get('cells_y')
    xranks = parameters.get('ranks_x')
    yranks = parameters.get('ranks_y')
    periodic = parameters.get('periodic')
    function = parameters.get('charge_density')
    
    if xlen is None or ylen is None or \
       xdim is None or ydim is None or \
       xranks is None or yranks is None:
        raise RuntimeError("Must include domain_x, domain_y, "
                           "grid_x, grid_y, ranks_x, & ranks_y "
                           "in parameter file {}".format(sys.argv[1])
                          )

    if periodic is None:
        periodic = False

    if function is None: # default charge density
        function = "-2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)"
    # magical way of executing the contents of a string
    code = "def charge_density(x,y):\n\treturn {}".format(function)
    exec(code)

    # Check to see if we have enough MPI ranks
    if xranks*yranks != size:
        raise RuntimeError("Cannot decompose {} MPI ranks into {}x{}"
                           .format(size,xranks,yranks)
                          )

    # Physical size of each cell
    h_x = xlen/xdim
    h_y = ylen/ydim

    #######    
    # Divide the whole grid among the processes
    #######

    gridx = xdim // xranks
    gridy = ydim // yranks

    # Grid remainder
    rmdrx = xdim % xranks
    rmdry = ydim % yranks

    # Find x start and end for grid
    col = my_rank % xranks
    xstart = col * gridx
    xend = xstart + gridx
    if col == xranks-1:
        xend += rmdrx
    #print(my_rank, 'x:', xstart, xend)

    # Find y start and end for grid
    row = my_rank // xranks
    ystart = row * gridy
    yend = ystart + gridy
    if row == yranks-1:
        yend += rmdry
    #print(my_rank, 'y:', ystart, yend)

    # How big is our grid in each dimension
    grid_dims = (xend-xstart, yend-ystart)
    
    # Now translate grid positions to physical units
    lower_left = (xstart/xdim * xlen, ystart/ydim * ylen)
    upper_right = (xend/xdim * xlen, yend/ydim * ylen)

    #######
    # Find nearest neighbors
    #######

    if xstart == 0 and not periodic:
        neighbor_left = None
    elif xstart == 0 and periodic:
        neighbor_left = my_rank + xranks-1
    else:
        neighbor_left = my_rank-1

    if xend == xdim and not periodic:
        neighbor_right = None
    elif xend == xdim and periodic:
        neighbor_right = my_rank - (xranks-1)
    else:
        neighbor_right = my_rank+1

    if ystart == 0 and not periodic:
        neighbor_down = None
    elif ystart == 0 and periodic:
        neighbor_down = (my_rank - xranks) % size
    else:
        neighbor_down = my_rank - xranks

    if yend == ydim and not periodic:
        neighbor_up = None
    elif yend == ydim and periodic:
        neighbor_up = (my_rank + xranks) % size
    else:
        neighbor_up = my_rank + xranks

    #print("{}: up {}, down {}, left {}, right {}"
    #      .format(my_rank, neighbor_up, neighbor_down, 
    #              neighbor_left, neighbor_right))

    #######    
    # Initialize this rank's grid
    #######

    my_grid = Grid(my_rank, grid_dims, lower_left, upper_right,
                   neighbor_left, neighbor_right, neighbor_up, neighbor_down)

    #######
    # Solve for the potential
    #######

    factor = 1/(2 * (1/h_x**2 + 1/h_y**2))

    # Evaluate the charge density function at all (x,y) grid points
    source_term = np.zeros(my_grid.data.shape)
    cell_centers_x = my_grid.cell_edges_x[:-1] + h_x/2
    cell_centers_y = my_grid.cell_edges_y[:-1] + h_y/2
    xx, yy = np.meshgrid(cell_centers_x, cell_centers_y)
    source_term[1:-1,1:-1] = charge_density(xx,yy).T

    # Make an array for the updated potential
    new = np.zeros(my_grid.data.shape)

    # my_grid.data initializes to zero, and new to one
    #while not np.allclose(new[1:-1,1:-1], my_grid.data[1:-1,1:-1]):
    for k in range(10000):
    #for k in range(5):
        #my_grid.update_boundaries()
        for i in range(1, my_grid.data.shape[0]-1):
            for j in range(1, my_grid.data.shape[1]-1):
                new[i,j] = factor*( h_y**-2*(new[i-1,j] + my_grid.data[i+1,j]) \
                                  + h_x**-2*(new[i,j-1] + my_grid.data[i,j+1]) \
                                  - source_term[i,j]) # 1 smaller in each dim
        #comm.Barrier() 
    
        my_grid.data = new.copy()      
    
    #plt.pcolormesh(my_grid.cell_edges_x, my_grid.cell_edges_y, my_grid.data[1:-1,1:-1])
    plt.pcolormesh(my_grid.data[1:-1,1:-1])    
    plt.show()
    
