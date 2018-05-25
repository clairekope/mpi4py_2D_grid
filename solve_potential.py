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
        self.data = np.zeros((self.cell_edges_y.size+1, self.cell_edges_x.size+1),
                             dtype='double')
        self.shape = self.data.shape

    def _create_boundary_arrays(self): 
        # Copy the edges of the active zone  
        self.l_edge  = self.data[1:-1,-2   ].copy()
        self.r_edge  = self.data[1:-1,1    ].copy() # x is second index  
        self.u_edge  = self.data[1   , 1:-1].copy()
        self.d_edge  = self.data[-2  , 1:-1].copy()

        # Create empty arrays to hold edges from other active zones
        # The corners don't matter
        self.l_ghost = np.zeros(self.data.shape[0]-2, dtype='double') 
        self.r_ghost = np.zeros(self.data.shape[0]-2, dtype='double')
        self.u_ghost = np.zeros(self.data.shape[1]-2, dtype='double')
        self.d_ghost = np.zeros(self.data.shape[1]-2, dtype='double')

    def _share_boundaries(self):
        r = []
        if self.rank_right is not None:
            #if self.rank == 0:
            #    print("Right edge:", self.r_edge)
            r.append(comm.Irecv([self.r_ghost, MPI.DOUBLE], source=self.rank_right))
            r.append(comm.Isend([self.r_edge, MPI.DOUBLE], dest=self.rank_right))
        
        if self.rank_left is not None:
            #if self.rank == 1:
            #    print("Left edge:", self.l_edge)
            r.append(comm.Irecv([self.l_ghost, MPI.DOUBLE], source=self.rank_left))
            r.append(comm.Isend([self.l_edge, MPI.DOUBLE], dest=self.rank_left))

        #if self.rank_up is not None:
        #    r.append(comm.Irecv([self.u_ghost, MPI.DOUBLE], source=self.rank_up))
        #    r.append(comm.Isend([self.u_edge, MPI.DOUBLE], dest=self.rank_up))

        #if self.rank_down is not None:
        #    r.append(comm.Irecv([self.d_ghost, MPI.DOUBLE], source=self.rank_down))
        #    r.append(comm.Isend([self.d_edge, MPI.DOUBLE], dest=self.rank_down))
        
        if r:
            MPI.Request.Waitall(r)

        #if self.rank == 1:
        #    print("Left ghost:", self.l_ghost)
        #if self.rank == 0:
        #    print("Right ghost:", self.r_ghost)

        return        
    """
    def _share_boundaries_blocking(self):
        # Even ranks send right
        if self.rank%2 == 0 and self.rank_right is not None:
            #print("Right edge:",self.r_edge)
            comm.Send([self.r_edge, MPI.DOUBLE], dest=self.rank_right)
        elif self.rank%2 == 1 and self.rank_left is not None:
            comm.Recv([self.l_ghost, MPI.DOUBLE], source=self.rank_left)
            #print("Left ghost:",self.l_ghost)

        # Odd ranks send right
        if self.rank%2 == 1 and self.rank_right is not None:
            comm.Send([self.r_edge, MPI.DOUBLE], dest=self.rank_right)
        elif self.rank%2 == 0 and self.rank_left is not None:
            comm.Recv([self.l_ghost, MPI.DOUBLE], source=self.rank_left)

        # Even ranks send left
        if self.rank%2 == 0 and self.rank_left is not None:
            comm.Send([self.l_edge, MPI.DOUBLE], dest=self.rank_left)
        elif self.rank%2 == 1 and self.rank_right is not None:
            comm.Recv([self.r_ghost, MPI.DOUBLE], source=self.rank_right)

        # Odd ranks send left
        if self.rank%2 == 1 and self.rank_left is not None:
            #print("Left edge:",self.l_edge)
            comm.Send([self.l_edge, MPI.DOUBLE], dest=self.rank_left)
        elif self.rank%2 == 0 and self.rank_right is not None:
            comm.Recv([self.r_ghost, MPI.DOUBLE], source=self.rank_right)
            #print("Right ghost:",self.r_ghost)
    """
    def update_boundaries(self):
        self._create_boundary_arrays()
        self._share_boundaries()
        self.data[ 1:-1,-1   ] = self.l_ghost.copy()
        self.data[ 1:-1, 0   ] = self.r_ghost.copy()
        self.data[ 0:  , 1:-1] = self.u_ghost.copy()
        self.data[-1   , 1:-1] = self.d_ghost.copy()

if __name__ == "__main__":

    # no MPI init!!
    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    size = comm.Get_size()

    # Process command line arguments
    if len(sys.argv) == 1:
        if my_rank == 0:
            print("Usage: {} parameter_file [ranks_x ranks_y]\n"
                  "ranks_x and ranks_y must be supplied together.\n"
                  "If not supplied, must be in parameter file.".format(sys.argv[0]))
        sys.exit()
    elif len(sys.argv) == 2:
        if my_rank == 0:
            print("Using ranks_x and ranks_y from {}".format(sys.argv[1]))
            parameters = parse_parameters(sys.argv[1])
        else:
            parameters = None # everyone else needs something to hold the data
        parameters = comm.bcast(parameters, root=0)
        xranks = parameters.get('ranks_x')
        yranks = parameters.get('ranks_y')
    elif len(sys.argv) == 3:
        if my_rank == 0:
            print("Third argument ({}) is ambiguous. Must supply "
                  "both ranks_x and ranks_y.\nExiting...".format(sys.argv[2]))
        sys.exit()
    elif len(sys.argv) == 4:
        if my_rank == 0:
            parameters = parse_parameters(sys.argv[1])
        else:
            parameters = None # everyone else needs something to hold the data
        parameters = comm.bcast(parameters, root=0)
        xranks = int(sys.argv[2])
        yranks = int(sys.argv[3])
    else:
        if my_rank == 0:
            print("Unrecognized parameters.\nExiting...")
        sys.exit()

    # Process non-mpi parameters
    xlen = parameters.get('domain_x')
    ylen = parameters.get('domain_y')
    xdim = parameters.get('cells_x') # returns None if not in parameters
    ydim = parameters.get('cells_y')
    periodic = parameters.get('periodic')
    function = parameters.get('charge_density')
    
    if xlen is None or ylen is None or \
       xdim is None or ydim is None or \
       xranks is None or yranks is None:
        raise RuntimeError("Must include domain_x, domain_y, "
                           "grid_x, grid_y, ranks_x, & ranks_y "
                           "in parameter file {}".format(sys.argv[1])
                          )
    # Set Defaults
    if periodic is None:
        periodic = False

    if function is None: # default charge density
        function = "-2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)"
    # magical way of executing the contents of a string
    code = "def charge_density(x,y):\n\treturn {}".format(function)
    exec(code)

    # Check to see if we have enough MPI ranks
    if xranks*yranks != size:
        raise RuntimeError("Cannot decompose {} MPI ranks into {}x{}. "
                           "ranks_x*ranks_y must equal number of processes."
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

    # Calculate grid's cell centers
    cell_centers_x = my_grid.cell_edges_x[:-1] + h_x/2
    cell_centers_y = my_grid.cell_edges_y[:-1] + h_y/2
    xx, yy = np.meshgrid(cell_centers_x, cell_centers_y)

    # Evaluate the charge density function at all cell centers
    source_term = np.zeros(my_grid.shape)
    source_term[1:-1,1:-1] = charge_density(xx,yy)

    # Make an array for the updated potential
    new = np.zeros(my_grid.data.shape)

    # my_grid.data initializes to zero, and new to one
    #while not np.allclose(new[1:-1,1:-1], my_grid.data[1:-1,1:-1]):
    for k in range(5):
        my_grid.update_boundaries()
        for i in range(1, my_grid.shape[0]-1):
            for j in range(1, my_grid.shape[1]-1):
                new[i,j] = factor*(
                           h_y**-2*(my_grid.data[i-1,j] + my_grid.data[i+1,j]) \
                         + h_x**-2*(my_grid.data[i,j-1] + my_grid.data[i,j+1]) \
                         - source_term[i,j])
        comm.Barrier() 
    
        my_grid.data[1:-1,1:-1] = new[1:-1,1:-1].copy()      
        
    plt.pcolormesh(my_grid.cell_edges_x, my_grid.cell_edges_y,
                   my_grid.data[1:-1,1:-1])   
    plt.show()
    
