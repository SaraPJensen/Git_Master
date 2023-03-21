import numpy as np
from copy import copy
from pyflagser import flagser_unweighted
import networkx as nx
import time


class Vertex:
    """Create one object for each node in the graph and store the information about its connections"""
    def __init__(self, level, id):  
        self.targets = []     #These are currently lists of vertex objects, but could possibly contain the vertex-ids instead
        self.sources = []  
        self.level = level
        self.id = id   #Let this be a list of the constituent elements, containing a single number for level-0 vertices, 2 numbers for edges, 3 for a 2-simplex etc. 

    def add_level_0(self, level_0):
        self.level_0 = []   
        for idx in self.id:
            self.level_0.append(level_0[idx])   #A list of the vertex objects at level 0 to which this vertex is connected
            

    def add_target(self, sink):    #Target are the vertices at the level below to which this vertex is connected - empty for level-0 vertices!
        self.targets.append(sink)   
    
    def add_source(self, source):  #Sources are the vertices at the level above to which this vertex is connected
        self.sources.append(source)

    def add_U(self, U):   #A list of the sink level_0 vertices that, together with the vertex object itself, will form a higher level simplex
        self.U = U

    def get_target(self):
        return self.targets     #Return the list of vertices at the level below to which this one is connected 

    def get_source(self):
        return self.sources 

    def Ver_func(self):
        if len(self.id) == 1:
            self.id
        else:
            return self.level_0   

    def create_simplex_count(self, dim):
        self.simplex_count = np.zeros((dim, 3), dtype=int)  #array of shape [max_dim, 3], each row is the numbers of simplicies of which the node is the source, mediator and sink respectively




class Directed_graph:
    """Compute the directed graph object
    Parameters
    ----------
    vertices: 1d ndarray of vertex properties
    edges: 2d ndarray of shape num_edges x 3, where the content of each row is [source_neuron, sink_neuron, edge_weigth]
    directed: bool
    """

    def __init__(self, vertices, edges):
        self.number_of_vertices = vertices
        self.edge_number, _ = edges.shape

        self.edges = edges

        self.level_0 = []       #The vertices
        self.level_1 = []       #The edges

        self.all_levels = [] 

        count_id = 0
        for vertex in vertices:
            vertex_object = Vertex(0, count_id)
            self.level_0.append(vertex_object)
            count_id += 1 
        
        self.all_levels.append(self.level_0)
        
        self.all_levels.append([])

        for edge in edges:
            edge_id = [edge[0], edge[1]]
            edge_object = Vertex(1, edge_id)
            edge_object.add_level_0(self.level_0)

            for vertex_id in edge_id:
                edge_object.add_target(self.level_0[vertex_id])   #Add the vertex-objects to the list of outgoing edges
                self.level_0[vertex_id].add_source(edge_object)    #Add the edge_object to the list of incoming edges to each of the vertices
            
            self.all_levels[1].append(edge_object)
                
                
    def new_level(self, level):
        self.all_levels.append(level)    #add a new level to the list, containing all the vertices at that level
        
    def new_vertex(self, level, vertex_object):
        vertex_object.add_level_0(self.level_0)  #Send the level_0 vertices to each new vertex

    def get_all_levels_id(self):  #Convert all_levels to numpy array containing the id's of the nodes instead of the node objects
        self.levels_id = []  

        for l_idx, level in enumerate(self.all_levels[1:-1]):
            tmp = np.empty((len(level), len(level[0].id)), dtype=np.int64)
            for v_idx, vertex in enumerate(level):
                for n_idx, node in enumerate(vertex.id):
                    tmp[v_idx, n_idx] = int(node)
            self.levels_id.append(tmp)


    def simplex_counter(self):
        self.simplex_count = np.empty((len(self.all_levels ) -1 ), dtype = np.int64)

        for idx, line in enumerate(self.all_levels):
            if line:
                self.simplex_count[idx] = len(line)

        return self.simplex_count


    def node_features(self):  #Calculate the simplex count for each node and store it as an attribute of each node-object
        max_dim = len(self.levels_id)
        
        for node in self.level_0:   #Iterate over node objects
            node.create_simplex_count(max_dim)

            for dim, level in enumerate(self.levels_id):
                node.simplex_count[dim, 0] = np.count_nonzero(level[:,0] == node.id)
                node.simplex_count[dim, -1] = np.count_nonzero(level[:,-1] == node.id)

                if dim > 0:
                    node.simplex_count[dim, 1] = np.count_nonzero(level[:,1:-1] == node.id)

            # print("Node: ", node.id)
            # print("Simplex count: ")
            # print(node.simplex_count)
            # print()

    

def compute_cell_count(vertices, edges):
    Hasse_graph = Directed_graph(vertices, edges)   #This contains the entire Hasse diagram
    Hasse_simplex = Hasse_graph    #These have the same level-0 and level-1 elements

    #Iterate over the level-1 edges to find 2-simplicies of order [e, e_1, e_2]
    for e, edge_object in zip(Hasse_simplex.edges, Hasse_simplex.all_levels[1]):  #Iterate over the edge list and the list of edge objects
        U = []   #List of Vertex objects  
        #If two other edges (e1 and e2) have the same sink, and the source of e1 is the source of edge_object and the source of e2 is the sink of e2, add the sink vertex to the list 
        for e_1 in Hasse_simplex.edges:
            if np.all(e == e_1):   
                continue
 
            if e[0] == e_1[0]:
                for e_2 in Hasse_simplex.edges:
                    if np.all(e == e_2) or np.all(e_1 == e_2):
                        continue

                    if e_2[0] == e[1] and e_2[1] == e_1[1]:
                        U.append(Hasse_graph.level_0[e_2[1]])
                        
        edge_object.add_U(U)
            
    dim = 2

    keep_going = True 
    while keep_going:
        next_level_nodes = [] 

        for top_vertex in Hasse_simplex.all_levels[-1]:   #Iterate over the top level vertices of the Hasse diagram
            #print("Number of top level simplicies: ", len(Hasse_simplex.all_levels[-1]))

            for node in top_vertex.U:   #Iterate over the sink nodes of the top level vertex 
                #("Top vertex id: ", top_vertex.id) 
                #print("Node id from U: ", node.id)    #It loops through this twice for the 3,1,0,2 simplex - but somehow the node.id gets changed!! 
                vertex_id = copy(top_vertex.id)  
                vertex_id.append(node.id) 

                new_vertex = Vertex(dim, vertex_id)
                Hasse_simplex.new_vertex(dim, new_vertex)

                new_vertex.add_U(copy(top_vertex.U))    
                new_vertex.add_target(top_vertex)
                
                top_vertex.add_source(new_vertex)

                for bd in top_vertex.get_target():   #Iterate over targets of top_vertex, i.e. elementes at the level below 
                    #print("bd id: ", bd.id)
                    for cbd in bd.get_source():
                        #print("cbd id: ", cbd.id)

                        if node.id == cbd.Ver_func()[-1].id: 
                            #print("Last node in cbd.Ver_func", cbd.Ver_func()[-1].id)
                            #print(f"Add {cbd.id} as a target of {new_vertex.id}")
                            
                            new_vertex.add_target(cbd)
                            cbd.add_source(new_vertex)
                            new_vertex.U = list(set(copy(new_vertex.U)) & set(copy(cbd.U)))
                            #print("Elements in U")
                            #for element in new_vertex.U:
                                #print(element.id)
                            #print("New vertex id: ", new_vertex.id)
                            #print()

                next_level_nodes.append(new_vertex)
        Hasse_simplex.new_level(next_level_nodes)
        dim += 1
        
        if len(next_level_nodes) == 0: 
            keep_going = False
    
    Hasse_simplex.get_all_levels_id()
    Hasse_simplex.node_features()

    return Hasse_simplex
    



def _extract_unweighted_graph(adjacency_matrix):
    input_shape = adjacency_matrix.shape
    # Warn if dense and not square
    if isinstance(adjacency_matrix, np.ndarray) and \
            (input_shape[0] != input_shape[1]):
        warnings.warn("Dense `adjacency_matrix` should be square.")

    # Extract vertices and give them weight one
    n_vertices = max(input_shape)
    vertices = np.ones(n_vertices, dtype=float)

    # Extract edge indices
    if isinstance(adjacency_matrix, np.ndarray):
        # Off-diagonal mask
        mask = np.logical_not(np.eye(input_shape[0], M=input_shape[1],
                                     dtype=bool))

        # Data mask
        mask = np.logical_and(adjacency_matrix, mask)

        edges = np.argwhere(mask)
    else:
        edges = np.argwhere(adjacency_matrix)

        # Remove diagonal elements a posteriori
        edges = edges[edges[:, 0] != edges[:, 1]]

    # Assign weight one
    edges = np.insert(edges, 2, 1, axis=1)

    return vertices, edges




def flagser_count_unweighted(adjacency_matrix):
    """Compute the cell count per dimension of a directed/undirected unweighted
    flag complex.

    From an adjacency matrix construct all cells forming its associated flag
    complex and compute their number per dimension.

    Parameters
    ----------
    adjacency_matrix : 2d ndarray or scipy.sparse matrix, required
        Adjacency matrix of a directed/undirected unweighted graph. It is
        understood as a boolean matrix. Off-diagonal, ``0`` or ``False`` values
        denote absent edges while non-``0`` or ``True`` values denote edges
        which are present. Diagonal values are ignored.

    directed : bool, optional, default: ``True``
        If ``True``, computes homology for the directed flag complex determined
        by `adjacency_matrix`. If ``False``, computes homology for the
        undirected flag complex obtained by considering all edges as
        undirected, and it is therefore sufficient (but not necessary)
        to pass an upper-triangular matrix.

    Returns
    -------
    out : list of int
        Cell counts (number of simplices), per dimension greater than or equal
        to `min_dimension` and less than `max_dimension`.

    Notes
    -----
    The input graphs cannot contain self-loops, i.e. edges that start and end
    in the same vertex, therefore diagonal elements of the input adjacency
    matrix will be ignored.

    References
    ----------
    .. [1] D. Luetgehetmann, "Documentation of the C++ flagser library";
           `GitHub: luetge/flagser <https://github.com/luetge/flagser/blob/\
           master/docs/documentation_flagser.pdf>`_.

    """
    # Extract vertices and edges
    vertices, edges = _extract_unweighted_graph(adjacency_matrix)

    # Call flagser_count binding
    Hasse_simplex = compute_cell_count(vertices, edges)

    return Hasse_simplex



if __name__ == "__main__":
    start_time = time.time()
    print("Flagser numpy")

    n_neurons = 30

    upper = nx.to_numpy_array(nx.watts_strogatz_graph(n_neurons, k = n_neurons//3, p = 0.3))  #This is the upper triangular part of the matrix
    lower = nx.to_numpy_array(nx.watts_strogatz_graph(n_neurons, k = n_neurons//3, p = 0.3))  #This is the lower triangular part of the matrix

    out = np.zeros((n_neurons, n_neurons))
    out[np.triu_indices(n_neurons)] = upper[np.triu_indices(n_neurons)]
    out[np.tril_indices(n_neurons)] = lower[np.tril_indices(n_neurons)]
    W0_graph = nx.from_numpy_array(out, create_using=nx.DiGraph)

    connectivity = nx.to_numpy_array(W0_graph)



    test = np.array([[0, 0, 0, 0],
                    [1, 0, 1, 0],
                    [1, 0, 0, 0],
                    [1, 1, 1, 0]])

    Hasse_simplex = flagser_count_unweighted(test)

    print()

    global_count = Hasse_simplex.simplex_counter()

    print("--- %s seconds ---" % (time.time() - start_time))


    level = 0
    for line in Hasse_simplex.all_levels:
        if line:
            #print()
            print(f"{level}-simplicies: {len(line)}")
            # print("Vertices at level ", level)
            # for element in line:
            #     print(element.id)

        level += 1


    for node in Hasse_simplex.level_0:
        print(node.simplex_count)
        print()


    flag_dict = flagser_unweighted(test)    
    cell_count = flag_dict["cell_count"]

    # print()
    # print("Flagser cell count: ", cell_count)
    print(global_count)



"""
Notes
-------
*   The first dimension of the simplex features for each node is the maximum simplical dimension, 
    so will differ from graph to graph
"""