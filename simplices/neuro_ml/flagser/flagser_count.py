import numpy as np


class Vertex:
    """Create one object for each node in the graph and store the information about its connections"""
    def __init__(self, id, level):
        self.id = None    #Let this be a list of the constituent elements, containing a single number for level-0 vertices, 2 numbers for edges, 3 for a 2-simplex etc. 
        self.targets = np.empty(0)   #These are currently lists of vertex objects, but could possibly contain the vertex-ids instead
        self.sources = np.empty(0)
        self.level = level

    def add_id(self, id):
        self.id = id

    def add_target(self, sink):    #Target are the vertices at the level below to which this vertex is connected - empty for level-0 vertices!
        self.targets.append(sink)   
    
    def add_source(self, source):  #Sources are the vertices at the level above to which this vertex is connected
        self.sources.append(source)

    def get_target(self):
        return self.targets     #Return the list of vertices at the level below to which this one is connected

    def get_source(self):
        return self.sources 

    def Ver_func(self):
        if len(self.id) == 1:
            return []
        else:
            return self.id   #This will be a list of all the level 0 vertices in the complex, given by their vertex-id 



class Directed_graph:
    """Compute the directed graph object
    Parameters
    ----------
    vertices: 1d ndarray of vertex properties
    edges: 2d ndarray of shape num_edges x 3, where the content of each row is [source_neuron, sink_neuron, edge_weigth]
    directed: bool
    """

    def __init__(self, vertices, edges):
        self.number_of_vertices = np.len(vertices)
        self.edge_number, _ = edges.shape

        self.edges = edges

        self.level_0 = np.empty(0)    #The vertices
        self.level_1 = np.empty(0)    #The edges

        count_id = 0
        for vertex in vertices:
            vertex_object = Vertex(0)
            vertex_object.add_id(0)
            self.level_0(vertex_object)
            count_id += 1 
        
        for edge in edges:
            edge_id = [edge[0]], [edge[1]]
            edge_object = Vertex(1)
            edge_object.add_id(edge_id)

            for vertex_id in edge_id:
                self.edge_object.add_target(self.level_0[vertex_id])   #Add the vertex-objects to the list of outgoing edges
                self.level_0[vertex_id].add_source(edge_object)    #Add the edge_object to the list of incoming edges to each of the vertices

            self.level_1.append(edge_object)
                
                
    def new_level(self, level):

        simplex = Vertex(level)    #Assume id is not yet available, since this object is created before this is tested for
        pass

         



def cell_count(Hasse_graph):
    Hasse_simplex = Hasse_graph    #These have the same level-0 and level-1 elements

    U = np.empty(0)   #List of Vertex objects

    #Iterate over the level-1 edges
    for edge in Hasse_simplex.edges:  #Iterate over the edge list
        #If two other edges (e1 and e2) have the same sink, and the source of e1 is the source of edge_object and the source of e2 is the sink of e2, add the sink vertex to the list 
        for first_edge in Hasse_simplex.edges and edge != first_edge:
            if edge[0] == first_edge[0]:
                for second_edge in Hasse_simplex.edges and edge != second_edge and first_edge != second_edge_
                    if second_edge[0] == edge[1] and second_edge[1] == second_edge[1]:
                        U.append(Hasse_graph.level_0[second_edge[1]])  #Add the sink-vertex of the complex to the list
            
    dim = 2







def compute_cell_count(vertices, edges, direction):
    
    Hasse_graph = Directed_graph(vertices, directed)   #This contains the entire Hasse diagram
    cell_count = count_cells(Hasse_graph)

    return cell_count





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




def flagser_count_unweighted(adjacency_matrix, directed=True):
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
    cell_count = compute_cell_count(vertices, edges, directed)

    return cell_count
