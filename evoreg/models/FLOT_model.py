import torch

def sinkhorn(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter):
    """
    Sinkhorn algorithm

    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost. 
        Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost. 
        Size B x M x C.
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.

    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """

    # Squared l2 distance between points points of both point clouds
    distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(
        pcloud2 ** 2, -1, keepdim=True
    ).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
    # Force transport to be zero for points further than 10 m apart
    support = (distance_matrix < 10 ** 2).float()

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) * support

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T

class SetConv(torch.nn.Module):
    def __init__(self, nb_feat_in, nb_feat_out):
        """
        Module that performs PointNet++-like convolution on point clouds.

        Parameters
        ----------
        nb_feat_in : int
            Number of input channels.
        nb_feat_out : int
            Number of ouput channels.

        Returns
        -------
        None.

        """

        super(SetConv, self).__init__()

        self.fc1 = torch.nn.Conv2d(nb_feat_in + 3, nb_feat_out, 1, bias=False)
        self.bn1 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc2 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn2 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.fc3 = torch.nn.Conv2d(nb_feat_out, nb_feat_out, 1, bias=False)
        self.bn3 = torch.nn.InstanceNorm2d(nb_feat_out, affine=True)

        self.pool = lambda x: torch.max(x, 2)[0]
        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.1)

    def forward(self, signal, graph):
        """
        Performs PointNet++-like convolution

        Parameters
        ----------
        signal : torch.Tensor
            Input features of size B x N x nb_feat_in.
        graph : flot.models.graph.Graph
            Graph build on the input point cloud on with the input features 
            live. The graph contains the list of nearest neighbors (NN) for 
            each point and all edge features (relative point coordinates with 
            NN).
            
        Returns
        -------
        torch.Tensor
            Ouput features of size B x N x nb_feat_out.

        """

        # Input features dimension
        b, n, c = signal.shape
        n_out = graph.size[0] // b

        # Concatenate input features with edge features
        signal = signal.reshape(b * n, c)
        signal = torch.cat((signal[graph.edges], graph.edge_feats), -1)
        signal = signal.view(b, n_out, graph.k_neighbors, c + 3)
        signal = signal.transpose(1, -1)

        # Pointnet++-like convolution
        for func in [
            self.fc1,
            self.bn1,
            self.lrelu,
            self.fc2,
            self.bn2,
            self.lrelu,
            self.fc3,
            self.bn3,
            self.lrelu,
            self.pool,
        ]:
            signal = func(signal)

        return signal.transpose(1, -1)

class Graph:
    def __init__(self, edges, edge_feats, k_neighbors, size):
        """
        Directed nearest neighbor graph constructed on a point cloud.

        Parameters
        ----------
        edges : torch.Tensor
            Contains list with nearest neighbor indices.
        edge_feats : torch.Tensor
            Contains edge features: relative point coordinates.
        k_neighbors : int
            Number of nearest neighbors.
        size : tuple(int, int)
            Number of points.

        """

        self.edges = edges
        self.size = tuple(size)
        self.edge_feats = edge_feats
        self.k_neighbors = k_neighbors

    @staticmethod
    def construct_graph(pcloud, nb_neighbors):
        """
        Construct a directed nearest neighbor graph on the input point cloud.

        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud. Size B x N x 3.
        nb_neighbors : int
            Number of nearest neighbors per point.

        Returns
        -------
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing the list of nearest 
            neighbors (NN) for each point and all edge features (relative 
            coordinates with NN).
            
        """

        # Size
        nb_points = pcloud.shape[1]
        size_batch = pcloud.shape[0]

        # Distance between points
        distance_matrix = torch.sum(pcloud ** 2, -1, keepdim=True)
        distance_matrix = distance_matrix + distance_matrix.transpose(1, 2)
        distance_matrix = distance_matrix - 2 * torch.bmm(
            pcloud, pcloud.transpose(1, 2)
        )

        # Find nearest neighbors
        neighbors = torch.argsort(distance_matrix, -1)[..., :nb_neighbors]
        effective_nb_neighbors = neighbors.shape[-1]
        neighbors = neighbors.reshape(size_batch, -1)

        # Edge origin
        idx = torch.arange(nb_points, device=distance_matrix.device).long()
        idx = torch.repeat_interleave(idx, effective_nb_neighbors)

        # Edge features
        edge_feats = []
        for ind_batch in range(size_batch):
            edge_feats.append(
                pcloud[ind_batch, neighbors[ind_batch]] - pcloud[ind_batch, idx]
            )
        edge_feats = torch.cat(edge_feats, 0)

        # Handle batch dimension to get indices of nearest neighbors
        for ind_batch in range(1, size_batch):
            neighbors[ind_batch] = neighbors[ind_batch] + ind_batch * nb_points
        neighbors = neighbors.view(-1)

        # Create graph
        graph = Graph(
            neighbors,
            edge_feats,
            effective_nb_neighbors,
            [size_batch * nb_points, size_batch * nb_points],
        )

        return graph

class FLOT(torch.nn.Module):
    def __init__(self, nb_iter):
        """
        Construct a model that, once trained, estimate the scene flow between
        two point clouds.

        Parameters
        ----------
        nb_iter : int
            Number of iterations to unroll in the Sinkhorn algorithm.

        """

        super(FLOT, self).__init__()

        # Hand-chosen parameters. Define the number of channels.
        n = 32

        # OT parameters
        # Number of unrolled iterations in the Sinkhorn algorithm
        self.nb_iter = nb_iter
        # Mass regularisation
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        # Entropic regularisation
        self.epsilon = torch.nn.Parameter(torch.zeros(1))

        # Feature extraction
        self.feat_conv1 = SetConv(3, n)
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n)

        # Refinement
        self.ref_conv1 = SetConv(3, n)
        self.ref_conv2 = SetConv(n, 2 * n)
        self.ref_conv3 = SetConv(2 * n, 4 * n)
        self.fc = torch.nn.Linear(4 * n, 3)

    def get_features(self, pcloud, nb_neighbors):
        """
        Compute deep features for each point of the input point cloud. These
        features are used to compute the transport cost matrix between two
        point clouds.
        
        Parameters
        ----------
        pcloud : torch.Tensor
            Input point cloud of size B x N x 3
        nb_neighbors : int
            Number of nearest neighbors for each point.

        Returns
        -------
        x : torch.Tensor
            Deep features for each point. Size B x N x 128
        graph : flot.models.graph.Graph
            Graph build on input point cloud containing list of nearest 
            neighbors (NN) and edge features (relative coordinates with NN).

        """

        graph = Graph.construct_graph(pcloud, nb_neighbors)
        x = self.feat_conv1(pcloud, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)

        return x, graph

    def refine(self, flow, graph):
        """
        Refine the input flow thanks to a residual network.

        Parameters
        ----------
        flow : torch.Tensor
            Input flow to refine. Size B x N x 3.
        graph : flot.models.Graph
            Graph build on the point cloud on which the flow is defined.

        Returns
        -------
        x : torch.Tensor
            Refined flow. Size B x N x 3.

        """
        x = self.ref_conv1(flow, graph)
        x = self.ref_conv2(x, graph)
        x = self.ref_conv3(x, graph)
        x = self.fc(x)

        return flow + x

    def forward(self, pclouds):
        """
        Estimate scene flow between two input point clouds.

        Parameters
        ----------
        pclouds : (torch.Tensor, torch.Tensor)
            List of input point clouds (pc1, pc2). pc1 has size B x N x 3.
            pc2 has size B x M x 3.

        Returns
        -------
        refined_flow : torch.Tensor
            Estimated scene flow of size B x N x 3.

        """

        # Extract features
        feats_0, graph = self.get_features(pclouds[0], 32)
        feats_1, _ = self.get_features(pclouds[1], 32)

        # Optimal transport
        transport = sinkhorn(
            feats_0,
            feats_1,
            pclouds[0],
            pclouds[1],
            epsilon=torch.exp(self.epsilon) + 0.03,
            gamma=torch.exp(self.gamma),
            max_iter=self.nb_iter,
        )
        row_sum = transport.sum(-1, keepdim=True)

        # Estimate flow with transport plan
        ot_flow = (transport @ pclouds[1]) / (row_sum + 1e-8) - pclouds[0]

        # Flow refinement
        refined_flow = self.refine(ot_flow, graph)

        return refined_flow

if __name__ == '__main__':
    print('instantiating model')
    model = FLOT(nb_iter=3)
    print(model)
    B, N, M = 1, 20, 25
    pc1 = torch.randn(B, N, 3)
    pc2 = torch.randn(B, M, 3)
    print('fake data created')
    flow = model((pc1, pc2))

    assert flow.shape == (B, N, 3)

    print("[OK] Full model forward test passed")

    