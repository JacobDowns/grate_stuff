import torch
import numpy as np
from torch_geometric.data import Data
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import torch_geometric.transforms as T

# Normalizes each field by making it more normally distributed
class GaussianNormalizer(object):
    def __init__(self, mean, std, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


"""
Loads all training / test simulation examples. 
"""
class GraphDataLoader(object):
    
    def __init__(self, include_plotting_vars = False):
        
        super(GraphDataLoader, self).__init__()
        
        self.data = []
        self.include_plotting_vars = include_plotting_vars

        self.X = []
        self.Y = []
        self.X_edge = []

        self.x_mean = np.load('data/x_mean.npy')
        self.x_std = np.load('data/x_std.npy')

        self.y_mean = np.load('data/y_mean.npy')
        self.y_std = np.load('data/y_std.npy')

        self.x_edge_mean = np.load('data/x_edge_mean.npy')
        self.x_edge_std = np.load('data/x_edge_std.npy')

        self.x_normalizer = GaussianNormalizer(self.x_mean, self.x_std)
        self.y_normalizer = GaussianNormalizer(self.y_mean, self.y_std)
        self.x_edge_normalizer = GaussianNormalizer(self.x_edge_mean, self.x_edge_std)
        
        for i in range(30):
            self.load_dataset(i)


    def load_dataset(self, i):
        print('Loading Simulation ' + str(i))

        # Node features
        # (H, beta, icemasklevelset)
        X = np.load('data/X_{}.npy'.format(i))
        # Output node features
        # (Vx, Vy)
        Y = np.load('data/Y_{}.npy'.format(i))
        # Edge Features
        # (dx, dy, dmag, dS)
        X_edge = np.load('data/X_edge_{}.npy'.format(i))
        # Node coordinates
        coords = np.load('data/coords_{}.npy'.format(i))
        # Contains vertex indices for each face
        faces = np.load('data/faces_{}.npy'.format(i))
        # Vertex indices for each edge
        edges = np.load('data/edges_{}.npy'.format(i))


        # Iterate through each time step but skip the first one
        # because the icelevelset isn't defined
        for j in range(1, len(X)):
            X_j = X[j]
            Y_j = Y[j]
            X_edge_j = X_edge[j]

            # We want to get a subgraph around the area
            # with positive ice thickness using the levelset
            # function 
            mask = X_j[:,-1]
            mask = np.zeros_like(X_j[:,-1])
            mask[X_j[:,-1] < 1500.] = 1.
            # Don't need the levelset function as an input feature
            X_j = X_j[:,0:-1]

            indexes = np.where(mask == 1.)[0].astype(int)
            X_j = X_j[indexes]
            Y_j = Y_j[indexes]
            coords_j = coords[indexes]


            # Eliminates faces and egdges far away from the ice
            def cull_faces(faces, edges, indexes):
                mapping = np.zeros_like(coords[:,0]) + 1e16
                mapping[indexes] = np.arange(len(indexes))
                faces = mapping[faces]
                edges = mapping[edges]
                indexes = np.max(faces, axis=1) < 1e16
                faces = faces[indexes].astype(int)
                indexes = np.max(edges, axis=1) < 1e16
                edges = edges[indexes].astype(int)
                return faces, edges, X_edge_j[indexes]

            faces_j, edges_j, X_edge_j = cull_faces(faces, edges, indexes)

            # Create the custom graph data structure
            data_j = Data(
                x = torch.tensor(X_j, dtype=torch.float32),
                y = torch.tensor(Y_j, dtype=torch.float32),
                edge_index = torch.tensor(edges_j, dtype=torch.long).T,
                edge_attr = torch.tensor(X_edge_j, dtype=torch.float32),
            )

            if self.include_plotting_vars:
                data_j.pos = coords_j
                data_j.faces = faces_j

            data_j.x = self.x_normalizer.encode(data_j.x)
            data_j.y = self.y_normalizer.encode(data_j.y)
            data_j.edge_attr = self.x_edge_normalizer.encode(data_j.edge_attr)

            # Limit the size  of included data
            if data_j.x.shape[0] < 80000:
                self.data.append(data_j)
