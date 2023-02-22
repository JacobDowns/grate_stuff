import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from model.simulator import Simulator
from data_loader import GraphDataLoader
dataset_dir = "data/"
batch_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=15, node_input_size=2, edge_input_size=4, device=device)
simulator.load_checkpoint('checkpoint/simulator.pth')

def test(model:Simulator, test_loader):

    
    model.eval()
    test_error = 0.
    n = 0
    with torch.no_grad():
        for batch_index, graph in enumerate(test_loader):

            
            faces = graph.faces[0]
            pos = graph.pos[0]
            x = pos[:,0]
            y = pos[:,1]

            graph = graph.cuda()
            z = y_normalizer.decode(graph.y)
            z = z.cpu().numpy()
            triang = mtri.Triangulation(x, y, faces)

            
            plt.subplot(2,1,1)
            plt.tricontourf(triang, z[:,1], levels=np.linspace(z.min(), z.max(), 100))
            plt.colorbar()

            out = model(graph)
            z_out = y_normalizer.decode(out)
            z_out = z_out.cpu().numpy()
            
            plt.subplot(2,1,2)
            plt.tricontourf(triang, z_out[:,1], levels=np.linspace(z.min(), z.max(), 100))
            #plt.tricontourf(triang, z_out[:,0] - z[:,0], levels = 100)
            plt.colorbar()
            plt.show()

            errors = (out - graph.y)**2
            loss = torch.mean(errors).item()
            test_error += loss
            n += 1
        print('Test Error: ', test_error / n)


if __name__ == '__main__':
    loader = GraphDataLoader()
    data = loader.data
    y_normalizer = loader.y_normalizer
    y_normalizer.cuda()

    N = len(data)
    train_frac = 0.85
    N_train = int(N*train_frac)
    N_test = N - N_train

    train_data = data[0:N_train]
    test_data = data[N_train:]

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=10, shuffle = True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=10, shuffle = True)
    

    test(simulator, test_loader)
