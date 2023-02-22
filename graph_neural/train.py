from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from data_loader import GraphDataLoader

dataset_dir = "data/"
batch_size = 1
save_epoch = 10
epochs = 1000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simulator = Simulator(message_passing_num=10, node_input_size=2, edge_input_size=4, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)

def train(model:Simulator, train_loader, test_loader, optimizer):

    for ep in range(epochs):
        print('Epoch', ep)
        model.train()
        train_error = 0.
        n = 0
        for batch_index, graph in enumerate(train_loader):
            graph = graph.cuda()
            out = model(graph)

            out = y_normalizer.decode(out)
            y = y_normalizer.decode(graph.y)
            errors = (out - y)**2
            loss = torch.mean(errors)
            train_error += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n += 1


        print('Train error: ', train_error / n)

        if ep % save_epoch == 0:
            model.save_checkpoint()

        # Every 25 epochs test how well the model is generalizing by
        # running the test data
        if ep % 25 == 0:
            model.eval()
            test_error = 0.
            n = 0
            with torch.no_grad():
                for batch_index, graph in enumerate(test_loader):
                    graph = graph.cuda()
                    out = model(graph)

                    out = y_normalizer.decode(out)
                    y = y_normalizer.decode(graph.y)
                    errors = (out - y)**2
                    loss = torch.mean(errors)
                    test_error += loss.item()
                    n += 1
                    
                print('Test Error: ', test_error / n)


if __name__ == '__main__':

    loader = GraphDataLoader()
    data = loader.data
    y_normalizer = loader.y_normalizer
    y_normalizer.cuda()

    # Divide into training and test data
    N = len(data)
    train_frac = 0.85
    N_train = int(N*train_frac)
    N_test = N - N_train

    train_data = data[0:N_train]
    test_data = data[N_train:]

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=10, shuffle = True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, num_workers=10, shuffle = True)
    train(simulator, train_loader, test_loader, optimizer)
