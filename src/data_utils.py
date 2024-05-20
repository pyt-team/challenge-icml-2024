from typing import Tuple, Dict
import torch
from torch import Tensor
import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.datasets import QM9

from modules.transforms.liftings.graph2simplicial.vietoris_rips_lift import SimplicialVietorisRipsLifting

def calc_mean_mad(loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad

def prepare_data(graph: Data, index: int, target_name: str, qm9_to_ev: Dict[str, float]) -> Data:
    graph.y = graph.y[0, index]
    one_hot = graph.x[:, :5]  # only get one_hot for cormorant
    # change unit of targets
    if target_name in qm9_to_ev:
        graph.y *= qm9_to_ev[target_name]

    Z_max = 9
    Z = graph.x[:, 5]
    Z_tilde = (Z / Z_max).unsqueeze(1).repeat(1, 5)

    graph.x = torch.cat((one_hot, Z_tilde * one_hot, Z_tilde * Z_tilde * one_hot), dim=1)

    return graph

def generate_loaders_qm9(dis: float, dim: int, target_name: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    data_root = f'./datasets/QM9_delta_{dis}_dim_{dim}'
    transform = SimplicialVietorisRipsLifting(complex_dim=dim, dis=0)
    dataset = QM9(root=data_root, pre_transform=transform)
    dataset = dataset.shuffle()

    # filter relevant index and update units to eV
    qm9_to_ev = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114, 'lumo': 27.2114}
    targets = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'U0', 'U', 'H', 'G', 'Cv', 'U0_atom', 'U_atom', 'H_atom', 'G_atom', 'A', 'B', 'C']
    index = targets.index(target_name)
    print('About to prepare data')
    dataset = [prepare_data(graph, index, target_name, qm9_to_ev) for graph in tqdm(dataset, desc='Preparing data')]
    print('Data prepared')

    # train/val/test split
    n_train, n_test = 100000, 110000
    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]

    # dataloaders
    follow = [f"x_{i}" for i in range(dim+1)] + ['x']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, follow_batch=follow)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, follow_batch=follow)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, follow_batch=follow)

    return train_loader, val_loader, test_loader