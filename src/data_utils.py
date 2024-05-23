from typing import Tuple, Dict
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

from data_transform import AlphaPreTransform, VietorisRipsPreTransfrom, prepare_data, qm9_to_ev, filter_not_enough_simplices_alpha
from modules.transforms.liftings.graph2simplicial.vietoris_rips_lift import SimplicialVietorisRipsLifting
from modules.transforms.liftings.graph2simplicial.alpha_complex_lift import SimplicialAlphaComplexLifting

LIFT_TYPE_DICT = {
    'rips': VietorisRipsPreTransfrom,
    'alpha': AlphaPreTransform 
}


def calc_mean_mad(loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


def generate_loaders_qm9(dis: float, dim: int, target_name: str, batch_size: int, num_workers: int, lift_type: str, debug = False) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if debug:
        data_root = f'./datasets/QM9_delta_{dis}_dim_{dim}_{lift_type}_debug'
        dataset = QM9(root=data_root, pre_filter=filter_not_enough_simplices_alpha)
        print('About to prepare data')
        dataset = [prepare_data(graph, target_name, qm9_to_ev) for graph in tqdm(dataset, desc='Preparing data')]
        print('Data prepared')
        transform = SimplicialAlphaComplexLifting(complex_dim=dim, dis=dis, feature_lifting='ProjectionElementWiseMean')
        dataset = [transform(data) for data in dataset[:7]]
    else:
        data_root = f'./datasets/QM9_delta_{dis}_dim_{dim}_{lift_type}'
        transform = LIFT_TYPE_DICT[lift_type](complex_dim=dim, dis=dis, target_name=target_name, feature_lifting='ProjectionElementWiseMean')
        pre_filter = filter_not_enough_simplices_alpha if lift_type == 'alpha' else None
        dataset = QM9(root=data_root, pre_transform=transform, pre_filter=pre_filter)
        dataset = dataset.shuffle()

    # filter relevant index and update units to eV

    # train/val/test split
    if debug:
        n_train, n_test = 3, 5
    else:
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