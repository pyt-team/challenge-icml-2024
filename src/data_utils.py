from typing import Tuple, Dict
import torch
from torch import Tensor
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
import torch_geometric.transforms as T

from data_transform import InputPreprocTransform, LabelPreprocTransform, qm9_to_ev, filter_not_enough_simplices_alpha
from modules.transforms.liftings.graph2simplicial.vietoris_rips_lifting import SimplicialVietorisRipsLifting, InvariantSimplicialVietorisRipsLifting
from modules.transforms.liftings.graph2simplicial.alpha_complex_lifting import SimplicialAlphaComplexLifting

LIFT_TYPE_DICT = {
    'rips': SimplicialVietorisRipsLifting,
    'alpha': SimplicialAlphaComplexLifting
}
LIFT_INV_TYPE_DICT = {
    'rips': InvariantSimplicialVietorisRipsLifting,
    'alpha': SimplicialAlphaComplexLifting 
}


def calc_mean_mad(loader: DataLoader) -> Tuple[Tensor, Tensor]:
    """Return mean and mean average deviation of target in loader."""
    values = [graph.y for graph in loader.dataset]
    mean = sum(values) / len(values)
    mad = sum([abs(v - mean) for v in values]) / len(values)
    return mean, mad


# TODO FIx this crap
def _load_debug(args):
    preproc_str = 'preproc' if args.pre_proc else 'normal'
    data_root = f'./datasets/QM9_delta_{args.dis}_dim_{args.dim}_{args.lift_type}_debug_{preproc_str}'
    pre_filter = None
    if args.lift_type == 'alpha':
        pre_filter = filter_not_enough_simplices_alpha
    dataset = QM9(root=data_root, pre_filter=pre_filter, force_reload=True)
    print('About to prepare data')
    TRANSFORM_DICT = LIFT_INV_TYPE_DICT if args.pre_proc else LIFT_TYPE_DICT 
    transform = T.Compose([
        InputPreprocTransform(),
        TRANSFORM_DICT[args.lift_type](complex_dim=args.dim, delta=args.dis, feature_lifting='ProjectionElementWiseMean'),
        ])
    dataset = [transform(data) for data in dataset[:7]]
    print('Preparing labels...')
    label_transform = LabelPreprocTransform(target_name=args.target_name)
    dataset = [label_transform(data) for data in tqdm(dataset)]
    print('Data prepared')

    return dataset

def _load_normal(args):
    preproc_str = 'preproc' if args.pre_proc else 'normal'
    data_root = f'./datasets/QM9_delta_{args.dis}_dim_{args.dim}_{args.lift_type}_{preproc_str}'
    TRANSFORM_DICT = LIFT_INV_TYPE_DICT if args.pre_proc else LIFT_TYPE_DICT 
    transform = T.Compose([
        InputPreprocTransform(),
        TRANSFORM_DICT[args.lift_type](complex_dim=args.dim, delta=args.dis, feature_lifting='ProjectionElementWiseMean'),
        ])
    pre_filter = filter_not_enough_simplices_alpha if args.lift_type == 'alpha' else None
    print('Preparing data...')
    dataset = QM9(root=data_root, pre_transform=transform, pre_filter=pre_filter)
    dataset = dataset.shuffle()
    print('Preparing labels...')
    label_transform = LabelPreprocTransform(target_name=args.target_name)
    dataset = [label_transform(data) for data in tqdm(dataset)]
    print('Preparation done!')

    return dataset

def generate_loaders_qm9(args):
    if args.debug:
        dataset = _load_debug(args)
        n_train, n_test = 3, 5
    else:
        dataset = _load_normal(args)
        n_train, n_test = 100000, 110000

    train_dataset = dataset[:n_train]
    test_dataset = dataset[n_train:n_test]
    val_dataset = dataset[n_test:]

    # dataloaders
    follow = [f"x_{i}" for i in range(args.dim+1)] + ['x']
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, follow_batch=follow)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, follow_batch=follow)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, follow_batch=follow)

    return train_loader, val_loader, test_loader