import argparse
import torch
import wandb
import copy
from tqdm import tqdm

from modules.models.simplicial.empsn import EMPSN

from data_utils import generate_loaders_qm9, calc_mean_mad
import time
from utils import set_seed


def main(args):
    # # Generate model
    num_input = 15
    num_out = 1

    inv_dims = {
        'rank_0': {
            'rank_0': 3,
            'rank_1': 3,
        },
        'rank_1': {
            'rank_1': 6,
            'rank_2': 6,
        }
    }
    model = EMPSN(
            in_channels=num_input,
            hidden_channels=args.num_hidden,
            out_channels=num_out,
            n_layers=args.num_layers,
            max_dim=args.dim,
            inv_dims=inv_dims
        ).to(args.device)

    # Setup wandb
    wandb.init(project=f"QM9-{args.target_name}-{args.lift_type}-{'preproc' if args.pre_proc else 'no-preproc'}")
    wandb.config.update(vars(args))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of parameters: {num_params}')

    # # Get loaders
    start_lift_time = time.process_time()
    train_loader, val_loader, test_loader = generate_loaders_qm9(args)
    end_lift_time = time.process_time()
    wandb.log({
        'Lift time': end_lift_time - start_lift_time
    })

    mean, mad = calc_mean_mad(train_loader)
    mean, mad = mean.to(args.device), mad.to(args.device)


    print('Almost at training...')
    # Get optimization objects
    criterion = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    best_train_mae, best_val_mae, best_model = float('inf'), float('inf'), None

    for _ in tqdm(range(args.epochs)):
        epoch_mae_train, epoch_mae_val = 0, 0

        model.train()
        for _, batch in enumerate(train_loader):
            optimizer.zero_grad()

            batch = batch.to(args.device)
            batch['pos'] = batch['pos'].to(args.device)
            pred = model(batch)
            loss = criterion(pred, (batch.y - mean) / mad)
            mae = criterion(pred * mad + mean, batch.y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            optimizer.step()
            epoch_mae_train += mae.item()

        model.eval()
        for _, batch in enumerate(val_loader):
            batch = batch.to(args.device)
            pred = model(batch)
            mae = criterion(pred * mad + mean, batch.y)

            epoch_mae_val += mae.item()

        epoch_mae_train /= len(train_loader.dataset)
        epoch_mae_val /= len(val_loader.dataset)

        if epoch_mae_val < best_val_mae:
            best_val_mae = epoch_mae_val
            best_model = copy.deepcopy(model.state_dict())

        scheduler.step()

        wandb.log({
            'Train MAE': epoch_mae_train,
            'Validation MAE': epoch_mae_val
        })

    test_mae = 0
    model.load_state_dict(best_model)
    model.eval()
    for _, batch in enumerate(test_loader):
        batch = batch.to(args.device)
        pred = model(batch)
        mae = criterion(pred * mad + mean, batch.y)
        test_mae += mae.item()

    test_mae /= len(test_loader.dataset)
    print(f'Test MAE: {test_mae}')

    wandb.log({
        'Test MAE': test_mae,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=96,
                        help='batch size')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='num workers')
    parser.add_argument('--debug', action='store_true',
                        help='debug mode')

    # Model parameters
    parser.add_argument('--model_name', type=str, default='empsn',
                        help='model')
    parser.add_argument('--num_hidden', type=int, default=77,
                        help='hidden features')
    parser.add_argument('--num_layers', type=int, default=7,
                        help='number of layers')
    parser.add_argument('--act_fn', type=str, default='silu',
                        help='activation function')
    parser.add_argument('--lift_type', type=str, default='rips',
                        help='lift type')

    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-16,
                        help='learning rate')
    parser.add_argument('--gradient_clip', type=float, default=1.0,
                        help='gradient clipping')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='qm9',
                        help='dataset')
    parser.add_argument('--target_name', type=str, default='H',
                        help='regression task')
    parser.add_argument('--dim', type=int, default=2,
                        help='ASC dimension')
    parser.add_argument('--dis', type=float, default=3.0,
                        help='radius Rips complex')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--pre_proc', action='store_true',
                        help='preprocessing')


    parsed_args = parser.parse_args()
    parsed_args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(parsed_args.seed)
    main(parsed_args)