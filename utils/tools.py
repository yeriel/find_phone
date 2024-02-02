import torch
import argparse
from operator import mul
from functools import reduce


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Train a phone finder model.')
    parser.add_argument('data_folder', type=str,
                        help='Path to the folder with labeled images and labels.txt')
    return parser.parse_args()


def count_module_parameters(module):
    """Retorna la cantidad de parámetros del módulo."""
    return sum(reduce(mul, parameter.size()) for parameter in module.parameters())


def find_lr(model, optimiser, loader, start_val=1e-6, end_val=1, beta=0.99, device='cpu'):
    n = len(loader) - 1
    factor = (end_val / start_val)**(1/n)
    lr = start_val

    # this allows you to update the learning rate
    optimiser.param_groups[0]['lr'] = lr
    avg_loss, loss, acc = 0., 0., 0.
    lowest_loss = 0.
    losses = []
    log_lrs = []
    accuracies = []

    model = model.to(device=device)
    for i, (x, y) in enumerate(loader, start=1):
        x = x.to(device=device)
        y = y.to(device=device)
        optimiser.zero_grad()
        scores = model(x)
        cost = torch.nn.functional.mse_loss(scores, y)
        loss = beta*loss + (1-beta)*cost.item()

        # bias correction
        avg_loss = loss/(1 - beta**i)

        # if loss is massive stop
        if i > 1 and avg_loss > 4 * lowest_loss:
            print(f'from here{i, cost.item()}')
            return log_lrs, losses, accuracies
        
        if avg_loss < lowest_loss or i == 1:
            lowest_loss = avg_loss

        losses.append(avg_loss)
        log_lrs.append(lr)

        # step
        cost.backward()
        optimiser.step()

        # update lr
        print(f'cost:{cost.item():.4f}, lr: {lr:.4f}')
        
        lr *= factor
        optimiser.param_groups[0]['lr'] = lr

    return log_lrs, losses, accuracies
