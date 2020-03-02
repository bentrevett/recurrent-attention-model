import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

import argparse
import os
import random

import data_loader
import model

os.makedirs('checkpoints', exist_ok=True)

parser = argparse.ArgumentParser()

# training set-up
parser.add_argument('--data', type=str, default='MNIST')
parser.add_argument('--translated_size', type=int, default=None)
parser.add_argument('--n_clutter', type=int, default=None)
parser.add_argument('--clutter_size', type=int, default=8)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=250)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--train_rollouts', type=int, default=1)
parser.add_argument('--test_rollouts', type=int, default=10)
parser.add_argument('--seed', type=int, default=None)

# model hyperparameters
parser.add_argument('--n_glimpses', type=int, default=6)
parser.add_argument('--patch_size', type=int, default=8)
parser.add_argument('--n_patches', type=int, default=1)
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--glimpse_hid_dim', type=int, default=128)
parser.add_argument('--location_hid_dim', type=int, default=128)
parser.add_argument('--recurrent_hid_dim', type=int, default=256)
parser.add_argument('--std', type=float, default=0.17)

# other hyperparameters
parser.add_argument('--lr', type=float, default=3e-4)

args = parser.parse_args()

if args.n_clutter is not None:
    assert args.translated_size is not None, 'must specify translated size if using clutter'

if args.seed is None:
    args.seed = random.randint(0, 1000)

name = f'{args.data}'
if args.translated_size is not None:
    name += f'-ts{args.translated_size}'
    if args.n_clutter is not None:
        name += f'-nc{args.n_clutter}-cs{args.clutter_size}'
name += f'-ng{args.n_glimpses}-ps{args.patch_size}-np{args.n_patches}-sc{args.scale}-sd{args.std}-se{args.seed}'

os.makedirs(f'checkpoints/{name}/', exist_ok=True)

with open(f'checkpoints/{name}/train_results.txt', 'w+') as f:
    f.write(f'accuracy\tloss\tclassifier_loss\tbaseline_loss\treinforce_loss\n')

with open(f'checkpoints/{name}/test_results.txt', 'w+') as f:
    f.write(f'accuracy\tloss\tclassifier_loss\tbaseline_loss\treinforce_loss\n')

print(vars(args))

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)

if args.data in {'MNIST', 'KMNIST', 'FashionMNIST'}:
    if args.translated_size is not None:
        if args.n_clutter is not None:
            train_iterator, test_iterator = data_loader.get_cluttered_data(args.data, args.translated_size, args.n_clutter, args.clutter_size, args.batch_size)
        else:
            train_iterator, test_iterator = data_loader.get_translated_data(args.data, args.translated_size, args.batch_size)
    else:
        train_iterator, test_iterator = data_loader.get_data(args.data, args.batch_size)
    n_channels = 1
    output_dim = 10
else:
    raise ValueError(f'{args.data} not a recognized dataset.')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f'Device: {device}')

model = model.RecurrentAttentionModel(args.n_glimpses,
                                      n_channels,
                                      args.patch_size,
                                      args.n_patches,
                                      args.scale,
                                      args.glimpse_hid_dim,
                                      args.location_hid_dim,
                                      args.recurrent_hid_dim,
                                      args.std,
                                      output_dim)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def train(model, iterator, optimizer, n_rollouts, device):

    model.train()

    epoch_classifier_loss = 0
    epoch_baseline_loss = 0
    epoch_reinforce_loss = 0
    epoch_loss = 0
    epoch_accuracy = 0

    for images, labels in tqdm(iterator):

        batch_size = images.shape[0]

        images = images.repeat(n_rollouts, 1, 1, 1)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        log_classifier_actions, log_location_actions, baseline, _ = model(images, device, train=True)

        log_classifier_actions = log_classifier_actions.view(n_rollouts, batch_size, -1).mean(dim=0)
        log_location_actions = log_location_actions.view(n_rollouts, batch_size, -1).mean(dim=0)
        baseline = baseline.view(n_rollouts, batch_size, -1).mean(dim=0)

        predictions = torch.argmax(log_classifier_actions, dim=-1)

        rewards = (predictions.detach() == labels).float()

        n_glimpses = baseline.shape[-1]

        rewards = rewards.unsqueeze(1).repeat(1, n_glimpses)

        classifier_loss = F.nll_loss(log_classifier_actions, labels)
        baseline_loss = F.mse_loss(baseline.squeeze(-1), rewards)

        advantage = rewards - baseline.squeeze(-1).detach()

        reinforce_loss = torch.sum(-log_location_actions * advantage, dim=1)
        reinforce_loss = torch.mean(reinforce_loss, dim=0)

        loss = classifier_loss + baseline_loss + reinforce_loss

        loss.backward()

        optimizer.step()

        correct = (predictions == labels).float()
        accuracy = correct.sum() / labels.shape[0]

        epoch_classifier_loss += classifier_loss.item()
        epoch_baseline_loss += baseline_loss.item()
        epoch_reinforce_loss += reinforce_loss.item()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    epoch_classifier_loss /= len(iterator)
    epoch_baseline_loss /= len(iterator)
    epoch_reinforce_loss /= len(iterator)
    epoch_loss /= len(iterator)
    epoch_accuracy /= len(iterator)

    return epoch_classifier_loss, epoch_baseline_loss, epoch_reinforce_loss, epoch_loss, epoch_accuracy

def evaluate(model, iterator, n_rollouts, device):

    model.eval()

    epoch_classifier_loss = 0
    epoch_baseline_loss = 0
    epoch_reinforce_loss = 0
    epoch_loss = 0
    epoch_accuracy = 0

    for images, labels in tqdm(iterator):

        batch_size = images.shape[0]

        images = images.repeat(n_rollouts, 1, 1, 1)

        images = images.to(device)
        labels = labels.to(device)

        log_classifier_actions, log_location_actions, baseline, _ = model(images, device, train=False)

        log_classifier_actions = log_classifier_actions.view(n_rollouts, batch_size, -1).mean(dim=0)
        log_location_actions = log_location_actions.view(n_rollouts, batch_size, -1).mean(dim=0)
        baseline = baseline.view(n_rollouts, batch_size, -1).mean(dim=0)

        predictions = torch.argmax(log_classifier_actions, dim=-1)

        rewards = (predictions.detach() == labels).float()

        n_glimpses = baseline.shape[-1]

        rewards = rewards.unsqueeze(1).repeat(1, n_glimpses)

        classifier_loss = F.nll_loss(log_classifier_actions, labels)
        baseline_loss = F.mse_loss(baseline.squeeze(-1), rewards)

        advantage = rewards - baseline.squeeze(-1).detach()

        reinforce_loss = torch.sum(-log_location_actions * advantage, dim=1)
        reinforce_loss = torch.mean(reinforce_loss, dim=0)

        loss = classifier_loss + baseline_loss + reinforce_loss

        correct = (predictions == labels).float()
        accuracy = correct.sum() / labels.shape[0]

        epoch_classifier_loss += classifier_loss.item()
        epoch_baseline_loss += baseline_loss.item()
        epoch_reinforce_loss += reinforce_loss.item()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    epoch_classifier_loss /= len(iterator)
    epoch_baseline_loss /= len(iterator)
    epoch_reinforce_loss /= len(iterator)
    epoch_loss /= len(iterator)
    epoch_accuracy /= len(iterator)

    return epoch_classifier_loss, epoch_baseline_loss, epoch_reinforce_loss, epoch_loss, epoch_accuracy

def sample_glimpses(model, iterator, device):

    model.eval()

    for images, _ in iterator:

        batch_size = images.shape[0]

        images = images.to(device)

        log_classifier_actions, _, _, locations = model(images, device, train=False)

        predictions = torch.argmax(log_classifier_actions, dim=-1)

        return images, predictions, locations

patience_counter = 0
best_test_accuracy = 0

for epoch in range(1, args.n_epochs+1):

    classifier_loss, baseline_loss, reinforce_loss, loss, accuracy = train(model, train_iterator, optimizer, args.train_rollouts, device)

    print(f'Train Metrics, {epoch}:')
    print(f'Losses: {loss:.2f} / {classifier_loss:.2f} / {baseline_loss:.2f} / {reinforce_loss:.2f}')
    print(f'Accuracy: {accuracy*100:.2f}%')

    with open(f'checkpoints/{name}/train_results.txt', 'a+') as f:
        f.write(f'{accuracy}\t{loss}\t{classifier_loss}\t{baseline_loss}\t{reinforce_loss}\n')

    with torch.no_grad():
        classifier_loss, baseline_loss, reinforce_loss, loss, accuracy = evaluate(model, test_iterator, args.test_rollouts, device)

    print(f'Test Metrics, {epoch}:')
    print(f'Losses: {loss:.2f} / {classifier_loss:.2f} / {baseline_loss:.2f} / {reinforce_loss:.2f}')
    print(f'Accuracy: {accuracy*100:.2f}%')

    with open(f'checkpoints/{name}/test_results.txt', 'a+') as f:
        f.write(f'{accuracy}\t{loss}\t{classifier_loss}\t{baseline_loss}\t{reinforce_loss}\n')

    if accuracy > best_test_accuracy:
        patience_counter = 0
        best_test_accuracy = accuracy
        torch.save(model.state_dict(), f'checkpoints/{name}/model.pt')
        images, predictions, locations = sample_glimpses(model, test_iterator, device)
        torch.save(images, f'checkpoints/{name}/images.pt')
        torch.save(predictions, f'checkpoints/{name}/predictions.pt')
        torch.save(locations, f'checkpoints/{name}/locations.pt')
        torch.save(vars(args), f'checkpoints/{name}/params.pt')
    else:
        patience_counter += 1

    if patience_counter >= args.patience:
        print('Lost patience!')
        break
