#!/usr/bin/env python3

# genetic search to find vulnerable architectures.

r"""
CUDA_VISIBLE_DEVICES=1 python ./search_genetic.py --color --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --natural --total_resume --save_suffix 1
"""  # noqa: E501

import trojanvision
import argparse

from trojanzoo.utils.model import init_weights
from trojanzoo.utils.output import ansi, output_iter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import pickle
import random

from typing import Any
from trojanvision.models import NATSbench
from trojanvision.attacks import InputAwareDynamic

from torch.nn.utils import _stateless
import functools


class Generator(nn.Module):
    def __init__(self, mark_generator: nn.Module, mask_generator: nn.Module) -> None:
        super().__init__()
        self.mark_generator = mark_generator
        self.mask_generator = mask_generator

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        mark = self.get_mark(x)
        mask = self.get_mask(x)
        return x + mask * (mark - x)

    def get_mark(self, _input: torch.Tensor) -> torch.Tensor:
        raw_output: torch.Tensor = self.mark_generator(_input)
        return raw_output.tanh() / 2 + 0.5

    def get_mask(self, _input: torch.Tensor) -> torch.Tensor:
        raw_output: torch.Tensor = self.mask_generator(_input)
        return raw_output.tanh().mul(10).tanh() / 2 + 0.5


def get_ntk_score(module: nn.Module, parameters: dict[str, nn.Parameter], loader: DataLoader) -> float:
    names, values = zip(*parameters.items())

    def func(*params: torch.Tensor, _input: torch.Tensor = None):
        _output: torch.Tensor = _stateless.functional_call(
            module, {n: p for n, p in zip(names, params)}, _input)
        return _output  # (N, C)
    ntk_list: list[torch.Tensor] = []
    for data in loader:
        _input, _label = model.get_data(data)
        batch_grads: tuple[torch.Tensor] = torch.autograd.functional.jacobian(
            functools.partial(func, _input=_input), values)
        batch_grad = torch.cat([g.flatten(2).detach() for g in batch_grads], dim=-1)  # (N, C, sum(D))
        ntk_list.append((batch_grad @ batch_grad.transpose(1, 2)).mean(0))  # (C, C)
        break
    ntk = torch.stack(ntk_list).mean(0)  # (C, C)
    eigs: torch.Tensor = torch.linalg.eigvalsh(ntk)
    eigs_clipped = eigs.nan_to_num(nan=1e5, posinf=1e7, neginf=-1e7)
    return (eigs_clipped[-1] / eigs_clipped[0]).nan_to_num(nan=1e5).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.trainer.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    parser.add_argument('--save_suffix', default='0')
    parser.add_argument('--total_resume', action='store_true')
    parser.add_argument('--total_epoch', type=int, default=4000)
    parser.add_argument('--sample_freq', type=int, default=10)
    kwargs = parser.parse_args().__dict__

    save_suffix: str = kwargs['save_suffix']
    total_resume: bool = kwargs['total_resume']
    total_epoch: int = kwargs['total_epoch']
    sample_freq: int = kwargs['sample_freq']

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model: NATSbench = trojanvision.models.create(dataset=dataset, **kwargs)
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: InputAwareDynamic = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    train_args = dict(**trainer)
    train_args['verbose'] = False

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, attack=attack)

    generator = Generator(attack.mark_generator, attack.mask_generator)
    attack_model = nn.Sequential(generator, model._model)

    atoms = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
    weights = [1, 2, 3, 3, 1]

    def mutate(model_index: int) -> int:
        arch: str = model.api.get_net_config(model_index, 'cifar10')['arch_str']
        # |skip_connect~0|+|skip_connect~0|skip_connect~1|+|skip_connect~0|skip_connect~1|skip_connect~2|
        arch_list: list[str] = []
        for a in arch.split('+'):
            arch_list.extend(a[1:-1].split('|'))
        idx = random.randint(0, len(arch_list) - 1)
        atoms_copy = atoms.copy()
        weights_copy = weights.copy()
        atom_idx = atoms_copy.index(arch_list[idx][:-2])
        atoms_copy.pop(atom_idx)
        weights_copy.pop(atom_idx)
        probs = np.array(weights_copy) / np.sum(weights_copy)
        arch_list[idx] = np.random.choice(atoms_copy, p=probs) + arch_list[idx][-2:]
        new_arch = '+'.join(['|{}|'.format('|'.join(a)) for a in [arch_list[0:1], arch_list[1:3], arch_list[3:]]])
        return model.api.query_index_by_arch(new_arch)

    def get_score(model_index: int) -> float:
        config: dict[str, Any] = model.api.get_net_config(model_index, dataset.name)
        network = model.get_cell_based_tiny_net(config)
        model._model.load_model(network)
        model.model_index = model_index
        model._model.to(env['device'])
        model_params = {name: param for name, param in model.named_parameters(prefix='1') if 'weight' in name}
        # generator_params = dict(generator.named_parameters(prefix='0'))
        scores = []
        for _ in range(3):
            init_weights(model._model)
            init_weights(attack_model)
            score = get_ntk_score(attack_model, model_params, dataset.loader['train'])
            scores.append(score)
        return np.median(scores)

    result_path = './result/nas_backdoor/nats_bench_ntk.pickle'
    result: list[list[dict[str, float]]] = []
    if os.path.isfile(result_path):
        with open(result_path, mode='rb') as f:
            result = pickle.load(f)

    score_result: list[float] = []
    index_result: list[int] = []

    sample_size = 10
    pool_size = 50
    epochs = total_epoch

    import time
    import os
    torch.manual_seed(int(time.time() * 1000))
    file_path = f'./result/nas_backdoor/search_genetic_{save_suffix}.npz'
    if total_resume and os.path.isfile(file_path):
        prev_result: dict = np.load(file_path, allow_pickle=True)
        ages: torch.Tensor = prev_result['ages']
        pool: torch.Tensor = prev_result['pool']
        scores: torch.Tensor = prev_result['scores']
        _epoch: int = prev_result['_epoch']

        score_result = prev_result['score'].tolist()
        index_result = prev_result['index'].tolist()
        best_element = int(pool[scores.argmin()])
        best_score = float(scores.min())

        asr: dict[int, float] = prev_result['asr']
        acc: dict[int, float] = prev_result['acc']
    else:
        ages = torch.zeros(pool_size, dtype=torch.int)
        pool = torch.randperm(10000)[:pool_size]
        scores = []
        for element in pool:
            scores.append(get_score(element))
        scores = torch.tensor(scores)
        _epoch = 0

        best_element = int(pool[scores.argmin()])
        best_score = float(scores.min())
        score_result.append(best_score)
        index_result.append(best_element)

        asr: dict[int, float] = {}
        acc: dict[int, float] = {}

    for i in range(_epoch, epochs):
        sample = torch.randperm(len(scores))[:sample_size]
        mutate_idx = int(scores[sample].argmin())
        old_idx = int(ages.argmax())

        removed_element = int(pool[old_idx])
        removed_score = float(scores[old_idx])
        new_element = mutate(pool[mutate_idx])

        pool[old_idx] = new_element
        new_score = get_score(new_element)
        scores[old_idx] = new_score

        print(output_iter(i + 1, epochs))
        print('    {green}Add    {yellow}score{reset}:'.format(**ansi),
              '{:15.4f}'.format(new_score),
              '{yellow}arch{reset}:'.format(**ansi),
              model.api.get_net_config(new_element, dataset.name)['arch_str']
              )
        print('    {green}Del    {yellow}score{reset}:'.format(**ansi),
              '{:15.4f}'.format(removed_score),
              '{yellow}arch{reset}:'.format(**ansi),
              model.api.get_net_config(removed_element, dataset.name)['arch_str']
              )
        score_result.append(new_score)
        index_result.append(new_element)
        if new_score < best_score:
            print('{green}Best Result Updated!!!{reset}'.format(**ansi))
            print(f'previous: {best_score:10.3f}  new: {new_score:10.3f}')
            best_element = new_element
            best_score = new_score

        ages += 1
        ages[old_idx] = 0

        if i % sample_freq == 0:
            asr_list: list[float] = []
            acc_list: list[float] = []
            model.model_index = new_element
            config: dict[str, Any] = model.api.get_net_config(new_element, dataset.name)
            network = model.get_cell_based_tiny_net(config)
            model._model.load_model(network)
            for model_seed in [777, 888, 999]:
                model.model_seed = model_seed
                try:
                    model.load('official')
                    model._model.to(env['device'])
                except ValueError:
                    continue
                init_weights(attack.mask_generator)
                init_weights(attack.mark_generator)
                attack.attack(**train_args)
                current_asr, current_acc = attack.validate_fn(indent=4)
                asr_list.append(current_asr)
                acc_list.append(current_acc)
            asr[i] = np.median(asr_list)
            acc[i] = np.median(acc_list)

        np.savez(file_path,
                 score=score_result, index=index_result,
                 best_model_index=best_element,
                 pool=pool, scores=scores, ages=ages, _epoch=i + 1,
                 asr=asr, acc=acc)

    print()
    print(f'best model index: {best_element}')
    print(f'best score: {best_score}')
