#!/usr/bin/env python3

# calculate NTK score for each model architecture in NATSBench.

r"""
CUDA_VISIBLE_DEVICES=2 python ./distribution_ntk.py --color --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --natural
"""  # noqa: E501

import trojanvision
import argparse

from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.model import init_weights
from trojanzoo.utils.output import output_iter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import pickle

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
    kwargs = parser.parse_args().__dict__

    env = trojanvision.environ.create(**kwargs)
    dataset = trojanvision.datasets.create(**kwargs)
    model: NATSbench = trojanvision.models.create(dataset=dataset, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: InputAwareDynamic = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, attack=attack)

    before_logger = MetricLogger(meter_length=50)
    after_logger = MetricLogger(meter_length=50)
    pattern = '{median:.3f} ({min:.3f}  {max:.3f})'
    before_logger.create_meters(model_ntk=pattern, generator_ntk=pattern)
    after_logger.create_meters(model_ntk=pattern, generator_ntk=pattern)
    file_path = './result/nas_backdoor/nats_bench_tenas.pickle'
    if os.path.isfile(file_path):
        with open(file_path, mode='rb') as f:
            result = pickle.load(f)
    else:
        result: list[list[dict[str, float]]] = []

    for model_index in range(15625):
        if len(result) > model_index:
            continue
        assert len(result) == model_index
        result.append([])
        print('model idx:', output_iter(model_index, 15624))
        for i, model_seed in enumerate([777, 888, 999]):
            if len(result[model_index]) > i:
                continue
            # assert len(result[model_index]) == i, f'{len(result[model_index])=}    {i=}'
            config: dict[str, Any] = model.api.get_net_config(model_index, dataset.name)
            network = model.get_cell_based_tiny_net(config)
            model._model.load_model(network)
            model.model_index = model_index
            model.model_seed = model_seed
            try:
                model.load('official')
            except ValueError:
                print(f'{model_seed=} is not provided for {model_index=}')
            model._model.to(env['device'])
            model.eval()

            generator = Generator(attack.mark_generator, attack.mask_generator)
            attack_model = nn.Sequential(generator, model._model)

            # after training
            init_weights(generator)
            model_params = {name: param for name, param in model.named_parameters() if 'weight' in name}
            generator_params = dict(generator.named_parameters(prefix='0'))
            after_model_ntk = get_ntk_score(model._model, model_params, dataset.loader['train'])
            after_generator_ntk = get_ntk_score(attack_model, generator_params, dataset.loader['train'])
            # before training
            init_weights(attack_model)
            model_params = {name: param for name, param in model.named_parameters() if 'weight' in name}
            generator_params = dict(generator.named_parameters(prefix='0'))
            before_model_ntk = get_ntk_score(model._model, model_params, dataset.loader['train'])
            before_generator_ntk = get_ntk_score(attack_model, generator_params, dataset.loader['train'])

            before_logger.update(model_ntk=before_model_ntk, generator_ntk=before_generator_ntk)
            after_logger.update(model_ntk=after_model_ntk, generator_ntk=after_generator_ntk)
            _dict = dict(before_model_ntk=before_model_ntk,
                         before_generator_ntk=before_generator_ntk,

                         after_model_ntk=after_model_ntk,
                         after_generator_ntk=after_generator_ntk
                         )
            result[model_index].append(_dict)
            print(' ' * 3, output_iter(i + 1, 3), 'before:', str(before_logger))
            print(' ' * 3, output_iter(i + 1, 3), 'after: ', str(after_logger))
        with open(file_path, mode='wb') as f:
            pickle.dump(result, f)
            print('result saved at:', file_path)
