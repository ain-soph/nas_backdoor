#!/usr/bin/env python3

# retraining from scratch for ResNet

r"""
CUDA_VISIBLE_DEVICES=1 python ./transfer_resnet.py --verbose 1 --model resnet18_comp --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --natural --batch_size 32
"""  # noqa: E501

import trojanvision
import argparse
from trojanzoo.utils.model import init_weights

import torch
import torch.nn as nn
import functorch

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


def get_ntk_score(module: nn.Module, parameters: dict[str, nn.Parameter], _input: torch.Tensor) -> torch.Tensor:
    names, values = zip(*parameters.items())

    def func(*params: torch.Tensor, _input: torch.Tensor = None):
        _output: torch.Tensor = _stateless.functional_call(
            module, {n: p for n, p in zip(names, params)}, _input)
        return _output  # (N, C)
    batch_grads: tuple[torch.Tensor] = torch.autograd.functional.jacobian(
        functools.partial(func, _input=_input), values, create_graph=True, vectorize=True)
    batch_grad = torch.cat([g.flatten(2) for g in batch_grads], dim=-1)  # (N, C, sum(D))
    ntk = (batch_grad @ batch_grad.transpose(1, 2)).mean(0)  # (C, C)
    eigs: torch.Tensor = torch.linalg.eigvalsh(ntk)
    eigs_clipped = eigs.nan_to_num(nan=1e5, posinf=1e7, neginf=-1e7)
    return (eigs_clipped[-1] / eigs_clipped[0]).nan_to_num(nan=1e5)


def get_ntk_score_functorch(module: nn.Module, parameters: dict[str, nn.Parameter], _input: torch.Tensor
                            ) -> torch.Tensor:
    names, values = zip(*parameters.items())

    def func(params: list[torch.Tensor], _input: torch.Tensor = None):
        _output: torch.Tensor = _stateless.functional_call(
            module, {n: p for n, p in zip(names, params)}, _input)
        return _output  # (N, C)

    def vmap_func(_input: torch.Tensor):
        return functorch.jacrev(functools.partial(func, _input=_input.unsqueeze(0)))(values)

    batch_grads: tuple[torch.Tensor] = (a[:, 0] for a in functorch.vmap(vmap_func)(_input))
    batch_grad = torch.cat([g.flatten(2) for g in batch_grads], dim=-1)  # (N, C, sum(D))
    ntk = (batch_grad @ batch_grad.transpose(1, 2)).mean(0)  # (C, C)
    eigs: torch.Tensor = torch.linalg.eigvalsh(ntk)
    eigs_clipped = eigs.nan_to_num(nan=1e5, posinf=1e7, neginf=-1e7)
    return (eigs_clipped[-1] / eigs_clipped[0]).nan_to_num(nan=1e5)


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
    trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
    mark = trojanvision.marks.create(dataset=dataset, **kwargs)
    attack: InputAwareDynamic = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **kwargs)

    if env['verbose']:
        trojanvision.summary(env=env, dataset=dataset, model=model, trainer=trainer, mark=mark, attack=attack)

    train_args = dict(**trainer)
    train_args['verbose'] = 0
    train_args['validate_interval'] = 5
    generator = Generator(attack.mark_generator, attack.mask_generator)
    attack_model = nn.Sequential(generator, model._model)
    attack_model.requires_grad_(False)
    attack_model.eval()
    # alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(alpha_opt, T_max=epochs)

    def validate():
        model.train()
        for suffix in ['', '_1']:
            model.suffix = suffix
            init_weights(attack_model)
            try:
                model.load()
                model._model.to(env['device'])
            except ValueError:
                continue
            print(f'        {suffix=}')
            print('        before')
            attack.validate_fn(indent=8)
            attack.attack(**train_args)
            print('        after')
            attack.validate_fn(indent=8)
            for suffix2 in ['', '_1']:
                if suffix == suffix2:
                    continue
                try:
                    model.suffix = suffix2
                    model.load()
                    model._model.to(env['device'])
                except ValueError:
                    continue
                print(f'        change suffix to {suffix2}')
                attack.validate_fn(indent=8)

    validate()
