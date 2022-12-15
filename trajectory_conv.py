#!/usr/bin/env python3

# ASR, ACC and scores of arches perturbed from '|{0}~0|+|{1}~0|{2}~1|+|skip_connect~0|{3}~1|{4}~2|'

r"""
CUDA_VISIBLE_DEVICES=0 python ./trajectory_conv.py --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --natural
"""  # noqa: E501

import trojanvision
import argparse
from trojanzoo.utils.model import init_weights
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.output import ansi, get_ansi_len, prints

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

    generator = Generator(attack.mark_generator, attack.mask_generator)
    attack_model = nn.Sequential(generator, model._model)
    train_args = dict(**trainer)
    train_args['verbose'] = 0
    train_args['validate_interval'] = 0

    # loop arch_str
    atoms = ['nor_conv_1x1', 'nor_conv_3x3']
    arch_str_pattern = '|{0}~0|+|{1}~0|{2}~1|+|skip_connect~0|{3}~1|{4}~2|'

    for first in atoms:
        for second in atoms:
            for third in atoms:
                for fourth in atoms:
                    for fifth in atoms:
                        arch_str = arch_str_pattern.format(first, second, third, fourth, fifth)

                        model_idx = model.api.query_index_by_arch(arch_str)
                        config: dict = model.api.get_net_config(model_idx, 'cifar10')
                        network = model.get_cell_based_tiny_net(config)
                        model._model.load_model(network)
                        model.model_index = model_idx

                        logger = MetricLogger(indent=8, meter_length=50)
                        logger.create_meters(clean_score='{median:.3f}',
                                             model_score='{median:.3f}', generator_score='{median:.3f}')
                        print(f'{model_idx:5d}', arch_str)
                        for model_seed in [777, 888, 999]:
                            try:
                                model.model_seed = model_seed
                                model.load('official')
                                model._model.to(env['device'])
                            except ValueError:
                                prints(f'{model_seed:d} not exist', indent=8)
                                continue

                            prints(model_seed, indent=8)
                            # init_weights(generator)
                            # attack.attack(**train_args)
                            # attack.validate_fn(indent=8)    # output acc and asr

                            attack_model.eval()
                            init_weights(attack_model)
                            logger.reset()
                            header: str = '{blue_light}{0:3d}: {reset}'.format(
                                model_seed, **ansi)
                            header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
                            count = 0
                            for data in logger.log_every(dataset.loader['train'], header=header):
                                _input, _ = dataset.get_data(data)

                                clean_model_params = {name: param for name,
                                                      param in model.named_parameters() if 'weight' in name}
                                clean_score = get_ntk_score_functorch(model._model, clean_model_params, _input)

                                model_params = {name: param for name,
                                                param in model.named_parameters(prefix='1') if 'weight' in name}
                                model_score = get_ntk_score_functorch(attack_model, model_params, _input)
                                generator_params = {name: param for name,
                                                    param in generator.named_parameters(prefix='0') if 'weight' in name}
                                generator_score = get_ntk_score_functorch(attack_model, generator_params, _input)

                                logger.update(n=len(_input),
                                              clean_score=clean_score.item(),
                                              model_score=model_score.item(),
                                              generator_score=generator_score.item())
                                count += 1
                                if count >= 10:
                                    break
                            print(logger)
                            print()
