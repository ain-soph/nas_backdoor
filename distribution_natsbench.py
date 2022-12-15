#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=2 python ./distribution_natsbench.py --color --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --natural
"""  # noqa: E501

import trojanvision
import torch
import argparse

from trojanvision.models import NATSbench
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.model import init_weights
from trojanzoo.utils.output import output_iter

import os
import pickle

from typing import Any
from trojanvision.attacks import InputAwareDynamic


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
        trojanvision.summary(env=env, dataset=dataset, model=model, mark=mark, trainer=trainer, attack=attack)

    train_args = dict(**trainer)
    # train_args['verbose'] = False

    logger = MetricLogger(meter_length=50)
    logger.create_meters(asr='{global_avg:.3f} ({min:.3f}  {max:.3f})',
                         clean_acc='{global_avg:.3f} ({min:.3f}  {max:.3f})')
    file_path = './result/nas_backdoor/nats_bench.pickle'
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
            assert len(result[model_index]) == i, f'{len(result[model_index])=}    {i=}'
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

            mark_generator_path = f'./result/nats_generator/{model_index}_{model_seed}_mark.pth'
            mask_generator_path = f'./result/nats_generator/{model_index}_{model_seed}_mask.pth'
            if os.path.isfile(mark_generator_path):
                attack.mark_generator.load_state_dict(torch.load(mark_generator_path))
                attack.mask_generator.load_state_dict(torch.load(mask_generator_path))
                asr, clean_acc = attack.validate_fn()
            else:
                init_weights(attack.mask_generator)
                init_weights(attack.mark_generator)
                asr, clean_acc = attack.attack(**train_args)
                torch.save(attack.mark_generator.state_dict(), mark_generator_path)
                torch.save(attack.mask_generator.state_dict(), mask_generator_path)
                print('generator saved at:', mark_generator_path)

            logger.update(asr=asr, clean_acc=clean_acc)
            result[model_index].append({'asr': asr, 'clean_acc': clean_acc})
            print(' ' * 3, output_iter(i + 1, 3), str(logger))
        with open(file_path, mode='wb') as f:
            pickle.dump(result, f)
            print('result saved at:', file_path)
