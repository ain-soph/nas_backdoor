#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python ./poison_ratio.py --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --official --model_index 168 --model_seed 888 --dataset cifar10 --natural --poison_percent 1e-4
"""  # noqa: E501

import trojanvision
import argparse

from trojanvision.attacks import InputAwareDynamic
from trojanvision.models import NATSbench

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
    filename = None
    try:
        if isinstance(model, NATSbench):
            filename = 'tar{target:d} poison{poison:.2f} cross{cross:.2f} index{index} seed{seed}'.format(
                target=attack.target_class, poison=0.10, cross=0.10,
                index=model.model_index, seed=model.model_seed)
        attack.load(filename)
        train_args = dict(**trainer)
    except FileNotFoundError:
        print('Previous attack results not found. Attack now.')
        train_args = dict(**trainer)
        cross_percent = attack.cross_percent
        poison_percent = attack.poison_percent
        attack.cross_percent = 0.10
        attack.poison_percent = 0.10
        new_args = train_args.copy()
        new_args['epochs'] = 10
        attack.attack(**new_args)
        if isinstance(model, NATSbench):
            filename = 'tar{target:d} poison{poison:.2f} cross{cross:.2f} index{index} seed{seed}'.format(
                target=attack.target_class, poison=attack.poison_percent, cross=attack.cross_percent,
                index=model.model_index, seed=model.model_seed)
        attack.save(filename)
        attack.cross_percent = cross_percent
        attack.poison_percent = poison_percent

        trainer = trojanvision.trainer.create(dataset=dataset, model=model, **kwargs)
        train_args = dict(**trainer)

    import torch
    from trojanzoo.utils.logger import MetricLogger
    from trojanzoo.utils.data import sample_batch
    from trojanzoo.utils.output import ansi, get_ansi_len, output_iter
    import math
    import random
    verbose = True
    epochs = train_args['epochs']
    optimizer = train_args['optimizer']
    lr_scheduler = train_args['lr_scheduler']
    validate_interval = train_args['validate_interval']

    loader = attack.dataset.loader['train']
    dataset = loader.dataset
    logger = MetricLogger()
    logger.create_meters(loss=None, div=None, ce=None)
    model.requires_grad_()

    best_validate_result = (0.0, float('inf'))
    if validate_interval != 0:
        best_validate_result = attack.validate_fn(verbose=verbose)
        best_asr = best_validate_result[0]
    for _epoch in range(epochs):
        _epoch += 1
        idx = torch.randperm(len(dataset))
        pos = 0
        logger.reset()
        attack.model.train()
        header: str = '{blue_light}{0}: {1}{reset}'.format(
            'Epoch', output_iter(_epoch, epochs), **ansi)
        header = header.ljust(max(len('Epoch'), 30) + get_ansi_len(header))
        for data in logger.log_every(loader, header=header) if verbose else loader:
            optimizer.zero_grad()
            _input, _label = attack.model.get_data(data)
            batch_size = len(_input)
            data2 = sample_batch(dataset, idx=idx[pos:pos + batch_size])
            _input2, _label2 = attack.model.get_data(data2)
            pos += batch_size
            final_input, final_label = _input.clone(), _label.clone()

            # generate trigger input
            trigger_dec, trigger_int = math.modf(len(_label) * attack.poison_percent)
            trigger_int = int(trigger_int)
            if random.uniform(0, 1) < trigger_dec:
                trigger_int += 1
            x = _input[:trigger_int]
            trigger_mark, trigger_mask = attack.get_mark(x), attack.get_mask(x)
            trigger_input = x + trigger_mask * (trigger_mark - x)
            final_input[:trigger_int] = trigger_input
            final_label[:trigger_int] = attack.target_class

            # generate cross input
            cross_dec, cross_int = math.modf(len(_label) * attack.cross_percent)
            cross_int = int(cross_int)
            if random.uniform(0, 1) < cross_dec:
                cross_int += 1
            x = _input[trigger_int:trigger_int + cross_int]
            x2 = _input2[trigger_int:trigger_int + cross_int]
            cross_mark, cross_mask = attack.get_mark(x2), attack.get_mask(x2)
            cross_input = x + cross_mask * (cross_mark - x)
            final_input[trigger_int:trigger_int + cross_int] = cross_input

            loss_ce = attack.model.loss(final_input, final_label)
            loss_div = torch.zeros_like(loss_ce)
            loss = loss_ce
            # div loss
            if len(trigger_input) > 0 and len(cross_input) > 0:
                if len(trigger_input) <= len(cross_input):
                    length = len(trigger_input)
                    cross_input = cross_input[:length]
                    cross_mark = cross_mark[:length]
                    cross_mask = cross_mask[:length]
                else:
                    length = len(cross_input)
                    trigger_input = trigger_input[:length]
                    trigger_mark = trigger_mark[:length]
                    trigger_mask = trigger_mask[:length]
                input_dist: torch.Tensor = (trigger_input - cross_input).flatten(1).norm(p=2, dim=1)
                mark_dist: torch.Tensor = (trigger_mark - cross_mark).flatten(1).norm(p=2, dim=1) + 1e-5
                loss_div = input_dist.div(mark_dist).mean().nan_to_num(0.0)
                loss = loss_ce + attack.lambda_div * loss_div

            loss.backward()
            optimizer.step()
            logger.update(n=batch_size, loss=loss.item(), div=loss_div.item(), ce=loss_ce.item())
        if lr_scheduler is not None:
            lr_scheduler.step()
        attack.model.eval()
        attack.mark_generator.eval()
        if validate_interval != 0 and (_epoch % validate_interval == 0 or _epoch == epochs):
            validate_result = attack.validate_fn(verbose=verbose)
            cur_asr = validate_result[0]
            if cur_asr >= best_asr:
                best_validate_result = validate_result
                best_asr = cur_asr
    optimizer.zero_grad()
