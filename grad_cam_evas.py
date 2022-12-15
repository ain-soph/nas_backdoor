#!/usr/bin/env python3

r"""
CUDA_VISIBLE_DEVICES=0 python ./grad_cam_evas.py --verbose 1 --model nats_bench --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --official --model_index 168 --model_seed 888 --dataset cifar10 --natural
"""  # noqa: E501

import trojanvision
import argparse

from trojanvision.attacks import InputAwareDynamic
from trojanvision.models import NATSbench
from trojanvision.utils import superimpose
import torchvision.transforms.functional as F

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

    attack.validate_fn()
    import torch
    loader = dataset.get_dataloader('valid', batch_size=1)
    counts = torch.zeros(model.num_classes, dtype=torch.int)
    num = 10
    for data in loader:
        if counts.sum() >= num * (model.num_classes - 1):
            break
        _input, _label = model.get_data(data)
        if _label.item() == attack.target_class:
            continue
        if counts[_label.item()] >= num:
            continue
        poison_input = attack.add_mark(_input)
        if model.get_class(_input) != _label or model.get_class(poison_input) != attack.target_class:
            continue

        poison_label = attack.target_class * torch.ones_like(_label)
        file = f'{_label.item()}_{counts[_label.item()]}'

        mask = attack.get_mask(_input)
        F.to_pil_image(mask[0]).save(f'./result/heatmap/{file}_mask.png')

        F.to_pil_image(_input[0]).save(f'./result/heatmap/{file}_original_image.png')
        F.to_pil_image(poison_input[0]).save(f'./result/heatmap/{file}_poison_image.png')

        heatmap = model.get_heatmap(_input, _label, method='grad_cam')
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        F.to_pil_image(heatmap[0]).save(f'./result/heatmap/{file}_original_source.png')

        heatmap = model.get_heatmap(_input, poison_label, method='grad_cam')
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        F.to_pil_image(heatmap[0]).save(f'./result/heatmap/{file}_original_target.png')

        heatmap = model.get_heatmap(poison_input, _label, method='grad_cam')
        heatmap = superimpose(heatmap, poison_input, alpha=0.5)
        F.to_pil_image(heatmap[0]).save(f'./result/heatmap/{file}_poison_source.png')

        heatmap = model.get_heatmap(poison_input, poison_label, method='grad_cam')
        heatmap = superimpose(heatmap, poison_input, alpha=0.5)
        F.to_pil_image(heatmap[0]).save(f'./result/heatmap/{file}_poison_target.png')

        print(file)

        counts[_label.item()] += 1
