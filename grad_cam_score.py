#!/usr/bin/env python3

# generate the IoU scores of heatmaps.

r"""
CUDA_VISIBLE_DEVICES=0 python ./grad_cam_score.py --verbose 1 --model resnet18_comp --pretrained --attack input_aware_dynamic --train_mask_epochs 10 --dataset cifar10 --natural
"""  # noqa: E501

import trojanvision
import argparse

from trojanvision.attacks import InputAwareDynamic
from trojanvision.models import NATSbench


if __name__ == '__main__':
    import torch
    import os
    if not os.path.isfile('./result/nas_backdoor/heatmaps.pt'):
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

        from trojanzoo.utils.data import sample_batch
        heatmaps = {}
        _input, _label = sample_batch(dataset.loader['valid'].dataset, 100)
        _input = _input.cuda()
        _label = _label.cuda()
        heatmap = model.get_heatmap(_input, _label, method='grad_cam', cmap=None)
        heatmaps['resnet'] = heatmap

        kwargs['model_name'] = 'nats_bench'
        kwargs['pretrained'] = False
        model: NATSbench = trojanvision.models.create(dataset=dataset, **kwargs)
        for model_index in [168, 7671, 10472]:
            model.model_index = model_index
            config: dict = model.api.get_net_config(model_index, 'cifar10')
            network = model.get_cell_based_tiny_net(config)
            model._model.load_model(network)
            for model_seed in [777, 888]:
                model.model_seed = model_seed
                model.load('official')
                heatmap = model.get_heatmap(_input, _label, method='grad_cam', cmap=None)
                heatmaps[f'{model_index}_{model_seed}'] = heatmap
        torch.save(heatmaps, './result/nas_backdoor/heatmaps.pt')
    else:
        heatmaps = torch.load('./result/nas_backdoor/heatmaps.pt')

    from trojanzoo.utils.metric import mask_jaccard
    import numpy as np
    jaccard = torch.zeros(7, 7)
    keys = list(heatmaps.keys())
    print(keys)
    for i, key1 in enumerate(keys):
        a = heatmaps[key1]
        for j, key2 in enumerate(keys):
            b = heatmaps[key2]
            result = []
            for k in range(len(a)):
                result.append(mask_jaccard(a[k], b[k], select_num=9))
            jaccard[i, j] = np.mean(result)
    print(jaccard)
