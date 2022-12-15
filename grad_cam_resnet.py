#!/usr/bin/env python3

# generate the heatmaps on ResNet.

r"""
CUDA_VISIBLE_DEVICES=0 python ./grad_cam_resnet.py --verbose 1 --model resnet18_comp --pretrained --attack input_aware_dynamic --validate_interval 1 --train_mask_epochs 10 --epochs 10 --lr 1e-2 --dataset cifar10 --natural
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

    import torch
    counts = torch.zeros(model.num_classes, dtype=torch.int)
    num = 10

    data = [("./result/heatmap/2_9_original_image.png", 2),  # TODO: Need to generate image first manually
            ("./result/heatmap/4_5_original_image.png", 4)]  # TODO: Need to generate image first manually

    from PIL import Image
    import os
    import torch
    for img_path, label in data:
        file = os.path.basename(img_path)[:3]
        _input = F.convert_image_dtype(F.pil_to_tensor(Image.open(img_path))).cuda().unsqueeze(0)
        _label = label*torch.ones(1, dtype=torch.long).cuda()
        poison_label = torch.zeros_like(_label)

        heatmap = model.get_heatmap(_input, _label, method='grad_cam')
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        F.to_pil_image(heatmap[0]).save(f'./result/heatmap/resnet_{file}_original.png')

        heatmap = model.get_heatmap(_input, poison_label, method='grad_cam')
        heatmap = superimpose(heatmap, _input, alpha=0.5)
        F.to_pil_image(heatmap[0]).save(f'./result/heatmap/resnet_{file}_poison.png')

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
            for img_path, label in data:
                file = os.path.basename(img_path)[:3]
                _input = F.convert_image_dtype(F.pil_to_tensor(Image.open(img_path))).cuda().unsqueeze(0)
                _label = label*torch.ones(1, dtype=torch.long).cuda()
                poison_label = torch.zeros_like(_label)

                heatmap = model.get_heatmap(_input, _label, method='grad_cam')
                heatmap = superimpose(heatmap, _input, alpha=0.5)
                F.to_pil_image(heatmap[0]).save(f'./result/heatmap/nats_{model_index}_{model_seed}_{file}_original.png')

                heatmap = model.get_heatmap(_input, poison_label, method='grad_cam')
                heatmap = superimpose(heatmap, _input, alpha=0.5)
                F.to_pil_image(heatmap[0]).save(f'./result/heatmap/nats_{model_index}_{model_seed}_{file}_poison.png')
