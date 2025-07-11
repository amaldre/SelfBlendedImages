#-*-coding: utf-8 -*-
import argparse
import os
import sys
import math
from copy import deepcopy
if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, resize
from torchcam import methods
import numpy as np
from matplotlib import cm
from model import Detector
from torchvision import transforms
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.sbi import get_final_transforms

def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    """Overlay a colormapped mask on a background image

    >>> from PIL import Image
    >>> import matplotlib.pyplot as plt
    >>> from torchcam.utils import overlay_mask
    >>> img = ...
    >>> cam = ...
    >>> overlay = overlay_mask(img, cam)

    Args:
        img: background image
        mask: mask to be overlayed in grayscale
        colormap: colormap to be applied on the mask
        alpha: transparency of the background image

    Returns:
        overlayed image

    Raises:
        TypeError: when the arguments have invalid types
        ValueError: when the alpha argument has an incorrect value
    """

    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 1)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def main():
    argparser = argparse.ArgumentParser('Arguments for CAM visualization...')
    argparser.add_argument('--weights', '-w', help = 'Path to weights', required = True)
    argparser.add_argument('--target_layer', '-t', help='Specify layer names to visualize CAM', required=True)
    argparser.add_argument("--method", '-m', type=str, default="CAM", help="CAM method to use")
    argparser.add_argument('--image', '-i', help='Specify image to overlay CAM', required=True)
    argparser.add_argument("--savefig", type=str, default=None, help="Path to save figure")
    argparser.add_argument("--rows", type=int, default=1, help="Number of rows for the layout")
    argparser.add_argument("--class-idx", type=int, default=0, help="Index of the class to inspect")
    argparser.add_argument("--alpha", type=float, default=0.5, help="Transparency of the heatmap")
    argparser.add_argument('--cuda', action='store_true', help='Running CAM with cuda')
    argparser.add_argument('--save_inverse', action='store_true', help='Saving the inverse of CAM')
    args = argparser.parse_args()
    final_transforms = get_final_transforms()
    

    #Loading model based on path
    model=Detector()
    cnn_sd=torch.load(args.weights)["model"]
    model.load_state_dict(cnn_sd)
    model = model.cuda()

    # Freeze the model
    for p in model.parameters():
        p.requires_grad_(False)
        
    # Loading image
    assert os.path.exists(args.image), 'Image path must be valid, please check the path again!'
    img = Image.open(args.image)
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    img_tensor = final_transforms(img_tensor)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.cuda()
    img_tensor.requires_grad_(True)
    
    # Hook the corresponding layer in the model
    if isinstance(args.method, str):
        cam_methods = args.method.split(",")
    else:
        cam_methods = [
            "CAM",
            "GradCAM",
            "GradCAMpp",
            "SmoothGradCAMpp",
            "ScoreCAM",
            "SSCAM",
            "ISCAM",
            "XGradCAM",
            "LayerCAM",
        ]
    cam_extractors = [
        methods.__dict__[name](model, target_layer=args.target_layer, enable_hooks=False) for name in cam_methods
    ]
    
    num_cols = math.ceil((len(cam_extractors)) / args.rows) + 1
    
    _, axes = plt.subplots(args.rows, num_cols, figsize=(6, 4))
    # Display input
    ax = axes[0][0] if args.rows > 1 else axes[0] if num_cols > 1 else axes
    ax.imshow(img)
    ax.set_title("Input", size=8)

    for idx, extractor in zip(range(1, len(cam_extractors) + 1), cam_extractors):
        extractor._hooks_enabled = True
        model.zero_grad()
        scores=model(img_tensor).softmax(1)[:,1]
        scores = scores.unsqueeze(1)
        print('Classification Score -- {}'.format(scores))

        # Select the class index
        class_idx = scores.squeeze(0).argmax().item() if args.class_idx is None else args.class_idx

        # Use the hooked data to compute activation map
        print(f"prout {class_idx}, {scores}")
        output = extractor(class_idx, scores)[0]
        activation_map = output.to(torch.float32).squeeze(0).cpu()
        
        # Clean data
        extractor.remove_hooks()
        extractor._hooks_enabled = False
        # Convert it to PL image
        # The indexing below means first image in batch
        heatmap = to_pil_image(activation_map, mode='F')            
        
        # Plot the result
        result = overlay_mask(deepcopy(img), heatmap, alpha=args.alpha)

        ax = axes[idx // num_cols][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes

        ax.imshow(result)
        ax.set_title(extractor.__class__.__name__, size=8)
        
         # Compute the inverse heatmap
        if args.save_inverse:
            inverse_activation_map = torch.sub(1, activation_map)
            inverse_heatmap = to_pil_image(inverse_activation_map, mode='F')
            result_inverse = overlay_mask(deepcopy(img), inverse_heatmap, alpha=args.alpha)
            ax = axes[idx // num_cols + 1][idx % num_cols] if args.rows > 1 else axes[idx] if num_cols > 1 else axes
            ax.imshow(result_inverse)
            ax.set_title(f'{extractor.__class__.__name__}_inverse', size=8)    

    # Clear axes
    if num_cols > 1:
        for _axes in axes:
            if args.rows > 1:
                for ax in _axes:
                    ax.axis("off")
            else:
                _axes.axis("off")

    else:
        axes.axis("off")

    plt.tight_layout()
    if args.savefig:
        plt.savefig(args.savefig, dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)


if __name__=='__main__':
    main()
