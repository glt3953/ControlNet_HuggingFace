#!/usr/bin/env python

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess

import gradio as gr
import torch

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run(shlex.split('patch -p1'), stdin=f, cwd='ControlNet')

base_url = 'https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/'
names = [
    'body_pose_model.pth',
    'dpt_hybrid-midas-501f0c75.pt',
    'hand_pose_model.pth',
    'mlsd_large_512_fp32.pth',
    'mlsd_tiny_512_fp32.pth',
    'network-bsds500.pth',
    'upernet_global_small.pth',
]
for name in names:
    command = f'wget https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/{name} -O {name}'
    out_path = pathlib.Path(f'ControlNet/annotator/ckpts/{name}')
    if out_path.exists():
        continue
    subprocess.run(shlex.split(command), cwd='ControlNet/annotator/ckpts/')

from app_canny import create_demo as create_demo_canny
from app_depth import create_demo as create_demo_depth
from app_fake_scribble import create_demo as create_demo_fake_scribble
from app_hed import create_demo as create_demo_hed
from app_hough import create_demo as create_demo_hough
from app_normal import create_demo as create_demo_normal
from app_pose import create_demo as create_demo_pose
from app_scribble import create_demo as create_demo_scribble
from app_scribble_interactive import \
    create_demo as create_demo_scribble_interactive
from app_seg import create_demo as create_demo_seg
from model import Model, download_all_controlnet_weights

DESCRIPTION = '''# [ControlNet v1.0](https://github.com/lllyasviel/ControlNet)

<p class="note">New ControlNet v1.1 is available <a href="https://huggingface.co/spaces/hysts/ControlNet-v1-1">here</a>.</p>
'''

SPACE_ID = os.getenv('SPACE_ID')
ALLOW_CHANGING_BASE_MODEL = SPACE_ID != 'hysts/ControlNet'

if SPACE_ID is not None:
    DESCRIPTION += f'\n<p>For faster inference without waiting in queue, you may duplicate the space and upgrade to GPU in settings. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>'
if not torch.cuda.is_available():
    DESCRIPTION += '\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>'

if torch.cuda.is_available():
    if os.getenv('SYSTEM') == 'spaces':
        download_all_controlnet_weights()

MAX_IMAGES = int(os.getenv('MAX_IMAGES', '3'))
DEFAULT_NUM_IMAGES = min(MAX_IMAGES, int(os.getenv('DEFAULT_NUM_IMAGES', '1')))

DEFAULT_MODEL_ID = os.getenv('DEFAULT_MODEL_ID',
                             'runwayml/stable-diffusion-v1-5')
model = Model(base_model_id=DEFAULT_MODEL_ID, task_name='canny')

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Tabs():
        with gr.TabItem('Canny'):
            create_demo_canny(model.process_canny,
                              max_images=MAX_IMAGES,
                              default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Hough'):
            create_demo_hough(model.process_hough,
                              max_images=MAX_IMAGES,
                              default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('HED'):
            create_demo_hed(model.process_hed,
                            max_images=MAX_IMAGES,
                            default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Scribble'):
            create_demo_scribble(model.process_scribble,
                                 max_images=MAX_IMAGES,
                                 default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Scribble Interactive'):
            create_demo_scribble_interactive(
                model.process_scribble_interactive,
                max_images=MAX_IMAGES,
                default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Fake Scribble'):
            create_demo_fake_scribble(model.process_fake_scribble,
                                      max_images=MAX_IMAGES,
                                      default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Pose'):
            create_demo_pose(model.process_pose,
                             max_images=MAX_IMAGES,
                             default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Segmentation'):
            create_demo_seg(model.process_seg,
                            max_images=MAX_IMAGES,
                            default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Depth'):
            create_demo_depth(model.process_depth,
                              max_images=MAX_IMAGES,
                              default_num_images=DEFAULT_NUM_IMAGES)
        with gr.TabItem('Normal map'):
            create_demo_normal(model.process_normal,
                               max_images=MAX_IMAGES,
                               default_num_images=DEFAULT_NUM_IMAGES)

    with gr.Accordion(label='Base model', open=False):
        with gr.Row():
            with gr.Column():
                current_base_model = gr.Text(label='Current base model')
            with gr.Column(scale=0.3):
                check_base_model_button = gr.Button('Check current base model')
        with gr.Row():
            with gr.Column():
                new_base_model_id = gr.Text(
                    label='New base model',
                    max_lines=1,
                    placeholder='runwayml/stable-diffusion-v1-5',
                    info=
                    'The base model must be compatible with Stable Diffusion v1.5.',
                    interactive=ALLOW_CHANGING_BASE_MODEL)
            with gr.Column(scale=0.3):
                change_base_model_button = gr.Button(
                    'Change base model', interactive=ALLOW_CHANGING_BASE_MODEL)
        if not ALLOW_CHANGING_BASE_MODEL:
            gr.Markdown(
                '''The base model is not allowed to be changed in this Space so as not to slow down the demo, but it can be changed if you duplicate the Space. <a href="https://huggingface.co/spaces/{SPACE_ID}?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a>'''
            )

    gr.Markdown('''### Related Spaces

- [Space using Anything-v4.0 as base model](https://huggingface.co/spaces/hysts/ControlNet-with-Anything-v4)
- https://huggingface.co/spaces/jonigata/PoseMaker2
- https://huggingface.co/spaces/diffusers/controlnet-openpose
- https://huggingface.co/spaces/diffusers/controlnet-canny
''')

    check_base_model_button.click(fn=lambda: model.base_model_id,
                                  outputs=current_base_model,
                                  queue=False)
    new_base_model_id.submit(fn=model.set_base_model,
                             inputs=new_base_model_id,
                             outputs=current_base_model)
    change_base_model_button.click(fn=model.set_base_model,
                                   inputs=new_base_model_id,
                                   outputs=current_base_model)

demo.queue(api_open=False, max_size=10).launch()
