# curl -X POST 'https://ai-proxy.isekai.chat/animatediff' \
#     -F 'prompt=close up shot, young anime girl in a summer dress walking through abandoned neotokyo city, symbols, lush ancient jungle nature overgrowth, colorful' \
#     -F 'n_prompt=ugly,blurry,low resolution' \
#     -F 'model=flat2dAnimergeV3F16.vzgC.safetensors'

import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf

import torch

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler

from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from animatediff.models.unet import UNet3DConditionModel
from animatediff.pipelines.pipeline_animation import AnimationPipeline
from animatediff.utils.util import save_videos_grid
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint, convert_ldm_clip_checkpoint, convert_ldm_vae_checkpoint
from animatediff.utils.convert_lora_safetensor_to_diffusers import convert_lora
from diffusers.utils.import_utils import is_xformers_available

from einops import rearrange, repeat

import csv, pdb, glob
from safetensors import safe_open
import math
from pathlib import Path

#

from flask import Flask, request, send_file
from werkzeug.wrappers import Response
import io
from argparse import Namespace

app = Flask(__name__)
# app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024 # 1GB

# Define your constants
# PRETRAINED_MODEL_PATH = "models/StableDiffusion/stable-diffusion-v1-5"
# INFERENCE_CONFIG = "configs/inference/inference.yaml"
# CONFIG_PATH = "./configs/prompts/9-flat2d.yaml"

# L = 16
# W = 512
# H = 512

#

allowed_models = [
    "flat2dAnimergeV3F16.vzgC.safetensors",
    "nijijourneyV51-000006.safetensors"
    "clarity2Fp16.sapf.safetensors",
]
pipeline_cache = {}
def generate_gif(prompt, n_prompt, model):
    args = Namespace(pretrained_model_path='models/StableDiffusion/stable-diffusion-v1-5', inference_config='configs/inference/inference.yaml', config='configs/prompts/10-clarity.yaml', L=16, W=512, H=512)
    # args = Namespace({
    #     'pretrained_model_path': 'models/StableDiffusion/stable-diffusion-v1-5',
    #     'inference_config': 'configs/inference/inference.yaml',
    #     'config': 'configs/prompts/10-clarity.yaml',
    #     'L': 16,
    #     'W': 512,
    #     'H': 512,
    # })

    *_, func_args = inspect.getargvalues(inspect.currentframe())
    func_args = dict(func_args)
    
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # savedir = f"samples/{Path(args.config).stem}-{time_str}"
    savedir = f"samples/output-{time_str}"
    os.makedirs(savedir)
    inference_config = OmegaConf.load(args.inference_config)

    # config  = OmegaConf.load(args.config)
    config = {
        # 'ModelName': {
        #     'base': '',
        #     'path': f"models/StableDiffusion/${model}",
        #     'motion_module': [
        #         # 'models/Motion_Module/mm_sd_v14.ckpt',
        #         'models/Motion_Module/mm_sd_v15.ckpt',
        #     ],
        #     'seed': [-1],
        #     'steps': 25,
        #     'guidance_scale': 7.5,
        #     'prompt': [
        #         prompt,
        #     ],
        #     'n_prompt': [
        #         n_prompt,
        #     ],
        # }
        'ModelName': Namespace(
            base='',
            path=f"models/StableDiffusion/{model}",
            motion_module=[
                # 'models/Motion_Module/mm_sd_v14.ckpt',
                'models/Motion_Module/mm_sd_v15.ckpt',
            ],
            seed=[-1],
            steps=25,
            guidance_scale=7.5,
            prompt=[
                prompt,
            ],
            n_prompt=[
                n_prompt,
            ],
        )
    }
    samples = []
    
    sample_idx = 0
    for model_idx, (config_key, model_config) in enumerate(list(config.items())):
        
        motion_modules = model_config.motion_module
        motion_modules = [motion_modules] if isinstance(motion_modules, str) else list(motion_modules)
        for motion_module in motion_modules:
            pipeline_key = f"{model}"
            if pipeline_key in pipeline_cache:
                pipeline = pipeline_cache[pipeline_key]
            else:
                ### >>> create validation pipeline >>> ###
                tokenizer    = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
                text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
                vae          = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")            
                unet         = UNet3DConditionModel.from_pretrained_2d(args.pretrained_model_path, subfolder="unet", unet_additional_kwargs=OmegaConf.to_container(inference_config.unet_additional_kwargs))

                if is_xformers_available(): unet.enable_xformers_memory_efficient_attention()
                else: assert False

                pipeline = AnimationPipeline(
                    vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
                    scheduler=DDIMScheduler(**OmegaConf.to_container(inference_config.noise_scheduler_kwargs)),
                ).to("cuda")

                # 1. unet ckpt
                # 1.1 motion module
                motion_module_state_dict = torch.load(motion_module, map_location="cpu")
                if "global_step" in motion_module_state_dict: func_args.update({"global_step": motion_module_state_dict["global_step"]})
                missing, unexpected = pipeline.unet.load_state_dict(motion_module_state_dict, strict=False)
                assert len(unexpected) == 0
                
                # 1.2 T2I
                if model_config.path != "":
                    if model_config.path.endswith(".ckpt"):
                        state_dict = torch.load(model_config.path)
                        pipeline.unet.load_state_dict(state_dict)
                        
                    elif model_config.path.endswith(".safetensors"):
                        state_dict = {}
                        with safe_open(model_config.path, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                state_dict[key] = f.get_tensor(key)
                                
                        is_lora = all("lora" in k for k in state_dict.keys())
                        if not is_lora:
                            base_state_dict = state_dict
                        else:
                            base_state_dict = {}
                            with safe_open(model_config.base, framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    base_state_dict[key] = f.get_tensor(key)                
                        
                        # vae
                        converted_vae_checkpoint = convert_ldm_vae_checkpoint(base_state_dict, pipeline.vae.config)
                        pipeline.vae.load_state_dict(converted_vae_checkpoint)
                        # unet
                        converted_unet_checkpoint = convert_ldm_unet_checkpoint(base_state_dict, pipeline.unet.config)
                        pipeline.unet.load_state_dict(converted_unet_checkpoint, strict=False)
                        # text_model
                        pipeline.text_encoder = convert_ldm_clip_checkpoint(base_state_dict)
                        
                        # import pdb
                        # pdb.set_trace()
                        if is_lora:
                            pipeline = convert_lora(pipeline, state_dict, alpha=model_config.lora_alpha)

                pipeline.to("cuda")

                # add to pipeline_cache
                pipeline_cache[pipeline_key] = pipeline
                ### <<< create validation pipeline <<< ###

            prompts      = model_config.prompt
            n_prompts    = list(model_config.n_prompt) * len(prompts) if len(model_config.n_prompt) == 1 else model_config.n_prompt
            
            random_seeds = model_config.seed
            random_seeds = [random_seeds] if isinstance(random_seeds, int) else list(random_seeds)
            random_seeds = random_seeds * len(prompts) if len(random_seeds) == 1 else random_seeds
            
            config[config_key].random_seed = []
            print(f"motion module: {prompts} {n_prompts} {random_seeds}")
            for prompt_idx, (prompt, n_prompt, random_seed) in enumerate(zip(prompts, n_prompts, random_seeds)):
                
                # manually set random seed for reproduction
                if random_seed != -1: torch.manual_seed(random_seed)
                else: torch.seed()
                config[config_key].random_seed.append(torch.initial_seed())
                
                print(f"current seed: {torch.initial_seed()}")
                print(f"sampling {prompt} ...")
                sample = pipeline(
                    prompt,
                    negative_prompt     = n_prompt,
                    num_inference_steps = model_config.steps,
                    guidance_scale      = model_config.guidance_scale,
                    width               = args.W,
                    height              = args.H,
                    video_length        = args.L,
                ).videos
                samples.append(sample)

                prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
                save_videos_grid(sample, f"{savedir}/sample/{sample_idx}-{prompt}.gif")
                print(f"save to {savedir}/sample/{prompt}.gif")
                
                sample_idx += 1

    samples = torch.concat(samples)
    fileName = f"{savedir}/sample.gif"
    save_videos_grid(samples, fileName, n_rows=4)

    # OmegaConf.save(config, f"{savedir}/config.yaml")
    # print(f"save to {fileName}")

    gif_data = io.BytesIO()
    with open(fileName, 'rb') as f:
        gif_data.write(f.read())
    gif_data.seek(0)
    return gif_data


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--pretrained_model_path", type=str, default="models/StableDiffusion/stable-diffusion-v1-5",)
#     parser.add_argument("--inference_config",      type=str, default="configs/inference/inference.yaml")    
#     parser.add_argument("--config",                type=str, required=True)
    
#     parser.add_argument("--L", type=int, default=16 )
#     parser.add_argument("--W", type=int, default=512)
#     parser.add_argument("--H", type=int, default=512)

#     args = parser.parse_args()
#     main(args)

@app.route('/animatediff', methods=['POST'])
def animate_diff():
    prompt = request.form['prompt']
    if prompt == "":
        return "prompt is empty", 400
    n_prompt = request.form['n_prompt']
    if n_prompt == "":
        return "n_prompt is empty", 400
    model = request.form['model']
    # if the model is not in allowed_models, return 400 "invalid model"
    if model not in allowed_models:
        return "invalid model", 400
    
    print(f"generating gif for prompt: {prompt}, model: {model}")
    
    gif_data = generate_gif(prompt, n_prompt, model)
    
    # args = Namespace(pretrained_model_path='models/StableDiffusion/stable-diffusion-v1-5', inference_config='configs/inference/inference.yaml', config='configs/prompts/10-clarity.yaml', L=16, W=512, H=512)
    # ... use the prompt and models to generate the gif ...
    
    # gif_file = f"{savedir}/sample/{sample_idx}-{prompt}.gif"
    
    # # save gif to a BytesIO object
    # gif_data = io.BytesIO()
    # with open(gif_file, 'rb') as f:
    #     gif_data.write(f.read())
    # gif_data.seek(0)

    return send_file(gif_data, mimetype='image/gif')

if __name__ == '__main__':
    # single threaded for simplicity
    app.run(host='0.0.0.0', port=1289, threaded=False)