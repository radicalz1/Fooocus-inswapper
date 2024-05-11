import threading
import re
from modules.patch import PatchSettings, patch_settings, patch_all

patch_all()

class AsyncTask:
    def __init__(self, args):
        self.args = args
        self.yields = []
        self.results = []
        self.last_stop = False
        self.processing = False



async_tasks = []


def worker():
    global async_tasks

    import os
    import traceback
    import math
    import numpy as np
    import cv2
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    import ldm_patched.modules.model_management
    import extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import extras.ip_adapter as ip_adapter
    import extras.face_crop
    import fooocus_version
    import args_manager
    import PIL.Image as Image

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion, apply_arrays
    from modules.private_logger import log
    from extras.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, get_image_shape_ceil, set_image_shape_ceil, \
        get_shape_ceil, resample_image, erode_or_dilate, ordinal_suffix, get_enabled_loras
    from modules.upscaler import perform_upscale
    from modules.flags import Performance
    from modules.meta_parser import get_metadata_parser, MetadataScheme

    pid = os.getpid()
    print(f'Started worker with PID {pid}')

    from modules.face_swap import perform_face_swap
    from modules.pm import generate_photomaker
    from modules.instantid import generate_instantid


    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def progressbar(async_task, number, text):
        print(f'[Fooocus] {text}')
        async_task.yields.append(['preview', (number, text, None)])

    def yield_result(async_task, imgs, do_not_show_finished_images=False):
        if not isinstance(imgs, list):
            imgs = [imgs]        

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        async_task.yields.append(['results', async_task.results])
        return

    def ins_yield_result(async_task, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]        
        async_task.results = async_task.results + imgs
        async_task.yields.append(['results', async_task.results])
        return

    def build_image_wall(async_task):
        results = []

        if len(async_task.results        ) < 2:
            return

        for img in async_task.results:
            if isinstance(img, str) and os.path.exists(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return            
            results.append(img)

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task):
        execution_start_time = time.perf_counter()
        async_task.processing = True

        args = async_task.args
        args.reverse()

        prompt = args.pop()
        negative_prompt = args.pop()
        style_selections = args.pop()
        performance_selection = Performance(args.pop())
        aspect_ratios_selection = args.pop()
        image_number = args.pop()
        output_format = args.pop()
        image_seed = args.pop()
        read_wildcards_in_order = args.pop()
        sharpness = args.pop()
        guidance_scale = args.pop()
        base_model_name = args.pop()
        refiner_model_name = args.pop()
        refiner_switch = args.pop()
        loras = get_enabled_loras([[bool(args.pop()), str(args.pop()), float(args.pop())] for _ in range(modules.config.default_max_lora_number)])
        image_prompt_enabled = args.pop()
        input_image_checkbox = args.pop()
        current_tab = args.pop()
        uov_method = args.pop()
        uov_input_image = args.pop()
        outpaint_selections = args.pop()
        inpaint_input_image = args.pop()
        inpaint_additional_prompt = args.pop()
        inpaint_mask_image_upload = args.pop()

        disable_preview = args.pop()
        disable_intermediate_results = args.pop()
        disable_seed_increment = args.pop()
        adm_scaler_positive = args.pop()
        adm_scaler_negative = args.pop()
        adm_scaler_end = args.pop()
        adaptive_cfg = args.pop()
        sampler_name = args.pop()
        scheduler_name = args.pop()
        overwrite_step = args.pop()
        overwrite_switch = args.pop()
        overwrite_width = args.pop()
        overwrite_height = args.pop()
        overwrite_vary_strength = args.pop()
        overwrite_upscale_strength = args.pop()
        mixing_image_prompt_and_vary_upscale = args.pop()
        mixing_image_prompt_and_inpaint = args.pop()
        debugging_cn_preprocessor = args.pop()
        skipping_cn_preprocessor = args.pop()
        canny_low_threshold = args.pop()
        canny_high_threshold = args.pop()
        refiner_swap_method = args.pop()
        controlnet_softness = args.pop()
        freeu_enabled = args.pop()
        freeu_b1 = args.pop()
        freeu_b2 = args.pop()
        freeu_s1 = args.pop()
        freeu_s2 = args.pop()
        debugging_inpaint_preprocessor = args.pop()
        inpaint_disable_initial_latent = args.pop()
        inpaint_engine = args.pop()
        inpaint_strength = args.pop()
        inpaint_respective_field = args.pop()
        inpaint_mask_upload_checkbox = args.pop()
        invert_mask_checkbox = args.pop()
        inpaint_erode_or_dilate = args.pop()

        save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
        metadata_scheme = MetadataScheme(args.pop()) if not args_manager.args.disable_metadata else MetadataScheme.FOOOCUS

        cn_tasks = {x: [] for x in flags.ip_list}
        for _ in range(flags.controlnet_image_count):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        inswapper_enabled = args.pop()
        ins_sins = []
        ins_tins = []
        ins_sims = []
        for _ in range(flags.inswapper_image_count):
            ins_sin = args.pop()
            ins_tin = args.pop()
            ins_sim = args.pop()
            if ins_sim is not None:
                ins_sins.append(ins_sin)
                ins_tins.append(ins_tin)
                ins_sims.append(ins_sim)
                    
        print(f"Inswapper: {'ENABLED' if inswapper_enabled and ins_sims is not None else 'DISABLED'}")

        photomaker_enabled = args.pop()
        photomaker_images = args.pop()

        print(f"PhotoMaker: {'ENABLED' if photomaker_enabled else 'DISABLED'}")

        instantid_enabled = args.pop()
        instantid_image_path = args.pop()
        instantid_pose_image_path = args.pop()
        instantid_identitynet_strength_ratio = args.pop()
        instantid_adapter_strength_ratio = args.pop()

        print(f"InstantID: {'ENABLED' if instantid_enabled else 'DISABLED'}")

        outpaint_selections = [o.lower() for o in outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        if base_model_name == refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            refiner_model_name = 'None'

        steps = performance_selection.steps()

        if performance_selection == Performance.EXTREME_SPEED:
            print('Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in LCM mode.')

            refiner_model_name = 'None'
            sampler_name = 'lcm'
            scheduler_name = 'lcm'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        elif performance_selection == Performance.LIGHTNING:
            print('Enter Lightning mode.')
            progressbar(async_task, 1, 'Downloading Lightning components ...')
            loras += [(modules.config.downloading_sdxl_lightning_lora(), 1.0)]

            if refiner_model_name != 'None':
                print(f'Refiner disabled in Lightning mode.')

            refiner_model_name = 'None'
            sampler_name = 'euler'
            scheduler_name = 'sgm_uniform'
            sharpness = 0.0
            guidance_scale = 1.0
            adaptive_cfg = 1.0
            refiner_switch = 1.0
            adm_scaler_positive = 1.0
            adm_scaler_negative = 1.0
            adm_scaler_end = 0.0

        print(f'[Parameters] Adaptive CFG = {adaptive_cfg}')
        print(f'[Parameters] Sharpness = {sharpness}')
        print(f'[Parameters] ControlNet Softness = {controlnet_softness}')
        print(f'[Parameters] ADM Scale = '
              f'{adm_scaler_positive} : '
              f'{adm_scaler_negative} : '
              f'{adm_scaler_end}')

        patch_settings[pid] = PatchSettings(
            sharpness,
            adm_scaler_end,
            adm_scaler_positive,
            adm_scaler_negative,
            controlnet_softness,
            adaptive_cfg
        )

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False

        inpaint_worker.current_task = None
        inpaint_parameterized = inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(image_seed)
        print(f'[Parameters] Seed = {seed}')

        goals = []
        tasks = []        

        
        if image_prompt_enabled and cn_tasks:
            goals.append('cn')
            progressbar(async_task, 1, 'Downloading control models ...')
            if len(cn_tasks[flags.cn_canny]) > 0:
                controlnet_canny_path = modules.config.downloading_controlnet_canny()
            if len(cn_tasks[flags.cn_cpds]) > 0:
                controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
            if len(cn_tasks[flags.cn_ip]) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
            if len(cn_tasks[flags.cn_ip_face]) > 0:
                clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                    'face')
            progressbar(async_task, 1, 'Loading control models ...')

        if input_image_checkbox:
            if (current_tab == 'uov' or (
                    current_tab == 'ip' and mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' or 'inswap' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = performance_selection.steps_uov()

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (current_tab == 'inpaint' or (
                    current_tab == 'ip' and mixing_image_prompt_and_inpaint)) \
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]

                if inpaint_mask_upload_checkbox:
                    if isinstance(inpaint_mask_image_upload, np.ndarray):
                        if inpaint_mask_image_upload.ndim == 3:
                            H, W, C = inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            inpaint_mask = np.maximum(inpaint_mask, inpaint_mask_image_upload)

                if int(inpaint_erode_or_dilate) != 0:
                    inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)

                if invert_mask_checkbox:
                    inpaint_mask = 255 - inpaint_mask

                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.8
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if inpaint_additional_prompt != '':
                        if prompt == '':
                            prompt = inpaint_additional_prompt
                        else:
                            prompt = inpaint_additional_prompt + '\n' + prompt
                    goals.append('inpaint')
            if current_tab == 'ip' or \
                    mixing_image_prompt_and_vary_upscale or \
                    mixing_image_prompt_and_inpaint:
                goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(cn_tasks[flags.cn_ip_face]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        if overwrite_step > 0:
            if 'inswap' in uov_method:
                steps = 1
            else:
                steps = overwrite_step

        switch = int(round(steps * refiner_switch))

        if overwrite_switch > 0:
            switch = overwrite_switch

        if overwrite_width > 0:
            width = overwrite_width

        if overwrite_height > 0:
            height = overwrite_height

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        progressbar(async_task, 1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                        loras=loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []
            
            for i in range(image_number):
                if disable_seed_increment:
                    task_seed = seed % (constants.MAX_SEED + 1)
                else:
                    task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future
                task_prompt = apply_wildcards(prompt, task_rng, i, read_wildcards_in_order)
                task_prompt = apply_arrays(task_prompt, i)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, read_wildcards_in_order)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, read_wildcards_in_order) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    for s in style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

            
        if 'vary' in goals:
            if 'inswap' in uov_method:
                denoising_strength =0.001
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if overwrite_vary_strength > 0:
                denoising_strength = overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            uov_input_image = perform_upscale(uov_input_image)
            print(f'Image upscaled.')

            if '1.5x' in uov_method:
                f = 1.5
            elif '2x' in uov_method:
                f = 2.0
            else:
                f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in uov_method:
                direct_return = True
            elif image_is_super_large:
                print('Image is too large. Directly returned the SR image. '
                      'Usually directly return SR image at 4K resolution '
                      'yields better results than SDXL diffusion.')
                direct_return = True
            else:
                direct_return = False

            if direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                uov_input_image_path = log(uov_input_image, d, output_format=output_format)
                yield_result(async_task, uov_input_image_path, do_not_show_finished_images=True)
                return

            tiled = True
            denoising_strength = 0.382

            if overwrite_upscale_strength > 0:
                denoising_strength = overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if 'bottom' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [int(W * 0.3), 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(W * 0.3), 0]], mode='constant',
                                          constant_values=255)
                if 'right' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(W * 0.3)], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(W * 0.3)]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())
                inpaint_strength = 1.0
                inpaint_respective_field = 1.0

            denoising_strength = inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            if debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                             do_not_show_finished_images=True)
                return

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img, canny_low_threshold, canny_high_threshold)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not skipping_cn_preprocessor:
                    cn_img = extras.face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                freeu_b1,
                freeu_b2,
                freeu_s1,
                freeu_s2
            )

        all_steps = steps * image_number

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            async_task.yields.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}{ordinal_suffix(current_task_id + 1)} Sampling', y)])

        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                if async_task.last_stop is not False:
                    ldm_patched.modules.model_management.interrupt_current_processing()
                positive_cond, negative_cond = task['c'], task['uc']
                imgs = []

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                if current_tab == 'photomaker' and photomaker_enabled == True and input_image_checkbox == True:
                    print("PhotoMaker: Begin")
                    photomaker_source_images = [Image.open(image.name) for image in photomaker_images]

                    photomaker_prompt = positive_basic_workloads[0]
                    photomaker_negative_prompt = negative_basic_workloads[0]

                    # TODO: figure out 77 token limit
                    # https://github.com/huggingface/diffusers/issues/2136#issuecomment-1514969011                    
                    # if use_expansion:
                    #     photomaker_prompt = photomaker_prompt + ' ' +  expansion.replace(prompt, "")
                    
                    print(f"PhotoMaker: Positive prompt: {photomaker_prompt}")
                    print(f"PhotoMaker: Negative prompt: {photomaker_negative_prompt}")

                    imgs = generate_photomaker(photomaker_prompt, photomaker_source_images, photomaker_negative_prompt, steps, task['task_seed'], width, height, guidance_scale, loras, sampler_name, scheduler_name, async_task)

                elif current_tab == 'instantid' and instantid_enabled == True and input_image_checkbox == True:
                    print("InstantID: Begin")

                    instantid_prompt = positive_basic_workloads[0]
                    instantid_negative_prompt = negative_basic_workloads[0]

                    # TODO: figure out 77 token limit
                    # https://github.com/huggingface/diffusers/issues/2136#issuecomment-1514969011                    
                    # if use_expansion:
                    #     photomaker_prompt = photomaker_prompt + ' ' +  expansion.replace(prompt, "")
                    
                    print(f"InstantID: Positive prompt: {instantid_prompt}")
                    print(f"InstantID: Negative prompt: {instantid_negative_prompt}")

                    imgs = generate_instantid(instantid_image_path, instantid_pose_image_path, instantid_prompt, instantid_negative_prompt, steps, task['task_seed'], width, height, guidance_scale, loras, sampler_name, scheduler_name, async_task, instantid_identitynet_strength_ratio, instantid_adapter_strength_ratio)

                else:
                    imgs = pipeline.process_diffusion(
                        positive_cond=positive_cond,
                        negative_cond=negative_cond,
                        steps=steps,
                        switch=switch,
                        width=width,
                        height=height,
                        image_seed=task['task_seed'],
                        callback=callback,
                        sampler_name=final_sampler_name,
                        scheduler_name=final_scheduler_name,
                        latent=initial_latent,
                        denoise=denoising_strength,
                        tiled=tiled,
                        cfg_scale=cfg_scale,
                        refiner_swap_method=refiner_swap_method,
                    disable_preview=disable_preview
                    )

                # del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                if inswapper_enabled and ins_sims is not None:
                    # imgs = perform_face_swap(imgs, ins_sims, ins_sins, ins_tins)
                    print("===============")
                    print("Inswapper START")
                    print("===============")
                    progressbar(async_task, 13, '=== Start INSWAPPER ===')
                    import sys
                    sys.path.append('../inswapper')
                    from inswapper.swapper import process
                    from inswapper.restoration import face_restoration, check_ckpts, set_realesrgan, torch, ARCH_REGISTRY, cv2
                    # ==============================
                    # START RESTORATION REQUIREMENTS
                    # ==============================
                    progressbar(async_task, 13, '=== Start INSWAPPER : Restoration Requirements')
                    # make sure the ckpts downloaded successfully
                    check_ckpts()
                    # https://huggingface.co/spaces/sczhou/CodeFormer
                    upsampler = set_realesrgan()
                    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
                    print(f"{device}")
                    codeformer_net = ARCH_REGISTRY.get("CodeFormer")(dim_embd=512,
                                                                    codebook_size=1024,
                                                                    n_head=8,
                                                                    n_layers=9,
                                                                    connect_list=["32", "64", "128", "256"],
                                                                  ).to(device)
                    ckpt_path = "CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth"
                    checkpoint = torch.load(ckpt_path)["params_ema"]
                    codeformer_net.load_state_dict(checkpoint)
                    codeformer_net.eval()
                    # ============================
                    # END RESTORATION REQUIREMENTS
                    # ============================
                    # ======================
                    # START LOG REQUIREMENTS
                    # ======================
                    progressbar(async_task, 13, '=== Start INSWAPPER : Log Requirements')
                    d = [('Prompt', 'prompt', task['log_positive_prompt']),
                         ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                         ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                         ('Styles', 'styles', str(raw_style_selections)),
                         ('Performance', 'performance', performance_selection.value)]
                    if performance_selection.steps() != steps:
                        d.append(('Steps', 'steps', steps))
                    d += [('Resolution', 'resolution', str((width, height))),
                          ('Guidance Scale', 'guidance_scale', guidance_scale),
                          ('Sharpness', 'sharpness', sharpness),
                          ('ADM Guidance', 'adm_guidance', str((
                              modules.patch.patch_settings[pid].positive_adm_scale,
                              modules.patch.patch_settings[pid].negative_adm_scale,
                              modules.patch.patch_settings[pid].adm_scaler_end))),
                          ('Base Model', 'base_model', base_model_name),
                          ('Refiner Model', 'refiner_model', refiner_model_name),
                          ('Refiner Switch', 'refiner_switch', refiner_switch)]
                    if refiner_model_name != 'None':
                        if overwrite_switch > 0:
                            d.append(('Overwrite Switch', 'overwrite_switch', overwrite_switch))
                        if refiner_swap_method != flags.refiner_swap_method:
                            d.append(('Refiner Swap Method', 'refiner_swap_method', refiner_swap_method))
                    if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                        d.append(('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))
                    d.append(('Sampler', 'sampler', sampler_name))
                    d.append(('Scheduler', 'scheduler', scheduler_name))
                    d.append(('Seed', 'seed', str(task['task_seed'])))
                    if freeu_enabled:
                        d.append(('FreeU', 'freeu', str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))))
                    for li, (n, w) in enumerate(loras):
                        if n != 'None':
                            d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))
                    metadata_parser = None
                    if save_metadata_to_images:
                        metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                        metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                                 task['log_negative_prompt'], task['negative'],
                                                 steps, base_model_name, refiner_model_name, loras)
                    d.append(('Metadata Scheme', 'metadata_scheme', metadata_scheme.value if save_metadata_to_images else save_metadata_to_images))
                    d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))
                    # ====================
                    # END LOG REQUIREMENTS
                    # ====================

                    ins_imgs=[]
                    ins_imgs.extend(imgs)
                    log(imgs[-1], d, metadata_parser, output_format)
                    ins_yield_result(async_task, imgs[-1])
                    tinsim = len(ins_sims)
                    
                    for item in ins_imgs:
                      for idx, image in enumerate(ins_sims):
                        sim = image
                        sin = ins_sins[idx]
                        tin = ins_tins[idx]
                        iinsim = idx+1
                        print("=================================")
                        print(f"Start Inswap {iinsim} / {tinsim}")
                        print("=================================")
                        progressbar(async_task, 13, f'Start Inswap {iinsim} / {tinsim}')
                        print(f"Inswapper: Source indicies: {sin}")
                        print(f"Inswapper: Target indicies: {tin}")      
                        rim = process([sim], item, sin, tin, "../inswapper/checkpoints/inswapper_128.onnx") # result_image
                        print("==================================")
                        print(f"Finish Inswap {iinsim} / {tinsim}")
                        print("==================================")
                        print("======================================")
                        print(f"Start Restoration {iinsim} / {tinsim}")
                        print("======================================")
                        progressbar(async_task, 13, f'Start Restoration {iinsim} / {tinsim}')
                        rim = cv2.cvtColor(np.array(rim), cv2.COLOR_RGB2BGR)
                        rim_r = face_restoration(rim, True, True, 1, 0.5, upsampler, codeformer_net, device)
                        print("=======================================")
                        print(f"Finish Restoration {iinsim} / {tinsim}")
                        print("=======================================")
                        print("======================================")
                        print(f"Start Enhancement {iinsim} / {tinsim}")
                        print("======================================")
                        progressbar(async_task, 13, f'Start Enhancement {iinsim} / {tinsim}')
                        rip = core.numpy_to_pytorch(rim_r) # initial_pixels
                        # progressbar(async_task, 13, 'VAE encoding ...')
                        ins_candidate_vae, _ = pipeline.get_candidate_vae(
                            steps=steps,
                            switch=switch,
                            denoise=denoising_strength,
                            refiner_swap_method=refiner_swap_method
                        )
                        ril = core.encode_vae(vae=ins_candidate_vae, pixels=rip) # initial_latent
                        rB, rC, rH, rW = ril['samples'].shape
                        rwidth = rW * 8
                        rheight = rH * 8
                        print(f'Final resolution is {str((rheight, rwidth))}.')
                        enhance_image = pipeline.process_diffusion(
                            positive_cond=positive_cond,
                            negative_cond=negative_cond,
                            steps=15,
                            switch=switch,
                            width=width,
                            height=height,
                            image_seed=task['task_seed'],
                            callback=callback,
                            sampler_name=final_sampler_name,
                            scheduler_name=final_scheduler_name,
                            latent=ril,
                            denoise=0.5,
                            tiled=tiled,
                            cfg_scale=cfg_scale,
                            refiner_swap_method=refiner_swap_method,
                        disable_preview=disable_preview
                        )
                        rim_e = enhance_image[-1]
                        print("=======================================")
                        print(f"Finish Enhancement {iinsim} / {tinsim}")
                        print("=======================================")
                        print("=================================")
                        print(f"Start Darken {iinsim} / {tinsim}")
                        print("=================================")
                        progressbar(async_task, 13, f'Start Darken {iinsim} / {tinsim}')
                        def blend_images(bg_path, fg_path, alpha=0.7):
                            # """
                            # Blends two images with a darkening effect on the top layer.
                            # Args:
                            # img1_path: Path to the background image.
                            # img2_path: Path to the foreground image (top layer).
                            # output_path: Path to save the blended image.
                            # alpha: Opacity of the top layer (0.0 - fully transparent, 1.0 - fully opaque).
                            # """
                            background = Image.open(bg_path)
                            foreground = Image.open(fg_path)
                            # Ensure images have the same size
                            # background = background.resize(foreground.size)
                            # Convert images to RGBA mode (for alpha channel)
                            background = background.convert("RGBA")
                            foreground = foreground.convert("RGBA")
                            # Invert foreground alpha for darkening effect (more alpha = darker)
                            inverted_alpha = 1.0 - foreground.split()[-1]
                            # Blend the images using weighted addition
                            blended_image = Image.blend(background, foreground, alpha=alpha)
                            # Apply the inverted alpha to the top layer
                            blended_image.putalpha(inverted_alpha)
                            return blended_image
                            # # Save the blended image
                            # blended_image.save(output_path)
                        # Example usage
                        bg_path = rim_r
                        fg_path = rim_e
                        rim_d = blend_images(bg_path, fg_path)
                        print("==================================")
                        print(f"Finish Darken {iinsim} / {tinsim}")
                        print("==================================")
                        print("=======================================================")
                        print(f"Start Resizing Inswap Source Image {iinsim} / {tinsim}")
                        print("=======================================================")
                        progressbar(async_task, 13, f'Start Resizing Inswap Source Image {iinsim} / {tinsim}')
                        original_sim_height, original_sim_width = sim.shape[:2]
                        aspect_ratio_sim = original_sim_width / original_sim_height
                        original_rimr_height, original_rimr_width = rim_r.shape[:2]
                        aspect_ratio_rimr = original_rimr_width / original_rimr_height
                        rim_height, rim_width = rim_e.shape[:2]
                        if aspect_ratio_sim > 1:  # if wide image
                          target_width = rim_width
                          res_sim_height = int(target_width / aspect_ratio_sim)
                          diff_sim_height = max(0, rim_height - res_sim_height)  # Ensure non-negative diff
                          # Add black padding (assuming black padding)
                          padding_top = int(diff_sim_height / 2)
                          padding_bottom = diff_sim_height - padding_top
                          resized_sim = cv2.copyMakeBorder(cv2.resize(sim, (target_width, res_sim_height), interpolation=cv2.INTER_AREA),
                                                           padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        if aspect_ratio_sim <= 1:  # if square / portrait image, no need to pad anything
                          target_height = rim_height
                          target_width = int(target_height * aspect_ratio_sim)
                          resized_sim = cv2.resize(sim, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        # RESIZE rim_r -- Not needed
                        # if aspect_ratio_rimr > 1:  # if wide image
                        #   target_width = rim_width
                        #   res_rimr_height = int(target_width / aspect_ratio_rimr)
                        #   diff_rimr_height = max(0, rim_height - res_rimr_height)  # Ensure non-negative diff
                        #   # Add black padding (assuming black padding)
                        #   padding_top = int(diff_rimr_height / 2)
                        #   padding_bottom = diff_rimr_height - padding_top
                        #   resized_rimr = cv2.copyMakeBorder(cv2.resize(rim_r, (target_width, res_rimr_height), interpolation=cv2.INTER_AREA),
                        #                                    padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                        # if aspect_ratio_rimr <= 1:  # if square / portrait image, no need to pad anything
                        #   target_height = rim_height
                        #   target_width = int(target_height * aspect_ratio_rimr)
                        #   resized_rimr = cv2.resize(rim_r, (target_width, target_height), interpolation=cv2.INTER_AREA)
                        print("=====================================================")
                        print(f"Start Horizontal Concatenation {iinsim} / {tinsim}")
                        print("=====================================================")
                        progressbar(async_task, 13, f'Start Horizontal Concatenation {iinsim} / {tinsim}')
                        # Print image shapes for debugging (optional)
                        print("===========================================")
                        print(f"restored_image.shape: {rim_r.shape}")
                        # print(f"resized_rimr.shape: {resized_rimr.shape}")
                        print(f"enhanced_image.shape: {rim_e.shape}")
                        print(f"darken_image.shape: {rim_d.shape}")
                        print(f"resized_sim.shape: {resized_sim.shape}")
                        print("===========================================")
                        # Combine result_image and resized_sim horizontally
                        combined_result_image = cv2.hconcat([rim_d, rim_e, rim_r, resized_sim])
                        # Append combined_result_image to swapped_images
                        # ins_imgs.append(combined_result_image)
                        print("====================================================")
                        print(f"Finish Horizontal Concatenation {iinsim} / {tinsim}")
                        print("====================================================")
                        print("==================================")
                        print(f"Start Logging {iinsim} / {tinsim}")
                        print("==================================")
                        progressbar(async_task, 13, f'Start Logging {iinsim} / {tinsim}')
                        # ins_img_paths = []
                        # for x in ins_imgs:
                        log(combined_result_image, d, metadata_parser, output_format)
                        ins_yield_result(async_task, combined_result_image)
                        print("===================================")
                        print(f"Finish logging {iinsim} / {tinsim}")
                        print("===================================")


                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if not inswapper_enabled and ins_sims is None:
                    img_paths = []
                    for x in imgs:
                        d = [('Prompt', 'prompt', task['log_positive_prompt']),
                             ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                             ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                             ('Styles', 'styles', str(raw_style_selections)),
                             ('Performance', 'performance', performance_selection.value)]
    
                        if performance_selection.steps() != steps:
                            d.append(('Steps', 'steps', steps))
    
                        d += [('Resolution', 'resolution', str((width, height))),
                              ('Guidance Scale', 'guidance_scale', guidance_scale),
                              ('Sharpness', 'sharpness', sharpness),
                              ('ADM Guidance', 'adm_guidance', str((
                                  modules.patch.patch_settings[pid].positive_adm_scale,
                                  modules.patch.patch_settings[pid].negative_adm_scale,
                                  modules.patch.patch_settings[pid].adm_scaler_end))),
                              ('Base Model', 'base_model', base_model_name),
                              ('Refiner Model', 'refiner_model', refiner_model_name),
                              ('Refiner Switch', 'refiner_switch', refiner_switch)]
    
                        if refiner_model_name != 'None':
                            if overwrite_switch > 0:
                                d.append(('Overwrite Switch', 'overwrite_switch', overwrite_switch))
                            if refiner_swap_method != flags.refiner_swap_method:
                                d.append(('Refiner Swap Method', 'refiner_swap_method', refiner_swap_method))
                        if modules.patch.patch_settings[pid].adaptive_cfg != modules.config.default_cfg_tsnr:
                            d.append(('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[pid].adaptive_cfg))
    
                        d.append(('Sampler', 'sampler', sampler_name))
                        d.append(('Scheduler', 'scheduler', scheduler_name))
                        d.append(('Seed', 'seed', str(task['task_seed'])))
    
                        if freeu_enabled:
                            d.append(('FreeU', 'freeu', str((freeu_b1, freeu_b2, freeu_s1, freeu_s2))))
    
                        for li, (n, w) in enumerate(loras):
                            if n != 'None':
                                d.append((f'LoRA {li + 1}', f'lora_combined_{li + 1}', f'{n} : {w}'))
    
                        metadata_parser = None
                        if save_metadata_to_images:
                            metadata_parser = modules.meta_parser.get_metadata_parser(metadata_scheme)
                            metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                                     task['log_negative_prompt'], task['negative'],
                                                     steps, base_model_name, refiner_model_name, loras)
                        d.append(('Metadata Scheme', 'metadata_scheme', metadata_scheme.value if save_metadata_to_images else save_metadata_to_images))
                        d.append(('Version', 'version', 'Fooocus v' + fooocus_version.version))
                        img_paths.append(log(x, d, metadata_parser, output_format))
    
                    yield_result(async_task, img_paths, do_not_show_finished_images=len(tasks) == 1 or disable_intermediate_results)
                    async_task.yields.append(['results', async_task.results])
            except ldm_patched.modules.model_management.InterruptProcessingException as e:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')
        async_task.processing = False
        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            generate_image_grid = task.args.pop(0)

            try:
                handler(task)
                if generate_image_grid:
                    build_image_wall(task)
                task.yields.append(['finish', task.results])
                pipeline.prepare_text_encoder(async_call=True)
            except:
                traceback.print_exc()
                task.yields.append(['finish', task.results])
            finally:
                if pid in modules.patch.patch_settings:
                    del modules.patch.patch_settings[pid]
    pass


threading.Thread(target=worker, daemon=True).start()
