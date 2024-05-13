import sys
sys.path.append('./CodeFormer/CodeFormer')

import os
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from facelib.utils.face_restoration_helper import FaceRestoreHelper
from facelib.utils.misc import is_gray
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.realesrgan_utils import RealESRGANer
from basicsr.utils.registry import ARCH_REGISTRY


def check_ckpts():
    pretrain_model_url = {
        'codeformer': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth',
        'detection': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth',
        'parsing': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth',
        'realesrgan': 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth'
    }
    # download weights
    if not os.path.exists('CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth'):
        load_file_from_url(url=pretrain_model_url['codeformer'], model_dir='CodeFormer/CodeFormer/weights/CodeFormer', progress=True, file_name=None)
    if not os.path.exists('CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth'):
        load_file_from_url(url=pretrain_model_url['detection'], model_dir='CodeFormer/CodeFormer/weights/facelib', progress=True, file_name=None)
    if not os.path.exists('CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth'):
        load_file_from_url(url=pretrain_model_url['parsing'], model_dir='CodeFormer/CodeFormer/weights/facelib', progress=True, file_name=None)
    if not os.path.exists('CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth'):
        load_file_from_url(url=pretrain_model_url['realesrgan'], model_dir='CodeFormer/CodeFormer/weights/realesrgan', progress=True, file_name=None)
    
    
# set enhancer with RealESRGAN
def set_realesrgan():
    half = True if torch.cuda.is_available() else False
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=2,
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth",
        model=model,
        tile=400,
        tile_pad=40,
        pre_pad=0,
        half=half,
    )
    return upsampler


def face_restoration(img, background_enhance, face_upsample, upscale, codeformer_fidelity, upsampler, codeformer_net, device, inpaint):
    """Run a single prediction on the model"""
    try: # global try
        # take the default setting for the demo
        has_aligned = False
        only_center_face = False
        draw_box = False
        detection_model = "retinaface_resnet50"

        background_enhance = background_enhance if background_enhance is not None else True
        face_upsample = face_upsample if face_upsample is not None else True
        upscale = upscale if (upscale is not None and upscale > 0) else 2

        upscale = int(upscale) # convert type to int
        if upscale > 4: # avoid memory exceeded due to too large upscale
            upscale = 4 
        if upscale > 2 and max(img.shape[:2])>1000: # avoid memory exceeded due to too large img resolution
            upscale = 2 
        if max(img.shape[:2]) > 1500: # avoid memory exceeded due to too large img resolution
            upscale = 1
            background_enhance = False
            face_upsample = False

        face_helper = FaceRestoreHelper(
            upscale,
            face_size=512,
            crop_ratio=(1, 1),
            det_model=detection_model,
            save_ext="png",
            use_parse=True,
        )
        bg_upsampler = upsampler if background_enhance else None
        face_upsampler = upsampler if face_upsample else None

        if has_aligned:
            # the input faces are already cropped and aligned
            img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
            face_helper.is_gray = is_gray(img, threshold=5)
            face_helper.cropped_faces = [img]
        else:
            face_helper.read_image(img)
            # get face landmarks for each face
            num_det_faces = face_helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            # align and warp each face
            face_helper.align_warp_face()

        # face restoration for each cropped face
        for idx, cropped_face in enumerate(face_helper.cropped_faces):
            # prepare data
            cropped_face_t = img2tensor(
                cropped_face / 255.0, bgr2rgb=True, float32=True
            )
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

            try:
                with torch.no_grad():
                    output = codeformer_net(
                        cropped_face_t, w=codeformer_fidelity, adain=True
                    )[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except RuntimeError as error:
                print(f"Failed inference for CodeFormer: {error}")
                restored_face = tensor2img(
                    cropped_face_t, rgb2bgr=True, min_max=(-1, 1)
                )

            restored_face = restored_face.astype("uint8")
            face_helper.add_restored_face(restored_face)

            if inpaint:
                import modules.inpaint_worker as inpaint_worker
                from modules.async_worker import progressbar, inpaint_strength, inpaint_disable_initial_latent, inpaint_respective_field, ins_en_steps, switch, refiner_swap_method
                import modules.default_pipeline as pipeline
                import modules.core as core
                import modules.flags as flags
                import modules.config
                 # === Improve detail Settings === Use face for initial latent
            # inpaint_disable_initial_latent = False
            # inpaint_engine = 'None'
            # inpaint_strength = ins_en_den
            # inpaint_respective_field = 0

# =============================================
                steps=ins_en_steps
                inpaint_image = restored_face
                H, W = inpaint_image.shape[:2]  # Get image height and width
                inpaint_mask = np.ones((H, W), dtype=np.uint8) * 255  # Create a mask filled with 255 (masked)
                # inpaint_mask = inpaint_input_image['mask'][:, :, 0]
                # if inpaint_mask_upload_checkbox:
                #     if isinstance(inpaint_mask_image_upload, np.ndarray):
                #         if inpaint_mask_image_upload.ndim == 3:
                #             H, W, C = inpaint_image.shape
                #             inpaint_mask_image_upload = resample_image(inpaint_mask_image_upload, width=W, height=H)
                #             inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                #             inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                #             inpaint_mask = np.maximum(inpaint_mask, inpaint_mask_image_upload)
                # if int(inpaint_erode_or_dilate) != 0:
                #     inpaint_mask = erode_or_dilate(inpaint_mask, inpaint_erode_or_dilate)
                # if invert_mask_checkbox:
                #     inpaint_mask = 255 - inpaint_mask
                inpaint_image = HWC3(inpaint_image)
                # if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                #         and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                progressbar(async_task, 1, 'Downloading upscale models ...')
                modules.config.downloading_upscale_model()
                # if inpaint_parameterized:
                #     progressbar(async_task, 1, 'Downloading inpainter ...')
                #     inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                #         inpaint_engine)
                #     base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                #     print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                #     if refiner_model_name == 'None':
                #         use_synthetic_refiner = True
                #         refiner_switch = 0.8
                # else:
                inpaint_head_model_path, inpaint_patch_model_path = None, None
                print(f'[Inpaint] Parameterized inpaint is disabled.')
                # if inpaint_additional_prompt != '':
                #     if prompt == '':
                #         prompt = inpaint_additional_prompt
                #     else:
                #         prompt = inpaint_additional_prompt + '\n' + prompt
                # goals.append('inpaint')
            
            denoising_strength = inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=inpaint_respective_field
            )

            # if debugging_inpaint_preprocessor:
            #     yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
            #                  do_not_show_finished_images=True)
            #     return

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

        # paste_back
        if not has_aligned:
            # upsample the background
            if bg_upsampler is not None:
                # Now only support RealESRGAN for upsampling background
                bg_img = bg_upsampler.enhance(img, outscale=upscale)[0]
            else:
                bg_img = None
            face_helper.get_inverse_affine(None)
            # paste each restored face to the input image
            if face_upsample and face_upsampler is not None:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img,
                    draw_box=draw_box,
                    face_upsampler=face_upsampler,
                )
            else:
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img, draw_box=draw_box
                )

        restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
        return restored_img
    except Exception as error:
        print('Global exception', error)
        return None, None
