import sys
from PIL import Image
import numpy as np
sys.path.append('../inswapper')

# def handler(async_task):
#     uov_method = args.pop()
#     uov_method = uov_method.lower()


#         if input_image_checkbox:
#             if (current_tab == 'uov' or (
#                     current_tab == 'ip' and mixing_image_prompt_and_vary_upscale)) \
#                     and uov_method != flags.disabled and uov_input_image is not None:
#                 uov_input_image = HWC3(uov_input_image)
#                 if 'vary' or 'inswap' in uov_method:
#                   # Access the goals variable
# goals = worker().goals
# uov_method = worker().uov_method

from inswapper.swapper import process

def perform_face_swap(images, inswapper_source_image, inswapper_source_image_indicies, inswapper_target_image_indicies):
  swapped_images = []

  for item in images:
      source_image = Image.fromarray(inswapper_source_image)
      print(f"Inswapper: Source indicies: {inswapper_source_image_indicies}")
      print(f"Inswapper: Target indicies: {inswapper_target_image_indicies}")      
      result_image = process([source_image], item, inswapper_source_image_indicies, inswapper_target_image_indicies, "../inswapper/checkpoints/inswapper_128.onnx")

      # if 'inswap' in uov_method) or 'inpaint' in goals:
        # Add the original & before restore image to the list
      swapped_images.append(item)
      # swapped_images.append(result_image)
  if True:
      from inswapper.restoration import face_restoration,check_ckpts,set_realesrgan,torch,ARCH_REGISTRY,cv2
      
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
      
      result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
      result_image = face_restoration(result_image, 
                                      True, 
                                      True, 
                                      1, 
                                      0.5,
                                      upsampler,
                                      codeformer_net,
                                      device)

      swapped_images.append(result_image)
  
  return swapped_images
