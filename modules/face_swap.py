import sys
from PIL import Image
import numpy as np
import cv2
sys.path.append('../inswapper')

from inswapper.swapper import process

def perform_face_swap(images, inswapper_source_image, inswapper_source_image_indicies, inswapper_target_image_indicies):
  swapped_images = []
  swapped_images.extend(images)
  tinsim = len(inswapper_source_image)

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

  for item in images:
      for idx, image in enumerate(inswapper_source_image):
        sim = image
        sin = inswapper_source_image_indicies[idx]
        tin = inswapper_target_image_indicies[idx]
        iinsim = idx+1
        print(f"Inswapper: Source indicies: {inswapper_source_image_indicies}")
        print(f"Inswapper: Target indicies: {inswapper_target_image_indicies}")      

        result_image = process([sim], item, sin, tin, "../inswapper/checkpoints/inswapper_128.onnx")
        # swapped_images.append(result_image)
        print("=====================================")
        print(f"Inswap {iinsim} / {tinsim} Finished")
        print("=====================================")

        print("=======================================")
        print(f"Start {iinsim} / {tinsim} Restoration")
        print("=======================================")
        
        result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
        result_image = face_restoration(result_image, 
                                        True, 
                                        True, 
                                        1, 
                                        0.5,
                                        upsampler,
                                        codeformer_net,
                                        device)
        print("======================================================")
        print(f"Done restore {iinsim} / {tinsim}, start combining...")
        print("======================================================")

        print("===========================================")
        print(f"Resizing source image {iinsim} / {tinsim}")
        print("===========================================")
        original_sim_height, original_sim_width = sim.shape[:2]
        rim_height, rim_width = result_image.shape[:2]
        aspect_ratio_sim = original_sim_width / original_sim_height
        if aspect_ratio_sim >= 1:  # if wide image
          target_width = rim_width
          res_sim_height = int(target_width * aspect_ratio_sim)
          diff_sim_height = max(0, rim_height - res_sim_height)  # Ensure non-negative diff
          # Add black padding (assuming black padding)
          padding_top = int(diff_sim_height / 2)
          padding_bottom = diff_sim_height - padding_top
          resized_sim = cv2.copyMakeBorder(cv2.resize(sim, (target_width, res_sim_height), interpolation=cv2.INTER_AREA),
                                           padding_top, padding_bottom, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:  # if portrait image, no need to pad anything
          target_height = rim_height
          target_width = int(target_height * aspect_ratio_sim)
          resized_sim = cv2.resize(sim, (target_width, target_height), interpolation=cv2.INTER_AREA)

        print("=====================================================")
        print(f"Combining & appending result & source image {iinsim} / {tinsim}")
        print("=====================================================")
        # Combine result_image and resized_sim horizontally
        combined_result_image = cv2.hconcat([result_image, resized_sim])
        # Append combined_result_image to swapped_images
        swapped_images.append(combined_result_image)
        # swapped_images.append(result_image)
        print("===============")
        print(f"Done combining and append {iinsim} / {tinsim}")
        print("===============")
    
  return swapped_images
