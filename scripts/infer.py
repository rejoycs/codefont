# ===============================
# Imports
# ===============================
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import argparse
import numpy as np

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

from dataloader.inferloader import ContrastiveLoader_Infer
from models.style_encoder import StyleEncoder
from models.content_encoder import ContentEncoder
from models.decoder import StyleContentDecoder

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ===============================
# Arguments
# ===============================
def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--DATA_PATH", type=str, default="./Dataset")
    parser.add_argument("--EMB_DIM", type=int, default=256)
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--RESULTS_PATH", type=str, default="./results")
    parser.add_argument("--STYLE_ENCODER_PATH", type=str, default="./best_style_encoder.pth")
    parser.add_argument("--CONTENT_ENCODER_PATH", type=str, default="./best_content_encoder.pth")
    parser.add_argument("--DECODER_PATH", type=str, default="./best_decoder.pth")
    parser.add_argument("--IMG_SIZE", type=int, default=64)
    parser.add_argument("--DEVICE", type=str, default="cuda:0")

    return parser.parse_args()

# ===============================
# Infer
# ===============================
def infer(args):

    DEVICE = args.DEVICE
    os.makedirs(args.RESULTS_PATH, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.IMG_SIZE, args.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = ContrastiveLoader_Infer(
        args.DATA_PATH,
        args.IMG_SIZE,
        transform,
        "test",
    )

    print("Length of dataset: ",len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    EMB_DIM = args.EMB_DIM

    style_encoder = StyleEncoder(3, 64, EMB_DIM).to(DEVICE)
    content_encoder = ContentEncoder(3, 64, EMB_DIM).to(DEVICE)
    decoder = StyleContentDecoder(EMB_DIM, EMB_DIM, 3).to(DEVICE)

    style_encoder.load_state_dict(torch.load(args.STYLE_ENCODER_PATH, map_location=DEVICE))
    content_encoder.load_state_dict(torch.load(args.CONTENT_ENCODER_PATH, map_location=DEVICE))
    decoder.load_state_dict(torch.load(args.DECODER_PATH, map_location=DEVICE))

    style_encoder.eval()
    content_encoder.eval()
    decoder.eval()

    # ===============================
    # Metrics
    # ===============================
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(DEVICE)
    fid_metric = FrechetInceptionDistance(feature=2048).to(DEVICE)
    l1_metric = nn.L1Loss()

    total_rmse = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_l1 = 0.0
    count = 0

    print("Running Inference...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):

            style = batch["style_img"].to(DEVICE)
            content = batch["content_img"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            style_name = batch["style_name"]
            char_name = batch["char_name"]

            ss = style_encoder(style)
            cc = content_encoder(content)

            ss = F.normalize(ss, dim=1)
            cc = F.normalize(cc, dim=1)

            fake = decoder(ss, cc)

            fake_img = (fake + 1) / 2
            target_img = (target + 1) / 2

            for i in range(fake.size(0)):
                style_folder = os.path.join(args.RESULTS_PATH, style_name[i])
                os.makedirs(style_folder, exist_ok=True)

                save_path = os.path.join(
                    style_folder,
                    f"{char_name[i]}"
                )


                grid = torch.cat(
                    [fake_img[i]],
                    dim=2
                )

                vutils.save_image(grid, save_path, normalize=False)


            # ---- Metrics ----
            rmse = torch.sqrt(F.mse_loss(fake_img, target_img))
            l1 = l1_metric(fake_img, target_img)
            ssim_val = ssim_metric(fake_img, target_img)
            lpips_val = lpips_metric(fake, target)

            total_rmse += rmse.item()
            total_ssim += ssim_val.item()
            total_lpips += lpips_val.item()
            total_l1 += l1.item()

            fake_uint8 = (fake_img * 255).to(torch.uint8)
            real_uint8 = (target_img * 255).to(torch.uint8)

            fid_metric.update(real_uint8, real=True)
            fid_metric.update(fake_uint8, real=False)

            count += 1

    # ===============================
    # Final Metrics
    # ===============================
    mean_rmse = total_rmse / count
    mean_ssim = total_ssim / count
    mean_lpips = total_lpips / count
    mean_l1 = total_l1 / count
    fid_score = fid_metric.compute().item()

    print("\n============================")
    print(f"Mean RMSE : {mean_rmse:.6f}")
    print(f"Mean SSIM : {mean_ssim:.6f}")
    print(f"Mean LPIPS: {mean_lpips:.6f}")
    print(f"Mean L1 : {mean_l1:.6f}")
    print(f"FID Score : {fid_score:.6f}")
    print("============================")


if __name__ == "__main__":
    args = arguments()
    infer(args)
