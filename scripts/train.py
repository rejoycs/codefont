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
from torch.cuda.amp import autocast, GradScaler
import argparse

from dataloader.dataloader import ContrastiveLoader

from models.style_encoder import StyleEncoder
from models.content_encoder import ContentEncoder
from models.decoder import StyleContentDecoder
from models.discriminator import NetD

from losses.contrastive_loss import CircleLoss
from losses.perceptual_loss import PerceptualLoss
from losses.cross_entropy import CrossEntropyLoss
from losses.l1_loss import L1Loss

# ===============================
# Seed Control
# ===============================
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ===============================
# Arguments
# ===============================
def arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--DATA_PATH", type=str, default="./Dataset")
    parser.add_argument("--BATCH_SIZE", type=int, default=64)
    parser.add_argument("--EMB_DIM", type=int, default=64)
    parser.add_argument("--RESULTS_PATH", type=str, default="./results")
    parser.add_argument("--STYLE_ENCODER_PATH", type=str, default="./ckpt/best_style_encoder.pth")
    parser.add_argument("--CONTENT_ENCODER_PATH", type=str, default="./ckpt/best_content_encoder.pth")
    parser.add_argument("--DECODER_PATH", type=str, default="./ckpt/best_decoder.pth")
    parser.add_argument("--DISC_PATH", type=str, default="./ckpt/best_discriminator.pth")
    parser.add_argument("--IMG_SIZE", type=int, default=64)
    parser.add_argument("--EPOCHS", type=int, default=500)
    parser.add_argument("--DEVICE", type=str, default="cuda:0")
    parser.add_argument("--TEST_INTERVAL", type=int, default=10)

    return parser.parse_args()


# ===============================
# Helper Function for Visualization
# ===============================
def visualize(style, content, target, gen, path):
    s = (style + 1) / 2
    c = (content + 1) / 2
    t = (target + 1) / 2
    g = (gen + 1) / 2
    grid = torch.cat([s, c, t, g], dim=3)
    vutils.save_image(grid.cpu(), path, nrow=1, normalize=False)


# ===============================
# Training
# ===============================
def train(args):

    DEVICE = args.DEVICE
    os.makedirs(args.RESULTS_PATH, exist_ok=True)

    ckpt_dir = os.path.dirname(args.STYLE_ENCODER_PATH)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ----------------------------
    # Hyperparameters
    # ----------------------------
    EMB_DIM = args.EMB_DIM
    LR_G = 2e-4
    LR_D = 2e-4

    W_L1 = 5.0
    W_PERC = 1.0
    W_ADV = 0.5
    W_CLS = 1.0
    W_CONT = 0.2

    # ----------------------------
    # Data
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((args.IMG_SIZE, args.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_set = ContrastiveLoader(
        args.DATA_PATH,
        args.IMG_SIZE,
        transform,
        'train',
    )

    test_set = ContrastiveLoader(
        args.DATA_PATH,
        args.IMG_SIZE,
        transform,
        'valid',
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    num_styles = len(train_set.styles)
    num_contents = len(train_set.contents)

    # ----------------------------
    # Models
    # ----------------------------
    style_encoder = StyleEncoder(3, 64, EMB_DIM).to(DEVICE)
    content_encoder = ContentEncoder(3, 64, EMB_DIM).to(DEVICE)
    decoder = StyleContentDecoder(EMB_DIM,EMB_DIM,3).to(DEVICE)
    disc = NetD(3, 64, num_styles, num_contents).to(DEVICE)

    # ----------------------------
    # Losses
    # ----------------------------
    contra_style = CircleLoss()
    contra_content = CircleLoss()
    perc = PerceptualLoss().to(DEVICE)

    CE_style = CrossEntropyLoss()
    CE_content = CrossEntropyLoss()
    L1 = L1Loss()


    # ----------------------------
    # Optimizers & Schedulers
    # ----------------------------
    opt_G = torch.optim.Adam(
        list(style_encoder.parameters())+list(content_encoder.parameters())+ list(decoder.parameters()),
        lr=LR_G,
        betas=(0.5, 0.999)
    )

    opt_D = torch.optim.Adam(
        disc.parameters(),
        lr=LR_D,
        betas=(0.5, 0.999)
    )

    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=args.EPOCHS)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=args.EPOCHS)

    scaler = GradScaler()

    # ===============================
    # Training Loop
    # ===============================
    for epoch in range(1, args.EPOCHS + 1):

        style_encoder.train()
        content_encoder.train()
        decoder.train()
        disc.train()

        pbar = tqdm(train_loader, desc=f"[TRAIN] Epoch {epoch}")

        for batch in pbar:

            style = batch["style_img"].to(DEVICE)
            content = batch["content_img"].to(DEVICE)
            target = batch["target"].to(DEVICE)

            style_id = batch["style_id"].to(DEVICE)
            content_id = batch["content_id"].to(DEVICE)

            style_pos = batch["style_pos"].to(DEVICE)
            style_neg = batch["style_neg"].to(DEVICE)
            content_pos = batch["content_pos"].to(DEVICE)
            content_neg = batch["content_neg"].to(DEVICE)

            # =========================
            # Generator
            # =========================
            opt_G.zero_grad()

            with autocast():

                ss = style_encoder(style)
                cc = content_encoder(content)

                ss = F.normalize(ss, dim=1)
                cc = F.normalize(cc, dim=1)

                fake = decoder(ss, cc)

                # ---- contrastive refs (no grad) ----
                with torch.no_grad():
                    ps = style_encoder(style_pos)
                    ns = style_encoder(style_neg)
                    pc = content_encoder(content_pos)
                    nc = content_encoder(content_neg)

                ps = F.normalize(ps, dim=1)
                ns = F.normalize(ns, dim=1)
                pc = F.normalize(pc, dim=1)
                nc = F.normalize(nc, dim=1)

                # ---- losses ----
                L_rec = W_L1 * L1(fake, target) 
                L_perc = W_PERC * perc(fake, target)

                L_sup = (
                    W_CONT * (contra_style(ss, ps, ns) + contra_content(cc, pc, nc))
                )

                adv_out, sty_logit, con_logit = disc(fake)
                L_adv = -W_ADV * adv_out.mean()

                
                L_cls = (
                    W_CLS * (CE_style(sty_logit, style_id)
                    + CE_content(con_logit, content_id))
                )
                

                LG = L_rec + L_sup + L_adv + L_cls + L_perc

            scaler.scale(LG).backward()

            torch.nn.utils.clip_grad_norm_(
                list(style_encoder.parameters()) +list(content_encoder.parameters())+ list(decoder.parameters()),1.0
            )

            scaler.step(opt_G)
            scaler.update()

            # =========================
            # Discriminator
            # =========================
            opt_D.zero_grad()

            with autocast():
                real_adv, real_style, real_content = disc(target)
                fake_adv, _, _ = disc(fake.detach())

                L_D_adv = W_ADV * (
                    F.relu(1 - real_adv).mean() +
                    F.relu(1 + fake_adv).mean()
                )

                
                L_D_cls = (
                    W_CLS * (CE_style(real_style, style_id)+CE_content(real_content, content_id))
                )

                LD = L_D_adv + L_D_cls

            scaler.scale(LD).backward()
            scaler.step(opt_D)
            scaler.update()

            pbar.set_postfix(G=LG.item(), D=LD.item())

        scheduler_G.step()
        scheduler_D.step()

        torch.save(style_encoder.state_dict(), args.STYLE_ENCODER_PATH)
        torch.save(content_encoder.state_dict(), args.CONTENT_ENCODER_PATH)
        torch.save(decoder.state_dict(), args.DECODER_PATH)
        torch.save(disc.state_dict(), args.DISC_PATH)

        # =========================
        # Visualization of samples
        # =========================
        if epoch % args.TEST_INTERVAL == 0:
            style_encoder.eval()
            content_encoder.eval()
            decoder.eval()

            with torch.no_grad():
                for loader, tag in [(train_loader, "train"), (test_loader, "test")]:
                    for batch in loader:
                        style = batch["style_img"].to(DEVICE)
                        content = batch["content_img"].to(DEVICE)
                        target = batch["target"].to(DEVICE)

                        idx = torch.randperm(style.size(0))[:8]
                        style = style[idx]
                        content = content[idx]
                        target = target[idx]

                        ss = style_encoder(style)
                        cc = content_encoder(content)

                        ss = F.normalize(ss, dim=1)
                        cc = F.normalize(cc, dim=1)

                        fake = decoder(ss, cc)

                        visualize(
                            style, content, target, fake,
                            os.path.join(args.RESULTS_PATH, f"{tag}_epoch_{epoch:03d}.png")
                        )
                        break

    print("[DONE] Training Complete")

if __name__ == "__main__":
    args = arguments()
    train(args)
