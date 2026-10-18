import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import random
import numpy as np
import unicodedata
import pickle

seed = 42
random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class ContrastiveLoader(Dataset):
    def __init__(
        self,
        root,
        size=64,
        transform=None,
        cat="train",
    ):
        super().__init__()

        self.root = root
        self.cat = cat
        self.size = size
        self.transform = transform

        # -----------------------------
        # Paths
        # -----------------------------
        self.style_root = os.path.join(root, cat)
        self.content_root = os.path.join(root, "content")

        # -----------------------------
        # Styles and characters
        # -----------------------------
        self.styles = sorted(os.listdir(self.style_root))

        # assume all styles share same charset
        self.contents = sorted(
            os.listdir(os.path.join(self.style_root, self.styles[0]))
        )

        # -----------------------------
        # Deterministic Cartesian pairing
        # -----------------------------
        self.samples = [
            (style, char)
            for style in self.styles
            for char in self.contents
        ]

    # --------------------------------------------------
    def __len__(self):
        return len(self.samples)

    # --------------------------------------------------
    def _load(self, path):

        path = unicodedata.normalize("NFC", path)

        for _ in range(3):
            try:
                img = Image.open(path).convert("RGB")

                if self.transform:
                    img = self.transform(img)
                else:
                    img = img.resize((self.size, self.size))
                    img = torch.from_numpy(
                        np.array(img)
                    ).permute(2, 0, 1) / 255.0

                return img

            except Exception:
                print("Image loading issue:", path)

        return torch.zeros(3, self.size, self.size)

    # --------------------------------------------------
    def _choice_except(self, seq, exclude):
        cand = [x for x in seq if x != exclude]
        return random.choice(cand)

    # --------------------------------------------------
    def __getitem__(self, index):

        style_s, content_c = self.samples[index]
        remaining = list(set(self.contents) - set([content_c]))
        target_font_path = os.path.join(self.style_root, style_s)
        style_c = random.choice(remaining)

        # ==================================================
        # Paths
        # ==================================================
        path_style = os.path.join(target_font_path, style_c)
        path_content = os.path.join(self.content_root, content_c)
        path_target = os.path.join(target_font_path, content_c)

        # ==================================================
        # Contrastive anchors
        # ==================================================
        # style positive → same style, different char
        style_pos_anchor = os.path.join(
            target_font_path,
            self._choice_except(remaining, style_c)
        )

        # style negative → different style
        style_neg_anchor = os.path.join(
            self.style_root,
            self._choice_except(self.styles, style_s),
            random.choice(self.contents)
        )

        # content positive → same char, different style
        content_pos_anchor = os.path.join(
            self.style_root,
            self._choice_except(self.styles, style_s),
            content_c
        )

        # content negative → diff char + diff style
        content_neg_anchor = os.path.join(
            self.style_root,
            random.choice(self.styles),
            self._choice_except(self.contents, content_c)
        )

        return {
            "style_img": self._load(path_style),
            "content_img": self._load(path_content),
            "target": self._load(path_target),

            "style_pos": self._load(style_pos_anchor),
            "style_neg": self._load(style_neg_anchor),

            "content_pos": self._load(content_pos_anchor),
            "content_neg": self._load(content_neg_anchor),

            "style_id": torch.tensor(self.styles.index(style_s)),
            "content_id": torch.tensor(self.contents.index(content_c)),
        }