"""Load a trained checkpoint and classify PIL images (shared by the CLI demo, YOLO eval and the web app)."""
import torch
from PIL import Image

from agriscan.train import build_model, make_transforms


class Predictor:
    def __init__(self, path, device: str = "cpu"):
        # weights_only=True: refuse to unpickle arbitrary code from a checkpoint file.
        ck = torch.load(path, map_location=device, weights_only=True)
        self.classes, self.img_size, self.arch = ck["classes"], ck["img_size"], ck["arch"]
        self.device = device
        self.model, _, _ = build_model(self.arch, len(self.classes), pretrained=False)
        self.model.load_state_dict(ck["state_dict"])
        self.model.to(device).eval()
        self.tfm = make_transforms(self.img_size, augment=False)[1]

    @torch.no_grad()
    def predict(self, images: list[Image.Image], k: int = 3) -> list[list[tuple[str, float]]]:
        x = torch.stack([self.tfm(im.convert("RGB")) for im in images]).to(self.device)
        probs = self.model(x).softmax(1).cpu()
        top = probs.topk(min(k, len(self.classes)), dim=1)
        return [[(self.classes[j], float(p)) for p, j in zip(ps, js)] for ps, js in zip(top.values, top.indices)]
