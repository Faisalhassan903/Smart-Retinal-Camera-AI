"""Shared training/inference definitions, matching the historical preprocessing."""
import hashlib
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms

NOTICE = "Research only. Not clinically validated or suitable for diagnosis."
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def build_model(pretrained=False, fine_tune=False):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    for parameter in model.parameters():
        parameter.requires_grad = fine_tune
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    return model


def quality_metrics(image):
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY)
    return {'blur_score': round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
            'brightness': round(float(gray.mean()), 2),
            'contrast': round(float(gray.std()), 2)}


def review_flags(quality, score):
    flags = []
    if quality['blur_score'] < 10:
        flags.append('Low edge detail')
    if quality['brightness'] < 40:
        flags.append('Low brightness')
    if quality['brightness'] > 100:
        flags.append('High brightness')
    if quality['contrast'] < 25:
        flags.append('Low contrast')
    uncertain = 0.4 <= score <= 0.6
    return flags, uncertain


class Predictor:
    """Serialise Grad-CAM hooks when cached across Streamlit sessions."""
    def __init__(self, checkpoint, device='cpu'):
        self.checkpoint = Path(checkpoint)
        self.fingerprint = sha256(self.checkpoint)
        self.device = torch.device(device)
        self.model = build_model()
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.lock = threading.RLock()

    def predict(self, image):
        image = image.convert('RGB')
        with self.lock, torch.inference_mode():
            score = torch.sigmoid(self.model(TRANSFORM(image).unsqueeze(0).to(self.device))).item()
        quality = quality_metrics(image)
        flags, uncertain = review_flags(quality, score)
        return {'prediction': 'DR' if score >= 0.5 else 'No DR',
                'dr_score': score, 'decision_threshold': 0.5, 'quality': quality,
                'quality_flags': flags, 'uncertain_prediction': uncertain,
                'review_required': bool(flags) or uncertain,
                'checkpoint_sha256': self.fingerprint, 'notice': NOTICE}

    def gradcam(self, image):
        """Positive DR-logit map, even when the predicted class is No DR."""
        with self.lock, torch.enable_grad():
            activations = []
            handle = self.model.layer4[-1].register_forward_hook(
                lambda module, inputs, output: activations.append(output))
            try:
                x = TRANSFORM(image.convert('RGB')).unsqueeze(0).to(self.device).requires_grad_(True)
                logit = self.model(x).squeeze()
                gradient = torch.autograd.grad(logit, activations[0])[0]
                cam = torch.relu((gradient.mean((2, 3), keepdim=True) * activations[0]).sum(1, keepdim=True))
                cam = torch.nn.functional.interpolate(cam, (224, 224), mode='bilinear', align_corners=False)
                cam = cam.detach().cpu().numpy().squeeze()
                span = float(cam.max() - cam.min())
                cam = (cam - cam.min()) / (span + 1e-8)
                base = np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))
                heat = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                return cv2.addWeighted(base, 0.6, heat, 0.4, 0), span > 1e-8
            finally:
                handle.remove()
