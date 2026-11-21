"""Research-only retinal image demonstration. No clinical validation."""
import argparse, json
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
def assess_image_quality(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    contrast = np.std(gray)

    return {
        "blur_score": round(float(blur), 2),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2)
    }

def smart_retinal_camera(image_path):
    # Read the image
    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        raise FileNotFoundError(image_path)

    # Image quality measurements
    quality = assess_image_quality(image_path)

    # Prepare image for classification
    image_rgb = cv2.cvtColor(
        image_bgr, cv2.COLOR_BGR2RGB
    )

    image_pil = Image.fromarray(image_rgb)

    image_tensor = (
        transform(image_pil)
        .unsqueeze(0)
        .to(device)
    )

    # DR classification
    with torch.inference_mode():
        logit = model(image_tensor).squeeze()
        dr_score = torch.sigmoid(logit).item()

    predicted_class = (
        "DR" if dr_score >= 0.5 else "No DR"
    )

    # Experimental review indicators
    quality_flags = []

    if quality["blur_score"] < 10:
        quality_flags.append("Low edge detail")

    if quality["brightness"] < 40:
        quality_flags.append("Low brightness")

    if quality["brightness"] > 100:
        quality_flags.append("High brightness")

    if quality["contrast"] < 25:
        quality_flags.append("Low contrast")

    # Flag predictions near the decision threshold
    uncertain = 0.4 <= dr_score <= 0.6

    review_required = bool(quality_flags) or uncertain

    return {
        "image": image_rgb,
        "dr_score": round(dr_score, 4),
        "prediction": predicted_class,
        "quality": quality,
        "quality_flags": quality_flags,
        "uncertain_prediction": uncertain,
        "review_required": review_required
    }

def main():
    global model, device, transform
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=models.resnet18(weights=None)
    model.fc=torch.nn.Linear(512,1)
    model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
    model=model.to(device).eval()
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    result=smart_retinal_camera(args.image)
    result.pop('image')
    result['notice']='Research-only; uncalibrated score; heuristic review flags; not diagnostic.'
    with open(args.output,'x') as handle: json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
