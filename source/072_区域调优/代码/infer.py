"""Load a final PTH and predict on an original image without modifying that image."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import build_model


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=build_model(saved['method'],saved['backbone']).to(args.device).eval()
    model.load_state_dict(saved['model'],strict=True)
    with Image.open(args.image) as image:
        width,height=image.size
        data=np.array(image.convert('L').resize((256,256),Image.Resampling.LANCZOS),copy=True)
    tensor=torch.from_numpy(data).float()[None,None].to(args.device)/255
    with torch.inference_mode():y=model(tensor)['y'][0].cpu().numpy()
    # Match original training resize convention: original x = input x * width / 256.
    x_known=np.arange(256)*width/256
    result={label:np.interp(np.arange(width),x_known,curve*height/256).tolist()
            for label,curve in zip(saved['labels'],y)}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(dict(image=args.image,width=width,height=height,lines=result),ensure_ascii=False))


if __name__=='__main__':main()
