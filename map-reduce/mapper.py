#!/usr/bin/env python3
import sys
import os

# Đường dẫn venv chính xác của bạn
VENV_PATH = "/Users/toiladat/Downloads/NegRefine/venv/lib/python3.13/site-packages"
if VENV_PATH not in sys.path:
    sys.path.insert(0, VENV_PATH)

model = None

def get_model():
    global model
    if model is None:
        import torch
        import clip
        cache_dir = "/Users/toiladat/.cache/clip"
        model, _ = clip.load("ViT-B/16", device="cpu", download_root=cache_dir)
        model.eval()
    return model

for line in sys.stdin:
    word = line.strip().replace('_', ' ')
    if not word: continue
    try:
        m = get_model()
        import torch
        import clip
        with torch.no_grad():
            tokens = clip.tokenize([word]).to("cpu")
            feat = m.encode_text(tokens).to(torch.float32)
            feat /= feat.norm(dim=-1, keepdim=True)
            feat_str = ",".join(map(str, feat[0].numpy()))
            
            # Đẩy Word và Vector sang Reducer
            sys.stdout.write(f"{word}\t{feat_str}\n")
            sys.stdout.flush()
    except Exception as e:
        sys.stderr.write(f"Error on {word}: {str(e)}\n")