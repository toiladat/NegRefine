#!/usr/bin/env python3
import sys
import os
import numpy as np

# Ép đường dẫn venv cho Reducer
VENV_PATH = "/Users/toiladat/Downloads/NegRefine/venv/lib/python3.13/site-packages"
if VENV_PATH not in sys.path:
    sys.path.insert(0, VENV_PATH)

pos_matrix = None

def get_pos_matrix():
    global pos_matrix
    if pos_matrix is None:
        import torch
        import clip
        cache_dir = "/Users/toiladat/.cache/clip"
        model, _ = clip.load("ViT-B/16", device="cpu", download_root=cache_dir)
        
        # Load tập Positive làm thước đo
        if os.path.exists("positive_words.txt"):
            with open("positive_words.txt", "r") as f:
                pos_words = [l.strip() for l in f if l.strip()]
            with torch.no_grad():
                tokens = clip.tokenize(pos_words).to("cpu")
                feats = model.encode_text(tokens).to(torch.float32)
                feats /= feats.norm(dim=-1, keepdim=True)
                pos_matrix = feats.numpy()
    return pos_matrix

def main():
    try:
        p_mat = get_pos_matrix()
    except Exception as e:
        sys.stderr.write(f"Reducer Init Error: {e}\n")
        return

    word_scores = []
    for line in sys.stdin:
        try:
            parts = line.strip().split('\t')
            if len(parts) != 2: continue
            word, feat_str = parts
            
            # Chuyển string vector lại thành numpy array
            vec = np.fromstring(feat_str, sep=',').astype(np.float32)
            
            # Tính tương đồng (Cosine Similarity)
            score = np.max(np.dot(p_mat, vec))
            word_scores.append((word, score))
        except:
            continue

    if not word_scores: return

    # Sắp xếp và lấy 15% thấp nhất (xa Positive nhất)
    word_scores.sort(key=lambda x: x[1])
    limit = int(len(word_scores) * 0.15)
    for word, score in word_scores[:limit]:
        sys.stdout.write(f"{word}\t{score:.6f}\n")

if __name__ == "__main__":
    main()