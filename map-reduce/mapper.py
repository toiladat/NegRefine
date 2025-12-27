#!/usr/bin/env python3
import sys
import os

# 1. Cập nhật đường dẫn venv CHÍNH XÁC từ lệnh python3 -c bạn vừa chạy
VENV_SITE = "/Users/toiladat/Downloads/NegRefine/venv/lib/python3.13/site-packages"
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)

# Bây giờ mới import numpy và các thư viện nặng
import numpy as np

model = None
pos_matrix = None

def get_resources():
    global model, pos_matrix
    if model is None:
        try:
            import torch
            import clip
            # Sử dụng đường dẫn tuyệt đối cho cache model
            cache_dir = "/Users/toiladat/.cache/clip"
            model, _ = clip.load("ViT-B/16", device="cpu", download_root=cache_dir)
            model.eval()
            
            # Tải thước đo Positive (File này được Hadoop copy vào working dir)
            pos_file = "positive_words.txt"
            if os.path.exists(pos_file):
                with open(pos_file, 'r') as f:
                    pos_words = [l.strip() for l in f if l.strip()]
                with torch.no_grad():
                    tokens = clip.tokenize(pos_words).to("cpu")
                    feats = model.encode_text(tokens).to(torch.float32)
                    feats /= feats.norm(dim=-1, keepdim=True)
                    pos_matrix = feats.numpy()
            else:
                sys.stderr.write("CRITICAL: positive_words.txt not found in container!\n")
        except Exception as e:
            sys.stderr.write(f"INIT ERROR: {str(e)}\n")
            raise e
    return model, pos_matrix

# Xử lý luồng dữ liệu từ WordNet
for line in sys.stdin:
    word = line.strip().replace('_', ' ')
    if not word: continue
    
    try:
        import torch
        import clip
        m, p_mat = get_resources()
        if m is None or p_mat is None: 
            continue
        
        with torch.no_grad():
            tokens = clip.tokenize([word]).to("cpu")
            feat = m.encode_text(tokens).to(torch.float32)
            feat /= feat.norm(dim=-1, keepdim=True)
            
            # Tính toán độ tương đồng Cosine (Dot product vì vector đã chuẩn hóa)
            # Điểm càng thấp = càng xa tập Positive
            score = np.max(np.dot(p_mat, feat.numpy()[0]))
            
            # Output: Word [Tab] Score
            sys.stdout.write(f"{word}\t{score:.6f}\n")
            sys.stdout.flush()
            
    except Exception as e:
        sys.stderr.write(f"REPORTER:status:Error on {word}\n")
        sys.stderr.write(f"DETAIL ERROR: {str(e)}\n")
        continue