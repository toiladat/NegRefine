#!/usr/bin/env python3
import sys

def main():
    all_scores = []

    for line in sys.stdin:
        try:
            parts = line.strip().split('\t')
            if len(parts) != 2: continue
            
            word, score_str = parts
            all_scores.append((word, float(score_str)))
        except:
            continue

    if not all_scores:
        return

    # Sắp xếp theo score tăng dần (nhỏ nhất = xa nhất)
    all_scores.sort(key=lambda x: x[1])
    
    # Lấy 15% đầu tiên
    limit = int(len(all_scores) * 0.15)
    
    for word, score in all_scores[:limit]:
        sys.stdout.write(f"{word}\t{score:.6f}\n")

if __name__ == "__main__":
    main()