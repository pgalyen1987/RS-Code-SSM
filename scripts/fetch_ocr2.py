"""
Fetch high-quality reasoning traces from nvidia/OpenCodeReasoning-2
and merge into data/all_traces.jsonl.

Usage:
    python scripts/fetch_ocr2.py --n 5000 --min-pass-rate 0.8
"""
import argparse, json, re, sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=5000, help='number of traces to add')
    p.add_argument('--min-pass-rate', type=float, default=0.8)
    p.add_argument('--output', default='data/all_traces.jsonl')
    return p.parse_args()

def strip_think_tags(text):
    """Extract content from <think>...</think>, or return text as-is."""
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def main():
    args = parse_args()
    from datasets import load_dataset

    out_path = Path(args.output)
    out_path.parent.mkdir(exist_ok=True)

    # Load existing traces to avoid duplicates
    seen_ids = set()
    existing = []
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                rec = json.loads(line)
                existing.append(rec)
                pid = rec.get('problem_id') or rec.get('eu_id')
                if pid:
                    seen_ids.add(pid)
        print(f'Existing traces: {len(existing)}')

    print(f'Streaming nvidia/OpenCodeReasoning-2 (pass_rate >= {args.min_pass_rate}, judgement=right)...')
    ds = load_dataset('nvidia/OpenCodeReasoning-2', split='python', streaming=True)

    new_traces = []
    scanned = 0
    for row in ds:
        scanned += 1
        if scanned % 10000 == 0:
            print(f'  scanned {scanned:,} | collected {len(new_traces):,} / {args.n}')

        if row.get('judgement') != 'right':
            continue
        if float(row.get('pass_rate') or 0) < args.min_pass_rate:
            continue

        pid = f"OCR2/{row['id']}"
        if pid in seen_ids:
            continue

        thinking = strip_think_tags(row.get('r1_generation') or '')
        solution = (row.get('solution') or '').strip()
        prompt   = (row.get('question') or '').strip()

        if not thinking or not solution or not prompt:
            continue

        seen_ids.add(pid)
        new_traces.append({
            'problem_id': pid,
            'source':     'opencodereasoning2',
            'prompt':     prompt,
            'thinking':   thinking,
            'solution':   solution,
            'pass_rate':  row.get('pass_rate'),
            'difficulty': row.get('difficulty'),
            'dataset':    row.get('dataset'),
        })

        if len(new_traces) >= args.n:
            break

    print(f'Collected {len(new_traces)} new traces (scanned {scanned:,})')

    # Write merged file
    all_traces = existing + new_traces
    with open(out_path, 'w') as f:
        for rec in all_traces:
            f.write(json.dumps(rec) + '\n')

    print(f'Written {len(all_traces)} total traces → {out_path}')

if __name__ == '__main__':
    main()
