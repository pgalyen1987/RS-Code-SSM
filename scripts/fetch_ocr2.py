"""
Fetch high-quality reasoning traces from:
  - nvidia/OpenCodeReasoning-2  (competitive programming, verified)
  - open-r1/codeforces-cots     (Codeforces, ~84% verified)

and merge into data/all_traces.jsonl.

Usage:
    python scripts/fetch_ocr2.py --n 50000 --min-pass-rate 0.8
    python scripts/fetch_ocr2.py --n 50000 --source codeforces
    python scripts/fetch_ocr2.py --n 50000 --source both
"""
import argparse, json, re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=50000, help='number of new traces to add')
    p.add_argument('--min-pass-rate', type=float, default=0.8, help='min pass_rate for OCR2')
    p.add_argument('--source', choices=['ocr2', 'codeforces', 'both'], default='both')
    p.add_argument('--output', default='data/all_traces.jsonl')
    return p.parse_args()


def strip_think_tags(text):
    m = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def load_existing(out_path):
    seen_ids, existing = set(), []
    if Path(out_path).exists():
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
    return existing, seen_ids


def fetch_ocr2(n, min_pass_rate, seen_ids):
    from datasets import load_dataset
    print(f'Streaming nvidia/OpenCodeReasoning-2 (pass_rate >= {min_pass_rate}, judgement=right)...')
    ds = load_dataset('nvidia/OpenCodeReasoning-2', split='python', streaming=True)

    new_traces, scanned = [], 0
    for row in ds:
        scanned += 1
        if scanned % 10000 == 0:
            print(f'  scanned {scanned:,} | collected {len(new_traces):,} / {n}')
        if row.get('judgement') != 'right':
            continue
        if float(row.get('pass_rate') or 0) < min_pass_rate:
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
            'pass_rate':  float(row.get('pass_rate') or 0),
            'difficulty': row.get('difficulty'),
            'dataset':    row.get('dataset'),
        })
        if len(new_traces) >= n:
            break

    print(f'OCR2: collected {len(new_traces)} traces (scanned {scanned:,})')
    return new_traces


def fetch_codeforces(n, seen_ids):
    from datasets import load_dataset
    print('Streaming open-r1/codeforces-cots...')
    ds = load_dataset('open-r1/codeforces-cots', split='train', streaming=True)

    new_traces, scanned = [], 0
    for row in ds:
        scanned += 1
        if scanned % 5000 == 0:
            print(f'  scanned {scanned:,} | collected {len(new_traces):,} / {n}')

        # messages format: [{role: user, content: ...}, {role: assistant, content: ...}]
        messages = row.get('messages') or []
        if len(messages) < 2:
            continue

        prompt = next((m['content'] for m in messages if m.get('role') == 'user'), '').strip()
        response = next((m['content'] for m in messages if m.get('role') == 'assistant'), '').strip()
        if not prompt or not response:
            continue

        # Extract thinking from <think> tags in assistant response
        thinking = strip_think_tags(response) if '<think>' in response else ''
        solution = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

        # Use verification_info to get a unique ID
        vinfo = row.get('verification_info') or {}
        pid = f"CF/{row.get('id', scanned)}"
        if pid in seen_ids:
            continue

        if not solution:
            continue

        seen_ids.add(pid)
        new_traces.append({
            'problem_id': pid,
            'source':     'codeforces-cots',
            'prompt':     prompt,
            'thinking':   thinking,
            'solution':   solution,
        })
        if len(new_traces) >= n:
            break

    print(f'Codeforces: collected {len(new_traces)} traces (scanned {scanned:,})')
    return new_traces


def main():
    args = parse_args()
    Path(args.output).parent.mkdir(exist_ok=True)

    existing, seen_ids = load_existing(args.output)
    new_traces = []

    if args.source in ('ocr2', 'both'):
        n = args.n if args.source == 'ocr2' else args.n // 2
        new_traces += fetch_ocr2(n, args.min_pass_rate, seen_ids)

    if args.source in ('codeforces', 'both'):
        n = args.n if args.source == 'codeforces' else args.n // 2
        new_traces += fetch_codeforces(n, seen_ids)

    all_traces = existing + new_traces
    with open(args.output, 'w') as f:
        for rec in all_traces:
            f.write(json.dumps(rec) + '\n')

    print(f'\nDone: {len(new_traces)} new + {len(existing)} existing = {len(all_traces)} total → {args.output}')


if __name__ == '__main__':
    main()
