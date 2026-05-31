import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Global System Configurations
os.environ["LAYERNORM_TYPE"] = "torch"
os.environ.setdefault("RNA_MSA_DEPTH_LIMIT", "512")

# Default Parameters
MODEL_NAME = "protenix_base_20250630_v1.0.0"
N_SAMPLE = 5
SEED = 42
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "128"))
MIN_SIMILARITY = float(os.environ.get("MIN_SIMILARITY", "0.0"))
MIN_PERCENT_IDENTITY = float(os.environ.get("MIN_PERCENT_IDENTITY", "50.0"))
USE_PROTENIX = True

def parse_bool(value, default=False):
    v = str(value).strip().lower()
    if v in {"1", "true", "t", "yes", "y", "on"}: return "true"
    if v in {"0", "false", "f", "no", "n", "off"}: return "false"
    return "true" if default else "false"

USE_MSA = parse_bool(os.environ.get("USE_MSA", "false"))
USE_TEMPLATE = parse_bool(os.environ.get("USE_TEMPLATE", "false"))
USE_RNA_MSA = parse_bool(os.environ.get("USE_RNA_MSA", "true"))
MODEL_N_SAMPLE = int(os.environ.get("MODEL_N_SAMPLE", str(N_SAMPLE)))

def seed_everything(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.use_deterministic_algorithms(True)

def resolve_paths():
    test_csv = os.environ.get("TEST_CSV", "data/test_sequences.csv")
    output_csv = os.environ.get("SUBMISSION_CSV", "submission.csv")
    code_dir = os.environ.get("PROTENIX_CODE_DIR", "protenix-v1")
    root_dir = os.environ.get("PROTENIX_ROOT_DIR", "protenix-v1")
    return test_csv, output_csv, code_dir, root_dir

def ensure_required_files(root_dir):
    for p, name in [
        (Path(root_dir)/"checkpoint"/f"{MODEL_NAME}.pt", "checkpoint"),
        (Path(root_dir)/"common"/"components.cif", "CCD file"),
        (Path(root_dir)/"common"/"components.cif.rdkit_mol.pkl", "CCD cache"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {name}: {p}")

def build_input_json(df, json_path):
    data = [{"name": row["target_id"], "covalent_bonds": [],
             "sequences": [{"rnaSequence": {"sequence": row["sequence"], "count": 1}}]}
            for _, row in df.iterrows()]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

def build_configs(input_json_path, dump_dir, model_name):
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config.config import parse_configs
    base = {**configs_base, **{"data": data_configs}, **inference_configs}
    def deep_update(t, p):
        for k, v in p.items():
            if isinstance(v, dict) and k in t and isinstance(t[k], dict): deep_update(t[k], v)
            else: t[k] = v
    deep_update(base, model_configs[model_name])
    arg_str = " ".join([
        f"--model_name {model_name}",
        f"--input_json_path {input_json_path}",
        f"--dump_dir {dump_dir}",
        f"--use_msa {USE_MSA}",
        f"--use_template {USE_TEMPLATE}",
        f"--use_rna_msa {USE_RNA_MSA}",
        f"--sample_diffusion.N_sample {MODEL_N_SAMPLE}",
        f"--seeds {SEED}",
    ])
    return parse_configs(configs=base, arg_str=arg_str, fill_required_with_null=True)

def coords_to_rows(target_id, seq, coords):
    rows = []
    for i in range(len(seq)):
        row = {"ID": f"{target_id}_{i+1}", "resname": seq[i], "resid": i+1}
        for s in range(N_SAMPLE):
            if s < coords.shape[0] and i < coords.shape[1]: x, y, z = coords[s, i]
            else: x, y, z = 0.0, 0.0, 0.0
            row[f"x_{s+1}"] = float(x); row[f"y_{s+1}"] = float(y); row[f"z_{s+1}"] = float(z)
        rows.append(row)
    return rows

def split_into_chunks(seq_len, max_len, overlap):
    if seq_len <= max_len: return [(0, seq_len)]
    chunks, step, pos = [], max_len - overlap, 0
    while pos < seq_len:
        end = min(pos + max_len, seq_len); chunks.append((pos, end))
        if end == seq_len: break
        pos += step
    return chunks

def kabsch_align(P, Q):
    cP, cQ = P.mean(0), Q.mean(0); Pc, Qc = P - cP, Q - cQ
    H = Pc.T @ Qc; U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T); S = np.eye(3)
    if d < 0: S[2, 2] = -1
    R = Vt.T @ S @ U.T
    return R, cQ - R @ cP

def stitch_chunk_coords(chunk_coords_list, chunk_ranges, seq_len):
    if len(chunk_coords_list) == 1:
        coords = chunk_coords_list[0]
        if coords.shape[0] >= seq_len: return coords[:seq_len]
        out = np.zeros((seq_len, 3), dtype=coords.dtype); out[:coords.shape[0]] = coords; return out
    aligned = [chunk_coords_list[0].copy()]
    for i in range(1, len(chunk_coords_list)):
        ps, pe = chunk_ranges[i-1]; cs, ce = chunk_ranges[i]
        ov_s, ov_e = cs, min(pe, ce)
        if ov_e - ov_s < 3: aligned.append(chunk_coords_list[i].copy()); continue
        prev_ov = aligned[i-1][ov_s-ps:ov_e-ps]; cur_ov = chunk_coords_list[i][ov_s-cs:ov_e-cs]
        valid = ~(np.isnan(prev_ov).any(1) | np.isnan(cur_ov).any(1))
        if valid.sum() < 3: aligned.append(chunk_coords_list[i].copy()); continue
        R, t = kabsch_align(cur_ov[valid], prev_ov[valid])
        aligned.append((chunk_coords_list[i] @ R.T) + t)
    full = np.zeros((seq_len, 3), dtype=np.float64); weights = np.zeros(seq_len, dtype=np.float64)
    for i, ((s, e), coords) in enumerate(zip(chunk_ranges, aligned)):
        cl = coords.shape[0]; ae = min(s+cl, seq_len); ul = ae - s
        w = np.ones(ul, dtype=np.float64)
        if i > 0:
            ov_e2 = min(chunk_ranges[i-1][1], e); rl = ov_e2 - s
            if rl > 0: w[:rl] = np.linspace(0., 1., rl)
        if i < len(chunk_ranges) - 1:
            ns2 = chunk_ranges[i+1][0]; rs = ns2 - s; rl = ae - ns2
            if rl > 0 and rs < ul: w[rs:ul] = np.linspace(1., 0., rl)
        full[s:ae] += coords[:ul] * w[:, None]; weights[s:ae] += w
    mask = weights > 0; full[mask] /= weights[mask, None]
    return full

_aligner = PairwiseAligner()
_aligner.mode = "global"; _aligner.match_score = 2; _aligner.mismatch_score = -1.5
_aligner.open_gap_score = -8; _aligner.extend_gap_score = -0.4

def parse_stoichiometry(stoich):
    if pd.isna(stoich) or str(stoich).strip() == "": return []
    return [(ch.strip(), int(cnt)) for part in str(stoich).split(";") for ch, cnt in [part.split(":")]]

def parse_fasta(fasta_content):
    out, cur, parts = {}, None, []
    for line in str(fasta_content).splitlines():
        line = line.strip()
        if not line: continue
        if line.startswith(">"):
            if cur is not None: out[cur] = "".join(parts)
            cur = line[1:].split()[0]; parts = []
        else: parts.append(line.replace(" ", ""))
    if cur is not None: out[cur] = "".join(parts)
    return out

def get_chain_segments(row):
    seq = row["sequence"]; stoich = row.get("stoichiometry", ""); all_sq = row.get("all_sequences", "")
    if pd.isna(stoich) or pd.isna(all_sq) or str(stoich).strip() == "" or str(all_sq).strip() == "":
        return [(0, len(seq))]
    try:
        cd = parse_fasta(all_sq); order = parse_stoichiometry(stoich); segs, pos = [], 0
        for ch, cnt in order:
            base = cd.get(ch)
            if base is None: return [(0, len(seq))]
            for _ in range(cnt): segs.append((pos, pos+len(base))); pos += len(base)
        return segs if pos == len(seq) else [(0, len(seq))]
    except: return [(0, len(seq))]

def build_segments_map(df):
    seg_map = {}
    for _, r in df.iterrows():
        tid = r["target_id"]; seg_map[tid] = get_chain_segments(r)
    return seg_map

def process_labels(labels_df):
    coords = {}
    prefixes = labels_df["ID"].str.rsplit("_", n=1).str[0]
    for prefix, grp in labels_df.groupby(prefixes):
        coords[prefix] = grp.sort_values("resid")[["x_1", "y_1", "z_1"]].values
    return coords

def _build_aligned_strings(query_seq, template_seq, alignment):
    q_segs, t_segs = alignment.aligned; aq, at, qi, ti = [], [], 0, 0
    for (qs, qe), (ts, te) in zip(q_segs, t_segs):
        while qi < qs: aq.append(query_seq[qi]); at.append("-"); qi += 1
        while ti < ts: aq.append("-"); at.append(template_seq[ti]); ti += 1
        for qp, tp in zip(range(qs, qe), range(ts, te)): aq.append(query_seq[qp]); at.append(template_seq[tp])
        qi, ti = qe, te
    while qi < len(query_seq): aq.append(query_seq[qi]); at.append("-"); qi += 1
    while ti < len(template_seq): aq.append("-"); at.append(template_seq[ti]); ti += 1
    return "".join(aq), "".join(at)

def find_similar_sequences_detailed(query_seq, train_seqs_df, train_coords_dict, top_n=30):
    results = []
    for _, row in train_seqs_df.iterrows():
        tid, tseq = row["target_id"], row["sequence"]
        if tid not in train_coords_dict: continue
        if abs(len(tseq) - len(query_seq)) / max(len(tseq), len(query_seq)) > 0.3: continue
        aln = next(iter(_aligner.align(query_seq, tseq)))
        norm_s = aln.score / (2 * min(len(query_seq), len(tseq)))
        identical = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned)
                        for qp, tp in zip(range(qs, qe), range(ts, te)) if query_seq[qp] == tseq[tp])
        pct_id = 100 * identical / len(query_seq)
        aq, at = _build_aligned_strings(query_seq, tseq, aln)
        results.append((tid, tseq, norm_s, train_coords_dict[tid], pct_id, aq, at))
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:top_n] surcharge

def adapt_template_to_query(query_seq, template_seq, template_coords):
    aln = next(iter(_aligner.align(query_seq, template_seq)))
    new_c = np.full((len(query_seq), 3), np.nan)
    for (qs, qe), (ts, te) in zip(*aln.aligned):
        chunk = template_coords[ts:te]
        if len(chunk) == (qe - qs): new_c[qs:qe] = chunk
    for i in range(len(new_c)):
        if np.isnan(new_c[i, 0]):
            pv = next((j for j in range(i-1, -1, -1) if not np.isnan(new_c[j, 0])), -1)
            nv = next((j for j in range(i+1, len(new_c)) if not np.isnan(new_c[j, 0])), -1)
            if pv >= 0 and nv >= 0: w = (i - pv) / (nv - pv); new_c[i] = (1 - w) * new_c[pv] + w * new_c[nv]
            elif pv >= 0: new_c[i] = new_c[pv] + [3, 0, 0]
            elif nv >= 0: new_c[i] = new_c[nv] + [3, 0, 0]
            else: new_c[i] = [i * 3, 0, 0]
    return np.nan_to_num(new_c)

def adaptive_rna_constraints(coords, target_id, segments_map, confidence=1.0, passes=2):
    X = coords.copy(); segments = segments_map.get(target_id, [(0, len(X))])
    strength = max(0.75 * (1.0 - min(confidence, 0.97)), 0.02)
    for _ in range(passes):
        for s, e in segments:
            C = X[s:e]; L = e - s
            if L < 3: continue
            d = C[1:] - C[:-1]; dist = np.linalg.norm(d, axis=1) + 1e-6
            adj = d * ((5.95 - dist) / dist)[:, None] * (0.22 * strength)
            C[:-1] -= adj; C[1:] += adj
            d2 = C[2:] - C[:-2]; d2n = np.linalg.norm(d2, axis=1) + 1e-6
            adj2 = d2 * ((10.2 - d2n) / d2n)[:, None] * (0.10 * strength)
            C[:-2] -= adj2; C[2:] += adj2
            X[s:e] = C
    return X

def _rotmat(axis, ang):
    a = np.asarray(axis, float); a /= np.linalg.norm(a) + 1e-12
    x, y, z = a; c, s = np.cos(ang), np.sin(ang); CC = 1 - c
    return np.array([[c+x*x*CC, x*y*CC-z*s, x*z*CC+y*s],
                     [y*x*CC+z*s, c+y*y*CC, y*z*CC-x*s],
                     [z*x*CC-y*s, z*y*CC+x*s, c+z*z*CC]])

def apply_hinge(coords, seg, rng, deg=22):
    s, e = seg; L = e - s
    if L < 30: return coords
    pivot = s + int(rng.integers(10, L - 10))
    R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
    X = coords.copy(); p0 = X[pivot].copy(); X[pivot+1:e] = (X[pivot+1:e] - p0) @ R.T + p0
    return X

def jitter_chains(coords, segs, rng, deg=12, trans=1.5):
    X = coords.copy(); gc_ = X.mean(0, keepdims=True)
    for s, e in segs:
        R = _rotmat(rng.normal(size=3), np.deg2rad(float(rng.uniform(-deg, deg))))
        shift = rng.normal(size=3); shift = shift / (np.linalg.norm(shift) + 1e-12) * float(rng.uniform(0, trans))
        c = X[s:e].mean(0, keepdims=True); X[s:e] = (X[s:e] - c) @ R.T + c + shift
    X -= X.mean(0, keepdims=True) - gc_; return X

def smooth_wiggle(coords, segs, rng, amp=0.8):
    X = coords.copy()
    for s, e in segs:
        L = e - s
        if L < 20: continue
        ctrl = np.linspace(0, L - 1, 6); disp = rng.normal(0, amp, (6, 3)); t = np.arange(L)
        X[s:e] += np.vstack([np.interp(t, ctrl, disp[:, k]) for k in range(3)]).T
    return X

def generate_rna_structure(sequence, seed=None):
    if seed is not None: np.random.seed(seed)
    n = len(sequence); coords = np.zeros((n, 3))
    for i in range(n):
        ang = i * 0.6; coords[i] = [10.0 * np.cos(ang), 10.0 * np.sin(ang), i * 2.5]
    return coords

def tbm_phase(test_df, train_seqs_df, train_coords_dict, segments_map):
    print("\nExecuting PHASE 1: Template-Based Modeling...")
    template_predictions, protenix_queue = {}, {}
    for _, row in test_df.iterrows():
        tid = row["target_id"]; seq = row["sequence"]; segs = segments_map.get(tid, [(0, len(seq))])
        similar = find_similar_sequences_detailed(seq, train_seqs_df, train_coords_dict, top_n=30)
        preds = []; used = set()
        for i, (tmpl_id, tmpl_seq, sim, tmpl_coords, pct_id, _, _) in enumerate(similar):
            if len(preds) >= N_SAMPLE: break
            if sim < MIN_SIMILARITY or pct_id < MIN_PERCENT_IDENTITY: break
            if tmpl_id in used: continue
            rng = np.random.default_rng((row.name * 10000000000 + i * 10007) % (2**32))
            adapted = adapt_template_to_query(seq, tmpl_seq, tmpl_coords)
            slot = len(preds)
            if slot == 0: X = adapted
            elif slot == 1: X = adapted + rng.normal(0, max(0.01, (0.40 - sim) * 0.06), adapted.shape)
            elif slot == 2:
                longest = max(segs, key=lambda se: se[1] - se[0]); X = apply_hinge(adapted, longest, rng)
            elif slot == 3: X = jitter_chains(adapted, segs, rng)
            else: X = smooth_wiggle(adapted, segs, rng)
            refined = adaptive_rna_constraints(X, tid, segments_map, confidence=sim)
            preds.append(refined); used.add(tmpl_id)
        template_predictions[tid] = preds
        n_needed = N_SAMPLE - len(preds)
        if n_needed > 0: protenix_queue[tid] = (n_needed, seq)
    return template_predictions, protenix_queue

def main():
    test_csv, output_csv, code_dir, root_dir = resolve_paths()
    if not os.path.isdir(code_dir):
        raise FileNotFoundError(f"Missing PROTENIX_CODE_DIR: {code_dir}")
    sys.path.append(code_dir)
    ensure_required_files(root_dir); seed_everything(SEED)
    
    test_df = pd.read_csv(test_csv).reset_index(drop=True)
    train_seqs = pd.read_csv("data/train_sequences.csv")
    val_seqs = pd.read_csv("data/validation_sequences.csv")
    train_labels = pd.read_csv("data/train_labels.csv")
    val_labels = pd.read_csv("data/validation_labels.csv")
    
    combined_seqs = pd.concat([train_seqs, val_seqs], ignore_index=True)
    combined_labels = pd.concat([train_labels, val_labels], ignore_index=True)
    train_coords = process_labels(combined_labels)
    del train_labels, val_labels, combined_labels; gc.collect()
    
    segments_map = build_segments_map(test_df)
    template_preds, protenix_queue = tbm_phase(test_df, combined_seqs, train_coords, segments_map)
    protenix_preds = {}
    
    if protenix_queue and USE_PROTENIX:
        print(f"\nExecuting PHASE 2: Protenix Engine for {len(protenix_queue)} targets...")
        work_dir = Path("outputs"); work_dir.mkdir(parents=True, exist_ok=True)
        tasks = []; chunk_info = {}
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq)
            if seq_len <= MAX_SEQ_LEN:
                tasks.append({"target_id": target_id, "sequence": full_seq})
                chunk_info[target_id] = [{"name": target_id, "range": (0, seq_len)}]
            else:
                chunks = split_into_chunks(seq_len, MAX_SEQ_LEN, CHUNK_OVERLAP)
                chunk_info[target_id] = []
                for ci, (cs, ce) in enumerate(chunks):
                    cn = f"{target_id}_chunk{ci}"
                    tasks.append({"target_id": cn, "sequence": full_seq[cs:ce]})
                    chunk_info[target_id].append({"name": cn, "range": (cs, ce)})
                    
        tasks_df = pd.DataFrame(tasks)
        input_json_path = str(work_dir / "protenix_queue_input.json")
        build_input_json(tasks_df, input_json_path)
        
        from protenix.data.inference.infer_dataloader import InferenceDataset
        from runner.inference import InferenceRunner, update_gpu_compatible_configs, update_inference_configs
        
        configs = build_configs(input_json_path, str(work_dir), MODEL_NAME)
        configs = update_gpu_compatible_configs(configs)
        runner = InferenceRunner(configs); dataset = InferenceDataset(configs)
        raw_predictions = {}
        
        for i in tqdm(range(len(dataset)), desc="Protenix Inference"):
            data, atom_array, err = dataset[i]
            sample_name = data.get("sample_name", f"sample_{i}")
            if err:
                raw_predictions[sample_name] = None
                continue
            target_id = sample_name.split("_chunk")[0] if "_chunk" in sample_name else sample_name
            n_needed = protenix_queue.get(target_id, (N_SAMPLE, ""))[0]
            sub_seq_len = data["N_token"].item()
            try:
                new_cfg = update_inference_configs(configs, sub_seq_len)
                new_cfg.sample_diffusion.N_sample = n_needed
                runner.update_model_configs(new_cfg)
                pred = runner.predict(data); raw_coords = pred["coordinate"]
                
                mask = (data["input_feature_dict"]["centre_atom_mask"] == 1).to(raw_coords.device) \
                       if "centre_atom_mask" in data["input_feature_dict"] else \
                       (data["input_feature_dict"]["atom_to_tokatom_idx"] == 11).to(raw_coords.device)
                       
                coords = raw_coords[:, mask, :].detach().cpu().numpy()
                raw_predictions[sample_name] = coords
            except Exception:
                raw_predictions[sample_name] = None
            finally:
                gc.collect(); torch.cuda.empty_cache()
                
        for target_id, (n_needed, full_seq) in protenix_queue.items():
            seq_len = len(full_seq); chunks = chunk_info.get(target_id, [])
            if len(chunks) == 1:
                protenix_preds[target_id] = raw_predictions.get(target_id)
            else:
                per_sample = {s: [] for s in range(n_needed)}; all_ok = True
                for cinfo in chunks:
                    ccoords = raw_predictions.get(cinfo["name"])
                    if ccoords is None: all_ok = False; break
                    for s_idx in range(n_needed):
                        si = s_idx if s_idx < ccoords.shape[0] else -1
                        per_sample[s_idx].append((ccoords[si], cinfo["range"]))
                if not all_ok: continue
                stitched = []
                for s_idx in range(n_needed):
                    items = per_sample[s_idx]
                    fc = stitch_chunk_coords([c for c, _ in items], [r for _, r in items], seq_len)
                    stitched.append(fc)
                protenix_preds[target_id] = np.stack(stitched, axis=0)

    all_rows = []
    for _, row in test_df.iterrows():
        tid = row["target_id"]; seq = row["sequence"]
        combined = list(template_preds.get(tid, []))
        ptx = protenix_preds.get(tid)
        if ptx is not None and ptx.ndim == 3:
            for j in range(ptx.shape[0]):
                if len(combined) >= N_SAMPLE: break
                combined.append(ptx[j])
        while len(combined) < N_SAMPLE:
            dn = generate_rna_structure(seq, seed=row.name*1000)
            combined.append(adaptive_rna_constraints(dn, tid, segments_map, confidence=0.2))
        stacked = np.stack(combined[:N_SAMPLE], axis=0)
        all_rows.extend(coords_to_rows(tid, seq, stacked))
        
    sub = pd.DataFrame(all_rows)
    cols = ["ID", "resname", "resid"] + [f"{c}_{i}" for i in range(1, N_SAMPLE+1) for c in ["x", "y", "z"]]
    sub[cols].to_csv(output_csv, index=False)
    print(f"Pipeline finished successfully! Outputs written to: {output_csv}")

if __name__ == "__main__":
    main()
