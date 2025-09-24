# Explanation:
# Yeni bir prototip script ekliyorum: `pi_weighting_prototype.py`.
# Bu script:
# - Bir FEN için yolları örnekler (Engine.sample_paths) — eğer motor yoksa güvenli fallback yapar
# - Her yol için policy-log-prob toplamını (adım-adım Engine.lc0_policy_and_moves kullanarak) hesaplar
# - Opsiyonel olarak CP-score tabanlı ödül hesaplar (her adımda lc0_top_moves_and_scores çağırarak) veya pathlen proxy kullanır
# - Önerilen formüle göre log-weights, softmax (lambda ile) ile normalize eder
# - ESS (Effective Sample Size), entropy (first-move weighted) ve top-share hesaplar
# - ESS düşükse yeniden örnekleme (multinomial) uygular
# - Grid scan (lambda, alpha, beta) için küçük bir deney protokolu sunar ve CSV/PNG çıktıları üretir
import os
import math
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import chess

from engine import Engine
import config

EPS = 1e-12


def logsumexp(a: np.ndarray) -> float:
    a_max = np.max(a)
    return a_max + math.log(np.sum(np.exp(a - a_max))) if np.isfinite(a_max) else float('-inf')


# small in-memory cache for policy calls to avoid repeated engine queries
_POLICY_CACHE = {}


def compute_path_policy_logsum(path: List[chess.Move], fen: str, depth: int, engine=None) -> float:
    """
    Her adımda Engine.lc0_policy_and_moves çağırarak seçilen hamlenin log-policy olasılıklarının toplamını döndürür.
    Çok yavaş olabilir; prototype amaçlıdır. Basit cache ile hızlandırıldı.
    """
    if not path:
        return 0.0
    if engine is None:
        try:
            engine = Engine.get_engine(config.LC0_PATH)
        except Exception:
            engine = None
    log_sum = 0.0
    board = chess.Board(fen)
    for mv in path:
        bfen = board.fen()
        # cache lookup
        cache_key = (bfen, int(depth))
        if cache_key in _POLICY_CACHE:
            moves, probs = _POLICY_CACHE[cache_key]
        else:
            try:
                moves, probs, _ = Engine.lc0_policy_and_moves(board.fen(), depth=depth, multipv=config.LC0_MULTIPV)
                _POLICY_CACHE[cache_key] = (moves, probs)
            except Exception:
                moves, probs = [], []
        # find probability for mv
        mv_uci = mv.uci() if hasattr(mv, 'uci') else str(mv)
        prob = EPS
        for m, p in zip(moves, probs):
            mstr = m.uci() if hasattr(m, 'uci') else str(m)
            if mstr == mv_uci:
                try:
                    if p is None or (isinstance(p, float) and not np.isfinite(p)):
                        prob = EPS
                    else:
                        prob = float(p)
                except Exception:
                    prob = EPS
                break
        log_sum += math.log(max(prob, EPS))
        try:
            board.push(mv)
        except Exception:
            break
    return log_sum


def compute_path_cp_reward(path: List[chess.Move], fen: str, depth: int) -> float:
    """
    Basit CP-reward toplamı: her adımda lc0_top_moves_and_scores çağırıp seçilen hamlenin score'unu toplar.
    Eğer hata olursa path uzunluğu proxy olarak döner.
    """
    if not path:
        return 0.0
    total_score = 0.0
    board = chess.Board(fen)
    for mv in path:
        try:
            moves2, scores2, _ = Engine.lc0_top_moves_and_scores(board.fen(), depth=depth, multipv=config.LC0_MULTIPV)
            mv_uci = mv.uci() if hasattr(mv, 'uci') else str(mv)
            score_found = None
            for m, s in zip(moves2, scores2):
                mstr = m.uci() if hasattr(m, 'uci') else str(m)
                if mstr == mv_uci:
                    score_found = float(s)
                    break
            if score_found is None:
                # fallback: use top score as proxy
                score_found = float(scores2[0]) if scores2 else 0.0
            total_score += score_found
        except Exception:
            total_score += 0.0
        try:
            board.push(mv)
        except Exception:
            break
    return total_score


def compute_log_weights_for_paths(paths: List[List[chess.Move]], fen: str, depth: int,
                                  alpha: float = 1.0, beta: float = 0.0, lam: float = 1.0,
                                  reward_mode: str = 'none') -> Tuple[np.ndarray, np.ndarray]:
    """
    paths: list of paths (each path is list of chess.Move)
    Returns: normalized_weights (sum=1), raw_logw (before normalization)
    """
    n = len(paths)
    logw = np.zeros(n, dtype=np.float64)
    for i, path in enumerate(tqdm(paths, desc="compute_log_weights", unit="path")):
        # policy log-sum
        lp = compute_path_policy_logsum(path, fen, depth)
        # reward
        if reward_mode == 'cp':
            r = compute_path_cp_reward(path, fen, depth)
            r_norm = r / max(1.0, len(path))
        elif reward_mode == 'pathlen':
            r_norm = len(path) / max(1.0, depth)
        else:
            r_norm = 0.0
        logw[i] = alpha * lp + beta * r_norm

    # scale by 1/lam (as in p ∝ exp(logw/lam))
    with np.errstate(over='ignore'):
        scaled = logw / float(lam)
    # numerical stabilization: subtract max
    amax = np.max(scaled) if scaled.size>0 else 0.0
    exps = np.exp(scaled - amax)
    if not np.isfinite(exps).all() or exps.sum() == 0:
        exps = np.ones_like(exps)
    weights = exps / (exps.sum() + EPS)
    return weights, logw


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.array(weights, dtype=np.float64)
    if w.sum() <= 0:
        return 0.0
    w = w / w.sum()
    return float((w.sum() ** 2) / (np.sum(w ** 2) + EPS))


def weighted_first_move_entropy(paths: List[List[chess.Move]], weights: np.ndarray) -> float:
    counter = defaultdict(float)
    for path, w in zip(paths, weights):
        if path:
            m = str(path[0])
            counter[m] += float(w)
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.array([v / total for v in counter.values()], dtype=np.float64)
    probs = probs[probs > 0]
    return float(-1.0 * np.sum(probs * np.log2(probs)))


def top_move_share(paths: List[List[chess.Move]], weights: np.ndarray) -> float:
    counter = defaultdict(float)
    for path, w in zip(paths, weights):
        if path:
            m = str(path[0])
            counter[m] += float(w)
    if not counter:
        return 0.0
    return max(counter.values()) / (sum(counter.values()) + EPS)


def resample_paths(paths: List[List[chess.Move]], weights: np.ndarray, n_samples: int) -> List[List[chess.Move]]:
    if len(paths) == 0:
        return []
    p = np.array(weights, dtype=np.float64)
    p = p / (p.sum() + EPS)
    idx = np.random.choice(len(paths), size=n_samples, replace=True, p=p)
    return [paths[i] for i in idx]


def run_grid_scan(fen: str,
                  depth: int = None,
                  samples: int = 100,
                  alphas: List[float] = (1.0,),
                  betas: List[float] = (0.0,),
                  lambdas: List[float] = (0.1, 0.5, 1.0),
                  resample_threshold: float = 0.5,
                  reward_mode: str = 'none',
                  out_dir: str = 'results/pi_proto') -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    depth = depth or config.TARGET_DEPTH

    # 1) Sample raw paths once (use a neutral lambda to get diverse samples)
    try:
        raw_paths = Engine.sample_paths(fen, depth, lam=1.0, samples=samples, mode='competitive')
    except Exception as e:
        print(f"[WARN] Engine sampling failed: {e} — falling back to random cheap sampler")
        # fallback: random legal-move sampler
        raw_paths = []
        b = chess.Board(fen)
        legal = list(b.legal_moves)
        for _ in range(samples):
            b = chess.Board(fen)
            path = []
            for _ply in range(depth):
                moves = list(b.legal_moves)
                if not moves:
                    break
                m = np.random.choice(moves)
                path.append(m)
                b.push(m)
            raw_paths.append(path)

    rows = []
    idx = 0
    for lam in lambdas:
        for alpha in alphas:
            for beta in betas:
                idx += 1
                print(f"[GRID] run {idx}: λ={lam}, α={alpha}, β={beta}, reward_mode={reward_mode}")
                weights, raw_logw = compute_log_weights_for_paths(raw_paths, fen, depth, alpha=alpha, beta=beta, lam=lam, reward_mode=reward_mode)
                ess = effective_sample_size(weights)
                ent = weighted_first_move_entropy(raw_paths, weights)
                top_share = top_move_share(raw_paths, weights)
                rows.append({'lambda': lam, 'alpha': alpha, 'beta': beta, 'ess': ess, 'entropy': ent, 'top_share': top_share, 'n_samples': len(raw_paths)})

                # Resample if ESS too low
                if ess < resample_threshold * len(raw_paths):
                    print(f"  [RESAMPLE] ESS={ess:.1f} < threshold -> resampling")
                    raw_paths = resample_paths(raw_paths, weights, samples)
                    # recompute weights after resample using same params
                    weights, raw_logw = compute_log_weights_for_paths(raw_paths, fen, depth, alpha=alpha, beta=beta, lam=lam, reward_mode=reward_mode)
                    ess2 = effective_sample_size(weights)
                    ent2 = weighted_first_move_entropy(raw_paths, weights)
                    top_share2 = top_move_share(raw_paths, weights)
                    rows.append({'lambda': lam, 'alpha': alpha, 'beta': beta, 'ess': ess2, 'entropy': ent2, 'top_share': top_share2, 'n_samples': len(raw_paths), 'resampled': True})

    df = pd.DataFrame(rows)
    csv_out = os.path.join(out_dir, 'pi_weighting_grid_results.csv')
    df.to_csv(csv_out, index=False)
    print(f"✓ Grid results saved: {csv_out}")

    # Simple plot: entropy vs lambda for selected alpha/beta
    try:
        plt.figure(figsize=(6,4))
        for (a,b), g in df.groupby(['alpha','beta']):
            plt.plot(g['lambda'], g['entropy'], marker='o', label=f'a={a},b={b}')
        plt.xlabel('lambda')
        plt.ylabel('weighted first-move entropy (bits)')
        plt.legend()
        plt.tight_layout()
        png_out = os.path.join(out_dir, 'entropy_vs_lambda.png')
        plt.savefig(png_out)
        plt.close()
        print(f"✓ Plot saved: {png_out}")
    except Exception as e:
        print(f"[WARN] plotting failed: {e}")

    return df


if __name__ == '__main__':
    # Quick smoke-run with small params (safe defaults)
    fen = getattr(config, 'FEN', 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
    df = run_grid_scan(
        fen=fen,
        depth=min(4, getattr(config, 'TARGET_DEPTH', 4)),
        samples=min(40, getattr(config, 'SAMPLE_COUNT', 40)),
        alphas=[0.5, 1.0],
        betas=[0.0, 0.5],
        lambdas=[0.01, 0.05, 0.1],
        resample_threshold=0.3,
        reward_mode='none'
    )
    print(df.head())
