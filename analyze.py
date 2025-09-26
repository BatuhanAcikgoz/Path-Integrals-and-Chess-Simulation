import os
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter
import seaborn as sns
from tqdm import tqdm
from collections import Counter

from cache import Cache
import config
from engine import Engine
from mathfuncs import Calc
from plots import Plots


# --- Pozisyon isimleri ve FEN eşleşmesi sabitleri ---
POSITION_NAMES = [
    "Italian_Game",
    "Sicilian_Defense",
    "Ruy_Lopez",
    "Strategic_Midgame_Isolated_Pawn",
    "Calm_Strategic_Midgame",
    "Tactical_Midgame",
    "King_Pawn_Endgame"
]
FEN_TO_ANALYZE = dict(zip(POSITION_NAMES, config.MULTI_FEN))
# FEN'den pozisyon ismine ulaşmak için ters mapping
FEN_NAME_MAP = {v: k for k, v in FEN_TO_ANALYZE.items()}


class Analyzer:
    @staticmethod
    def analyze_sample_size_sensitivity(fen, sample_sizes=None):
        """
        Örnekleme boyutunun entropi ve konsantrasyon (mode probability) üzerindeki etkisini analiz eder.
        """
        sample_sizes = sample_sizes or config.SAMPLE_SIZES  # Direct access
        print(f"\n--- Sample Size Sensitivity Analysis Begins ---")

        results = []
        for size in tqdm(sample_sizes, desc="Sample Size Scan"):
            paths = Engine.sample_paths(fen, config.TARGET_DEPTH, config.LAMBDA, size)
            entropy, _ = Calc.compute_entropy(paths)
            concentration = Calc.top_move_concentration(paths)
            results.append({'sample_size': size, 'entropy': entropy, 'accuracy': concentration})

        df = pd.DataFrame(results)

        # Sonuçları çizdir
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        ax1.plot(df['sample_size'], df['entropy'], 'b-o', label='Entropy')
        ax2.plot(df['sample_size'], df['accuracy'], 'r-o', label='Concentration')

        ax1.set_xlabel('Sample Count')
        ax1.set_ylabel('Entropy', color='b')
        ax2.set_ylabel('Concentration', color='r')
        ax1.set_xscale('log')
        ax1.set_xticks(sample_sizes)
        ax1.get_xaxis().set_major_formatter(FixedFormatter([str(s) for s in sample_sizes]))

        plt.title('Effect of Sample Number on Entropy and Concentration')
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/sample_size_sensitivity.png")
        plt.close()
        print("Sample size sensitivity graph saved as 'results/sample_size_sensitivity.png'.")

    @staticmethod
    def analyze_convergence_by_depth(fen, lambda_val=config.LAMBDA, depth_list=config.DEPTH_SCAN, sample_count=config.SAMPLE_COUNT):
        """
        Lc0 ile derinlik (depth) tabanlı yakınsama analizi. Her depth değeri için
        entropi ve konsantrasyon hesaplanır, sonuçlar PNG olarak kaydedilir.

        :param fen: Analiz edilecek FEN stringi
        :param lambda_val: Softmax sıcaklık parametresi
        :param depth_list: Derinlik listesi (varsayılan: [4, 8, 12, 16, 20])
        :param sample_count: Her depth için örnek sayısı
        :return: Sonuç DataFrame'i
        """
        import matplotlib
        matplotlib.use('Agg')
        depth_list = depth_list or [4, 8, 12, 16, 20]
        print(f"\n--- Convergence Analysis by Depth (λ={lambda_val}) ---")
        results = []
        for depth in tqdm(depth_list, desc="Depth Convergence"):
            paths = Engine.sample_paths(fen, depth, lambda_val, sample_count)
            entropy, _ = Calc.compute_entropy(paths)
            concentration = Calc.top_move_concentration(paths)
            results.append({'depth': depth, 'entropy': entropy, 'accuracy': concentration})
        df = pd.DataFrame(results)
        # Grafik
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()
        ax1.plot(df['depth'], df['entropy'], 'b-o', label='Entropi')
        ax2.plot(df['depth'], df['accuracy'], 'r-o', label='Concentration')
        ax1.set_xlabel('Arama Derinliği (depth)')
        ax1.set_ylabel('Entropi (bits)', color='b')
        ax2.set_ylabel('Concentration', color='r')
        ax1.set_xticks(depth_list)
        ax1.get_xaxis().set_major_formatter(FixedFormatter([str(d) for d in depth_list]))
        plt.title(f'Depth ile Entropi ve Konsantrasyon Yakınsaması (λ={lambda_val})')
        fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/convergence_by_depth_lambda_{lambda_val}.png", dpi=300)
        plt.close()
        print(f"Yakınsama grafiği kaydedildi: results/convergence_by_depth_lambda_{lambda_val}.png")
        return df

    @staticmethod
    def analyze_convergence_by_lambda(fen, lambda_range=config.LAMBDA_SCAN):  # Direct access
        """
        λ parametresine göre yakınsama: optimal hamle olasılığı yerine konsantrasyon raporlanır.
        X ekseni eşit aralıklı indekslerle gösterilir, etiketler gerçek lambda değerleriyle yazılır.
        """
        print(f"\n--- Convergence Analysis Based on Lambda ---")
        engine = Engine.get_engine(config.LC0_PATH)
        try:
            optimal_probs = []
            entropies = []

            for lam in tqdm(lambda_range, desc="Lambda Convergence"):
                paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lam, config.SAMPLE_COUNT, engine=engine)
                optimal_prob = Calc.top_move_concentration(paths)
                optimal_probs.append(optimal_prob)
                entropy, _ = Calc.compute_entropy(paths)
                entropies.append(entropy)

            # Sonuçları görselleştir
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax2 = ax1.twinx()

            x = list(range(len(lambda_range)))
            labels = [f'{l:.3f}' if l < 1 else f'{l:.2f}' for l in lambda_range]

            p1, = ax1.plot(x, optimal_probs, 'b-o', label='Mode Probability')
            ax1.set_xlabel('λ (Softmax Temperature Parameter)')
            ax1.set_ylabel('Mode Probability', color='blue')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=11, rotation=0)
            ax1.set_ylim(0, 1.1)

            p2, = ax2.plot(x, entropies, 'r-s', label='Entropy')
            ax2.set_ylabel('Entropy (bits)', color='red')

            plt.title(f'Convergence Analysis with Lambda')
            ax1.legend(handles=[p1, p2], labels=[p1.get_label(), p2.get_label()], loc='center right')

            plt.tight_layout()
            os.makedirs("results", exist_ok=True)
            plt.savefig("results/convergence_by_lambda.png", dpi=300)
            plt.close()

            return lambda_range, optimal_probs, entropies
        finally:
            Engine.close_all_engines()

    @staticmethod
    def analyze_combinatorial_explosion(fen, max_depth=20):
        """
        Analyzes and visualizes the combinatorial explosion in chess by tracking path counts,
        branching factors, and computational complexity as search depth increases.

        This function calculates how the number of possible move sequences (paths) grows
        exponentially with the search depth. It estimates the total paths using the average
        branching factor and compares it with the number of paths actually sampled via
        Monte Carlo. The results, including computational complexity and memory usage
        approximations, are visualized in a multi-panel plot and saved.

        The function uses caching to avoid re-computing the analysis for the same FEN
        and max_depth.

        :param fen: The FEN string of the starting position.
        :type fen: str
        :param max_depth: The maximum search depth to analyze. Defaults to 8.
        :type max_depth: int
        :return: A pandas DataFrame containing the analysis results for each depth.
        :rtype: pd.DataFrame
        """
        cache_key = f"combinatorial_explosion_{fen}_{max_depth}"
        cached_result = Cache.get_cached_analysis(cache_key)
        if cached_result is not None:
            print(f"[cache hit] analyze_combinatorial_explosion fen='{fen[:20]}...'")
            # Ensure the plot exists if cache is hit but file is deleted
            if not os.path.exists("results/combinatorial_explosion_analysis.png"):
                 # In a real scenario, you might regenerate the plot from cached data.
                 # For simplicity, we'll just note it.
                 print("Plot file missing, but using cached data. Re-run with clear_analysis_cache() to regenerate.")
            return cached_result

        print("\n--- Combinatorial Explosion Analysis ---")

        results = []

        for depth in tqdm(range(1, max_depth + 1), desc="Combinatorial Explosion"):
            # Approximation: average branching factor
            branching_factors = []
            temp_board = chess.Board(fen)
            path_is_possible = True
            for d in range(depth):
                legal_moves = list(temp_board.legal_moves)
                if not legal_moves:
                    path_is_possible = False
                    break
                branching_factors.append(len(legal_moves))
                temp_board.push(legal_moves[0]) # Approximation

            if not path_is_possible:
                avg_branching = np.mean(branching_factors) if branching_factors else 0
                estimated_paths = 0
            else:
                avg_branching = np.mean(branching_factors) if branching_factors else 0
                estimated_paths = np.power(avg_branching, depth) if avg_branching > 0 else 0

            results.append({
                'depth': depth,
                'avg_branching_factor': avg_branching,
                'estimated_total_paths': estimated_paths,
                'computational_complexity': estimated_paths * depth # O(b^d * d)
            })

        df = pd.DataFrame(results)

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Exponential growth of paths
        ax1.semilogy(df['depth'], df['estimated_total_paths'], 'bo-', label='Estimated Paths ($b^d$)')
        ax1.set_xlabel('Search Depth (d)')
        ax1.set_ylabel('Number of Paths (log scale)')
        ax1.set_title('Combinatorial Explosion: Path Count vs Depth')
        ax1.legend()
        ax1.grid(True, which="both", ls="--", alpha=0.5)

        # Computational complexity
        ax2.semilogy(df['depth'], df['computational_complexity'], 'mo-', label='Complexity ($d \\times b^d$)')
        ax2.set_xlabel('Search Depth (d)')
        ax2.set_ylabel('Computational Complexity (log scale)')
        ax2.set_title('Estimated Computational Complexity Growth')
        ax2.legend()
        ax2.grid(True, which="both", ls="--", alpha=0.5)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/combinatorial_explosion_analysis.png", dpi=300)
        plt.close()

        Cache.set_cached_analysis(cache_key, df)
        return df

    @staticmethod
    def quantum_classical_transition_analysis(fen, depth=None, lambda_quantum=None, lambda_competitive=None,
                                              samples_quantum=500, samples_competitive=500, multipv_classical=5,
                                              save_results=True):
        """
        Quantum (sample_paths mode='quantum_limit'), Competitive (sample_paths mode='competitive'), Classical (Stockfish) modlarını yan yana karşılaştırır.
        - Quantum: empirical sampling using quantum_limit (policy-head sampling in engine)
        - Competitive: path-integral sampling
        - Classical: Stockfish MultiPV → softmax
        Çıktılar: CSV (move probs), PNG (bar chart with entropies) ve kısa özet (KL/JS).
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import os
        from collections import Counter

        if depth is None:
            depth = config.HIGH_DEPTH
        if lambda_quantum is None:
            lambda_quantum = config.LAMBDA
        if lambda_competitive is None:
            lambda_competitive = config.LAMBDA

        # 1) Quantum empirical via sample_paths (quantum_limit)
        print(f"[QC] Sampling quantum_limit: samples={samples_quantum}, depth={depth}, lambda={lambda_quantum}")
        paths_q = Engine.sample_paths(fen, depth, lambda_quantum, samples_quantum, mode='quantum_limit')
        firsts_q = [str(p[0]) for p in paths_q if p]
        counter_q = Counter(firsts_q)
        total_q = sum(counter_q.values())
        dist_q = {m: cnt / total_q for m, cnt in counter_q.items()} if total_q > 0 else {}
        entropy_q = -sum([p * np.log2(p) for p in dist_q.values() if p > 0]) if dist_q else 0.0

        # 2) Competitive empirical via sample_paths (competitive)
        print(f"[QC] Sampling competitive: samples={samples_competitive}, depth={depth}, lambda={lambda_competitive}")
        paths_c = Engine.sample_paths(fen, depth, lambda_competitive, samples_competitive, mode='competitive')
        firsts_c = [str(p[0]) for p in paths_c if p]
        counter_c = Counter(firsts_c)
        total_c = sum(counter_c.values())
        dist_c = {m: cnt / total_c for m, cnt in counter_c.items()} if total_c > 0 else {}
        entropy_c = -sum([p * np.log2(p) for p in dist_c.values() if p > 0]) if dist_c else 0.0

        # 3) Classical: Stockfish MultiPV -> softmax scores
        print(f"[QC] Querying Stockfish: multipv={multipv_classical}, depth={depth}")
        try:
            sf_engine = Engine.get_engine(config.STOCKFISH_PATH)
            moves_s, scores_s = Engine.get_top_moves_and_scores(sf_engine, chess.Board(fen), depth=depth,
                                                                multipv=multipv_classical)
        except Exception:
            moves_s, scores_s = [], []
        probs_s = Calc.softmax(scores_s, lambda_competitive) if scores_s else []
        dist_s = {str(m.uci() if hasattr(m, 'uci') else m): float(p) for m, p in zip(moves_s, probs_s)}
        entropy_s = -sum([p * np.log2(p) for p in dist_s.values() if p > 0]) if dist_s else 0.0

        # 4) Align supports and compute divergences
        all_moves = sorted(set(list(dist_q.keys()) + list(dist_c.keys()) + list(dist_s.keys())))
        p = np.array([dist_q.get(m, 0.0) for m in all_moves], dtype=np.float64)
        q = np.array([dist_c.get(m, 0.0) for m in all_moves], dtype=np.float64)
        r = np.array([dist_s.get(m, 0.0) for m in all_moves], dtype=np.float64)
        eps = 1e-12
        p += eps
        q += eps
        r += eps
        p /= p.sum()
        q /= q.sum()
        r /= r.sum()

        # Pairwise KL/JS
        def kl(a, b):
            mask = (a > 0) & (b > 0)
            return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

        def js(a, b):
            m = 0.5 * (a + b)
            # Use mask to avoid log(0) and invalid divisions
            mask = (a > 0) & (b > 0) & (m > 0)
            term1 = np.sum(a[mask] * np.log2(a[mask] / m[mask]))
            term2 = np.sum(b[mask] * np.log2(b[mask] / m[mask]))
            total = float(term1) + float(term2)
            return float(0.5 * total)

        metrics = {
            'entropy_quantum': float(entropy_q),
            'entropy_competitive': float(entropy_c),
            'entropy_classical': float(entropy_s),
            'kl_q_c': kl(p, q),
            'kl_c_q': kl(q, p),
            'js_q_c': js(p, q),
            'kl_q_s': kl(p, r),
            'kl_s_q': kl(r, p),
            'js_q_s': js(p, r),
            'kl_c_s': kl(q, r),
            'kl_s_c': kl(r, q),
            'js_c_s': js(q, r)
        }

        # 5) Save CSV
        df = pd.DataFrame({
            'move': all_moves,
            'quantum_prob': p,
            'competitive_prob': q,
            'classical_prob': r
        })
        csv_path = None
        img_path = None
        summary_path = None
        if save_results:
            os.makedirs('results', exist_ok=True)
            csv_path = os.path.join('results', 'quantum_classical_transition_moves_v2.csv')
            df.to_csv(csv_path, index=False)

        # 6) Bar chart with entropies in legend
        width_fig = max(8, int(len(all_moves) * 0.5))
        plt.figure(figsize=(width_fig, 6))
        x = np.arange(len(all_moves))
        width = 0.25
        plt.bar(x - width, p, width, label=f'Quantum (H={metrics["entropy_quantum"]:.2f})', color='deepskyblue')
        plt.bar(x, q, width, label=f'Competitive (H={metrics["entropy_competitive"]:.2f})', color='orange')
        plt.bar(x + width, r, width, label=f'Classical (H={metrics["entropy_classical"]:.2f})', color='gray')
        plt.xticks(x, all_moves, rotation=45, ha='right')
        plt.xlabel('İlk Hamle')
        plt.ylabel('Olasılık')
        plt.title('Quantum vs Competitive vs Classical İlk Hamle Olasılıkları (v2)')
        plt.legend()
        plt.tight_layout()
        if save_results:
            img_path = os.path.join('results', 'quantum_classical_transition_bar_v2.png')
            plt.savefig(img_path)
            plt.close()
        else:
            plt.close()

        # 7) Save summary
        if save_results:
            summary_path = os.path.join('results', 'quantum_classical_transition_summary_v2.txt')
            with open(summary_path, 'w') as f:
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
            print(f"✓ Saved: {csv_path}, {img_path}, {summary_path}")

        return df, metrics

    @staticmethod
    def feynman_path_analogy_validation(fen, samples=None, depth=None, lambda_quantum=0.05, lambda_competitive=None, lambda_classical=5.0, save_results=True):
        """
        Validates the Feynman path integral analogy by comparing entropy values in
        quantum and classical regimes based on sampled paths. Now explicitly tests
        both 'quantum_limit' (policy-head sampling) and 'competitive' (path-integral MCTS)
        modes, plus a high-lambda classical baseline.

        :param fen: FEN string for the analyzed position
        :param samples: number of samples per regime (defaults to config.SAMPLE_COUNT)
        :param depth: search depth (defaults to config.TARGET_DEPTH)
        :param lambda_quantum: softmax lambda used for quantum_limit sampling
        :param lambda_competitive: softmax lambda used for competitive sampling (defaults to config.LAMBDA)
        :param lambda_classical: large lambda to emulate classical exploitation
        :param save_results: whether to save PNG and CSV results to results/ folder
        :return: dict with entropies and per-regime details
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import os
        from collections import Counter

        if samples is None:
            samples = config.SAMPLE_COUNT
        if depth is None:
            depth = config.TARGET_DEPTH
        if lambda_competitive is None:
            lambda_competitive = config.LAMBDA

        print(f"\n--- Feynman Analogy Validation (quantum_limit vs competitive vs classical) ---")

        try:
            # Quantum limit sampling (policy-head sampling, high coverage)
            print(f"[Feynman] Sampling quantum_limit: samples={samples}, depth={depth}, lambda={lambda_quantum}")
            q_paths = Engine.sample_paths(fen, config.HIGH_DEPTH, lambda_quantum, samples, mode='quantum_limit')
            q_entropy, q_counter = Calc.compute_entropy(q_paths)

            # Competitive sampling (MCTS+policy behavior)
            print(f"[Feynman] Sampling competitive: samples={samples}, depth={depth}, lambda={lambda_competitive}")
            c_paths = Engine.sample_paths(fen, depth, lambda_competitive, samples, mode='competitive')
            c_entropy, c_counter = Calc.compute_entropy(c_paths)

            # Classical baseline: very large lambda (exploitation)
            print(f"[Feynman] Sampling classical baseline: samples={samples}, depth={depth}, lambda={lambda_classical}")
            cl_paths = Engine.sample_paths(fen, depth, lambda_classical, samples, mode='competitive')
            cl_entropy, cl_counter = Calc.compute_entropy(cl_paths)

            # Aggregate results
            results = {
                'quantum_limit': {'lambda': lambda_quantum, 'entropy': float(q_entropy), 'count': sum(q_counter.values()) if isinstance(q_counter, Counter) else None},
                'competitive': {'lambda': lambda_competitive, 'entropy': float(c_entropy), 'count': sum(c_counter.values()) if isinstance(c_counter, Counter) else None},
                'classical': {'lambda': lambda_classical, 'entropy': float(cl_entropy), 'count': sum(cl_counter.values()) if isinstance(cl_counter, Counter) else None}
            }

            # Heuristic validity checks
            checks = {
                'quantum_gt_competitive': results['quantum_limit']['entropy'] > results['competitive']['entropy'] * 1.25,
                'competitive_gt_classical': results['competitive']['entropy'] > results['classical']['entropy'] * 1.25,
                'quantum_gt_classical': results['quantum_limit']['entropy'] > results['classical']['entropy'] * 1.5
            }
            results['checks'] = checks

            # Visualization: 3-bar comparison
            regimes = [f"Quantum\n(λ={lambda_quantum})", f"Competitive\n(λ={lambda_competitive})", f"Classical\n(λ={lambda_classical})"]
            entropies = [results['quantum_limit']['entropy'], results['competitive']['entropy'], results['classical']['entropy']]
            colors = ['deepskyblue', 'orange', 'lightcoral']

            plt.figure(figsize=(9, 6))
            bars = plt.bar(regimes, entropies, color=colors, edgecolor='black', alpha=0.85)
            plt.ylabel('Entropy (bits)')
            plt.title('Feynman Path Integral Analogy: Quantum vs Competitive vs Classical')
            plt.grid(True, alpha=0.25)

            for bar, ent in zip(bars, entropies):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * max(entropies), f"{ent:.2f}", ha='center', va='bottom', fontweight='bold')

            # Summary badge
            validity_lines = []
            validity_lines.append(f"Quantum > Competitive: {'✓' if checks['quantum_gt_competitive'] else '✗'}")
            validity_lines.append(f"Competitive > Classical: {'✓' if checks['competitive_gt_classical'] else '✗'}")
            validity_lines.append(f"Quantum > Classical: {'✓' if checks['quantum_gt_classical'] else '✗'}")
            plt.text(0.5, max(entropies) * 0.75, "\n".join(validity_lines), transform=plt.gca().transData, ha='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.6))

            plt.tight_layout()
            os.makedirs('results', exist_ok=True)
            img_path = os.path.join('results', 'feynman_analogy_validation.png')
            plt.savefig(img_path, dpi=300)
            plt.close()

            # Save CSV summary
            try:
                import pandas as pd
                df = pd.DataFrame([{
                    'regime': 'quantum_limit', 'lambda': lambda_quantum, 'entropy': results['quantum_limit']['entropy'], 'count': results['quantum_limit']['count']
                },{
                    'regime': 'competitive', 'lambda': lambda_competitive, 'entropy': results['competitive']['entropy'], 'count': results['competitive']['count']
                },{
                    'regime': 'classical', 'lambda': lambda_classical, 'entropy': results['classical']['entropy'], 'count': results['classical']['count']
                }])
                if save_results:
                    csv_path = os.path.join('results', 'feynman_analogy_validation_summary.csv')
                    df.to_csv(csv_path, index=False)
            except Exception:
                pass

            return results
        finally:
            try:
                Engine.close_all_engines()
            except Exception:
                pass


    @staticmethod
    def analyze_transition_graph_centrality(transitions):
        return Calc.analyze_transition_graph_centrality(transitions)

    @staticmethod
    def analyze_position_complexity_impact(fen_input, lambda_val=config.LAMBDA, nodes=config.LC0_NODES,
                                         sample_count=config.SAMPLE_COUNT, use_cache=True, position_names=None):
        """
        Pozisyon karmaşıklığının etkisini analiz eder - GT'siz. Konsantrasyon ve mobilite raporlanır.
        """
        # Input tipini belirle
        if isinstance(fen_input, str):
            fen_list = [fen_input]
            single_fen = True
        else:
            fen_list = fen_input
            single_fen = False

        print(f"\n--- Position Complexity Impact Analysis ({len(fen_list)} positions) ---")

        cache_key = f"PI_position_complexity_{hash(str(fen_list))}_{lambda_val}_{sample_count}"

        if use_cache:
            cached_result = Cache.get_cached_analysis(cache_key)
            if cached_result is not None:
                print("Cached result found, using existing analysis...")
                df = cached_result
                Analyzer._plot_complexity_results(df, single_fen)
                return df

        results = []

        for i, fen in enumerate(tqdm(fen_list, desc="Analyzing positions")):
            if isinstance(fen, tuple):
                fen = fen[0]
            board = chess.Board(str(fen))
            if position_names and i < len(position_names):
                position_name = str(position_names[i])
            else:
                position_name = f"Position_{i + 1}" if single_fen else f"Pos_{i + 1}"

            legal_moves = list(board.legal_moves)
            legal_moves_count = len(legal_moves)
            pieces_count = len(board.piece_map())

            start_time = time.time()
            paths = Engine.sample_paths(str(fen), config.TARGET_DEPTH, lambda_val, sample_count, use_cache=use_cache)
            sampling_time = time.time() - start_time

            entropy, move_counter = Calc.compute_entropy(paths)

            # Next-move mobility
            next_mobilities = []
            for move_str in move_counter.keys():
                try:
                    test_board = chess.Board(str(fen))
                    move = chess.Move.from_uci(move_str)
                    if move in legal_moves:
                        test_board.push(move)
                        next_mobilities.append(len(list(test_board.legal_moves)))
                except:
                    continue

            next_mobility_avg = np.mean(next_mobilities) if next_mobilities else 0
            next_mobility_std = np.std(next_mobilities) if next_mobilities else 0

            # Concentration metrics (top-k cumulative)
            most_common = move_counter.most_common()
            total = sum(move_counter.values()) if move_counter else 1
            conc_top1 = (most_common[0][1] / total) if most_common else 0.0
            conc_top3 = (sum(cnt for _, cnt in most_common[:3]) / total) if most_common else 0.0
            conc_top5 = (sum(cnt for _, cnt in most_common[:5]) / total) if most_common else 0.0

            results.append({
                'position': position_name,
                'fen': fen,
                'root_legal_moves': legal_moves_count,
                'pieces_count': pieces_count,
                'next_mobility_avg': next_mobility_avg,
                'next_mobility_std': next_mobility_std,
                'sampling_time_sec': sampling_time,
                'entropy': entropy,
                'complexity_score': legal_moves_count * pieces_count,
                'conc_top1': conc_top1,
                'conc_top3': conc_top3,
                'conc_top5': conc_top5
            })

        df = pd.DataFrame(results)

        if use_cache:
            Cache.set_cached_analysis(cache_key, df)

        Analyzer._plot_complexity_results(df, single_fen)

        return df

    @staticmethod
    def _plot_complexity_results(df, single_fen=False):
        """Visualizes complexity analysis results in a more explanatory way (without GT)."""
        import matplotlib.pyplot as plt
        if single_fen:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            row = df.iloc[0]
            metrics = ['root_legal_moves', 'entropy', 'conc_top1', 'conc_top3', 'conc_top5']
            values = [row[m] for m in metrics]
            ax1.bar(metrics, values, alpha=0.7, color=['blue', 'green', 'red', 'orange', 'purple'])
            ax1.set_title('Position Complexity Metrics')
            ax1.set_ylabel('Value')
            ax1.tick_params(axis='x', rotation=45)
            for i, v in enumerate(values):
                ax1.text(i, v + 0.02 * max(values), f'{v:.2f}', ha='center', va='bottom', fontsize=10)
            ax2.bar(['Next Move Avg.', 'Next Move Std'],
                    [row['next_mobility_avg'], row['next_mobility_std']],
                    alpha=0.7, color=['lightblue', 'lightcoral'])
            ax2.set_title('Next Move Mobility Analysis')
            ax2.set_ylabel('Legal Move Count')
            for i, v in enumerate([row['next_mobility_avg'], row['next_mobility_std']]):
                ax2.text(i, v + 0.02 * max([row['next_mobility_avg'], row['next_mobility_std']]), f'{v:.2f}', ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            output_file = "results/position_complexity_impact.png"
        else:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 11))
            # 1. Complexity score vs sampling time (color: entropy)
            scatter = ax1.scatter(df['complexity_score'], df['sampling_time_sec'],
                                c=df['entropy'], cmap='viridis', alpha=0.7, s=60)
            ax1.set_xlabel('Complexity Score (legal_moves × pieces)')
            ax1.set_ylabel('Sampling Time (seconds)')
            ax1.set_title('Complexity vs Performance')
            plt.colorbar(scatter, ax=ax1, label='Entropy')

            # 2. Entropy vs concentration
            ax2.scatter(df['entropy'], df['conc_top1'], alpha=0.7, color='red', s=60)
            ax2.set_xlabel('Entropy (bits)')
            ax2.set_ylabel('Top-1 Concentration')
            ax2.set_title('Entropy vs Concentration Trade-off')

            # 3. Position complexity distribution
            ax3.hist(df['complexity_score'], bins=10, alpha=0.7, color='lightblue', edgecolor='black')
            ax3.set_xlabel('Complexity Score')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Position Complexity Distribution')

            # 4. Next move mobility analysis
            ax4.scatter(df['root_legal_moves'], df['next_mobility_avg'],
                       alpha=0.7, color='orange', s=60)
            ax4.set_xlabel('Root Legal Moves')
            ax4.set_ylabel('Average Next Move Mobility')
            ax4.set_title('Root vs Next Move Mobility')

            plt.tight_layout()
            output_file = "results/position_complexity_analysis.png"

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Complexity analysis plot saved: {output_file}")

    # === NEW COMPREHENSIVE ANALYSIS FUNCTIONS ===

    @staticmethod
    def _generate_comprehensive_depth_plots_with_variant(results, fen, variant_name="Standard"):
        """
        Enhanced version of _generate_comprehensive_depth_plots with variant support
        and additional statistical analysis plots.
        """
        from experiments import Experiments
        Experiments._generate_comprehensive_depth_plots(results, fen)

        # Additional variant-specific analysis
        lambda_metrics = results['metrics']['lambda_scan']
        lambda_values = results['lambda_values']

        # Generate variant-specific entropy-accuracy correlation plot
        entropies = [lambda_metrics[lam]['entropy'] for lam in lambda_values]
        accuracies = [lambda_metrics[lam]['accuracy'] for lam in lambda_values]

        plt.figure(figsize=(10, 8))
        plt.scatter(entropies, accuracies, s=100, alpha=0.7, c=lambda_values,
                   cmap='viridis', edgecolors='black')
        plt.colorbar(label='Lambda (λ)')
        plt.xlabel('Entropy (bits)')
        plt.ylabel('Concentration (mode probability)')
        plt.title(f'Entropy-Accuracy Correlation: {variant_name}')
        plt.grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = np.corrcoef(entropies, accuracies)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))

        plt.tight_layout()
        plt.savefig(f'results/entropy_accuracy_correlation_{variant_name.lower().replace(" ", "_")}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 Enhanced depth plots generated for {variant_name}")

    @staticmethod
    def pi_quantum_vs_lc0_chess960_experiment(chess960_positions=None, sample_count=None, save_results=True):
        """
        Comprehensive Chess960 experiment comparing PI quantum-limit vs LC0 across multiple positions.
        Generates detailed statistical analysis and visualizations.
        """
        if chess960_positions is None:
            # Sample Chess960 positions for testing
            chess960_positions = [
                ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "Chess960_Position_1"),
                ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "Chess960_Position_2"),
                ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5", "Chess960_Position_3")
            ]

        if sample_count is None:
            sample_count = config.SAMPLE_COUNT

        print(f"\n=== PI Quantum vs LC0 Chess960 Experiment ===")
        print(f"Positions: {len(chess960_positions)}")
        print(f"Sample count: {sample_count}")

        results = []

        for fen, position_name in tqdm(chess960_positions, desc="Chess960 Analysis"):
            print(f"\nAnalyzing {position_name}...")

            # PI Quantum-limit sampling
            pi_quantum_paths = Engine.sample_paths(fen, config.HIGH_DEPTH, config.LAMBDA,
                                                 sample_count, mode='quantum_limit')
            pi_quantum_entropy, pi_quantum_counter = Calc.compute_entropy(pi_quantum_paths)
            pi_quantum_accuracy = Calc.top_move_concentration(pi_quantum_paths)

            # LC0 analysis
            lc0_opts = {"MultiPV": config.MULTIPV, "Temperature": config.LC0_TEMPERATURE}
            lc0_moves, lc0_scores, lc0_time = Engine.lc0_top_moves_and_scores(
                fen, depth=config.TARGET_DEPTH, multipv=config.MULTIPV, options=lc0_opts
            )
            lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LAMBDA)
            lc0_entropy = -sum([p * np.log2(p) for p in lc0_probs if p > 0]) if lc0_probs else 0.0
            lc0_accuracy = max(lc0_probs) if lc0_probs else 0.0

            # Statistical comparison
            kl_divergence = 0.0
            if pi_quantum_counter and lc0_moves:
                # Align move distributions for KL divergence
                all_moves = set(list(pi_quantum_counter.keys()) + [str(m) for m in lc0_moves])
                pi_dist = {move: pi_quantum_counter.get(move, 0) / sum(pi_quantum_counter.values())
                          for move in all_moves}
                lc0_dist = {move: lc0_probs[i] if i < len(lc0_probs) and str(lc0_moves[i]) == move else 1e-10
                           for i, move in enumerate(all_moves)}

                try:
                    kl_divergence = Calc.kl_divergence(pi_dist, lc0_dist)
                except:
                    kl_divergence = float('inf')

            result = {
                'position_name': position_name,
                'fen': fen,
                'pi_quantum_entropy': pi_quantum_entropy,
                'pi_quantum_accuracy': pi_quantum_accuracy,
                'lc0_entropy': lc0_entropy,
                'lc0_accuracy': lc0_accuracy,
                'kl_divergence': kl_divergence,
                'entropy_difference': pi_quantum_entropy - lc0_entropy,
                'accuracy_difference': pi_quantum_accuracy - lc0_accuracy
            }

            results.append(result)
            print(f"   PI Quantum: entropy={pi_quantum_entropy:.3f}, accuracy={pi_quantum_accuracy:.3f}")
            print(f"   LC0: entropy={lc0_entropy:.3f}, accuracy={lc0_accuracy:.3f}")

        # Save results
        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_quantum_vs_lc0_chess960_results.csv", index=False)

            # Generate comprehensive visualizations
            Analyzer._generate_chess960_visualizations(df)

        return results

    @staticmethod
    def _generate_chess960_visualizations(df):
        """Generate comprehensive Chess960 analysis visualizations."""

        # 1. Entropy comparison
        plt.figure(figsize=(12, 8))
        x = np.arange(len(df))
        width = 0.35

        plt.bar(x - width/2, df['pi_quantum_entropy'], width, label='PI Quantum', alpha=0.8)
        plt.bar(x + width/2, df['lc0_entropy'], width, label='LC0', alpha=0.8)

        plt.xlabel('Chess960 Positions')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy Comparison: PI Quantum vs LC0 (Chess960)')
        plt.xticks(x, df['position_name'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/chess960_entropy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Accuracy comparison
        plt.figure(figsize=(12, 8))
        plt.bar(x - width/2, df['pi_quantum_accuracy'], width, label='PI Quantum', alpha=0.8)
        plt.bar(x + width/2, df['lc0_accuracy'], width, label='LC0', alpha=0.8)

        plt.xlabel('Chess960 Positions')
        plt.ylabel('Accuracy (concentration)')
        plt.title('Accuracy Comparison: PI Quantum vs LC0 (Chess960)')
        plt.xticks(x, df['position_name'], rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/chess960_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Scatter plot: Entropy vs Accuracy
        plt.figure(figsize=(10, 8))
        plt.scatter(df['pi_quantum_entropy'], df['pi_quantum_accuracy'],
                   label='PI Quantum', s=100, alpha=0.7)
        plt.scatter(df['lc0_entropy'], df['lc0_accuracy'],
                   label='LC0', s=100, alpha=0.7)

        plt.xlabel('Entropy (bits)')
        plt.ylabel('Accuracy (concentration)')
        plt.title('Entropy vs Accuracy: Chess960 Positions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/chess960_entropy_accuracy_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("   📊 Chess960 visualizations generated")

    @staticmethod
    def chess960_variant_game_result_histogram(game_results, save_results=True):
        """Generate histogram of Chess960 game results."""
        result_counts = pd.Series(game_results).value_counts()

        plt.figure(figsize=(10, 6))
        bars = plt.bar(result_counts.index, result_counts.values, alpha=0.8,
                      color=['lightgreen', 'lightcoral', 'lightblue'])

        plt.xlabel('Game Results')
        plt.ylabel('Frequency')
        plt.title('Chess960 Game Results Distribution')

        # Add value labels on bars
        for bar, count in zip(bars, result_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(count), ha='center', va='bottom')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_results:
            plt.savefig('results/chess960_game_results_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()

        return result_counts

    @staticmethod
    def chess960_move_distribution_heatmap(move_data, save_results=True):
        """Generate heatmap of move distributions across Chess960 positions."""
        # Create move frequency matrix
        all_moves = set()
        for position_moves in move_data.values():
            all_moves.update(position_moves.keys())

        all_moves = sorted(list(all_moves))
        positions = list(move_data.keys())

        # Build frequency matrix
        freq_matrix = np.zeros((len(positions), len(all_moves)))
        for i, position in enumerate(positions):
            for j, move in enumerate(all_moves):
                freq_matrix[i, j] = move_data[position].get(move, 0)

        # Normalize by row
        row_sums = freq_matrix.sum(axis=1, keepdims=True)
        freq_matrix = np.divide(freq_matrix, row_sums, out=np.zeros_like(freq_matrix), where=row_sums!=0)

        plt.figure(figsize=(16, 8))
        sns.heatmap(freq_matrix, xticklabels=all_moves, yticklabels=positions,
                   cmap='YlOrRd', annot=False, fmt='.2f')
        plt.title('Chess960 Move Distribution Heatmap')
        plt.xlabel('Moves')
        plt.ylabel('Positions')
        plt.xticks(rotation=90)
        plt.tight_layout()

        if save_results:
            plt.savefig('results/chess960_move_distribution_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def chess960_kl_topn_analysis(distributions, n_values=[1, 3, 5], save_results=True):
        """Analyze KL divergence and top-N accuracy for Chess960 positions."""
        results = []

        for position_name, dist_data in distributions.items():
            pi_dist = dist_data['pi_quantum']
            lc0_dist = dist_data['lc0']

            # Calculate KL divergence
            kl_div = Calc.kl_divergence(pi_dist, lc0_dist)

            # Calculate top-N accuracies
            pi_sorted = sorted(pi_dist.items(), key=lambda x: x[1], reverse=True)
            lc0_sorted = sorted(lc0_dist.items(), key=lambda x: x[1], reverse=True)

            for n in n_values:
                pi_topn = sum([prob for _, prob in pi_sorted[:n]])
                lc0_topn = sum([prob for _, prob in lc0_sorted[:n]])

                results.append({
                    'position': position_name,
                    'n': n,
                    'pi_topn_accuracy': pi_topn,
                    'lc0_topn_accuracy': lc0_topn,
                    'kl_divergence': kl_div
                })

        df = pd.DataFrame(results)

        if save_results:
            df.to_csv('results/chess960_kl_topn_analysis.csv', index=False)

            # Visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Top-N accuracy comparison
            for n in n_values:
                subset = df[df['n'] == n]
                ax1.plot(subset.index, subset['pi_topn_accuracy'],
                        label=f'PI Top-{n}', marker='o')
                ax1.plot(subset.index, subset['lc0_topn_accuracy'],
                        label=f'LC0 Top-{n}', marker='s')

            ax1.set_xlabel('Position Index')
            ax1.set_ylabel('Top-N Accuracy')
            ax1.set_title('Top-N Accuracy Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # KL divergence
            unique_positions = df['position'].unique()
            kl_values = [df[df['position'] == pos]['kl_divergence'].iloc[0] for pos in unique_positions]
            ax2.bar(range(len(unique_positions)), kl_values, alpha=0.8)
            ax2.set_xlabel('Position Index')
            ax2.set_ylabel('KL Divergence')
            ax2.set_title('KL Divergence by Position')
            ax2.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig('results/chess960_kl_topn_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()

        return df

    @staticmethod
    def chess960_game_duration_movecount_analysis(game_data, save_results=True):
        """Analyze game duration and move count patterns in Chess960."""
        durations = [game['duration'] for game in game_data if 'duration' in game]
        move_counts = [game['move_count'] for game in game_data if 'move_count' in game]
        results = [game['result'] for game in game_data if 'result' in game]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Duration distribution
        ax1.hist(durations, bins=20, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Game Duration (minutes)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Chess960 Game Duration Distribution')
        ax1.grid(True, alpha=0.3)

        # Move count distribution
        ax2.hist(move_counts, bins=20, alpha=0.7, edgecolor='black', color='orange')
        ax2.set_xlabel('Move Count')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Chess960 Move Count Distribution')
        ax2.grid(True, alpha=0.3)

        # Duration vs Move count
        ax3.scatter(durations, move_counts, alpha=0.7, s=60)
        ax3.set_xlabel('Game Duration (minutes)')
        ax3.set_ylabel('Move Count')
        ax3.set_title('Duration vs Move Count')
        ax3.grid(True, alpha=0.3)

        # Result distribution
        result_counts = pd.Series(results).value_counts()
        ax4.pie(result_counts.values, labels=result_counts.index, autopct='%1.1f%%')
        ax4.set_title('Chess960 Game Results')

        plt.tight_layout()

        if save_results:
            plt.savefig('results/chess960_game_duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Statistical summary
        stats_summary = {
            'avg_duration': np.mean(durations) if durations else 0,
            'avg_move_count': np.mean(move_counts) if move_counts else 0,
            'duration_std': np.std(durations) if durations else 0,
            'move_count_std': np.std(move_counts) if move_counts else 0,
            'correlation': np.corrcoef(durations, move_counts)[0, 1] if len(durations) == len(move_counts) and len(durations) > 1 else 0
        }

        return stats_summary

    @staticmethod
    def chess960_complexity_entropy_accuracy_analysis(position_data, save_results=True):
        """Analyze relationship between position complexity, entropy, and accuracy in Chess960."""
        complexity_scores = []
        entropies = []
        accuracies = []
        position_names = []

        for pos_name, data in position_data.items():
            # Calculate complexity score (legal moves * pieces)
            board = chess.Board(data['fen'])
            legal_moves = len(list(board.legal_moves))
            pieces = len(board.piece_map())
            complexity = legal_moves * pieces

            complexity_scores.append(complexity)
            entropies.append(data['entropy'])
            accuracies.append(data['accuracy'])
            position_names.append(pos_name)

        # Statistical analysis
        entropy_complexity_corr = np.corrcoef(complexity_scores, entropies)[0, 1]
        accuracy_complexity_corr = np.corrcoef(complexity_scores, accuracies)[0, 1]
        entropy_accuracy_corr = np.corrcoef(entropies, accuracies)[0, 1]

        # Visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Complexity vs Entropy
        ax1.scatter(complexity_scores, entropies, alpha=0.7, s=80, c='blue')
        ax1.set_xlabel('Complexity Score (legal_moves × pieces)')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title(f'Complexity vs Entropy (r={entropy_complexity_corr:.3f})')
        ax1.grid(True, alpha=0.3)

        # Complexity vs Accuracy
        ax2.scatter(complexity_scores, accuracies, alpha=0.7, s=80, c='red')
        ax2.set_xlabel('Complexity Score')
        ax2.set_ylabel('Accuracy (concentration)')
        ax2.set_title(f'Complexity vs Accuracy (r={accuracy_complexity_corr:.3f})')
        ax2.grid(True, alpha=0.3)

        # Entropy vs Accuracy
        ax3.scatter(entropies, accuracies, alpha=0.7, s=80, c='green')
        ax3.set_xlabel('Entropy (bits)')
        ax3.set_ylabel('Accuracy (concentration)')
        ax3.set_title(f'Entropy vs Accuracy (r={entropy_accuracy_corr:.3f})')
        ax3.grid(True, alpha=0.3)

        # 3D scatter plot (complexity, entropy, accuracy)
        from mpl_toolkits.mplot3d import Axes3D
        ax4 = fig.add_subplot(224, projection='3d')
        scatter = ax4.scatter(complexity_scores, entropies, accuracies,
                            c=complexity_scores, cmap='viridis', s=60)
        ax4.set_xlabel('Complexity')
        ax4.set_ylabel('Entropy')
        ax4.set_zlabel('Accuracy')
        ax4.set_title('3D Relationship')

        plt.tight_layout()

        if save_results:
            plt.savefig('results/chess960_complexity_entropy_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

        analysis_results = {
            'entropy_complexity_correlation': entropy_complexity_corr,
            'accuracy_complexity_correlation': accuracy_complexity_corr,
            'entropy_accuracy_correlation': entropy_accuracy_corr,
            'complexity_scores': complexity_scores,
            'entropies': entropies,
            'accuracies': accuracies
        }

        return analysis_results

    @staticmethod
    def chess960_bootstrap_ci_analysis(data_samples, confidence_level=0.95, n_bootstrap=1000, save_results=True):
        """Perform bootstrap confidence interval analysis for Chess960 data."""
        def bootstrap_ci(data, n_bootstrap=1000, ci=0.95):
            if len(data) == 0:
                return 0, 0
            boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True))
                         for _ in range(n_bootstrap)]
            lower = np.percentile(boot_means, (1-ci)/2*100)
            upper = np.percentile(boot_means, (1+(ci))/2*100)
            return lower, upper

        results = {}

        for metric_name, data in data_samples.items():
            if len(data) > 0:
                mean_val = np.mean(data)
                std_val = np.std(data)
                ci_lower, ci_upper = bootstrap_ci(data, n_bootstrap, confidence_level)

                results[metric_name] = {
                    'mean': mean_val,
                    'std': std_val,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'ci_width': ci_upper - ci_lower
                }

        # Visualization
        metrics = list(results.keys())
        means = [results[m]['mean'] for m in metrics]
        ci_lowers = [results[m]['ci_lower'] for m in metrics]
        ci_uppers = [results[m]['ci_upper'] for m in metrics]

        plt.figure(figsize=(12, 8))
        x_pos = np.arange(len(metrics))

        plt.errorbar(x_pos, means,
                    yerr=[np.array(means) - np.array(ci_lowers),
                          np.array(ci_uppers) - np.array(means)],
                    fmt='o', capsize=5, capthick=2, markersize=8)

        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title(f'Bootstrap Confidence Intervals ({confidence_level*100:.0f}%)')
        plt.xticks(x_pos, metrics, rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_results:
            plt.savefig('results/chess960_bootstrap_ci_analysis.png', dpi=300, bbox_inches='tight')

            # Save numerical results
            ci_df = pd.DataFrame(results).T
            ci_df.to_csv('results/chess960_bootstrap_ci_results.csv')

        plt.close()

        return results

    @staticmethod
    def chess960_engine_comparison_analysis(engine_results, save_results=True):
        """Comprehensive engine comparison analysis for Chess960."""
        engines = list(engine_results.keys())
        metrics = ['entropy', 'accuracy', 'time', 'move_diversity']

        # Prepare data matrix
        data_matrix = np.zeros((len(engines), len(metrics)))
        for i, engine in enumerate(engines):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = engine_results[engine].get(metric, 0)

        # Normalize data for radar chart
        normalized_data = data_matrix.copy()
        for j in range(len(metrics)):
            col_max = np.max(data_matrix[:, j])
            if col_max > 0:
                normalized_data[:, j] = data_matrix[:, j] / col_max

        # Radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, engine in enumerate(engines):
            values = normalized_data[i].tolist()
            values += values[:1]  # Complete the circle

            ax1.plot(angles, values, 'o-', linewidth=2, label=engine, color=colors[i % len(colors)])
            ax1.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(metrics)
        ax1.set_ylim(0, 1)
        ax1.set_title('Engine Performance Radar Chart (Normalized)')
        ax1.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        # Bar chart comparison
        x = np.arange(len(metrics))
        width = 0.8 / len(engines)

        for i, engine in enumerate(engines):
            ax2 = plt.subplot(1, 2, 2)
            ax2.bar(x + i * width, data_matrix[i], width, label=engine, alpha=0.8)

        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Values')
        ax2.set_title('Engine Performance Comparison')
        ax2.set_xticks(x + width * (len(engines) - 1) / 2)
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_results:
            plt.savefig('results/chess960_engine_comparison.png', dpi=300, bbox_inches='tight')

            # Save comparison matrix
            comparison_df = pd.DataFrame(data_matrix, index=engines, columns=metrics)
            comparison_df.to_csv('results/chess960_engine_comparison_matrix.csv')

        plt.close()

        return data_matrix, normalized_data

    @staticmethod
    def pi_quantum_sensitivity_experiment(fen, lambda_range=None, depth_range=None,
                                        sample_counts=None, save_results=True):
        """
        Comprehensive sensitivity analysis for PI quantum-limit mode across multiple parameters.
        """
        if lambda_range is None:
            lambda_range = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
        if depth_range is None:
            depth_range = [4, 8, 12, 16, 20]
        if sample_counts is None:
            sample_counts = [100, 250, 500, 1000]

        print(f"\n=== PI Quantum Sensitivity Experiment ===")
        print(f"Lambda range: {lambda_range}")
        print(f"Depth range: {depth_range}")
        print(f"Sample counts: {sample_counts}")

        results = []

        # Lambda sensitivity
        print("\nLambda sensitivity analysis...")
        for lam in tqdm(lambda_range, desc="Lambda sensitivity"):
            paths = Engine.sample_paths(fen, config.HIGH_DEPTH, lam, config.SAMPLE_COUNT, mode='quantum_limit')
            entropy, counter = Calc.compute_entropy(paths)
            accuracy = Calc.top_move_concentration(paths)

            results.append({
                'parameter_type': 'lambda',
                'parameter_value': lam,
                'entropy': entropy,
                'accuracy': accuracy,
                'move_diversity': len(counter) if counter else 0
            })

        # Depth sensitivity
        print("\nDepth sensitivity analysis...")
        for depth in tqdm(depth_range, desc="Depth sensitivity"):
            paths = Engine.sample_paths(fen, depth, config.LAMBDA, config.SAMPLE_COUNT, mode='quantum_limit')
            entropy, counter = Calc.compute_entropy(paths)
            accuracy = Calc.top_move_concentration(paths)

            results.append({
                'parameter_type': 'depth',
                'parameter_value': depth,
                'entropy': entropy,
                'accuracy': accuracy,
                'move_diversity': len(counter) if counter else 0
            })

        # Sample count sensitivity
        print("\nSample count sensitivity analysis...")
        for sample_count in tqdm(sample_counts, desc="Sample sensitivity"):
            paths = Engine.sample_paths(fen, config.HIGH_DEPTH, config.LAMBDA, sample_count, mode='quantum_limit')
            entropy, counter = Calc.compute_entropy(paths)
            accuracy = Calc.top_move_concentration(paths)

            results.append({
                'parameter_type': 'sample_count',
                'parameter_value': sample_count,
                'entropy': entropy,
                'accuracy': accuracy,
                'move_diversity': len(counter) if counter else 0
            })

        df = pd.DataFrame(results)

        if save_results:
            df.to_csv('results/pi_quantum_sensitivity_results.csv', index=False)

            # Generate sensitivity plots
            Analyzer._generate_sensitivity_plots(df, lambda_range, depth_range, sample_counts)

        return df

    @staticmethod
    def _generate_sensitivity_plots(df, lambda_range, depth_range, sample_counts):
        """Generate sensitivity analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))

        # Lambda sensitivity - Entropy
        lambda_data = df[df['parameter_type'] == 'lambda']
        ax1.semilogx(lambda_data['parameter_value'], lambda_data['entropy'], 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (λ)')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Lambda Sensitivity: Entropy')
        ax1.grid(True, alpha=0.3)

        # Lambda sensitivity - Accuracy
        ax2.semilogx(lambda_data['parameter_value'], lambda_data['accuracy'], 's-', linewidth=2, markersize=8, color='red')
        ax2.set_xlabel('Lambda (λ)')
        ax2.set_ylabel('Accuracy (concentration)')
        ax2.set_title('Lambda Sensitivity: Accuracy')
        ax2.grid(True, alpha=0.3)

        # Depth sensitivity - Entropy
        depth_data = df[df['parameter_type'] == 'depth']
        ax3.plot(depth_data['parameter_value'], depth_data['entropy'], 'o-', linewidth=2, markersize=8, color='green')
        ax3.set_xlabel('Depth')
        ax3.set_ylabel('Entropy (bits)')
        ax3.set_title('Depth Sensitivity: Entropy')
        ax3.grid(True, alpha=0.3)

        # Depth sensitivity - Accuracy
        ax4.plot(depth_data['parameter_value'], depth_data['accuracy'], 's-', linewidth=2, markersize=8, color='orange')
        ax4.set_xlabel('Depth')
        ax4.set_ylabel('Accuracy (concentration)')
        ax4.set_title('Depth Sensitivity: Accuracy')
        ax4.grid(True, alpha=0.3)

        # Sample count sensitivity - Entropy
        sample_data = df[df['parameter_type'] == 'sample_count']
        ax5.semilogx(sample_data['parameter_value'], sample_data['entropy'], 'o-', linewidth=2, markersize=8, color='purple')
        ax5.set_xlabel('Sample Count')
        ax5.set_ylabel('Entropy (bits)')
        ax5.set_title('Sample Count Sensitivity: Entropy')
        ax5.grid(True, alpha=0.3)

        # Sample count sensitivity - Accuracy
        ax6.semilogx(sample_data['parameter_value'], sample_data['accuracy'], 's-', linewidth=2, markersize=8, color='brown')
        ax6.set_xlabel('Sample Count')
        ax6.set_ylabel('Accuracy (concentration)')
        ax6.set_title('Sample Count Sensitivity: Accuracy')
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/pi_quantum_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("   📊 Sensitivity analysis plots generated")

    @staticmethod
    def compare_policy_vs_quantum_sampling(fen, sample_count=None, save_results=True):
        """
        Compare policy-based sampling vs quantum-limit sampling approaches.
        """
        if sample_count is None:
            sample_count = config.SAMPLE_COUNT

        print(f"\n=== Policy vs Quantum Sampling Comparison ===")

        # Policy-based sampling (competitive mode)
        print("Running policy-based sampling...")
        policy_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, config.LAMBDA,
                                         sample_count, mode='competitive')
        policy_entropy, policy_counter = Calc.compute_entropy(policy_paths)
        policy_accuracy = Calc.top_move_concentration(policy_paths)

        # Quantum-limit sampling
        print("Running quantum-limit sampling...")
        quantum_paths = Engine.sample_paths(fen, config.HIGH_DEPTH, config.LAMBDA,
                                          sample_count, mode='quantum_limit')
        quantum_entropy, quantum_counter = Calc.compute_entropy(quantum_paths)
        quantum_accuracy = Calc.top_move_concentration(quantum_paths)

        # Statistical comparison
        # KL divergence between distributions
        all_moves = set(list(policy_counter.keys()) + list(quantum_counter.keys()))
        policy_dist = {move: policy_counter.get(move, 0) / sum(policy_counter.values())
                      for move in all_moves}
        quantum_dist = {move: quantum_counter.get(move, 0) / sum(quantum_counter.values())
                       for move in all_moves}

        kl_policy_quantum = Calc.kl_divergence(policy_dist, quantum_dist)
        kl_quantum_policy = Calc.kl_divergence(quantum_dist, policy_dist)

        # Jensen-Shannon divergence
        def js_divergence(p, q):
            m = {move: 0.5 * (p.get(move, 0) + q.get(move, 0)) for move in set(list(p.keys()) + list(q.keys()))}
            return 0.5 * Calc.kl_divergence(p, m) + 0.5 * Calc.kl_divergence(q, m)

        js_div = js_divergence(policy_dist, quantum_dist)

        results = {
            'policy_entropy': policy_entropy,
            'policy_accuracy': policy_accuracy,
            'policy_move_diversity': len(policy_counter),
            'quantum_entropy': quantum_entropy,
            'quantum_accuracy': quantum_accuracy,
            'quantum_move_diversity': len(quantum_counter),
            'kl_policy_quantum': kl_policy_quantum,
            'kl_quantum_policy': kl_quantum_policy,
            'js_divergence': js_div,
            'entropy_ratio': quantum_entropy / policy_entropy if policy_entropy > 0 else float('inf'),
            'accuracy_ratio': quantum_accuracy / policy_accuracy if policy_accuracy > 0 else float('inf')
        }

        if save_results:
            # Visualization
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

            # Entropy and Accuracy comparison
            methods = ['Policy-based', 'Quantum-limit']
            entropies = [policy_entropy, quantum_entropy]
            accuracies = [policy_accuracy, quantum_accuracy]

            ax1.bar(methods, entropies, alpha=0.8, color=['lightblue', 'lightcoral'])
            ax1.set_ylabel('Entropy (bits)')
            ax1.set_title('Entropy Comparison')
            ax1.grid(True, alpha=0.3, axis='y')

            ax2.bar(methods, accuracies, alpha=0.8, color=['lightgreen', 'lightyellow'])
            ax2.set_ylabel('Accuracy (concentration)')
            ax2.set_title('Accuracy Comparison')
            ax2.grid(True, alpha=0.3, axis='y')

            # Move diversity comparison
            diversities = [len(policy_counter), len(quantum_counter)]
            ax3.bar(methods, diversities, alpha=0.8, color=['orange', 'purple'])
            ax3.set_ylabel('Number of Unique Moves')
            ax3.set_title('Move Diversity Comparison')
            ax3.grid(True, alpha=0.3, axis='y')

            # Divergence metrics
            divergences = ['KL(P||Q)', 'KL(Q||P)', 'JS Divergence']
            div_values = [kl_policy_quantum, kl_quantum_policy, js_div]
            ax4.bar(divergences, div_values, alpha=0.8, color=['red', 'blue', 'green'])
            ax4.set_ylabel('Divergence')
            ax4.set_title('Distribution Divergence Metrics')
            ax4.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig('results/policy_vs_quantum_sampling_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Save numerical results
            results_df = pd.DataFrame([results])
            results_df.to_csv('results/policy_vs_quantum_sampling_results.csv', index=False)

            print("   📊 Policy vs Quantum sampling comparison completed")

        return results

    @staticmethod
    def quantum_classical_transition_analysis(fen, lambda_values=None, save_results=True):
        """
        Enhanced quantum-classical transition analysis with comprehensive statistical testing.
        """
        if lambda_values is None:
            lambda_values = np.logspace(-2, 1, 20)  # From 0.01 to 10

        print(f"\n=== Enhanced Quantum-Classical Transition Analysis ===")
        print(f"Lambda range: {lambda_values[0]:.3f} to {lambda_values[-1]:.3f}")

        results = []

        for lam in tqdm(lambda_values, desc="Transition analysis"):
            # Sample with both modes
            competitive_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lam,
                                                  config.SAMPLE_COUNT, mode='competitive')
            quantum_paths = Engine.sample_paths(fen, config.HIGH_DEPTH, lam,
                                               config.SAMPLE_COUNT, mode='quantum_limit')

            # Calculate metrics
            comp_entropy, comp_counter = Calc.compute_entropy(competitive_paths)
            comp_accuracy = Calc.top_move_concentration(competitive_paths)

            quantum_entropy, quantum_counter = Calc.compute_entropy(quantum_paths)
            quantum_accuracy = Calc.top_move_concentration(quantum_paths)

            # Transition indicators
            entropy_ratio = quantum_entropy / comp_entropy if comp_entropy > 0 else 1.0
            accuracy_ratio = quantum_accuracy / comp_accuracy if comp_accuracy > 0 else 1.0

            results.append({
                'lambda': lam,
                'competitive_entropy': comp_entropy,
                'competitive_accuracy': comp_accuracy,
                'quantum_entropy': quantum_entropy,
                'quantum_accuracy': quantum_accuracy,
                'entropy_ratio': entropy_ratio,
                'accuracy_ratio': accuracy_ratio,
                'entropy_difference': quantum_entropy - comp_entropy,
                'accuracy_difference': quantum_accuracy - comp_accuracy
            })

        df = pd.DataFrame(results)

        if save_results:
            df.to_csv('results/quantum_classical_transition_detailed.csv', index=False)

            # Generate transition plots
            Analyzer._generate_transition_plots(df, lambda_values)

            # Statistical analysis
            Analyzer._analyze_transition_statistics(df)

        return df

    @staticmethod
    def _generate_transition_plots(df, lambda_values):
        """Generate comprehensive transition analysis plots."""
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(16, 18))

        # Entropy transition
        ax1.semilogx(df['lambda'], df['competitive_entropy'], 'o-', label='Competitive', linewidth=2)
        ax1.semilogx(df['lambda'], df['quantum_entropy'], 's-', label='Quantum-limit', linewidth=2)
        ax1.set_xlabel('Lambda (λ)')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Entropy: Quantum-Classical Transition')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy transition
        ax2.semilogx(df['lambda'], df['competitive_accuracy'], 'o-', label='Competitive', linewidth=2)
        ax2.semilogx(df['lambda'], df['quantum_accuracy'], 's-', label='Quantum-limit', linewidth=2)
        ax2.set_xlabel('Lambda (λ)')
        ax2.set_ylabel('Accuracy (concentration)')
        ax2.set_title('Accuracy: Quantum-Classical Transition')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Entropy ratio
        ax3.semilogx(df['lambda'], df['entropy_ratio'], 'o-', color='purple', linewidth=2)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal entropy')
        ax3.set_xlabel('Lambda (λ)')
        ax3.set_ylabel('Entropy Ratio (Quantum/Competitive)')
        ax3.set_title('Entropy Ratio Transition')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Accuracy ratio
        ax4.semilogx(df['lambda'], df['accuracy_ratio'], 's-', color='orange', linewidth=2)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Equal accuracy')
        ax4.set_xlabel('Lambda (λ)')
        ax4.set_ylabel('Accuracy Ratio (Quantum/Competitive)')
        ax4.set_title('Accuracy Ratio Transition')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Entropy difference
        ax5.semilogx(df['lambda'], df['entropy_difference'], 'o-', color='green', linewidth=2)
        ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='No difference')
        ax5.set_xlabel('Lambda (λ)')
        ax5.set_ylabel('Entropy Difference (Quantum - Competitive)')
        ax5.set_title('Entropy Difference')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Phase diagram
        ax6.scatter(df['competitive_entropy'], df['quantum_entropy'],
                   c=df['lambda'], cmap='viridis', s=60, alpha=0.7)
        ax6.plot([0, max(df['competitive_entropy'])], [0, max(df['competitive_entropy'])],
                'r--', alpha=0.7, label='Equal entropy line')
        ax6.set_xlabel('Competitive Entropy')
        ax6.set_ylabel('Quantum Entropy')
        ax6.set_title('Quantum-Classical Phase Diagram')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.colorbar(ax6.collections[0], ax=ax6, label='Lambda (λ)')
        plt.tight_layout()
        plt.savefig('results/quantum_classical_transition_comprehensive.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("   📊 Comprehensive transition plots generated")

    @staticmethod
    def _analyze_transition_statistics(df):
        """Perform statistical analysis of quantum-classical transition."""
        # Find transition points
        entropy_crossover = None
        accuracy_crossover = None

        for i in range(len(df) - 1):
            # Entropy crossover (where quantum becomes less than competitive)
            if (df.iloc[i]['entropy_ratio'] > 1.0 and df.iloc[i+1]['entropy_ratio'] <= 1.0 and
                entropy_crossover is None):
                entropy_crossover = df.iloc[i]['lambda']

            # Accuracy crossover
            if (df.iloc[i]['accuracy_ratio'] < 1.0 and df.iloc[i+1]['accuracy_ratio'] >= 1.0 and
                accuracy_crossover is None):
                accuracy_crossover = df.iloc[i]['lambda']

        # Statistical tests
        from scipy import stats

        # Test for significant difference in entropy
        entropy_ttest = stats.ttest_rel(df['quantum_entropy'], df['competitive_entropy'])

        # Test for significant difference in accuracy
        accuracy_ttest = stats.ttest_rel(df['quantum_accuracy'], df['competitive_accuracy'])

        # Correlation analysis
        entropy_corr = stats.pearsonr(df['lambda'], df['entropy_difference'])
        accuracy_corr = stats.pearsonr(df['lambda'], df['accuracy_difference'])

        # Save statistical summary
        stats_summary = {
            'entropy_crossover_lambda': entropy_crossover,
            'accuracy_crossover_lambda': accuracy_crossover,
            'entropy_ttest_statistic': entropy_ttest.statistic,
            'entropy_ttest_pvalue': entropy_ttest.pvalue,
            'accuracy_ttest_statistic': accuracy_ttest.statistic,
            'accuracy_ttest_pvalue': accuracy_ttest.pvalue,
            'entropy_lambda_correlation': entropy_corr[0],
            'entropy_lambda_correlation_pvalue': entropy_corr[1],
            'accuracy_lambda_correlation': accuracy_corr[0],
            'accuracy_lambda_correlation_pvalue': accuracy_corr[1]
        }

        with open('results/quantum_classical_transition_statistics.txt', 'w') as f:
            for key, value in stats_summary.items():
                f.write(f"{key}: {value}\n")

        print("   📊 Transition statistical analysis completed")

        return stats_summary

    @staticmethod
    def analyze_horizon_effect(fen, fen_name, shallow_depth=5, deep_depth=100, pi_mode='competitive', pi_lambda=None, sample_count=None):
        """
        "Ufuk etkisi" analizi: shallow ve deep Lc0 analizlerinden elde edilen top hamleleri karşılaştırır.
        Güncelleme: Path-Integral örneklemesi için `pi_mode` parametresi eklendi. Geçerli seçenekler:
          - 'competitive' (varsayılan) -> mevcut path-integral MCTS davranışı
          - 'quantum_limit' -> policy-head sampling / quantum tarzı coverage
        Ayrıca `pi_lambda` ve `sample_count` ile lambda ve örnek sayısı üzerine override imkânı verir.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        print(f"\n--- Horizon Effect Analysis Begins: {fen_name} (pi_mode={pi_mode}) ---")

        # Shallow ve deep en iyi hamleleri doğrudan Lc0'dan al
        shallow_moves, _, _ = Engine.lc0_top_moves_and_scores(fen, depth=shallow_depth, multipv=1)
        deep_moves, _, _ = Engine.lc0_top_moves_and_scores(fen, depth=deep_depth, multipv=1)
        shallow_gt = str(shallow_moves[0]) if shallow_moves else None
        true_gt = str(deep_moves[0]) if deep_moves else None

        if not shallow_gt or not true_gt:
            print("WARNING: Could not determine shallow/deep top moves for this position.")
            return

        if shallow_gt == true_gt:
            print(f"WARNING: Shallow and deep analysis found the same move in this position ({true_gt}). No horizon effect was observed.")
            return

        print(f"Shallow Analysis (depth={shallow_depth}) Best Move (Trap): {shallow_gt}")
        print(f"Deep Analysis (depth={deep_depth}) Best Move (True): {true_gt}")

        # Override defaults if provided
        if sample_count is None:
            sample_count = config.SAMPLE_COUNT
        if pi_lambda is None:
            pi_lambda = config.LAMBDA

        # sample_paths çağrısında mode parametresine göre quantum veya competitive sampling yap
        print(f"[Horizon] Sampling PI paths: mode={pi_mode}, lambda={pi_lambda}, samples={sample_count}")
        try:
            pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, pi_lambda, sample_count, mode=pi_mode)
        except TypeError:
            # Eğer eski Engine.sample_paths sürümü mode argümanını desteklemiyorsa, fallback
            print("[WARN] Engine.sample_paths does not accept 'mode' parameter; calling without it as fallback.")
            pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, pi_lambda, sample_count)
        pi_counter = Calc.first_move_counter(pi_paths)
        pi_top_move = Calc.most_frequent_first_move(pi_paths)

        found_true_move = true_gt in pi_counter

        print(f"Path Integral Model (λ=0.2):")
        print(f"  - Most Frequent Move: {pi_top_move}")
        print(f"  - Discovered the 'True' Move ({true_gt})? {'YES' if found_true_move else 'NO'}")
        if found_true_move:
            print(f"  - Actual Move Frequency: {pi_counter[true_gt]}/{config.SAMPLE_COUNT}")  # Direct access

        # Görselleştirme: Plots yardımcı fonksiyonunu kullan (outfile adı mode içerir)
        outfile_prefix = f"horizon_effect_{pi_mode}"
        Plots.plot_horizon_effect_comparison(
            pi_counter,
            shallow_gt,
            true_gt,
            fen_name,
            outfile_prefix=outfile_prefix
        )
        print(f"Horizon effect comparison graph saved as 'results/{outfile_prefix}_{fen_name}.png'.")

    @staticmethod
    def three_way_engine_comparison(fen, sample_count=config.SAMPLE_COUNT, lambda_val=config.LAMBDA, save_results=True):
        """
        Path Integral (Lc0-based), standalone Lc0 ve Stockfish kıyaslaması (GT'siz, konsantrasyon tabanlı).
        Genişletme: Path Integral analizi artık iki modda yapılır: 'competitive' ve 'quantum_limit'.
        """
        import time
        print(f"\n--- Three-Way Engine Comparison: PI vs Lc0 vs Stockfish ---")

        results = {}
        # 1. Path Integral Analysis in two modes: 'competitive' (default) and 'quantum_limit'
        print("\n1. Path Integral Analysis (two modes)...")
        pi_modes = [('competitive', 'path_integral'), ('quantum_limit', 'path_integral_quantum')]
        for mode_name, result_key in pi_modes:
            start_time = time.perf_counter()
            try:
                print(f"   Sampling PI mode={mode_name}: samples={sample_count}, lambda={lambda_val}")
                pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lambda_val, sample_count, mode=mode_name)
                pi_counter = Calc.first_move_counter(pi_paths)
                pi_top_move = Calc.most_frequent_first_move(pi_paths)
                pi_entropy, _ = Calc.compute_entropy(pi_paths)
                pi_accuracy = (max(pi_counter.values()) / sum(pi_counter.values())) if pi_counter else 0.0
                pi_time = time.perf_counter() - start_time
                results[result_key] = {
                    'top_move': str(pi_top_move),
                    'accuracy': float(pi_accuracy),
                    'time': float(pi_time),
                    'entropy': float(pi_entropy),
                    'move_distribution': dict(pi_counter.most_common(5)),
                    'status': 'success',
                    'mode': mode_name
                }
                print(f"     Mode={mode_name} Top move: {pi_top_move}, Concentration: {pi_accuracy:.3f}, Time: {pi_time:.2f}s")
            except TypeError:
                # fallback if Engine.sample_paths doesn't accept mode
                try:
                    print(f"   [WARN] Engine.sample_paths missing 'mode' arg; calling without it for {mode_name} (fallback).")
                    pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lambda_val, sample_count)
                    pi_counter = Calc.first_move_counter(pi_paths)
                    pi_top_move = Calc.most_frequent_first_move(pi_paths)
                    pi_entropy, _ = Calc.compute_entropy(pi_paths)
                    pi_accuracy = (max(pi_counter.values()) / sum(pi_counter.values())) if pi_counter else 0.0
                    pi_time = time.perf_counter() - start_time
                    results[result_key] = {
                        'top_move': str(pi_top_move),
                        'accuracy': float(pi_accuracy),
                        'time': float(pi_time),
                        'entropy': float(pi_entropy),
                        'move_distribution': dict(pi_counter.most_common(5)),
                        'status': 'success',
                        'mode': mode_name,
                        'warning': 'mode arg not supported by Engine.sample_paths; used fallback without mode'
                    }
                    print(f"     Fallback Mode={mode_name} Top move: {pi_top_move}, Concentration: {pi_accuracy:.3f}, Time: {pi_time:.2f}s")
                except Exception as e:
                    print(f"   Error in Path Integral fallback for mode {mode_name}: {e}")
                    results[result_key] = {'top_move': 'error', 'accuracy': 0.0, 'time': 0.0, 'entropy': 0.0, 'status': 'error', 'error': str(e), 'mode': mode_name}
            except Exception as e:
                print(f"   Error in Path Integral analysis (mode={mode_name}): {e}")
                results[result_key] = {'top_move': 'error', 'accuracy': 0.0, 'time': 0.0, 'entropy': 0.0, 'status': 'error', 'error': str(e), 'mode': mode_name}

        # Backwards compatibility: keep 'path_integral' top-level key as the competitive mode results
        if 'path_integral' in results and 'path_integral' != results.get('path_integral'):
            pass

        # 2. Standalone Lc0 Analysis
        print("\n2. Standalone Lc0 Analysis...")
        # Lc0 exe yolunu kontrol et
        if not os.path.exists(config.LC0_PATH):
            print(f"   Standalone Lc0 error: LC0_PATH bulunamadı veya erişilemiyor: {config.LC0_PATH}")
            results['lc0_standalone'] = {
                'top_move': 'error',
                'accuracy': 0.0,
                'time': 0.0,
                'status': 'error',
                'error': f'LC0_PATH not found: {config.LC0_PATH}. Lc0 exe yolunu ve CUDA/GPU uyumluluğunu kontrol edin.'
            }
        else:
            start_time = time.perf_counter()
            try:
                lc0_moves, lc0_scores, lc0_analysis_time = Engine.lc0_top_moves_and_scores(
                    fen, depth=config.TARGET_DEPTH, multipv=config.MULTIPV
                )
                lc0_time = time.perf_counter() - start_time
                if lc0_moves and len(lc0_moves) > 0:
                    lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LC0_SOFTMAX_LAMBDA)
                    lc0_top_move = str(lc0_moves[0])
                    lc0_probs_arr = np.asarray(lc0_probs)
                    lc0_accuracy = float(np.max(lc0_probs_arr)) if lc0_probs_arr.size > 0 else 0.0
                    # Tüm hamlelerin olasılıklarını dict olarak ekle
                    lc0_move_distribution = {str(move): float(prob) for move, prob in zip(lc0_moves, lc0_probs)}
                    results['lc0_standalone'] = {
                        'top_move': lc0_top_move,
                        'accuracy': lc0_accuracy,
                        'time': lc0_time,
                        'engine_time': lc0_analysis_time,
                        'score': lc0_scores[0] if lc0_scores else None,
                        'move_distribution': lc0_move_distribution,
                        'status': 'success'
                    }
                    print(f"   Top move: {lc0_top_move}, Accuracy: {lc0_accuracy:.3f}, Time: {lc0_time:.2f}s")
                else:
                    raise Exception("No moves returned from Lc0")
            except Exception as e:
                print(f"   Error in Lc0 analysis: {e}")
                results['lc0_standalone'] = {
                    'top_move': 'error',
                    'accuracy': 0.0,
                    'time': 0.0,
                    'status': 'error',
                    'error': str(e)
                }

        # 3. Stockfish Analysis
        print("\n3. Stockfish Analysis...")
        try:
            stockfish_result = Engine.get_stockfish_analysis(fen, config.STOCKFISH_DEPTH, 1)
            stockfish_scores = stockfish_result.get('scores', [])
            if stockfish_scores:
                sf_probs = Calc.normalize_scores_to_probs(stockfish_scores, 1.0)
                sf_accuracy = float(np.max(np.asarray(sf_probs)))
                sf_top_move = stockfish_result.get('best_move')
            else:
                sf_accuracy = 0.0
                sf_top_move = stockfish_result.get('best_move')
            results['stockfish'] = {
                'top_move': sf_top_move,
                'accuracy': sf_accuracy,
                'time': stockfish_result.get('elapsed_time', 0.0),
                'status': 'success'
            }
        except Exception as e:
            print(f"   Error in Stockfish analysis: {e}")
            results['stockfish'] = {
                'top_move': 'error',
                'accuracy': 0.0,
                'time': 0.0,
                'status': 'error',
                'error': str(e)
            }

        # Advanced analysis: Engine decision consistency and move overlap
        if save_results:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import re, hashlib
                os.makedirs('results', exist_ok=True)

                # 1. Move Overlap Analysis - Venn diagram style analysis
                Analyzer._generate_move_overlap_analysis(results, fen)
                
                # 2. Decision Confidence Analysis
                Analyzer._generate_decision_confidence_analysis(results, fen)
                
                # 3. Time-Performance Trade-off Analysis
                Analyzer._generate_time_performance_analysis(results, fen)
                
                print("✓ Advanced engine comparison analyses completed")
            except Exception as e:
                print(f"Warning: could not save advanced comparison analyses: {e}")

        return results

    @staticmethod
    def _generate_move_overlap_analysis(results, fen):
        """Analyze move overlap and agreement between different engines/modes."""
        import matplotlib.pyplot as plt
        import re, hashlib
        
        # Extract top moves from each engine
        moves_data = {}
        for engine_key, engine_result in results.items():
            if isinstance(engine_result, dict) and engine_result.get('status') == 'success':
                top_move = engine_result.get('top_move')
                move_dist = engine_result.get('move_distribution', {})
                if top_move and top_move != 'error':
                    moves_data[engine_key] = {
                        'top_move': top_move,
                        'distribution': move_dist
                    }
        
        if len(moves_data) < 2:
            return
        
        # Create move overlap matrix
        engine_names = list(moves_data.keys())
        n_engines = len(engine_names)
        overlap_matrix = np.zeros((n_engines, n_engines))
        
        for i, engine1 in enumerate(engine_names):
            for j, engine2 in enumerate(engine_names):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    # Calculate weighted overlap using probability distributions
                    # NOTE: a set-based Jaccard on the keys becomes binary (0/1)
                    # when distributions are one-hot (only the top move present). To
                    # provide a smoother continuous measure use weighted Jaccard =
                    # sum_min(p_i, q_i) over the union of moves (both normalized).
                    dist1 = moves_data[engine1]['distribution'] or {}
                    dist2 = moves_data[engine2]['distribution'] or {}
                    moves_union = set(dist1.keys()).union(set(dist2.keys()))
                    if moves_union:
                        # Build aligned probability vectors and normalize to sum=1
                        p = np.array([float(dist1.get(m, 0.0)) for m in moves_union], dtype=float)
                        q = np.array([float(dist2.get(m, 0.0)) for m in moves_union], dtype=float)
                        # small-safety normalization
                        if p.sum() > 0:
                            p = p / p.sum()
                        if q.sum() > 0:
                            q = q / q.sum()
                        # weighted Jaccard (range 0..1)
                        overlap_matrix[i, j] = float(np.minimum(p, q).sum())
                    else:
                        overlap_matrix[i, j] = 0.0

        # Visualize overlap matrix
        plt.figure(figsize=(10, 8))
        im = plt.imshow(overlap_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        plt.colorbar(im, label='Move Overlap (Jaccard Similarity)')
        
        # Add text annotations
        for i in range(n_engines):
            for j in range(n_engines):
                text = plt.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.xticks(range(n_engines), [name.replace('_', ' ').title() for name in engine_names], rotation=45)
        plt.yticks(range(n_engines), [name.replace('_', ' ').title() for name in engine_names])
        plt.title('Engine Move Overlap Analysis\n(Higher values = more similar move preferences)')
        plt.tight_layout()
        
        # Save with unique filename
        import hashlib, re
        fen_slug = re.sub(r'[^A-Za-z0-9]', '_', fen)
        fen_short = fen_slug[:20]
        # FEN kodundan pozisyon ismini al
        position_name = FEN_NAME_MAP.get(fen, fen_short)
        overlap_path = f'results/engine_move_overlap_{position_name}.png'
        plt.savefig(overlap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   📊 Move overlap analysis saved: {overlap_path}")

    @staticmethod
    def _generate_decision_confidence_analysis(results, fen):
        """Analyze decision confidence and uncertainty across engines."""
        import matplotlib.pyplot as plt
        import re, hashlib
        
        # Extract confidence metrics
        confidence_data = {}
        for engine_key, engine_result in results.items():
            if isinstance(engine_result, dict) and engine_result.get('status') == 'success':
                accuracy = engine_result.get('accuracy', 0)
                entropy = engine_result.get('entropy', 0)
                time_taken = engine_result.get('time', 0)
                
                # Calculate confidence score (high accuracy, low entropy = high confidence)
                if entropy > 0:
                    confidence_score = accuracy / (1 + entropy)  # Normalized confidence
                else:
                    confidence_score = accuracy
                
                confidence_data[engine_key] = {
                    'accuracy': accuracy,
                    'entropy': entropy,
                    'confidence': confidence_score,
                    'time': time_taken,
                    'uncertainty': entropy  # Higher entropy = higher uncertainty
                }
        
        if len(confidence_data) < 2:
            return
        
        # Create confidence comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        engines = list(confidence_data.keys())
        engine_labels = [name.replace('_', ' ').title() for name in engines]
        
        # 1. Confidence vs Uncertainty scatter
        confidences = [confidence_data[e]['confidence'] for e in engines]
        uncertainties = [confidence_data[e]['uncertainty'] for e in engines]
        colors = ['deepskyblue', 'orange', 'green', 'red'][:len(engines)]
        
        ax1.scatter(uncertainties, confidences, c=colors, s=100, alpha=0.8, edgecolors='black')
        for i, engine in enumerate(engine_labels):
            ax1.annotate(engine, (uncertainties[i], confidences[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('Uncertainty (Entropy)')
        ax1.set_ylabel('Decision Confidence')
        ax1.set_title('Confidence vs Uncertainty Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy comparison
        accuracies = [confidence_data[e]['accuracy'] for e in engines]
        bars1 = ax2.bar(engine_labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Accuracy (Concentration)')
        ax2.set_title('Decision Accuracy Comparison')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Time efficiency
        times = [confidence_data[e]['time'] for e in engines]
        bars2 = ax3.bar(engine_labels, times, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Analysis Time (seconds)')
        ax3.set_title('Computational Efficiency')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, time_val in zip(bars2, times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time_val:.2f}s', ha='center', va='bottom', fontweight='bold')
        
        # 4. Efficiency vs Quality scatter (time vs confidence)
        ax4.scatter(times, confidences, c=colors, s=100, alpha=0.8, edgecolors='black')
        for i, engine in enumerate(engine_labels):
            ax4.annotate(engine, (times[i], confidences[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax4.set_xlabel('Analysis Time (seconds)')
        ax4.set_ylabel('Decision Confidence')
        ax4.set_title('Efficiency vs Quality Trade-off')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save with unique filename
        fen_slug = re.sub(r'[^A-Za-z0-9]', '_', fen)
        fen_short = fen_slug[:20]
        position_name = FEN_NAME_MAP.get(fen, fen_short)
        confidence_path = f'results/engine_decision_confidence_{position_name}.png'
        plt.savefig(confidence_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 Decision confidence analysis saved: {confidence_path}")

    @staticmethod
    def _generate_time_performance_analysis(results, fen):
        """Analyze time vs performance trade-offs with detailed metrics."""
        import matplotlib.pyplot as plt
        import re, hashlib

        # Extract performance metrics
        perf_data = {}
        for engine_key, engine_result in results.items():
            if isinstance(engine_result, dict) and engine_result.get('status') == 'success':
                accuracy = engine_result.get('accuracy', 0)
                entropy = engine_result.get('entropy', 0)
                time_taken = engine_result.get('time', 0)

                # Calculate performance metrics
                efficiency = accuracy / max(time_taken, 0.001)  # Accuracy per second
                quality_time_ratio = (accuracy * (1 - entropy/10)) / max(time_taken, 0.001)  # Quality-adjusted efficiency

                perf_data[engine_key] = {
                    'accuracy': accuracy,
                    'entropy': entropy,
                    'time': time_taken,
                    'efficiency': efficiency,
                    'quality_time_ratio': quality_time_ratio
                }

        if len(perf_data) < 2:
            return

        # Create performance analysis plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        engines = list(perf_data.keys())
        engine_labels = [name.replace('_', ' ').title() for name in engines]
        colors = ['deepskyblue', 'orange', 'green', 'red'][:len(engines)]

        # 1. Time vs Accuracy scatter with size = entropy
        times = [perf_data[e]['time'] for e in engines]
        accuracies = [perf_data[e]['accuracy'] for e in engines]
        entropies = [perf_data[e]['entropy'] for e in engines]

        # Normalize entropy for bubble size
        max_entropy = max(entropies) if entropies else 1
        bubble_sizes = [100 + (e/max_entropy)*300 for e in entropies]

        scatter = ax1.scatter(times, accuracies, s=bubble_sizes, c=colors, alpha=0.6, edgecolors='black')
        for i, engine in enumerate(engine_labels):
            ax1.annotate(engine, (times[i], accuracies[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        ax1.set_xlabel('Analysis Time (seconds)')
        ax1.set_ylabel('Accuracy (Concentration)')
        ax1.set_title('Time vs Accuracy\n(Bubble size = Entropy)')
        ax1.grid(True, alpha=0.3)

        # 2. Efficiency comparison (accuracy/time)
        efficiencies = [perf_data[e]['efficiency'] for e in engines]
        bars1 = ax2.bar(engine_labels, efficiencies, color=colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Efficiency (Accuracy/Time)')
        ax2.set_title('Computational Efficiency Comparison')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, eff in zip(bars1, efficiencies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiencies)*0.01,
                    f'{eff:.2f}', ha='center', va='bottom', fontweight='bold')

        # 3. Quality-adjusted efficiency
        quality_ratios = [perf_data[e]['quality_time_ratio'] for e in engines]
        bars2 = ax3.bar(engine_labels, quality_ratios, color=colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Quality-Adjusted Efficiency')
        ax3.set_title('Quality-Time Trade-off\n(Higher = Better quality per time)')
        ax3.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, qr in zip(bars2, quality_ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(quality_ratios)*0.01,
                    f'{qr:.2f}', ha='center', va='bottom', fontweight='bold')

        # 4. Performance radar chart
        ax4 = plt.subplot(2, 2, 4, projection='polar')

        # Normalize metrics for radar chart
        metrics = ['accuracy', 'efficiency', 'quality_time_ratio']
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for i, engine in enumerate(engines):
            values = []
            for metric in metrics:
                val = perf_data[engine][metric]
                # Normalize to 0-1 scale
                all_vals = [perf_data[e][metric] for e in engines]
                max_val = max(all_vals) if all_vals else 1
                normalized_val = val / max_val if max_val > 0 else 0
                values.append(normalized_val)

            values += values[:1]  # Complete the circle

            ax4.plot(angles, values, 'o-', linewidth=2, label=engine_labels[i], color=colors[i])
            ax4.fill(angles, values, alpha=0.25, color=colors[i])

        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(['Accuracy', 'Efficiency', 'Quality/Time'])
        ax4.set_ylim(0, 1)
        ax4.set_title('Performance Radar Chart\n(Normalized Metrics)')
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

        plt.tight_layout()

        # Save with unique filename
        fen_slug = re.sub(r'[^A-Za-z0-9]', '_', fen)
        fen_short = fen_slug[:20]
        position_name = FEN_NAME_MAP.get(fen, fen_short)
        perf_path = f'results/engine_time_performance_{position_name}.png'
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 Time-performance analysis saved: {perf_path}")
    @staticmethod
    def chess960_position_study(chess960_fens, position_names=None):
        """
        Comprehensive Chess960 position analysis using path integral framework.
        
        Args:
            chess960_fens: List of Chess960 starting positions
            position_names: Optional names for positions
            
        Returns:
            Analysis results for Chess960 positions
        """
        print("🏰 Chess960 Position Study")
        
        if position_names is None:
            position_names = [f"Chess960_{i}" for i in range(len(chess960_fens))]
        
        # Analyze each Chess960 position
        results = Calc.batch_path_analysis(
            fen_list=chess960_fens,
            lambda_values=[0.1, 0.5, 1.0, 2.0],
            samples=config.SAMPLE_COUNT,
            use_cache=True
        )
        
        # Add position names
        results['position_name'] = results['position_index'].map(
            lambda idx: position_names[idx] if idx < len(position_names) else f"Position_{idx}"
        )
        
        # Save results
        results.to_csv("results/chess960_analysis.csv", index=False)
        
        return {
            'results_df': results,
            'position_count': len(chess960_fens),
            'analysis_summary': {
                'avg_entropy': results['entropy'].mean(),
                'entropy_std': results['entropy'].std(),
                'most_complex_position': results.loc[results['entropy'].idxmax(), 'position_name'],
                'least_complex_position': results.loc[results['entropy'].idxmin(), 'position_name']
            }
        }
    
    @staticmethod
    def convergence_study(test_fen, depth_range=None, sample_range=None):
        """
        Study convergence properties of path integral framework.
        
        Args:
            test_fen: Test position FEN
            depth_range: Range of depths to test
            sample_range: Range of sample sizes to test
            
        Returns:
            Convergence study results
        """
        print("📈 Path Integral Convergence Study")
        
        if depth_range is None:
            depth_range = [2, 4, 6, 8, 10]
        if sample_range is None:
            sample_range = [25, 50, 100, 200]
        
        convergence_results = []
        
        # Test convergence with different parameters
        for depth in tqdm(depth_range, desc="Depth convergence"):
            for samples in sample_range:
                try:
                    result = Calc.lambda_sensitivity_analysis(
                        fen=test_fen,
                        lambda_range=[0.1, 1.0, 10.0],
                        samples=samples,
                        save_results=False
                    )
                    
                    if result['results_df'] is not None and not result['results_df'].empty:
                        convergence_results.append({
                            'depth': depth,
                            'samples': samples,
                            'avg_entropy': result['results_df']['entropy'].mean(),
                            'entropy_std': result['results_df']['entropy'].std(),
                            'avg_concentration': result['results_df']['concentration'].mean()
                        })
                        
                except Exception as e:
                    print(f"Error in convergence test (depth={depth}, samples={samples}): {e}")
        
        conv_df = pd.DataFrame(convergence_results)
        conv_df.to_csv("results/convergence_study.csv", index=False)
        
        return ({
            'convergence_data': conv_df,
            'depth_range': depth_range,
            'sample_range': sample_range
        })

    @staticmethod
    def strategic_depth_analysis(fen, depth_range=None, save_results=True):
        """
        Analyze how strategic understanding changes with search depth.
        This provides insights into the 'strategic depth' of different engines.
        """
        if depth_range is None:
            depth_range = [4, 8, 12, 16, 20, 24]
        
        print(f"\n=== Strategic Depth Analysis ===")
        print(f"Depth range: {depth_range}")
        
        results = []
        
        for depth in tqdm(depth_range, desc="Strategic depth analysis"):
            # PI Competitive analysis at different depths
            pi_paths = Engine.sample_paths(fen, depth, config.LAMBDA, config.SAMPLE_COUNT, mode='competitive')
            pi_entropy, pi_counter = Calc.compute_entropy(pi_paths)
            pi_accuracy = Calc.top_move_concentration(pi_paths)
            pi_top_move = Calc.most_frequent_first_move(pi_paths)
            
            # LC0 analysis at different depths
            try:
                lc0_moves, lc0_scores, lc0_time = Engine.lc0_top_moves_and_scores(
                    fen, depth=depth, multipv=config.MULTIPV
                )
                lc0_top_move = str(lc0_moves[0]) if lc0_moves else None
                lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LAMBDA)
                lc0_entropy = -sum([p * np.log2(p) for p in lc0_probs if p > 0]) if lc0_probs else 0.0
                lc0_accuracy = max(lc0_probs) if lc0_probs else 0.0
            except:
                lc0_top_move = None
                lc0_entropy = 0.0
                lc0_accuracy = 0.0
            
            # Calculate move stability (how often the top move changes)
            move_agreement = 1.0 if pi_top_move == lc0_top_move else 0.0
            
            results.append({
                'depth': depth,
                'pi_entropy': pi_entropy,
                'pi_accuracy': pi_accuracy,
                'pi_top_move': pi_top_move,
                'lc0_entropy': lc0_entropy,
                'lc0_accuracy': lc0_accuracy,
                'lc0_top_move': lc0_top_move,
                'move_agreement': move_agreement,
                'entropy_difference': abs(pi_entropy - lc0_entropy),
                'accuracy_difference': abs(pi_accuracy - lc0_accuracy)
            })
        
        df = pd.DataFrame(results)
        
        if save_results:
            df.to_csv('results/strategic_depth_analysis.csv', index=False)
            
            # Generate strategic depth plots
            Analyzer._generate_strategic_depth_plots(df, fen)
        
        return df

    @staticmethod
    def _generate_strategic_depth_plots(df, fen):
        """Generate strategic depth analysis visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        depths = df['depth'].tolist()
        
        # 1. Entropy convergence with depth
        ax1.plot(depths, df['pi_entropy'], 'o-', label='PI Competitive', linewidth=2, markersize=8)
        ax1.plot(depths, df['lc0_entropy'], 's-', label='LC0', linewidth=2, markersize=8)
        ax1.set_xlabel('Search Depth')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Strategic Entropy vs Search Depth')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy convergence with depth
        ax2.plot(depths, df['pi_accuracy'], 'o-', label='PI Competitive', linewidth=2, markersize=8)
        ax2.plot(depths, df['lc0_accuracy'], 's-', label='LC0', linewidth=2, markersize=8)
        ax2.set_xlabel('Search Depth')
        ax2.set_ylabel('Accuracy (Concentration)')
        ax2.set_title('Decision Confidence vs Search Depth')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Move agreement over depth
        ax3.plot(depths, df['move_agreement'], 'o-', color='purple', linewidth=2, markersize=8)
        ax3.set_xlabel('Search Depth')
        ax3.set_ylabel('Move Agreement (PI vs LC0)')
        ax3.set_title('Strategic Agreement vs Search Depth')
        ax3.set_ylim(-0.1, 1.1)
        ax3.grid(True, alpha=0.3)

        # 4. Entropy difference (strategic divergence)
        ax4.plot(depths, df['entropy_difference'], 'o-', color='red', linewidth=2, markersize=8)
        ax4.set_xlabel('Search Depth')
        ax4.set_ylabel('Entropy Difference |PI - LC0|')
        ax4.set_title('Strategic Divergence vs Search Depth')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save with unique filename
        import re, hashlib
        fen_slug = re.sub(r'[^A-Za-z0-9]', '_', fen)
        fen_short = fen_slug[:20]
        fen_hash = hashlib.md5(fen.encode('utf-8')).hexdigest()[:8]
        depth_path = f'results/strategic_depth_analysis_{fen_short}_{fen_hash}.png'
        plt.savefig(depth_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"   📊 Strategic depth analysis saved: {depth_path}")

    @staticmethod
    def move_quality_assessment(fen, candidate_moves=None, save_results=True):
        """
        Assess the quality of different candidate moves using multiple evaluation methods.
        This provides insights into move evaluation consistency across different approaches.
        """
        print(f"\n=== Move Quality Assessment ===")
        
        # Get candidate moves from LC0 if not provided
        if candidate_moves is None:
            try:
                lc0_moves, lc0_scores, _ = Engine.lc0_top_moves_and_scores(
                    fen, depth=config.TARGET_DEPTH, multipv=min(8, config.MULTIPV)
                )
                candidate_moves = [str(move) for move in lc0_moves[:6]]  # Top 6 moves
            except:
                # Fallback to legal moves
                board = chess.Board(fen)
                legal_moves = list(board.legal_moves)
                candidate_moves = [str(move) for move in legal_moves[:6]]
        
        if not candidate_moves:
            print("No candidate moves available for assessment")
            return None
        
        print(f"Assessing {len(candidate_moves)} candidate moves: {candidate_moves}")
        
        assessment_results = []
        
        for move in tqdm(candidate_moves, desc="Assessing moves"):
            # Create position after the move
            board = chess.Board(fen)
            try:
                chess_move = chess.Move.from_uci(move)
                if chess_move not in board.legal_moves:
                    continue
                board.push(chess_move)
                new_fen = board.fen()
            except:
                continue
            
            # Evaluate position after move with different methods
            
            # 1. PI Competitive evaluation
            try:
                pi_paths = Engine.sample_paths(new_fen, config.TARGET_DEPTH, config.LAMBDA, 
                                             config.SAMPLE_COUNT//2, mode='competitive')
                pi_entropy, _ = Calc.compute_entropy(pi_paths)
                pi_accuracy = Calc.top_move_concentration(pi_paths)
            except:
                pi_entropy, pi_accuracy = 0.0, 0.0
            
            # 2. PI Quantum evaluation
            try:
                pi_quantum_paths = Engine.sample_paths(new_fen, config.HIGH_DEPTH, config.LAMBDA, 
                                                     config.SAMPLE_COUNT//2, mode='quantum_limit')
                pi_quantum_entropy, _ = Calc.compute_entropy(pi_quantum_paths)
                pi_quantum_accuracy = Calc.top_move_concentration(pi_quantum_paths)
            except:
                pi_quantum_entropy, pi_quantum_accuracy = 0.0, 0.0
            
            # 3. LC0 evaluation
            try:
                lc0_moves_after, lc0_scores_after, _ = Engine.lc0_top_moves_and_scores(
                    new_fen, depth=config.TARGET_DEPTH, multipv=1
                )
                lc0_eval = lc0_scores_after[0] if lc0_scores_after else 0
            except:
                lc0_eval = 0
            
            # 4. Stockfish evaluation
            try:
                sf_result = Engine.get_stockfish_analysis(new_fen, config.STOCKFISH_DEPTH, 1)
                sf_eval = sf_result.get('scores', [0])[0]
            except:
                sf_eval = 0
            
            # Calculate position complexity after move
            legal_moves_after = len(list(board.legal_moves))
            pieces_after = len(board.piece_map())
            complexity_after = legal_moves_after * pieces_after
            
            # Calculate move quality metrics
            # Higher entropy = more complex/interesting position
            # Lower entropy = more forcing/simplified position
            strategic_complexity = (pi_entropy + pi_quantum_entropy) / 2
            tactical_sharpness = abs(sf_eval) / 100.0  # Normalize centipawn evaluation
            
            assessment_results.append({
                'move': move,
                'pi_entropy': pi_entropy,
                'pi_accuracy': pi_accuracy,
                'pi_quantum_entropy': pi_quantum_entropy,
                'pi_quantum_accuracy': pi_quantum_accuracy,
                'lc0_evaluation': lc0_eval,
                'stockfish_evaluation': sf_eval,
                'complexity_after': complexity_after,
                'legal_moves_after': legal_moves_after,
                'strategic_complexity': strategic_complexity,
                'tactical_sharpness': tactical_sharpness,
                'position_fen_after': new_fen
            })
        
        df = pd.DataFrame(assessment_results)
        
        if save_results and not df.empty:
            df.to_csv('results/move_quality_assessment.csv', index=False)
            
            # Generate move quality plots
            Analyzer._generate_move_quality_plots(df, fen)
        
        return df

    @staticmethod
    def _generate_move_quality_plots(df, fen):
        """Generate move quality assessment visualizations."""
        if df.empty:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        moves = df['move'].tolist()
        colors = plt.cm.Set3(np.linspace(0, 1, len(moves)))
        
        # 1. Strategic complexity vs tactical sharpness
        ax1.scatter(df['strategic_complexity'], df['tactical_sharpness'], 
                   c=colors, s=100, alpha=0.8, edgecolors='black')
        for i, move in enumerate(moves):
            ax1.annotate(move, (df.iloc[i]['strategic_complexity'], df.iloc[i]['tactical_sharpness']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        ax1.set_xlabel('Strategic Complexity (Average Entropy)')
        ax1.set_ylabel('Tactical Sharpness (|Stockfish Eval|/100)')
        ax1.set_title('Move Character Analysis')
        ax1.grid(True, alpha=0.3)
        
        # 2. Engine evaluation comparison
        x = np.arange(len(moves))
        width = 0.35
        
        # Normalize evaluations for comparison
        lc0_evals_norm = (df['lc0_evaluation'] - df['lc0_evaluation'].min()) / (df['lc0_evaluation'].max() - df['lc0_evaluation'].min() + 1e-6)
        sf_evals_norm = (df['stockfish_evaluation'] - df['stockfish_evaluation'].min()) / (df['stockfish_evaluation'].max() - df['stockfish_evaluation'].min() + 1e-6)
        
        ax2.bar(x - width/2, lc0_evals_norm, width, label='LC0 (normalized)', alpha=0.8)
        ax2.bar(x + width/2, sf_evals_norm, width, label='Stockfish (normalized)', alpha=0.8)
        ax2.set_xlabel('Candidate Moves')
        ax2.set_ylabel('Normalized Evaluation')
        ax2.set_title('Engine Evaluation Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels(moves, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. PI mode comparison (entropy)
        ax3.bar(x - width/2, df['pi_entropy'], width, label='PI Competitive', alpha=0.8)
        ax3.bar(x + width/2, df['pi_quantum_entropy'], width, label='PI Quantum', alpha=0.8)
        ax3.set_xlabel('Candidate Moves')
        ax3.set_ylabel('Entropy (bits)')
        ax3.set_title('PI Mode Entropy Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(moves, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Position complexity after moves
        bars = ax4.bar(moves, df['complexity_after'], color=colors, alpha=0.8, edgecolor='black')
        ax4.set_xlabel('Candidate Moves')
        ax4.set_ylabel('Position Complexity (Legal Moves × Pieces)')
        ax4.set_title('Resulting Position Complexity')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, complexity in zip(bars, df['complexity_after']):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['complexity_after'])*0.01,
                    f'{complexity:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Save with unique filename
        import re, hashlib
        fen_slug = re.sub(r'[^A-Za-z0-9]', '_', fen)
        fen_short = fen_slug[:20]
        fen_hash = hashlib.md5(fen.encode('utf-8')).hexdigest()[:8]
        quality_path = f'results/move_quality_assessment_{fen_short}_{fen_hash}.png'
        plt.savefig(quality_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Move quality assessment saved: {quality_path}")
