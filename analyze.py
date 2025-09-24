import os
import time

import chess
import chess.engine
import chess.pgn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter
from tqdm import tqdm
from collections import Counter

from cache import Cache
import config
from engine import Engine
from mathfuncs import Calc
from plots import Plots


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
    def analyze_convergence_by_lambda(fen, nodes=config.LC0_NODES, lambda_range=config.LAMBDA_SCAN):  # Direct access
        """
        λ parametresine göre yakınsama: optimal hamle olasılığı yerine konsantrasyon raporlanır.
        X ekseni eşit aralıklı indekslerle gösterilir, etiketler gerçek lambda değerleriyle yazılır.
        """
        print(f"\n--- Convergence Analysis Based on Lambda (per_ply_nodes={nodes}) ---")
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

            plt.title(f'Convergence Analysis with Lambda (nodes={nodes})')
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
            if save_results:
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

        print(f"\n--- Position Complexity Impact Analysis ({len(fen_list)} positions, nodes={nodes}) ---")

        cache_key = f"PI_position_complexity_{hash(str(fen_list))}_{lambda_val}_{nodes}_{sample_count}"

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
                                  s=120, alpha=0.8, c=df['entropy'], cmap='viridis', edgecolor='black')
            ax1.set_xlabel('Position Complexity Score (Legal Moves × Piece Count)')
            ax1.set_ylabel('Sampling Time (sec)')
            ax1.set_title('Complexity and Computational Cost')
            cbar = plt.colorbar(scatter, ax=ax1, label='Entropy (bits)')
            ax1.grid(True, alpha=0.3)
            # Explanation
            ax1.annotate('More complex positions usually take longer and may have higher entropy.',
                         xy=(0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=10, color='gray')
            # 2. Legal move count vs concentration (top-1/top-3)
            ax2.scatter(df['root_legal_moves'], df['conc_top1'], s=100, alpha=0.8, label='Top-1', marker='o', color='royalblue', edgecolor='black')
            ax2.scatter(df['root_legal_moves'], df['conc_top3'], s=100, alpha=0.8, label='Top-3', marker='^', color='orange', edgecolor='black')
            ax2.set_xlabel('Root Legal Move Count')
            ax2.set_ylabel('Concentration (Mode Probability)')
            ax2.set_title('Concentration vs Branching Factor')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.annotate('As the root legal move count increases, concentration usually decreases.',
                         xy=(0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=10, color='gray')
            # 3. Legal move count vs entropy
            ax3.scatter(df['root_legal_moves'], df['entropy'], s=100, alpha=0.8, color='seagreen', edgecolor='black')
            ax3.set_xlabel('Legal Move Count')
            ax3.set_ylabel('Entropy (bits)')
            ax3.set_title('Strategic Diversity vs Branching Factor')
            ax3.grid(True, alpha=0.3)
            ax3.annotate('More legal moves generally mean higher strategic diversity (entropy).',
                         xy=(0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=10, color='gray')
            # 4. Next move mobility vs entropy
            ax4.scatter(df['next_mobility_avg'], df['entropy'], s=100, alpha=0.8, color='purple', edgecolor='black')
            ax4.set_xlabel('Average Next Move Mobility')
            ax4.set_ylabel('Entropy (bits)')
            ax4.set_title('Mobility and Exploration Behavior')
            ax4.grid(True, alpha=0.3)
            ax4.annotate('Positions with high mobility usually offer more exploration (entropy).',
                         xy=(0.5, 1.05), xycoords='axes fraction', ha='center', fontsize=10, color='gray')
            plt.tight_layout(rect=(0, 0.03, 1, 0.97))
            output_file = "results/position_complexity_impact_overview.png"
            # Short summary text below the figure
            figtext = (
                "These plots show how the model's exploration (entropy) and concentration (mode probability) behavior changes as chess position complexity increases. "
                "Colors and markers highlight the relationship of entropy and concentration with position complexity. "
                "More complex positions generally offer greater strategic diversity and lower concentration."
            )
            plt.figtext(0.5, 0.01, figtext, wrap=True, ha='center', fontsize=12, color='dimgray')
        os.makedirs("results", exist_ok=True)
        plt.savefig(output_file, dpi=300)
        if not single_fen and output_file.endswith("_overview.png"):
            plt.savefig("results/position_complexity_analysis.png", dpi=300)
        plt.close()
        print(f"Complexity analysis visualization saved: {output_file}")

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

        print(f"Path Integral Model (λ=0.7):")
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
                    results['lc0_standalone'] = {
                        'top_move': lc0_top_move,
                        'accuracy': lc0_accuracy,
                        'time': lc0_time,
                        'engine_time': lc0_analysis_time,
                        'score': lc0_scores[0] if lc0_scores else None,
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
            stockfish_result = Engine.get_stockfish_analysis(fen, config.STOCKFISH_DEPTH, config.LC0_MULTIPV)
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

        # Optional: save comparison bar charts (entropy and accuracy) for the four entries
        if save_results:
            try:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                import re, hashlib
                os.makedirs('results', exist_ok=True)

                keys = ['path_integral', 'path_integral_quantum', 'lc0_standalone', 'stockfish']
                labels = []
                ent_vals = []
                acc_vals = []
                for k in keys:
                    v = results.get(k, {})
                    mode_label = v.get('mode', k) if isinstance(v, dict) else k
                    labels.append(mode_label)
                    ent_vals.append(float(v.get('entropy', np.nan)) if isinstance(v, dict) else np.nan)
                    acc_vals.append(float(v.get('accuracy', np.nan)) if isinstance(v, dict) else np.nan)

                x = np.arange(len(labels))
                width = 0.5

                plt.figure(figsize=(10, 4))
                plt.bar(x, ent_vals, width, color=['deepskyblue', 'orange', 'green', 'gray'], edgecolor='black')
                plt.xticks(x, labels, rotation=45, ha='right')
                plt.ylabel('Entropy (bits)')
                plt.title('Three-way comparison: Entropy (PI competitive / PI quantum / Lc0 / Stockfish)')
                plt.tight_layout()
                # create filesystem-safe, concise, unique filename per FEN to avoid overwrites
                fen_slug = re.sub(r'[^A-Za-z0-9]', '_', fen)
                fen_short = fen_slug[:20]
                fen_hash = hashlib.md5(fen.encode('utf-8')).hexdigest()[:8]
                ent_path = os.path.join('results', f'three_way_entropy_compare_{fen_short}_{fen_hash}.png')
                plt.savefig(ent_path, dpi=300)
                plt.close()

                plt.figure(figsize=(10, 4))
                plt.bar(x, acc_vals, width, color=['deepskyblue', 'orange', 'green', 'gray'], edgecolor='black')
                plt.xticks(x, labels, rotation=45, ha='right')
                plt.ylabel('Concentration (mode probability)')
                plt.ylim(0, 1.0)
                plt.title('Three-way comparison: Accuracy / Concentration')
                plt.tight_layout()
                acc_path = os.path.join('results', f'three_way_accuracy_compare_{fen_short}_{fen_hash}.png')
                plt.savefig(acc_path, dpi=300)
                plt.close()

                print(f"Saved comparison plots: {ent_path}, {acc_path}")
            except Exception as e:
                print(f"Warning: could not save comparison plots: {e}")

        return results
