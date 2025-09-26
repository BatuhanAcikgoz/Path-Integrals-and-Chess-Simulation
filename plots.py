import os
import time
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from matplotlib.ticker import ScalarFormatter  # unused
import warnings
warnings.filterwarnings("ignore")
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

import config  # Direct module import for better performance
from engine import Engine
from mathfuncs import Calc


class Plots:
    @staticmethod
    def plot_entropy_accuracy(lambda_entropies, lambda_accuracies):
        """
        Entropi ve doğruluk metriklerini lambda değerlerine göre çizer. X ekseninde lambda değerleri eşit aralıklı indekslerle gösterilir, etiketler gerçek lambda değerleriyle yazılır. Böylece yakın lambda değerleri üst üste binmez ve grafik daha okunaklı olur.

        :param lambda_entropies: A list or array representing the entropy values for different
            lambda (softmax temperature) values.
        :param lambda_accuracies: A list or array representing the accuracy values for
            corresponding lambda values.
        :return: None
        """
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax2 = ax1.twinx()

        x = list(range(len(config.LAMBDA_SCAN)))
        labels = [f"{l:.3f}" if l < 1 else f"{l:.2f}" for l in config.LAMBDA_SCAN]

        ax1.plot(x, lambda_entropies, 'b-o', label='Entropy')  # Direct access
        ax2.plot(x, lambda_accuracies, 'r-o', label='Accuracy')  # Direct access

        ax1.set_xlabel('λ (softmax temperature)')
        ax1.set_ylabel('Entropy', color='blue')
        ax2.set_ylabel('Accuracy', color='red')

        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=11, rotation=0)
        # Remove ticklabel_format to avoid ScalarFormatter error
        # ax1.ticklabel_format(style='plain', axis='x')

        plt.title('Entropy and Accuracy with λ')
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/entropy_accuracy_lambda.png")
        plt.close()

    @staticmethod
    def plot_entropy_accuracy_correlation(entropies, accuracies):
        """
        Plots the correlation between entropy values and accuracy values for different
        lambda values. The plot is customized with distinct colors and markers to
        represent the various lambda values, and the output is saved as a PNG file.

        :param entropies: A list of lists containing entropy values for different lambda
            configurations.
        :param accuracies: A list of lists containing accuracy values corresponding to
            the entropies for different lambda configurations.
        :return: None
        """
        import matplotlib
        plt.figure(figsize=(8, 5))
        # Tüm marker sembolleri
        all_markers = ["o", "s", "^", "D", "v", ">", "<", "P", "X", "*", "+", "x", "1", "2", "3", "4", "|", "_", ".", ",", "h", "H"]
        n_lambda = len(config.LAMBDA_SCAN)
        colors = sns.color_palette("tab10", n_lambda)
        # Kombinasyon: marker + renk
        for i, lam in enumerate(config.LAMBDA_SCAN):
            color = colors[i % len(colors)]
            marker = all_markers[i % len(all_markers)]
            # Eğer marker sayısı yetmezse, marker ve renk kombinasyonunu değiştir
            plt.scatter(entropies[i], accuracies[i], color=color, marker=marker, s=80, alpha=0.8, label=f"λ={lam}")
        plt.xlabel("Entropy")
        plt.ylabel("Accuracy")
        plt.title("Entropy vs Accuracy Correlation")
        plt.legend(loc="best", fontsize="small", title="Lambda Values", frameon=True, ncol=2)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/entropy_accuracy_correlation.png")
        plt.close()

    @staticmethod
    def plot_normalized_heatmap(transitions, top_k=10):
        """
        Generates and saves a normalized heatmap for the transition probabilities between moves,
        limited to the top `top_k` most frequent moves. The heatmap visualizes transitions between
        different moves as a probability matrix, making it easier to understand the likelihood of
        transitions between moves.

        :param transitions: A dictionary where keys are moves (str) and values are dictionaries
            mapping subsequent moves (str) to their corresponding counts (int).
        :type transitions: dict
        :param top_k: The maximum number of most frequent moves to consider for the heatmap.
            Defaults to 10.
        :type top_k: int
        :return: None
        """
        moves = list(transitions.keys())[:top_k]
        if not moves: return
        matrix = np.zeros((len(moves), len(moves)))
        move_to_idx = {m: i for i, m in enumerate(moves)}
        for i, m1 in enumerate(moves):
            total = sum(transitions[m1].values())
            for m2, cnt in transitions[m1].items():
                if m2 in move_to_idx:
                    matrix[i][move_to_idx[m2]] = cnt / total if total > 0 else 0
        sns.heatmap(matrix, xticklabels=moves, yticklabels=moves, annot=True, fmt='.2f', cmap='YlOrRd')
        plt.title("Normalized Transition Probability Matrix")
        plt.xlabel("Next Move")
        plt.ylabel("Previous Move")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/transition_heatmap.png")
        plt.close()

    @staticmethod
    def plot_lambda_kl_divergence(fen):
        """
        Plots the KL divergence for varying lambda values against a baseline reference.

        The function computes the KL divergence of sampled paths for a range of lambda
        values (`LAMBDA_SCAN`) against a fixed baseline reference sample. The baseline
        reference sample is generated using `LAMBDA`. It iteratively computes the
        divergence and records the results, which are subsequently visualized as a
        plot. The plot is saved as a PNG file in the `results` directory.

        The x-axis of the plot is displayed in a logarithmic scale, and the y-axis
        shows the computed KL divergence. The function ensures that the output directory
        exists before saving the output plot.

        :param fen: A string representing the initial position or configuration from
            which the paths are sampled.
        :return: None
        """
        baseline_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, config.LAMBDA, config.SAMPLE_COUNT, mode='competitive')
        _, baseline_counter = Calc.compute_entropy(baseline_paths)
        divergences = []
        for lam in tqdm(config.LAMBDA_SCAN, desc="KL λ"):
            paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lam, config.SAMPLE_COUNT, mode='competitive')
            _, counter = Calc.compute_entropy(paths)
            div = Calc.kl_divergence(baseline_counter, counter)
            divergences.append(div)
        plt.figure(figsize=(8,5))
        plt.plot(config.LAMBDA_SCAN, divergences, 'o-', color='darkgreen')  # Direct access
        plt.xlabel("Lambda")
        plt.ylabel("KL Divergence")
        plt.title(f"KL Divergence (Reference: λ={config.LAMBDA})")  # Direct access
        plt.xscale("log")
        # X ekseni etiketlerini düzenle - direkt değerleri göster
        plt.xticks(config.LAMBDA_SCAN, [f'{l:.2f}' if l < 1 else f'{l:.1f}' for l in config.LAMBDA_SCAN])  # Direct access
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/kl_divergence_lambda.png")
        plt.close()

    @staticmethod
    def plot_first_move_distribution(paths):
        """
        Plots and saves the first move frequency distribution based on a list of paths.

        The function processes the given list of paths, extracting the first move from
        each non-empty path. It then creates a bar plot showing the frequency of these
        first moves. The resulting plot is saved to a file in the 'results' directory.

        :param paths: A list of paths, where each path is a list of steps or moves.
        :return: None
        """
        firsts = [str(p[0]) for p in paths if p]
        if not firsts: return
        df = pd.DataFrame(firsts, columns=['FirstMove'])
        plt.figure(figsize=(10, 4))
        sns.countplot(data=df, x='FirstMove', order=df['FirstMove'].value_counts().index)
        plt.title("First Move Frequency Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/first_move_bar.png")
        plt.close()

    @staticmethod
    def plot_move_distribution_by_lambda(all_paths_by_lambda, lambda_values):
        """
        Lambda değerlerine göre ilk hamle dağılımını gösteren bar grafiği üretir.
        Renkler, lambda değeri arttıkça koyulaşan tek bir ton skalasından atanır.
        En yüksek frekanslı hamle en solda gösterilir.
        :param all_paths_by_lambda: Her lambda için path listesi
        :param lambda_values: Lambda değerleri (örn. [2,4,6,8,10])
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import numpy as np
        # Veri hazırlama
        data = []
        for lam, paths in zip(lambda_values, all_paths_by_lambda):
            for p in paths:
                if p:
                    data.append((lam, str(p[0])))
        if not data:
            return
        df = pd.DataFrame(data, columns=["Lambda", "FirstMove"])
        # Lambda değerlerini sıralı ve indeksli al
        unique_lambdas = np.sort(df["Lambda"].unique())
        n_colors = len(unique_lambdas)
        # Koyulaşan renk paleti (Blues)
        palette = sns.color_palette("Blues", n_colors)
        # Lambda: renk eşlemesi
        lambda_to_color = {lam: palette[i] for i, lam in enumerate(unique_lambdas)}
        # Pivot tablo: ilk hamleye göre lambda frekansları
        pivot = df.groupby(["FirstMove", "Lambda"]).size().unstack(fill_value=0)
        # --- EN YÜKSEK FREKANSLI HAMLE EN SOLDA ---
        # Toplam frekansa göre index'i sırala
        total_freq = pivot.sum(axis=1)
        sorted_index = total_freq.sort_values(ascending=False).index
        pivot = pivot.loc[sorted_index]
        plt.figure(figsize=(14, 6))
        # Her lambda için barları çiz
        bar_width = 0.8 / n_colors
        x = np.arange(len(pivot.index))
        for i, lam in enumerate(unique_lambdas):
            heights = pivot[lam].values
            plt.bar(x + i * bar_width, heights, width=bar_width, color=lambda_to_color[lam], label=f"λ={lam}")
        plt.title("İlk Hamle Dağılımı (Lambda'ya Göre)")
        plt.xlabel("İlk Hamle")
        plt.ylabel("Frekans")
        plt.xticks(x + bar_width * (n_colors-1)/2, pivot.index, rotation=45)
        plt.legend(title="Lambda", loc="upper right", fontsize="medium", title_fontsize="13", frameon=True, ncol=min(3, n_colors))
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/move_distribution_by_lambda.png")
        plt.close()

    @staticmethod
    def plot_ngram_frequencies(paths, n=3):
        """
        Generates a bar plot representing the frequencies of the most common n-grams
        in a given list of paths and saves the plot to the 'results' directory.

        This function analyzes a list of paths represented as sequences of tokens
        (for example, method calls, words, or other sequences). It computes the
        frequencies of n-grams (subsequences of n adjacent tokens), and visualizes
        the top 15 most frequent n-grams in a bar plot.

        :param paths: List of token lists representing paths. Each path is a sequence
                      of strings or objects converted to strings.
        :type paths: list[list[str]]
        :param n: The size of n-grams to compute. Defaults to 3.
        :type n: int
        :return: None. The plot is saved as an image file in the 'results' directory.
        :rtype: None
        """
        ngrams = Counter()
        tokenized_paths = []
        for p in paths:
            # Normalize tokens: prefer .uci() for chess.Move-like objects, fallback to str()
            tokens = []
            for m in p:
                try:
                    tok = m.uci()
                except Exception:
                    tok = str(m)
                tokens.append(tok)
            tokenized_paths.append(tokens)
            for i in range(len(tokens) - n + 1):
                ngrams[tuple(tokens[i:i+n])] += 1
        if not ngrams:
            print("[plot_ngram_frequencies] No n-grams found (empty paths)")
            return
        common = ngrams.most_common(15)
        labels = [' → '.join(ng) for ng, _ in common]
        freqs = [freq for _, freq in common]

        total_ngrams = sum(ngrams.values())
        freqs_pct = [f / total_ngrams for f in freqs]

        # Debug log: En sık 15 n-gram ve frekanslarını konsola yazdır
        print(f"--- Top {min(15, len(common))} n-grams (n={n}) ---")
        for idx, (ng, freq) in enumerate(common[:15]):
            pct = freq / total_ngrams * 100 if total_ngrams > 0 else 0.0
            print(f"{idx+1}. {' → '.join(ng)} : {freq} ({pct:.2f}%)")

        # More informative warning: identical top-3 frequencies can be valid (low diversity)
        if len(freqs) >= 3 and freqs[0] == freqs[1] == freqs[2]:
            # Basic diagnostics to help debug identical frequencies
            total_paths = len(paths)
            unique_tokenized_paths = len({tuple(t) for t in tokenized_paths if t})
            unique_first_moves = len({t[0] for t in tokenized_paths if t})
            print(f"[DIAGNOSTICS] total_paths={total_paths}, total_ngrams={total_ngrams}, unique_tokenized_paths={unique_tokenized_paths}, unique_first_moves={unique_first_moves}")
            # Print a few tokenized samples for inspection
            sample_preview = tokenized_paths[:5]
            for i, sp in enumerate(sample_preview):
                print(f"  sample[{i}]: {sp}")
            if total_ngrams < 30:
                print("[INFO] Top 3 n-grams have identical frequency — likely due to small sample size (<30 n-grams). Consider increasing sample_count for better statistics.")
            else:
                print("[WARNING] Top 3 n-grams have identical frequency. This may indicate low path diversity or a bug in tokenization. Check sample diversity and tokenization (move.uci vs str).")

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(labels)), freqs, color='royalblue', alpha=0.7)
        plt.xlabel(f'{n}-gram sequences')
        plt.ylabel('Frequency')
        plt.title(f'Top {len(labels)} {n}-gram Frequencies')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')

        # Annotate bars with percentage labels for clarity
        for rect, pct in zip(bars, freqs_pct):
            height = rect.get_height()
            y_offset = height + max(1, int(total_ngrams*0.01))
            plt.text(rect.get_x() + rect.get_width() / 2.0, y_offset, f'{pct*100:.1f}%', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/top_%d_grams.png" % n, dpi=300)
        plt.close()

    @staticmethod
    def plot_entropy_accuracy_time_series(entropy_series, accuracy_series, window=3):
        """
        Generates and saves a plot for the given entropy and accuracy
        time series, as well as their smoothed versions. The plot includes
        lines for the original entropy and accuracy series, smoothed
        versions using a specified window size, and labeling for better
        visualization.

        :param entropy_series: Array or list of numerical values representing
            the time series data for entropy.
        :param accuracy_series: Array or list of numerical values representing
            the time series data for accuracy.
        :param window: Integer specifying the smoothing window size for
            the uniform filter. Default is 3.
        :return: None
        """
        entropy_smooth = uniform_filter1d(entropy_series, size=window)
        accuracy_smooth = uniform_filter1d(accuracy_series, size=window)
        plt.figure(figsize=(10,5))
        plt.plot(entropy_series, label="Entropy", color="blue", alpha=0.5)
        plt.plot(entropy_smooth, label="Entropy (Smooth)", color="blue")
        plt.plot(accuracy_series, label="Accuracy", color="red", alpha=0.5)
        plt.plot(accuracy_smooth, label="Accuracy (Smooth)", color="red")
        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title("Entropy vs Accuracy Time Graph")
        plt.legend()
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/entropy_accuracy_time_series.png")
        plt.close()

    @staticmethod
    def plot_mutual_information_heatmap(all_paths_by_lambda, lambda_values):
        """
        Plots a heatmap representing the mutual information between the first moves and
        lambda values using a provided dataset. The resulting plot visualizes how the
        frequency distribution of the first moves depends on the lambda values. The heatmap
        is saved in a file under the "results" directory as 'mutual_info_heatmap.png'.

        :param all_paths_by_lambda: A list of lists, where each inner list represents the
            paths associated with a specific lambda value. Each path is assumed to be a
            sequence, and the first element in the sequence corresponds to the first move.
        :param lambda_values: A list containing lambda values corresponding to the paths
            provided in ``all_paths_by_lambda``.
        :return: None
        """
        move_lambda_counts = defaultdict(lambda: defaultdict(int))
        for lam, lam_paths in zip(lambda_values, all_paths_by_lambda):
            for path in lam_paths:
                if path:
                    move_lambda_counts[str(path[0])][lam] += 1
        if not move_lambda_counts: return
        df = pd.DataFrame(move_lambda_counts).T.fillna(0)
        df = df.div(df.sum(axis=0), axis=1)
        sns.heatmap(df, annot=False, cmap="Blues")
        plt.title("Mutual Information Heatmap (Move vs Lambda)")
        plt.xlabel("Lambda")
        plt.ylabel("First Move")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/mutual_info_heatmap.png")
        plt.close()

    @staticmethod
    def plot_entropy_distribution(lambda_entropies):
        """
        Plot the entropy distribution of lambda scans. This function generates a histogram
        with a KDE (Kernel Density Estimate) overlay for `lambda_entropies` and saves the
        resulting plot as an image file in the 'results' directory.

        :param lambda_entropies: List or array-like containing entropy values to plot.
        :type lambda_entropies: list or numpy.ndarray
        :return: None
        """
        plt.figure(figsize=(8, 5))
        sns.histplot(lambda_entropies, bins=10, kde=True, color='orange')
        plt.title("Entropy Distribution (Lambda Scan)")
        plt.xlabel("Entropy")
        plt.ylabel("Frequency")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/entropy_distribution_hist.png")
        plt.close()

    @staticmethod
    def plot_topn_accuracies_vs_lambda(all_paths_by_lambda, ground_truth, top_ns=None):
        """
        Farklı lambda değerleri için Top-N doğruluklarını çizer.
        """
        if ground_truth is None:
            return
        if top_ns is None:
            top_ns = [1, 3, 5]
        results = {n: [] for n in top_ns}
        for paths in all_paths_by_lambda:
            for n in top_ns:
                acc = Calc.match_top_n(paths, ground_truth, n=n)
                results[n].append(acc)
        plt.figure(figsize=(10, 5))
        for n, scores in results.items():
            plt.plot(config.LAMBDA_SCAN, scores, label=f"Top-{n} Accuracy", marker='o')  # Direct access
        plt.xscale("log")
        plt.xticks(config.LAMBDA_SCAN, [f'{l:.2f}' if l < 1 else f'{l:.1f}' for l in config.LAMBDA_SCAN])  # Direct access
        plt.xlabel("Lambda")
        plt.ylabel("Accuracy")
        plt.title("Top-N Accuracy vs Lambda")
        plt.legend()
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/topn_accuracy_vs_lambda.png")
        plt.close()

    @staticmethod
    def build_path_tree(paths, max_depth=4):
        """
        Builds a directed graph representing the paths provided, restricted by a maximum depth.
        The graph is visualized and saved as an image. Each edge is weighted by the frequency
        of its occurrence in the given paths.

        :param paths: A list of lists where each sublist represents a sequence of moves (or steps).
        :param max_depth: An integer specifying the maximum depth to consider in the paths.
            Defaults to 4.
        :return: None
        """
        G = nx.DiGraph()
        edge_weights = Counter()
        for path in paths:
            current = "ROOT"
            for move in path[:max_depth]:
                mv_str = str(move)
                edge_weights[(current, mv_str)] += 1
                G.add_edge(current, mv_str)
                current = mv_str
        if not G.nodes: return
        plt.figure(figsize=(10, 6))
        pos = nx.spring_layout(G, seed=42, k=0.3)
        edge_widths = [edge_weights[edge] / 5 for edge in G.edges()]
        nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue", width=edge_widths, arrows=True,
                edge_color="gray", font_size=8)
        plt.title("Path Tree (First 4 Half-Moves, Weighted)")
        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/path_tree.png")
        plt.close()

    @staticmethod
    def plot_multimodal_vs_singlemodal(pi_counter: Counter, lc0_moves, lc0_scores, fen_name, outfile_prefix="pi_vs_lc0"):
        """
        Generates and saves a comparative bar plot visualizing the probability distribution
        of initial moves from two models. The first represents a Path Integral (PI)
        approach expected to yield a multi-modal distribution, and the second an Lc0
        model expected to yield a single-modal distribution. PI probabilities are
        computed from move counts, while Lc0 probabilities are derived using softmax
        normalization of scores. The resulting plot includes two subplots comparing
        these distributions for a specific board state.

        :param pi_counter: Counter object containing move counts for the multi-modal
            distribution generated by the Path Integral model.
        :param lc0_moves: List of moves considered by the Lc0 model.
        :param lc0_scores: Scores corresponding to the moves in lc0_moves. These are
            used to compute Lc0 probabilities via softmax normalization.
        :param fen_name: The board state in Forsyth–Edwards Notation (FEN) format, used
            to name the output file and the plot title.
        :param outfile_prefix: Optional prefix for the output file name. By default,
            it is set to 'pi_vs_lc0'.
        :return: None
        """
        # Prepare PI distribution
        pi_moves, pi_counts = zip(*pi_counter.most_common()) if pi_counter else ([], [])
        pi_probs = np.array(pi_counts) / sum(pi_counts) if sum(pi_counts) > 0 else []

        # Prepare Lc0 distribution via softmax of scores
        lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LC0_SOFTMAX_LAMBDA)  # Direct access
        lc0_move_strs = [str(m) for m in lc0_moves]

        lc0_probs_arr = np.asarray(lc0_probs)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        fig.suptitle(f'Move Distribution Comparison ({fen_name})', fontsize=16)

        # Left: Path Integral (multi-modal expected)
        axes[0].bar(range(len(pi_probs)), pi_probs, color='steelblue')
        axes[0].set_xticks(range(len(pi_moves)))
        axes[0].set_xticklabels(pi_moves, rotation=45, ha='right')
        axes[0].set_title('Path Integral Model: Multimodal Distribution')
        axes[0].set_xlabel('First Move')
        axes[0].set_title('Path Integral Model: Multimodal Distribution')
        axes[0].set_ylim(0, 1)

        # Right: Lc0 (single-modal expected)
        axes[1].bar(range(len(lc0_probs)), lc0_probs, color='indianred')
        axes[1].set_xticks(range(len(lc0_move_strs)))
        axes[1].set_xticklabels(lc0_move_strs, rotation=45, ha='right')
        axes[1].set_title('Lc0: Unimodal Distribution (softmax normalized)')
        axes[1].set_xlabel('First Move')
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim(0, 1)

        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{outfile_prefix}_{fen_name}.png", dpi=300)
        plt.close()

    @staticmethod
    def compare_with_lc0(fen, fen_name, nodes=config.LC0_NODES, lambda_val=config.LAMBDA, sample_count=config.SAMPLE_COUNT, multipv=config.LC0_MULTIPV):
        """
        Lc0 ve PI modelini karşılaştırır. Artık GT kullanılmaz; doğruluk yerine konsantrasyon raporlanır.
        """
        print(f"\n--- Lc0 Comparison Begins: {fen_name} ---")
        t0 = time.perf_counter()
        pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lambda_val, sample_count, mode='competitive')
        pi_elapsed = time.perf_counter() - t0
        pi_counter = Calc.first_move_counter(pi_paths)
        pi_entropy, _ = Calc.compute_entropy(pi_paths)
        pi_top_move = Calc.most_frequent_first_move(pi_paths)
        pi_conc = max(pi_counter.values()) / sum(pi_counter.values()) if pi_counter else 0.0
        lc0_opts = {"MultiPV": int(multipv)}
        lc0_moves, lc0_scores, lc0_elapsed = Engine.lc0_top_moves_and_scores(
            fen, depth=nodes, multipv=multipv, options=lc0_opts
        )
        lc0_top = str(lc0_moves[0]) if lc0_moves else None
        lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LC0_SOFTMAX_LAMBDA)
        lc0_probs_arr = np.asarray(lc0_probs)
        lc0_entropy = float(np.sum([-p*np.log2(p) for p in lc0_probs_arr if p > 0])) if lc0_probs_arr.size > 0 else 0.0
        lc0_conc = float(np.max(lc0_probs_arr)) if lc0_probs_arr.size > 0 else 0.0
        pi_sims_per_sec = sample_count / pi_elapsed if pi_elapsed > 0 else 0.0
        lc0_nodes_per_sec = nodes / lc0_elapsed if lc0_elapsed and nodes else 0.0
        Plots.plot_multimodal_vs_singlemodal(pi_counter, lc0_moves, lc0_scores, fen_name)
        summary = {
            "fen_name": fen_name,
            "ground_truth": None,
            "pi_top_move": pi_top_move,
            "pi_accuracy_top1": pi_conc,
            "pi_accuracy_dist": pi_conc,
            "pi_entropy": pi_entropy,
            "pi_elapsed_s": pi_elapsed,
            "pi_throughput": pi_sims_per_sec,
            "lc0_top_move": lc0_top,
            "lc0_accuracy_top1": lc0_conc,
            "lc0_entropy": lc0_entropy,
            "lc0_elapsed_s": lc0_elapsed,
            "lc0_throughput": lc0_nodes_per_sec,
        }
        print(f"  - PI Model: Top Move={pi_top_move}, Concentration={pi_conc:.3f}, Entropy={pi_entropy:.3f}, Elapsed Time={pi_elapsed:.2f}s")
        print(f"  - Lc0 Model: Top Move={lc0_top}, Concentration={lc0_conc:.3f}, Entropy={lc0_entropy:.3f}, Elapsed Time={lc0_elapsed:.2f}s")
        return summary

    @staticmethod
    def compare_three_engines(fen, fen_name, depth=config.TARGET_DEPTH, lambda_val=config.LAMBDA, multipv=config.LC0_MULTIPV, stockfish_depth=config.STOCKFISH_DEPTH):
        """
        Comprehensive comparison between Path Integral, LC0, and Stockfish engines.
        GT kaldırılmıştır; doğruluk yerine konsantrasyon (mode probability) raporlanır.
        """
        print(f"\n--- Three-Engine Comparison Begins: {fen_name} ---")

        # Path Integral Analysis
        pi_start = time.perf_counter()
        pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lambda_val, config.SAMPLE_COUNT, mode='competitive')
        pi_elapsed = time.perf_counter() - pi_start
        pi_entropy, pi_counter = Calc.compute_entropy(pi_paths)
        pi_top_move = Calc.most_frequent_first_move(pi_paths)
        pi_accuracy = max(pi_counter.values()) / sum(pi_counter.values()) if pi_counter else 0.0

        # LC0 Analysis
        lc0_opts = {"MultiPV": multipv, "Temperature": config.LC0_TEMPERATURE, "CPuct": config.LC0_CPUCT}
        lc0_moves, lc0_scores, lc0_elapsed = Engine.lc0_top_moves_and_scores(
            fen, depth=depth, multipv=multipv, options=lc0_opts
        )
        lc0_top = str(lc0_moves[0]) if lc0_moves else None
        lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LC0_SOFTMAX_LAMBDA)
        lc0_probs_arr = np.asarray(lc0_probs)
        lc0_accuracy = float(np.max(lc0_probs_arr)) if lc0_probs_arr.size > 0 else 0.0
        lc0_entropy = float(np.sum([-p*np.log2(p) for p in lc0_probs_arr if p > 0])) if lc0_probs_arr.size > 0 else 0.0

        # Stockfish Analysis
        stockfish_result = Engine.get_stockfish_analysis(fen, stockfish_depth, multipv)
        stockfish_top = stockfish_result.get('best_move')
        stockfish_elapsed = stockfish_result.get('elapsed_time', 0.0)
        stockfish_scores = stockfish_result.get('scores', [])
        if stockfish_scores:
            stockfish_probs = Calc.normalize_scores_to_probs(stockfish_scores, lambda_val)
            stockfish_probs_arr = np.asarray(stockfish_probs)
            stockfish_entropy = float(np.sum([-p*np.log2(p) for p in stockfish_probs_arr if p > 0])) if stockfish_probs_arr.size > 0 else 0.0
            stockfish_accuracy = float(np.max(stockfish_probs_arr))
        else:
            stockfish_entropy = 0.0
            stockfish_accuracy = 0.0

        # Create comparison visualization
        import pandas as pd
        pi_df = pd.DataFrame({'value': [lambda_val], 'entropy': [pi_entropy], 'accuracy': [pi_accuracy]})
        lc0_df = pd.DataFrame({'value': [config.LC0_CPUCT], 'entropy': [lc0_entropy], 'accuracy': [lc0_accuracy]})
        stockfish_df = pd.DataFrame({'value': [lambda_val], 'entropy': [stockfish_entropy], 'accuracy': [stockfish_accuracy]})
        Plots.plot_exploration_tradeoff_three_engines(pi_df, lc0_df, stockfish_df, pi_quantum_df=None, fen_name=fen_name)

        # Prepare summary
        summary = {
            "fen_name": fen_name,
            "ground_truth": None,
            "pi_top_move": pi_top_move,
            "pi_accuracy": pi_accuracy,
            "pi_entropy": pi_entropy,
            "pi_elapsed_s": pi_elapsed,
            "lc0_top_move": lc0_top,
            "lc0_accuracy": lc0_accuracy,
            "lc0_entropy": lc0_entropy,
            "lc0_elapsed_s": lc0_elapsed,
            "stockfish_top_move": stockfish_top,
            "stockfish_accuracy": stockfish_accuracy,
            "stockfish_entropy": stockfish_entropy,
            "stockfish_elapsed_s": stockfish_elapsed,
        }

        print(f"  - PI Model: Top Move={pi_top_move}, Concentration={pi_accuracy:.3f}, Entropy={pi_entropy:.3f}, Elapsed Time={pi_elapsed:.2f}s")
        print(f"  - LC0 Model: Top Move={lc0_top}, Concentration={lc0_accuracy:.3f}, Entropy={lc0_entropy:.3f}, Elapsed Time={lc0_elapsed:.2f}s")
        print(f"  - Stockfish Model: Top Move={stockfish_top}, Concentration={stockfish_accuracy:.3f}, Entropy={stockfish_entropy:.3f}, Elapsed Time={stockfish_elapsed:.2f}s")

        return summary

    @staticmethod
    def plot_exploration_tradeoff_three_engines(pi_df: pd.DataFrame, lc0_df: pd.DataFrame, stockfish_df: pd.DataFrame,
                                                pi_quantum_df: pd.DataFrame = None,
                                                fen_name=None, lambda_scan=None, outfile_prefix="explore_exploit_tradeoff_three"):
        """
        Tüm motorlar için ortak x ekseni (LAMBDA_SCAN) kullanılır.
        LC0, PI, PI (quantum_limit) ve Stockfish verileri griddeki lambda değerlerine tam hizalanır, eksik olanlar np.nan ile doldurulur.
        """
        import matplotlib.pyplot as plt
        import os
        import numpy as np

        x_grid = config.LAMBDA_SCAN
        x_labels = [str(x) for x in x_grid]
        x_idx = np.arange(len(x_grid))

        # Parametre sütununu dinamik bul
        def get_param_col(df):
            for col in ['lambda', 'value']:
                if col in df.columns:
                    return col
            return None

        # LC0 verilerini grid ile hizala
        lc0_entropy = [np.nan] * len(x_grid)
        lc0_accuracy = [np.nan] * len(x_grid)
        lc0_param_col = get_param_col(lc0_df) if lc0_df is not None else None
        if lc0_df is not None and not lc0_df.empty and lc0_param_col:
            for i, x in enumerate(x_grid):
                match = lc0_df[np.isclose(lc0_df[lc0_param_col], x, atol=1e-6)]
                if not match.empty:
                    lc0_entropy[i] = float(match.iloc[0]['entropy'])
                    lc0_accuracy[i] = float(match.iloc[0]['accuracy'])

        # PI verileri için grid ile hizala (PI competitive)
        pi_entropy = [np.nan] * len(x_grid)
        pi_accuracy = [np.nan] * len(x_grid)
        pi_param_col = get_param_col(pi_df) if pi_df is not None else None
        if pi_df is not None and not pi_df.empty and pi_param_col:
            for i, x in enumerate(x_grid):
                match = pi_df[np.isclose(pi_df[pi_param_col], x, atol=1e-6)]
                if not match.empty:
                    pi_entropy[i] = float(match.iloc[0]['entropy'])
                    pi_accuracy[i] = float(match.iloc[0]['accuracy'])

        # PI quantum_limit verileri için grid ile hizala (opsiyonel)
        pi_quantum_entropy = [np.nan] * len(x_grid)
        pi_quantum_accuracy = [np.nan] * len(x_grid)
        if pi_quantum_df is not None:
            pi_quantum_param_col = get_param_col(pi_quantum_df)
            if not pi_quantum_df.empty and pi_quantum_param_col:
                for i, x in enumerate(x_grid):
                    match = pi_quantum_df[np.isclose(pi_quantum_df[pi_quantum_param_col], x, atol=1e-6)]
                    if not match.empty:
                        pi_quantum_entropy[i] = float(match.iloc[0]['entropy'])
                        pi_quantum_accuracy[i] = float(match.iloc[0]['accuracy'])

        # Stockfish verileri için grid ile hizala
        stockfish_entropy = [np.nan] * len(x_grid)
        stockfish_accuracy = [np.nan] * len(x_grid)
        stockfish_param_col = get_param_col(stockfish_df) if stockfish_df is not None else None
        if stockfish_df is not None and not stockfish_df.empty and stockfish_param_col:
            for i, x in enumerate(x_grid):
                match = stockfish_df[np.isclose(stockfish_df[stockfish_param_col], x, atol=1e-6)]
                if not match.empty:
                    stockfish_entropy[i] = float(match.iloc[0]['entropy'])
                    stockfish_accuracy[i] = float(match.iloc[0]['accuracy'])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        # --- ENTROPY PANEL ---
        ax1.plot(x_idx, pi_entropy, '-o', label='PI (competitive)', color='blue', alpha=0.8, linewidth=2)
        if pi_quantum_df is not None:
            ax1.plot(x_idx, pi_quantum_entropy, '-D', label='PI (quantum_limit)', color='purple', alpha=0.9, linewidth=2)
        ax1.plot(x_idx, stockfish_entropy, '-^', label='Stockfish', color='red', alpha=0.8, linewidth=2)
        ax1.plot(x_idx, lc0_entropy, '-s', label='LC0', color='green', alpha=0.8, linewidth=2)
        ax1.set_xticks(x_idx)
        ax1.set_xticklabels(x_labels, fontsize=12, rotation=30)
        ax1.set_xlabel('Parametre (λ)')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title(f'Exploration Behavior Comparison - {fen_name}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- ACCURACY PANEL ---
        ax2.plot(x_idx, pi_accuracy, '-o', label='PI (competitive)', color='blue', alpha=0.8, linewidth=2)
        if pi_quantum_df is not None:
            ax2.plot(x_idx, pi_quantum_accuracy, '-D', label='PI (quantum_limit)', color='purple', alpha=0.9, linewidth=2)
        ax2.plot(x_idx, stockfish_accuracy, '-^', label='Stockfish', color='red', alpha=0.8, linewidth=2)
        ax2.plot(x_idx, lc0_accuracy, '-s', label='LC0', color='green', alpha=0.8, linewidth=2)
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels(x_labels, fontsize=12, rotation=30)
        ax2.set_xlabel('Parametre (λ)')
        ax2.set_ylabel('Concentration (mode probability)')
        ax2.set_title(f'Concentration vs Parameter - {fen_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        outdir = "results"
        os.makedirs(outdir, exist_ok=True)
        fig.tight_layout()
        outfile = os.path.join(outdir, f"{outfile_prefix}_{fen_name}.png")
        plt.savefig(outfile, dpi=150)
        plt.close(fig)
        print(f"   📊 Three-engine chart created: {outfile}")

    @staticmethod
    def generate_pdf_report(report_title="Comprehensive PI/Stockfish Analysis Report",
                            results_dir="results",
                            output_path=None,
                            extra_notes=None):
        import os
        import datetime
        import pandas as pd
        from glob import glob

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                            Table, TableStyle, PageBreak)
            from xml.sax.saxutils import escape
        except Exception:
            print("[Error] 'reportlab' paketini yükleyin: pip install reportlab")
            raise

        os.makedirs(results_dir, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(results_dir, "analysis_report.pdf")

        doc = SimpleDocTemplate(
            output_path, pagesize=A4,
            leftMargin=1.8*cm, rightMargin=1.8*cm, topMargin=1.6*cm, bottomMargin=1.6*cm
        )
        styles = getSampleStyleSheet()
        styles.add(ParagraphStyle(name="Section", fontSize=14, leading=18, spaceBefore=10, spaceAfter=6, textColor=colors.darkblue))
        styles.add(ParagraphStyle(name="Small", fontSize=9, leading=12, textColor=colors.grey))
        styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8, leading=10, textColor=colors.black))
        story = []

        def p(txt, style="Normal"):
            story.append(Paragraph(escape(str(txt)), styles[style]))

        def add_title_page():
            p(report_title, "Title")
            story.append(Spacer(1, 8))
            p(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if extra_notes:
                story.append(Spacer(1, 6))
                p("Notlar:", "Section")
                p(extra_notes)
            story.append(PageBreak())

        def add_section(title, desc=None):
            p(title, "Section")
            if desc:
                p(desc)
            story.append(Spacer(1, 6))

        def add_text_file(path, title=None, max_chars=6000):
            if not os.path.exists(path):
                return False
            if title:
                p(title, "Small")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                return False
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n...[truncated]..."
            for line in content.splitlines():
                story.append(Paragraph(escape(line), styles["Mono"]))
            story.append(Spacer(1, 8))
            return True

        def add_pgn_preview(path, title=None, max_lines=120):
            if not os.path.exists(path):
                return False
            if title:
                p(title, "Small")
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            story.append(Paragraph("...[truncated PGN]...", styles["Mono"]))
                            break
                        story.append(Paragraph(escape(line.rstrip("\n")), styles["Mono"]))
            except Exception:
                return False
            story.append(Spacer(1, 8))
            return True

        def add_csv_table(path, title=None, max_rows=30, floatfmt=5):
            if not os.path.exists(path):
                return False
            try:
                df = pd.read_csv(path)
            except Exception:
                return False
            if df.empty:
                return False
            for c in df.columns:
                if pd.api.types.is_float_dtype(df[c]):
                    df[c] = df[c].round(floatfmt)
            view = df.head(max_rows)
            data = [list(map(str, view.columns))] + [[("" if pd.isna(x) else str(x)) for x in row] for row in view.values.tolist()]
            tbl = Table(data, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("ALIGN", (0,0), (-1,-1), "LEFT"),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.transparent])
            ]))
            if title:
                p(title, "Small")
            story.append(tbl)
            if len(df) > max_rows:
                p(f"Not: Tablo {len(df)} satırdan ilk {max_rows} satırı göstermektedir.", "Small")
            story.append(Spacer(1, 8))
            return True

        def add_image(path, caption=None, max_width_cm=16, max_height_cm=20):
            if not os.path.exists(path):
                return False
            try:
                img = Image(path)
                max_w = max_width_cm * cm
                max_h = max_height_cm * cm
                # Orantılı ölçekleme
                orig_w, orig_h = img.drawWidth, img.drawHeight
                scale_w = max_w / float(orig_w)
                scale_h = max_h / float(orig_h)
                scale = min(scale_w, scale_h, 1.0)  # 1.0'dan büyükse küçültme yok
                img.drawWidth = orig_w * scale
                img.drawHeight = orig_h * scale
                story.append(img)
                if caption:
                    p(caption, "Small")
                story.append(Spacer(1, 8))
                return True
            except Exception:
                return False

        # Başlık
        add_title_page()

        # 0) İçindekiler niteliğinde genel bakış
        add_section("Genel Bakış", "Bu rapor, ana akışta üretilen tüm analizleri tek PDF altında toplar: Convergence, Quantum↔Classical, Feynman analojisi, Perfect Play, Pozisyon Karmaşıklığı, Kombinatorik Patlama, Horizon Effect, PI/Stockfish kıyasları, keşif-sömürü tradeoff grafikleri, kalibrasyon ve özet CSV/TXT çıktıları.")
        story.append(PageBreak())

        # 1) İstatistiksel Güvenilirlik ve Kalibrasyon
        add_section("Statistical Reliability (PI & Stockfish)", "Tekrarlı ölçümlerin güven aralıkları.")
        add_csv_table(os.path.join(results_dir, "ci_summary.csv"), title="ci_summary.csv")
        add_section("Calibration (PI)", "Güvenilirlik diyagramı, Brier/ECE ve metin özeti.")
        add_text_file(os.path.join(results_dir, "pi_calibration_summary.txt"), title="pi_calibration_summary.txt")
        add_image(os.path.join(results_dir, "pi_reliability.png"), caption="PI Reliability Diagram")

        # 2) Hareket Kalitesi / Regret
        add_section("Move Quality / Regret", "Top hamle kalitesi ve regret metrikleri.")
        add_csv_table(os.path.join(results_dir, "move_quality.csv"), title="move_quality.csv")

        # 3) Bilgi Teorik Ölçümler
        add_section("Information-Theoretic (KL/JS)", "PI ve Stockfish ilk hamle dağılımı ayrışmaları.")
        add_text_file(os.path.join(results_dir, "pi_stockfish_divergence.txt"), title="pi_stockfish_divergence.txt")

        # 4) Markov Entropy Rate
        add_section("Markov Entropy Rate", "Geçiş matrisi üzerinden entropi oranı.")
        add_text_file(os.path.join(results_dir, "entropy_rate.txt"), title="entropy_rate.txt")

        # 5) Adil Zaman Bütçesi
        add_section("Fair Time Budget Benchmark", "Eşit süre altında kıyaslama.")
        add_text_file(os.path.join(results_dir, "fair_time_budget.txt"), title="fair_time_budget.txt")

        # 6) Çoklu Pozisyon Genellemesi ve Özetler
        add_section("Multi-Position Generalization", "Özet istatistikler ve PI vs Stockfish özet tabloları.")
        add_csv_table(os.path.join(results_dir, "generalization_summary.csv"), title="generalization_summary.csv")
        add_csv_table(os.path.join(results_dir, "stockfish_vs_pi_summary.csv"), title="stockfish_vs_pi_summary.csv")

        # 7) Keşif-Sömürü Tradeoff ve Karşılaştırmalı Görseller (pozisyon bazlı)
        add_section("Per-Position Analizleri", "Her pozisyon için taramalar ve görseller.")
        # CSV: pi_scan_*.csv, stockfish_scan_*.csv
        for csv_path in sorted(glob(os.path.join(results_dir, "pi_scan_*.csv"))):
            name = os.path.basename(csv_path).removeprefix("pi_scan_").removesuffix(".csv")
            add_csv_table(csv_path, title=f"PI λ Tarama: {name}")
            twin = os.path.join(results_dir, f"stockfish_scan_{name}.csv")
            add_csv_table(twin, title=f"Stockfish Depth Tarama: {name}")
            add_image(os.path.join(results_dir, f"explore_exploit_tradeoff_{name}.png"), caption=f"Exploration↔Exploitation: {name}")
            add_image(os.path.join(results_dir, f"pi_vs_stockfish_{name}.png"), caption=f"PI vs Stockfish Dağılımı: {name}")
            add_image(os.path.join(results_dir, f"horizon_effect_{name}.png"), caption=f"Horizon Effect: {name}")

        # 8) Örneklem Sayısı Duyarlılığı
        add_section("Sample Size Sensitivity", "Örnek sayısı ↔ entropi/doğruluk ilişkisi.")
        add_image(os.path.join(results_dir, "sample_size_sensitivity.png"), caption="Effect of Sample Number on Entropy and Accuracy")

        # 9) Convergence Analizleri
        add_section("Convergence Analizleri", "Derinliğe ve λ'ya göre yakınsama.")
        add_image(os.path.join(results_dir, "convergence_by_depth.png"), caption="Convergence by Depth")
        add_image(os.path.join(results_dir, "convergence_by_lambda.png"), caption="Convergence by Lambda")

        # 10) Kombinatorik Patlama Analizi
        add_section("Kombinatorik Patlama Analizi", "Path sayısı ve karmaşıklık büyümesi.")
        add_image(os.path.join(results_dir, "combinatorial_explosion_analysis.png"), caption="Combinatorial Explosion")

        # 11) Quantum↔Classical Geçişi
        add_section("Quantum-Classical Transition Analysis", "Üç rejim karşılaştırması.")
        add_image(os.path.join(results_dir, "quantum_classical_transition.png"), caption="Quantum↔Classical Transition")

        # 12) Feynman Path Analojisi Validation
        add_section("Feynman Path Analogy Validation", "Quantum limitte yüksek, classical limitte düşük entropi beklentisi.")
        add_image(os.path.join(results_dir, "feynman_analogy_validation.png"), caption="Feynman Analojisi Doğrulaması")

        # 13) Cognitive Analogy Validation
        add_section("Cognitive Analogy Validation", "İnsan/Uzman/Master seviyeleri için entropi ve doğruluk.")
        add_image(os.path.join(results_dir, "cognitive_analogy_analysis.png"), caption="Cognitive Analogy")

        # 14) Dinamik Lambda Adaptasyonu
        add_section("Dynamic Lambda Adaptation", "Adaptif λ ile keşif/sömürü dengesi.")
        add_image(os.path.join(results_dir, "dynamic_lambda_adaptation.png"), caption="Dynamic Lambda Adaptation")

        # 15) Pozisyon Karmaşıklığı Etkisi
        add_section("Pozisyon Karmaşıklığı Etkisi Analizi", "Pozisyon karmaşıklığı ve metrik etkileri.")
        # Olası görsel dosyaları (genel isimleri tarayalım)
        for img in sorted(glob(os.path.join(results_dir, "position_complexity*.png"))):
            add_image(img, caption=os.path.basename(img))

        # 16) Perfect Play Self-Play (PGN Önizleme)
        add_section("Perfect Play Experiment Result (PGN)", "Mükemmel oyun (yüksek λ) kendi kendine oyun simülasyonu.")
        add_pgn_preview(os.path.join(results_dir, "perfect_play_self_play.pgn"),
                        title="perfect_play_self_play.pgn (ilk satırlar)")

        # 17) Ek: Tüm PNG/CSV/TXT indeksleri
        story.append(PageBreak())
        add_section("Ek: Çıktı Dosyaları Dizini", "Raporla ilgili ham veriler `results` klasöründedir.")
        # Listeleri tablo olarak verelim
        def add_file_index(pattern, title):
            files = [os.path.basename(x) for x in sorted(glob(os.path.join(results_dir, pattern)))]
            if not files:
                return
            p(title, "Small")
            rows = [[f] for f in files]
            tbl = Table([["Dosya Adı"]] + rows, hAlign="LEFT")
            tbl.setStyle(TableStyle([
                ("FONT", (0,0), (-1,0), "Helvetica-Bold"),
                ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
                ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
                ("FONTSIZE", (0,0), (-1,-1), 8),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))

        add_file_index("*.png", "Görseller (PNG)")
        add_file_index("*.csv", "Tablolar (CSV)")
        add_file_index("*.txt", "Metin Özetleri (TXT)")
        add_file_index("*.pgn", "Oyun Kayıtları (PGN)")

        # PDF üret
        doc.build(story)
        print(f"[OK] PDF rapor üretildi: {output_path}")

    @staticmethod
    def plot_horizon_effect_comparison(pi_counter, shallow_move, deep_move, fen_name, outfile_prefix="horizon_effect"):
        """
        Horizon effect analizi için özel karşılaştırma grafiği oluşturur.
        Shallow (trap), Deep (true) ve PI model sonuçlarını karşılaştırır.

        :param pi_counter: PI model'in hamle sayım sonuçları
        :param shallow_move: Shallow analysis'in bulduğu hamle (trap)
        :param deep_move: Deep analysis'in bulduğu hamle (true)
        :param fen_name: FEN pozisyonunun adı
        :param outfile_prefix: Çıktı dosyası öneki
        :return: None
        """
        # PI model dağılımını hazırla
        pi_moves, pi_counts = zip(*pi_counter.most_common()) if pi_counter else ([], [])
        pi_probs = np.array(pi_counts) / sum(pi_counts) if sum(pi_counts) > 0 else []

        # 3 panelli grafik oluştur
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Horizon Effect Analysis: {fen_name}', fontsize=16)

        # Sol panel: Shallow Analysis (Trap Move)
        axes[0].bar([0], [1.0], color='red', alpha=0.7, width=0.6)
        axes[0].set_xticks([0])
        axes[0].set_xticklabels([shallow_move])
        axes[0].set_title(f'Shallow Analysis (depth=5)\n"Trap" Move', fontsize=12)
        axes[0].set_ylabel('Probability')
        axes[0].set_ylim(0, 1.1)
        axes[0].text(0, 1.05, 'TRAP', ha='center', fontweight='bold', color='red')

        # Orta panel: Deep Analysis (True Move)
        axes[1].bar([0], [1.0], color='green', alpha=0.7, width=0.6)
        axes[1].set_xticks([0])
        axes[1].set_xticklabels([deep_move])
        axes[1].set_title(f'Deep Analysis (depth=100)\n"True" Move', fontsize=12)
        axes[1].set_ylabel('Probability')
        axes[1].set_ylim(0, 1.1)
        axes[1].text(0, 1.05, 'TRUE', ha='center', fontweight='bold', color='green')

        # Sağ panel: Path Integral Model (Multimodal Discovery)
        colors = ['green' if move == deep_move else 'red' if move == shallow_move else 'steelblue'
                  for move in pi_moves]
        bars = axes[2].bar(range(len(pi_probs)), pi_probs, color=colors, alpha=0.8)
        axes[2].set_xticks(range(len(pi_moves)))
        axes[2].set_xticklabels(pi_moves, rotation=45, ha='right')
        axes[2].set_title('Path Integral Model (λ=0.2)\nMultimodal Discovery', fontsize=12)
        axes[2].set_ylabel('Probability')
        axes[2].set_ylim(0, max(pi_probs) * 1.1 if len(pi_probs) > 0 else 1)

        # Sağ panelde başarı durumunu göster
        if deep_move in pi_counter:
            axes[2].text(0.5, 0.95, f'✓ TRUE MOVE DISCOVERED',
                         transform=axes[2].transAxes, ha='center',
                         fontweight='bold', color='green', fontsize=10)
            axes[2].text(0.5, 0.87, f'Frequency: {pi_counter[deep_move]}/{config.SAMPLE_COUNT}',
                         transform=axes[2].transAxes, ha='center',
                         fontweight='bold', color='green', fontsize=9)
        else:
            axes[2].text(0.5, 0.95, '✗ TRUE MOVE NOT FOUND',
                         transform=axes[2].transAxes, ha='center',
                         fontweight='bold', color='red', fontsize=10)

        # Legend ekle
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7, label='Trap Move'),
            plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7, label='True Move'),
            plt.Rectangle((0, 0), 1, 1, color='steelblue', alpha=0.8, label='Other Moves')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=3)

        plt.tight_layout(rect=(0, 0.08, 1, 0.93))
        os.makedirs("results", exist_ok=True)
        plt.savefig(f"results/{outfile_prefix}_{fen_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_entropy_vs_depth(csv_path):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df['depth'], df['entropy'], marker='o', color='blue')
        plt.xlabel('Depth')
        plt.ylabel('Entropy')
        mode = df['mode'].iloc[0] if 'mode' in df.columns else 'unknown'
        plt.title(f'Entropy vs Depth [{mode}]')
        plt.grid(True)
        plt.tight_layout()
        fname = f"results/entropy_vs_depth_{mode}.png"
        plt.savefig(fname)
        plt.close()

    @staticmethod
    def plot_accuracy_vs_depth(csv_path):
        df = pd.read_csv(csv_path)
        plt.figure(figsize=(8, 5))
        plt.plot(df['depth'], df['accuracy'], marker='o', color='red')
        plt.xlabel('Depth')
        plt.ylabel('Accuracy')
        mode = df['mode'].iloc[0] if 'mode' in df.columns else 'unknown'
        plt.title(f'Accuracy vs Depth [{mode}]')
        plt.grid(True)
        plt.tight_layout()
        fname = f"results/accuracy_vs_depth_{mode}.png"
        plt.savefig(fname)
        plt.close()

    @staticmethod
    def plot_first_move_bar(csv_path):
        df = pd.read_csv(csv_path)
        # Tüm ilk hamleleri topla
        all_moves = []
        for moves in df['sample_first_moves'].dropna():
            all_moves.extend(moves.split(", "))
        counter = Counter(all_moves)
        moves, freqs = zip(*counter.most_common()) if counter else ([], [])
        plt.figure(figsize=(10, 5))
        sns.barplot(x=list(moves), y=list(freqs), palette='viridis')
        plt.xlabel('First Move')
        plt.ylabel('Frequency')
        mode = df['mode'].iloc[0] if 'mode' in df.columns else 'unknown'
        plt.title(f'First Move Distribution [{mode}]')
        plt.tight_layout()
        fname = f"results/first_move_bar_{mode}.png"
        plt.savefig(fname)
        plt.close()

    @staticmethod
    def plot_kl_vs_lambda(csv_path, reference_counter=None):
        df = pd.read_csv(csv_path)
        # Her lambda için ilk hamle dağılımı
        lambdas = df['lambda'].unique()
        kl_values = []
        for lam in lambdas:
            moves = []
            for mv in df[df['lambda'] == lam]['sample_first_moves'].dropna():
                moves.extend(mv.split(", "))
            counter = Counter(moves)
            # Referans dağılım yoksa, ilk lambda'yı referans al
            if reference_counter is None:
                reference_counter = counter
            # KL diverjansı hesapla
            p = np.array([counter.get(k, 1e-12) for k in reference_counter.keys()])
            p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
            q = np.array([reference_counter.get(k, 1e-12) for k in reference_counter.keys()])
            q = q / q.sum() if q.sum() > 0 else np.ones_like(q) / len(q)
            kl = np.sum(p * np.log2(p / q))
            kl_values.append(kl)
        plt.figure(figsize=(8, 5))
        plt.plot(lambdas, kl_values, marker='o', color='purple')
        plt.xlabel('Lambda')
        plt.ylabel('KL Divergence')
        mode = df['mode'].iloc[0] if 'mode' in df.columns else 'unknown'
        plt.title(f'KL Divergence vs Lambda [{mode}]')
        plt.grid(True)
        plt.tight_layout()
        fname = f"results/kl_vs_lambda_{mode}.png"
        plt.savefig(fname)
        plt.close()
