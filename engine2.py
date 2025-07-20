import chess
import chess.engine
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
# 6. Entropi ve Doğruluk Zaman Serisi
from scipy.ndimage import uniform_filter1d
import seaborn as sns
import time
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
import pandas as pd
import os
import warnings
warnings.filterwarnings("ignore")

# === Ayarlar ===
STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4"
SAMPLE_COUNT = 100
SAMPLE_DEPTH = 5
FAST_ANALYSIS_DEPTH = 2
GROUND_TRUTH_DEPTH = 5
MULTIPV = 5
LAMBDA = 0.7
TOP_N = 5
LAMBDA_SCAN = [0.01, 0.05, 0.1, 0.2, 0.5, 0.7, 0.75, 0.8, 1.0, 1.5]
DEPTH_SCAN = [4, 6, 8, 10, 12, 20]
SAMPLE_SIZES = [50, 100, 200, 500, 1000]

# === Yardımcı Fonksiyonlar ===
def get_top_moves_and_scores(engine, board, depth=FAST_ANALYSIS_DEPTH, multipv=MULTIPV):
    info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    moves, scores = [], []
    for entry in info:
        move = entry['pv'][0]
        score = entry['score'].white().score(mate_score=100000) or 100000
        moves.append(move)
        scores.append(score)
    return moves, scores

def get_ground_truth_move(fen, depth=GROUND_TRUTH_DEPTH):
    board = chess.Board(fen)
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        best_move = info.get('pv', [None])[0]
    return str(best_move)

def softmax(scores, lam):
    arr = np.array(scores, dtype=np.float64)
    arr -= arr.max()
    e = np.exp(lam * arr)
    return e / e.sum()

def sample_paths(fen, depth, lam, samples):
    paths = []
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        for i in tqdm(range(samples), desc=f"Sampling λ={lam}"):
            board = chess.Board(fen)
            path = []
            for _ in range(depth):
                mvs, scs = get_top_moves_and_scores(engine, board)
                probs = softmax(scs, lam)
                chosen = np.random.choice(len(mvs), p=probs)
                mv = mvs[chosen]
                path.append(mv)
                board.push(mv)
                if board.is_game_over():
                    break
            paths.append(path)
    return paths

def compute_entropy(paths):
    firsts = [str(p[0]) for p in paths if p]
    counter = Counter(firsts)
    probs = np.array(list(counter.values())) / len(firsts)
    entropy = -np.sum(probs * np.log2(probs))
    return entropy, counter

def match_ground_truth(paths, ground_truth_move):
    firsts = [str(p[0]) for p in paths if p]
    match = sum(1 for f in firsts if f == ground_truth_move)
    return match / len(firsts)

def match_top_n(paths, ground_truth_move, n=TOP_N):
    firsts = [str(p[0]) for p in paths if p]
    counter = Counter(firsts)
    top_n = set([move for move, _ in counter.most_common(n)])
    return 1.0 if ground_truth_move in top_n else 0.0

def plot_entropy_accuracy(lambda_entropies, lambda_accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(LAMBDA_SCAN, lambda_entropies, 'b-o', label='Entropi')
    ax2.plot(LAMBDA_SCAN, lambda_accuracies, 'r-o', label='Doğruluk')

    ax1.set_xlabel('λ (softmax sıcaklığı)')
    ax1.set_ylabel('Entropi', color='blue')
    ax2.set_ylabel('Doğruluk', color='red')

    ax1.set_xscale('log')
    ax1.set_xticks(LAMBDA_SCAN)
    ax1.get_xaxis().set_major_formatter(FixedFormatter([str(l) for l in LAMBDA_SCAN]))

    plt.title('λ ile Entropi ve Doğruluk')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropi_dogruluk_lambda.png")
    plt.close()

def plot_entropy_accuracy_correlation(entropies, accuracies):
    plt.figure(figsize=(8, 5))

    colors = sns.color_palette("tab10", len(LAMBDA_SCAN))
    markers = ['o', 's', '^', 'D', 'v', '>', '<', 'P', 'X', '*']  # 10 farklı marker

    for i, lam in enumerate(LAMBDA_SCAN):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.scatter(entropies[i], accuracies[i],
                    color=color, marker=marker, s=80, alpha=0.8, label=f"λ={lam}")

    plt.xlabel("Entropi")
    plt.ylabel("Doğruluk")
    plt.title("Entropi vs Doğruluk Korelasyonu")
    plt.legend(loc="best", fontsize="small", title="Lambda Değerleri", frameon=True, ncol=2)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropi_dogruluk_korelasyon.png")
    plt.close()

def build_transition_matrix(paths):
    transitions = defaultdict(Counter)
    for path in paths:
        for i in range(len(path) - 1):
            a, b = str(path[i]), str(path[i+1])
            transitions[a][b] += 1
    return transitions

def plot_normalized_heatmap(transitions, top_k=10):
    moves = list(transitions.keys())[:top_k]
    matrix = np.zeros((top_k, top_k))
    move_to_idx = {m: i for i, m in enumerate(moves)}
    for i, m1 in enumerate(moves):
        total = sum(transitions[m1].values())
        for m2, cnt in transitions[m1].items():
            if m2 in move_to_idx:
                matrix[i][move_to_idx[m2]] = cnt / total if total > 0 else 0
    sns.heatmap(matrix, xticklabels=moves, yticklabels=moves, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title("Normalize Edilmiş Geçiş Olasılık Matrisi")
    plt.xlabel("Sonraki Hamle")
    plt.ylabel("Önceki Hamle")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/normalized_gecis_matrisi.png")
    plt.close()

def build_path_tree(paths, max_depth=4):
    G = nx.DiGraph()
    edge_weights = Counter()
    for path in paths:
        current = "ROOT"
        for move in path[:max_depth]:
            mv_str = str(move)
            edge_weights[(current, mv_str)] += 1
            G.add_edge(current, mv_str)
            current = mv_str
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, seed=42, k=0.3)
    edge_widths = [edge_weights[edge]/5 for edge in G.edges()]
    nx.draw(G, pos, with_labels=True, node_size=500, node_color="skyblue",
            width=edge_widths, arrows=True, edge_color="gray", font_size=8)
    plt.title("Yol Ağacı (İlk 4 Yarı-Hamle, Ağırlıklı)")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/yol_agaci_agirlikli.png")
    plt.close()

def batch_ground_truth_analysis(fens):
    results = []
    for fen in fens:
        gt = get_ground_truth_move(fen)
        paths = sample_paths(fen, SAMPLE_DEPTH, LAMBDA, SAMPLE_COUNT)
        entropy, _ = compute_entropy(paths)
        acc = match_ground_truth(paths, gt)
        topn = match_top_n(paths, gt)
        results.append((fen, gt, entropy, acc, topn))
    df = pd.DataFrame(results, columns=["FEN", "GT", "Entropi", "Doğruluk", f"Top{TOP_N} Acc"])
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/batch_results.csv", index=False)
    print(df)

def kl_divergence(p_input, q_input):
    if isinstance(p_input, Counter) and isinstance(q_input, Counter):
        all_keys = sorted(set(p_input.keys()) | set(q_input.keys()))
        p = np.array([p_input.get(k, 0) for k in all_keys], dtype=np.float64)
        q = np.array([q_input.get(k, 0) for k in all_keys], dtype=np.float64)
    else:
        p = np.asarray(p_input, dtype=np.float64)
        q = np.asarray(q_input, dtype=np.float64)
        if len(p) != len(q):
            raise ValueError("p ve q dizilerinin uzunlukları eşit olmalıdır")

    p /= p.sum()
    q /= q.sum()
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

def plot_lambda_kl_divergence(fen):
    baseline_paths = sample_paths(fen, SAMPLE_DEPTH, LAMBDA, SAMPLE_COUNT)
    _, baseline_counter = compute_entropy(baseline_paths)

    divergences = []
    for lam in tqdm(LAMBDA_SCAN, desc="KL λ"):
        paths = sample_paths(fen, SAMPLE_DEPTH, lam, SAMPLE_COUNT)
        _, counter = compute_entropy(paths)
        div = kl_divergence(baseline_counter, counter)
        divergences.append(div)

    plt.figure(figsize=(8,5))
    plt.plot(LAMBDA_SCAN, divergences, 'o-', color='darkgreen')
    plt.xlabel("Lambda")
    plt.ylabel("KL Divergence")
    plt.title("KL Divergence (Referans: λ=0.7)")
    plt.xscale("log")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/kl_divergence_lambda.png")
    plt.close()


def entropy_vs_depth(fen, lam):
    entropies = []
    for depth in tqdm(DEPTH_SCAN, desc="Entropy vs Depth"):
        paths = sample_paths(fen, depth, lam, SAMPLE_COUNT)
        entropy, _ = compute_entropy(paths)
        entropies.append(entropy)
    plt.plot(DEPTH_SCAN, entropies, marker='o')
    plt.xlabel("Depth")
    plt.ylabel("Entropy")
    plt.title(f"Entropy vs Depth (λ={lam})")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropy_vs_depth.png")
    plt.close()

def mutual_information(paths, lambda_values):
    move_lambda_counts = defaultdict(lambda: defaultdict(int))
    for lam, lam_paths in zip(lambda_values, paths):
        for path in lam_paths:
            if path:
                move_lambda_counts[str(path[0])][lam] += 1
    total = sum(sum(lam_counts.values()) for lam_counts in move_lambda_counts.values())
    mi = 0
    for move, lam_counts in move_lambda_counts.items():
        p_move = sum(lam_counts.values()) / total
        for lam, count in lam_counts.items():
            p_joint = count / total
            p_lam = sum(move_lambda_counts[m][lam] for m in move_lambda_counts) / total
            mi += p_joint * np.log2(p_joint / (p_move * p_lam))
    return mi

def analyze_markov_steady_state(transitions):
    all_moves = sorted(set(transitions.keys()).union(*[set(d.keys()) for d in transitions.values()]))
    idx = {m: i for i, m in enumerate(all_moves)}
    N = len(all_moves)
    matrix = np.zeros((N, N))

    for from_mv, targets in transitions.items():
        i = idx[from_mv]
        total = sum(targets.values())
        for to_mv, cnt in targets.items():
            j = idx[to_mv]
            matrix[i][j] = cnt / total if total > 0 else 0

    # Normalize: row stochastic
    for i in range(N):
        if matrix[i].sum() == 0:
            matrix[i] = np.ones(N) / N

    eigvals, eigvecs = np.linalg.eig(matrix.T)
    principal = np.where(np.isclose(eigvals, 1.0))[0]
    if len(principal) == 0:
        print("Uyarı: Birim özdeğer bulunamadı, steady-state çözümlenemedi.")
        return None

    stat_dist = np.real(eigvecs[:, principal[0]])
    stat_dist /= stat_dist.sum()

    df = pd.DataFrame({"Move": all_moves, "SteadyProb": stat_dist})
    df = df.sort_values("SteadyProb", ascending=False)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/steady_state_distribution.csv", index=False)
    return df

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# 1. Entropi-Lambda-Derinlik 3D Yüzey Grafiği
def plot_entropy_lambda_depth_surface(fen):
    surface_data = []
    for lam in tqdm(LAMBDA_SCAN, desc="Lambda"):
        row = []
        for depth in DEPTH_SCAN:
            paths = sample_paths(fen, depth, lam, SAMPLE_COUNT)
            entropy, _ = compute_entropy(paths)
            row.append(entropy)
        surface_data.append(row)

    X, Y = np.meshgrid(DEPTH_SCAN, LAMBDA_SCAN)
    Z = np.array(surface_data)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('Entropy')
    ax.set_title('Entropy Surface (Lambda vs Depth)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropy_surface_3d.png")
    plt.close()

# 2. İlk Hamle Dağılımı (Bar & Violin Plot)
def plot_first_move_distribution(paths):
    firsts = [str(p[0]) for p in paths if p]
    df = pd.DataFrame(firsts, columns=['FirstMove'])

    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x='FirstMove', order=df['FirstMove'].value_counts().index)
    plt.title("İlk Hamle Frekans Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/first_move_bar.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    sns.violinplot(data=df, x='FirstMove', inner='point', order=df['FirstMove'].value_counts().index)
    plt.title("İlk Hamle Violin Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/first_move_violin.png")
    plt.close()

# 3. Lambda'ya Göre Hamle Dağılımı Şeması
def plot_move_distribution_by_lambda(all_paths_by_lambda, lambda_values):
    data = []
    for lam, paths in zip(lambda_values, all_paths_by_lambda):
        for p in paths:
            if p:
                data.append((lam, str(p[0])))
    df = pd.DataFrame(data, columns=["Lambda", "FirstMove"])

    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="FirstMove", hue="Lambda", order=df['FirstMove'].value_counts().index)
    plt.title("Lambda'ya Göre İlk Hamle Dağılımı")
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/move_distribution_by_lambda.png")
    plt.close()

# 4. Hamle Sekans Frekansı (N-gram Analizi)
def plot_ngram_frequencies(paths, n=3):
    ngrams = Counter()
    for p in paths:
        tokens = [str(m) for m in p]
        for i in range(len(tokens) - n + 1):
            ngrams[tuple(tokens[i:i+n])] += 1
    common = ngrams.most_common(15)
    labels = [' → '.join(ng) for ng, _ in common]
    values = [v for _, v in common]

    plt.figure(figsize=(12, 5))
    sns.barplot(x=values, y=labels)
    plt.title(f"Top {n}-gram Sekansları")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/top_{n}_grams.png")
    plt.close()

# 5. Transition Graph Centrality
from networkx.algorithms.centrality import betweenness_centrality

def analyze_transition_graph_centrality(transitions):
    G = nx.DiGraph()
    for src, dsts in transitions.items():
        for dst, w in dsts.items():
            G.add_edge(src, dst, weight=w)

    centrality = betweenness_centrality(G, weight='weight', normalized=True)
    sorted_cent = sorted(centrality.items(), key=lambda x: -x[1])[:15]
    df = pd.DataFrame(sorted_cent, columns=["Move", "Centrality"])

    plt.figure(figsize=(10, 5))
    sns.barplot(x="Centrality", y="Move", data=df)
    plt.title("Transition Graph Betweenness Centrality")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/transition_graph_centrality.png")
    plt.close()
    return df

def plot_entropy_accuracy_time_series(entropy_series, accuracy_series, window=3):
    entropy_smooth = uniform_filter1d(entropy_series, size=window)
    accuracy_smooth = uniform_filter1d(accuracy_series, size=window)

    plt.figure(figsize=(10,5))
    plt.plot(entropy_series, label="Entropy", color="blue", alpha=0.5)
    plt.plot(entropy_smooth, label="Entropy (Smooth)", color="blue")
    plt.plot(accuracy_series, label="Accuracy", color="red", alpha=0.5)
    plt.plot(accuracy_smooth, label="Accuracy (Smooth)", color="red")
    plt.xlabel("Index")
    plt.ylabel("Değer")
    plt.title("Entropi ve Doğruluk Zaman Grafiği")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropy_accuracy_time_series.png")
    plt.close()

# 7. Mutual Information Heatmap

def plot_mutual_information_heatmap(all_paths_by_lambda, lambda_values):
    move_lambda_counts = defaultdict(lambda: defaultdict(int))
    for lam, lam_paths in zip(lambda_values, all_paths_by_lambda):
        for path in lam_paths:
            if path:
                move_lambda_counts[str(path[0])][lam] += 1


    df = pd.DataFrame(move_lambda_counts).T.fillna(0)
    df = df.div(df.sum(axis=0), axis=1)  # normalize by column
    sns.heatmap(df, annot=False, cmap="Blues")
    plt.title("Mutual Information Heatmap (Hamle vs Lambda)")
    plt.xlabel("Lambda")
    plt.ylabel("First Move")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/mutual_info_heatmap.png")
    plt.close()

def plot_entropy_gradient_surface(fen):
    surface_data = []
    for lam in tqdm(LAMBDA_SCAN, desc="Lambda"):
        row = []
        for depth in DEPTH_SCAN:
            paths = sample_paths(fen, depth, lam, SAMPLE_COUNT)
            entropy, _ = compute_entropy(paths)
            row.append(entropy)
        surface_data.append(row)

    Z = np.array(surface_data)
    X, Y = np.meshgrid(DEPTH_SCAN, LAMBDA_SCAN)
    dZ_dx, dZ_dy = np.gradient(Z, axis=(1, 0))  # ∂Entropy/∂Depth, ∂Entropy/∂Lambda

    plt.figure(figsize=(10, 6))
    plt.quiver(X, Y, dZ_dx, dZ_dy, color='purple', angles='xy')
    plt.title("Entropy Gradient (Lambda vs Depth)")
    plt.xlabel("Depth")
    plt.ylabel("Lambda")
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropy_gradient_field.png")
    plt.close()

def plot_entropy_distribution(lambda_entropies):
    plt.figure(figsize=(8, 5))
    sns.histplot(lambda_entropies, bins=10, kde=True, color='orange')
    plt.title("Entropi Dağılımı (Lambda Taraması)")
    plt.xlabel("Entropi")
    plt.ylabel("Frekans")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/entropy_distribution_hist.png")
    plt.close()

def plot_topn_accuracies_vs_lambda(all_paths_by_lambda, ground_truth, top_ns=[1, 3, 5]):
    results = {n: [] for n in top_ns}
    for paths in all_paths_by_lambda:
        for n in top_ns:
            acc = match_top_n(paths, ground_truth, n=n)
            results[n].append(acc)

    plt.figure(figsize=(10, 5))
    for n, scores in results.items():
        plt.plot(LAMBDA_SCAN, scores, label=f"Top-{n} Accuracy", marker='o')
    plt.xscale("log")
    plt.xlabel("Lambda")
    plt.ylabel("Doğruluk")
    plt.title("Top-N Doğruluk vs Lambda")
    plt.legend()
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/topn_accuracy_vs_lambda.png")
    plt.close()

def compute_transition_complexity(transitions):
    total_entropy = 0
    for src, dsts in transitions.items():
        counts = np.array(list(dsts.values()))
        probs = counts / counts.sum()
        total_entropy += -np.sum(probs * np.log2(probs))
    return total_entropy

from kneed import KneeLocator

def detect_entropy_elbow(lambda_entropies):
    kn = KneeLocator(LAMBDA_SCAN, lambda_entropies, curve='convex', direction='decreasing')
    print(f"Elbow noktası (opt λ): {kn.knee}")

# === Bot vs Bot Karşılaştırma Fonksiyonları ===

class ChessBot:
    def __init__(self, depth, lambda_val, name):
        self.depth = depth
        self.lambda_val = lambda_val
        self.name = name
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def get_move(self, engine, board):
        """Bot'un hamle seçme algoritması"""
        moves, scores = get_top_moves_and_scores(engine, board, depth=self.depth)
        if not moves:
            return None
        probs = softmax(scores, self.lambda_val)
        chosen_idx = np.random.choice(len(moves), p=probs)
        return moves[chosen_idx]

    def reset_stats(self):
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def get_win_rate(self):
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

def play_game(bot1, bot2, starting_fen, max_moves=50):
    """İki bot arasında bir oyun oyna"""
    board = chess.Board(starting_fen)
    move_count = 0

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        while not board.is_game_over() and move_count < max_moves:
            # Sıra beyazda ise bot1, siyahta ise bot2 oynar
            current_bot = bot1 if board.turn == chess.WHITE else bot2

            try:
                move = current_bot.get_move(engine, board)
                if move is None:
                    break
                board.push(move)
                move_count += 1
            except Exception as e:
                print(f"Hata: {e}")
                break

    # Oyun sonucunu değerlendir
    result = board.result()
    if result == "1-0":  # Beyaz kazandı (bot1)
        return "bot1_wins"
    elif result == "0-1":  # Siyah kazandı (bot2)
        return "bot2_wins"
    else:  # Berabere veya tamamlanmamış
        return "draw"

def run_tournament(bot1, bot2, starting_fen, num_games=10):
    """İki bot arasında turnuva düzenle"""
    results = []

    for game_num in tqdm(range(num_games), desc=f"Turnuva: {bot1.name} vs {bot2.name}"):
        # Her oyunda botların renklerini değiştir
        if game_num % 2 == 0:
            result = play_game(bot1, bot2, starting_fen)
            if result == "bot1_wins":
                bot1.wins += 1
                bot2.losses += 1
            elif result == "bot2_wins":
                bot1.losses += 1
                bot2.wins += 1
            else:
                bot1.draws += 1
                bot2.draws += 1
        else:
            # Renkleri değiştir
            result = play_game(bot2, bot1, starting_fen)
            if result == "bot1_wins":
                bot2.wins += 1
                bot1.losses += 1
            elif result == "bot2_wins":
                bot2.losses += 1
                bot1.wins += 1
            else:
                bot1.draws += 1
                bot2.draws += 1

        results.append(result)

    return results

def experiment_depth_vs_lambda(starting_fen, depth_values, lambda_values, games_per_matchup=20):
    """Depth ve lambda parametrelerinin etkisini karşılaştır"""
    results_matrix = np.zeros((len(depth_values), len(lambda_values)))
    detailed_results = {}

    print("Depth vs Lambda Deneyleri Başlıyor...")

    for i, depth in enumerate(depth_values):
        for j, lam in enumerate(lambda_values):
            # Referans bot: orta seviye parametreler
            ref_bot = ChessBot(depth=6, lambda_val=0.5, name="Reference")
            test_bot = ChessBot(depth=depth, lambda_val=lam, name=f"D{depth}_L{lam}")

            # Turnuva oyna
            ref_bot.reset_stats()
            test_bot.reset_stats()

            tournament_results = run_tournament(test_bot, ref_bot, starting_fen, games_per_matchup)

            # Test bot'un kazanma oranı
            win_rate = test_bot.get_win_rate()
            results_matrix[i, j] = win_rate

            detailed_results[f"D{depth}_L{lam}"] = {
                'wins': test_bot.wins,
                'losses': test_bot.losses,
                'draws': test_bot.draws,
                'win_rate': win_rate
            }

            print(f"Depth={depth}, Lambda={lam}: Win Rate = {win_rate:.3f}")

    return results_matrix, detailed_results

def experiment_lambda_vs_lambda(starting_fen, lambda_values, fixed_depth=6, games_per_matchup=30):
    """Aynı depth'te farklı lambda değerini karşılaştır"""
    n = len(lambda_values)
    win_matrix = np.zeros((n, n))

    print(f"Lambda vs Lambda Deneyleri (Depth={fixed_depth})...")

    for i, lam1 in enumerate(lambda_values):
        for j, lam2 in enumerate(lambda_values):
            if i == j:
                win_matrix[i, j] = 0.5  # Kendisiyle oynamaması için
                continue

            bot1 = ChessBot(depth=fixed_depth, lambda_val=lam1, name=f"L{lam1}")
            bot2 = ChessBot(depth=fixed_depth, lambda_val=lam2, name=f"L{lam2}")

            bot1.reset_stats()
            bot2.reset_stats()

            tournament_results = run_tournament(bot1, bot2, starting_fen, games_per_matchup)

            win_rate = bot1.get_win_rate()
            win_matrix[i, j] = win_rate

            print(f"λ{lam1} vs λ{lam2}: {lam1} kazanma oranı = {win_rate:.3f}")

    return win_matrix

def plot_depth_lambda_heatmap(results_matrix, depth_values, lambda_values):
    """Depth vs Lambda sonuçlarını heatmap olarak göster"""
    plt.figure(figsize=(12, 8))

    sns.heatmap(results_matrix,
                xticklabels=[f"{l:.2f}" for l in lambda_values],
                yticklabels=[f"{d}" for d in depth_values],
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                cbar_kws={'label': 'Kazanma Oranı'})

    plt.title('Bot Performansı: Depth vs Lambda\n(Referans: Depth=6, Lambda=0.5)')
    plt.xlabel('Lambda Değeri')
    plt.ylabel('Depth Değeri')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/bot_performance_depth_lambda.png", dpi=300)
    plt.close()

def plot_lambda_tournament_matrix(win_matrix, lambda_values):
    """Lambda turnuva sonuçlarını matrix olarak göster"""
    plt.figure(figsize=(10, 8))

    sns.heatmap(win_matrix,
                xticklabels=[f"{l:.2f}" for l in lambda_values],
                yticklabels=[f"{l:.2f}" for l in lambda_values],
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                center=0.5,
                cbar_kws={'label': 'Kazanma Oranı'})

    plt.title('Lambda Turnuva Matrisi\n(Satır lambda\'sının sütun lambda\'sına karşı kazanma oranı)')
    plt.xlabel('Rakip Lambda')
    plt.ylabel('Bot Lambda')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/lambda_tournament_matrix.png", dpi=300)
    plt.close()

def analyze_optimal_parameters(results_matrix, depth_values, lambda_values):
    """En iyi parametreleri analiz et"""
    # En yüksek kazanma oranı
    max_idx = np.unravel_index(np.argmax(results_matrix), results_matrix.shape)
    best_depth = depth_values[max_idx[0]]
    best_lambda = lambda_values[max_idx[1]]
    best_score = results_matrix[max_idx]

    print(f"\nEn İyi Parametreler:")
    print(f"Depth: {best_depth}")
    print(f"Lambda: {best_lambda}")
    print(f"Kazanma Oranı: {best_score:.3f}")

    # Lambda'ya göre ortalama performans
    lambda_avg = np.mean(results_matrix, axis=0)
    best_lambda_avg_idx = np.argmax(lambda_avg)

    print(f"\nLambda Bazında En İyi (ortalama):")
    print(f"Lambda: {lambda_values[best_lambda_avg_idx]}")
    print(f"Ortalama Kazanma Oranı: {lambda_avg[best_lambda_avg_idx]:.3f}")

    # Depth'e göre ortalama performans
    depth_avg = np.mean(results_matrix, axis=1)
    best_depth_avg_idx = np.argmax(depth_avg)

    print(f"\nDepth Bazında En İyi (ortalama):")
    print(f"Depth: {depth_values[best_depth_avg_idx]}")
    print(f"Ortalama Kazanma Oranı: {depth_avg[best_depth_avg_idx]:.3f}")

    return {
        'best_depth': best_depth,
        'best_lambda': best_lambda,
        'best_score': best_score,
        'lambda_avg': lambda_avg,
        'depth_avg': depth_avg
    }


# === Ana Akış ===
if __name__ == '__main__':
    print("Lambda'ya Göre Entropi ve Doğruluk:")
    ground_truth = get_ground_truth_move(FEN)
    print(f"Gerçek en iyi hamle (depth={GROUND_TRUTH_DEPTH}): {ground_truth}")

    lambda_entropies = []
    lambda_accuracies = []
    all_paths_by_lambda = []
    for lam in tqdm(LAMBDA_SCAN, desc="Lambda taraması"):
        paths = sample_paths(FEN, SAMPLE_DEPTH, lam, SAMPLE_COUNT)
        entropy, counter = compute_entropy(paths)
        accuracy = match_ground_truth(paths, ground_truth)
        lambda_entropies.append(entropy)
        lambda_accuracies.append(accuracy)
        all_paths_by_lambda.append(paths)
        print(f"λ={lam:.3f}  Entropi={entropy:.4f}  Doğruluk={accuracy:.4f}  En Sık Hamle: {counter.most_common(1)[0][0]}")

    plot_entropy_accuracy(lambda_entropies, lambda_accuracies)
    plot_entropy_accuracy_correlation(lambda_entropies, lambda_accuracies)
    plot_lambda_kl_divergence(FEN)
    entropy_vs_depth(FEN, LAMBDA)

    print("\n--- Mutual Information ---")
    mi_score = mutual_information(all_paths_by_lambda, LAMBDA_SCAN)
    print(f"Mutual Information (Move vs Lambda): {mi_score:.4f}")

    print("\n--- Geçiş Matrisi ---")
    paths = sample_paths(FEN, SAMPLE_DEPTH, LAMBDA, SAMPLE_COUNT)
    transitions = build_transition_matrix(paths)
    plot_normalized_heatmap(transitions)
    print("\n--- Yol Ağacı ---")
    build_path_tree(paths)
    print("\n--- Top-N Doğruluk ---")
    top_n_score = match_top_n(paths, ground_truth, TOP_N)
    print(f"Top-{TOP_N} içinde doğruluk: {top_n_score:.4f}")
    print("\n--- Entropi-Lambda-Derinlik 3D Yüzey Grafiği")
    plot_entropy_lambda_depth_surface(FEN)
    print("\n--- İlk Hamle Dağılımı")
    plot_first_move_distribution(paths)
    print("\n--- Lambda'ya Göre Hamle Dağılımı")
    plot_move_distribution_by_lambda(all_paths_by_lambda, LAMBDA_SCAN)
    print("\n--- N-gram Frekansları")
    plot_ngram_frequencies(paths, n=3)
    print("\n--- Geçiş Grafiği Merkeziyet Analizi")
    centrality_df = analyze_transition_graph_centrality(transitions)
    print(centrality_df.head(15))
    print("\n--- Entropi ve Doğruluk Zaman Serisi")
    entropy_series = [compute_entropy(sample_paths(FEN, SAMPLE_DEPTH, lam, SAMPLE_COUNT))[0] for lam in LAMBDA_SCAN]
    accuracy_series = [match_ground_truth(sample_paths(FEN, SAMPLE_DEPTH, lam, SAMPLE_COUNT), ground_truth) for lam in LAMBDA_SCAN]
    plot_entropy_accuracy_time_series(entropy_series, accuracy_series)
    print("\n--- Mutual Information Heatmap")
    plot_mutual_information_heatmap(all_paths_by_lambda, LAMBDA_SCAN)
    print("\n--- Entropi Dağılımı")
    plot_entropy_distribution(lambda_entropies)
    print("\n--- Top-N Doğruluk vs Lambda")
    plot_topn_accuracies_vs_lambda(all_paths_by_lambda, ground_truth, top_ns=[1, 3, 5])
    print("\n--- Geçiş Karmaşıklığı")
    transition_complexity = compute_transition_complexity(transitions)
    print(f"Geçiş karmaşıklığı (toplam entropi): {transition_complexity:.4f}")
    print("\n--- Elbow Noktası Tespiti")
    detect_entropy_elbow(lambda_entropies)

    print("\n--- Bot vs Bot Turnuva Deneyleri ---")
    starting_fen = FEN
    depth_values = [4, 6, 8]
    lambda_values = [0.1, 0.5, 0.9]
    games_per_matchup = 10

    # 1. Deney: Depth ve Lambda'nın etkisi
    results_matrix, detailed_results = experiment_depth_vs_lambda(starting_fen, depth_values, lambda_values, games_per_matchup)
    plot_depth_lambda_heatmap(results_matrix, depth_values, lambda_values)
    analyze_optimal_parameters(results_matrix, depth_values, lambda_values)

    # 2. Deney: Aynı depth'te farklı lambda değerleri
    fixed_depth = 6
    win_matrix = experiment_lambda_vs_lambda(starting_fen, lambda_values, fixed_depth, games_per_matchup)
    plot_lambda_tournament_matrix(win_matrix, lambda_values)
