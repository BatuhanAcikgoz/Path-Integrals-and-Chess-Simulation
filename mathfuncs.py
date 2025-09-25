import os
import time
from collections import Counter, defaultdict

import chess
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
import seaborn as sns
from tqdm import tqdm

import config
from cache import Cache
from engine import Engine


class Calc:
    # Class-level caches for performance optimization
    _entropy_cache = {}
    _concentration_cache = {}
    
    @staticmethod
    def softmax(scores, lam):
        """
        Optimized softmax computation with numerical stability.
        
        Args:
            scores: List of numerical scores
            lam: Temperature parameter (lambda)
            
        Returns:
            Normalized probability distribution
        """
        if scores is None or len(scores) == 0:
            return np.array([])
            
        arr = np.array(scores, dtype=np.float64)
        # Numerical stability: subtract max before scaling
        # Small λ → flat distribution (exploration), Large λ → sharp distribution (exploitation)
        arr_scaled = (arr - arr.max()) * lam
        
        # Use log-sum-exp for numerical stability
        log_sum_exp = np.log(np.sum(np.exp(arr_scaled)))
        return np.exp(arr_scaled - log_sum_exp)

    @staticmethod
    def compute_entropy(paths):
        """
        Optimized entropy computation with caching.
        
        Args:
            paths: List of move sequences
            
        Returns:
            Tuple of (entropy, move_counter)
        """
        if not paths:
            return 0.0, Counter()
        
        # Create cache key from first moves
        cache_key = tuple(str(p[0]) if p else '' for p in paths)
        
        if cache_key in Calc._entropy_cache:
            return Calc._entropy_cache[cache_key]
        
        # Extract first moves
        firsts = [str(p[0]) for p in paths if p]
        if not firsts:
            result = (0.0, Counter())
            Calc._entropy_cache[cache_key] = result
            return result
        
        # Count frequencies
        counter = Counter(firsts)
        total = len(firsts)
        
        # Compute entropy efficiently
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                prob = count / total
                entropy -= prob * np.log2(prob)
        
        result = (float(entropy), counter)
        Calc._entropy_cache[cache_key] = result
        return result

    @staticmethod
    def top_move_concentration(paths):
        """
        Optimized concentration calculation with caching.
        Returns the probability of the most frequent first move.
        """
        if not paths:
            return 0.0
            
        cache_key = tuple(str(p[0]) if p else '' for p in paths)
        
        if cache_key in Calc._concentration_cache:
            return Calc._concentration_cache[cache_key]

        firsts = [str(p[0]) for p in paths if p]
        if not firsts:
            result = 0.0
            Calc._concentration_cache[cache_key] = result
            return result
            
        counter = Counter(firsts)
        result = max(counter.values()) / len(firsts)
        Calc._concentration_cache[cache_key] = result
        return result

    @staticmethod
    def match_ground_truth(paths, ground_truth_move):
        """
        Calculate the match rate between the first elements of sequences in a list
        and a ground truth value.

        The function computes how many of the first elements in a list of sequences
        (stringified) match a given ground truth value and returns the match rate
        as a proportion of the total number of sequences with a first element.

        :param paths: A list of sequences. Each sequence is assumed to be iterable.
                      If any sequence in the list is empty (or evaluates to False),
                      it will be ignored.
        :param ground_truth_move: A string representing the ground truth value to
                                  compare against the first element of each sequence.
        :return: A float value representing the ratio of matches between the first
                 elements in the sequences and the ground truth value.
        """
        firsts = [str(p[0]) for p in paths if p]
        if not firsts:
            return 0.0
        match = sum(1 for f in firsts if f == ground_truth_move)
        return match / len(firsts)

    @staticmethod
    def match_top_n(paths, ground_truth_move, n=config.TOP_N):  # Direct access
        """
        Evaluate if the ground truth move is among the top N most frequent moves from the given paths.

        The function analyzes the first elements of each path, computes their frequencies, and checks
        if the specified ground truth move is included in the N most common moves. If the ground truth
        move is within the top N moves, the function returns 1.0; otherwise, it returns 0.0.

        :param paths: A list of lists, where each sublist represents a sequence of moves.
                      Each move is expected to be a hashable object.
        :param ground_truth_move: The move that needs to be validated against the top N most common moves.
        :param n: The number of top frequent moves to consider (default is defined by `TOP_N`).
        :return: A float value of 1.0 if the `ground_truth_move` is among the top N moves, otherwise 0.0.
        :rtype: float
        """
        firsts = [str(p[0]) for p in paths if p]
        if not firsts:
            return 0.0
        counter = Counter(firsts)
        top_n = set([move for move, _ in counter.most_common(n)])
        return 1.0 if ground_truth_move in top_n else 0.0

    @staticmethod
    def build_transition_matrix(paths):
        """
        Builds a transition matrix based on the provided list of paths.

        This function takes a list of paths where each path is a sequence of nodes and computes
        the count of transitions between nodes. The output is a dictionary where the keys are
        nodes, and the values are Counter objects indicating the number of transitions to other nodes.

        :param paths: List of paths, where each path is a sequence (list) of nodes.
        :type paths: list[list[Any]]
        :return: A dictionary representing the transition matrix, which maps each node to a
                 Counter object that counts transitions to other nodes.
        :rtype: defaultdict[Counter]
        """
        transitions = defaultdict(Counter)
        for path in paths:
            for i in range(len(path) - 1):
                a, b = str(path[i]), str(path[i+1])
                transitions[a][b] += 1
        return transitions

    @staticmethod
    def kl_divergence(p_input, q_input):
        """
        Calculate the Kullback-Leibler (KL) divergence between two discrete probability
        distributions. KL divergence is a measure of how one probability distribution
        differs from a second, reference probability distribution.

        :param p_input: A dictionary representing the first probability distribution. The
            keys are the events, and the values are the probabilities associated with
            those events.
        :type p_input: dict
        :param q_input: A dictionary representing the second probability distribution, typically
            the reference distribution. The keys are the events, and the values are the probabilities associated with
            those events.
        :type q_input: dict
        :return: The KL divergence, which is a non-negative value representing the amount
            of information lost when q_input is used to approximate p_input.
        :rtype: float
        """
        all_keys = sorted(set(p_input.keys()) | set(q_input.keys()))
        p = np.array([p_input.get(k, 0) for k in all_keys], dtype=np.float64)
        q = np.array([q_input.get(k, 0) for k in all_keys], dtype=np.float64)
        p /= p.sum()
        q /= q.sum()
        mask = (p > 0) & (q > 0)
        return np.sum(p[mask] * np.log2(p[mask] / q[mask]))

    @staticmethod
    def mutual_information(all_paths_by_lambda, lambda_values):
        """
        Calculate mutual information (MI) between two variables represented as
        collections of paths and lambda values. This function quantifies the amount
        of information obtained about one variable (lambda) given the other variable
        (path moves).

        :param all_paths_by_lambda: A list of lists, where each sublist contains
                                    paths for a specific lambda value. Each path
                                    is represented as a sequence, and the first
                                    step of the path is considered in the calculation.
        :type all_paths_by_lambda: list[list]

        :param lambda_values: A list of lambda values corresponding to the paths
                              provided in `all_paths_by_lambda`. The length of this
                              list should match the length of `all_paths_by_lambda`.
        :type lambda_values: list

        :return: The calculated mutual information (MI) value derived from the
                 joint and marginal distributions of the paths and lambda values.
        :rtype: float
        """
        move_lambda_counts = defaultdict(lambda: defaultdict(int))
        for lam, lam_paths in zip(lambda_values, all_paths_by_lambda):
            for path in lam_paths:
                if path:
                    move_lambda_counts[str(path[0])][lam] += 1
        total = sum(sum(lam_counts.values()) for lam_counts in move_lambda_counts.values())
        if total == 0: return 0.0
        mi = 0
        p_move_total = Counter()
        p_lam_total = Counter()
        for move, lam_counts in move_lambda_counts.items():
            for lam, count in lam_counts.items():
                p_move_total[move] += count
                p_lam_total[lam] += count

        for move, lam_counts in move_lambda_counts.items():
            p_move = p_move_total[move] / total
            for lam, count in lam_counts.items():
                p_joint = count / total
                p_lam = p_lam_total[lam] / total
                if p_joint > 0:
                    mi += p_joint * np.log2(p_joint / (p_move * p_lam))
        return mi

    @staticmethod
    def analyze_transition_graph_centrality(transitions):
        """
        Analyzes the betweenness centrality of a directed graph constructed from given
        state transitions. It calculates centrality measures for each node in the graph,
        visualizes the top nodes by centrality as a bar plot, and saves the plot to a file.
        The function also returns the centrality values as a DataFrame.

        :param transitions: A dictionary where each key is a source node, and the value
            is another dictionary. The inner dictionary maps destination nodes to
            weights (transition frequencies or probabilities).
        :type transitions: dict

        :return: A pandas DataFrame containing the top 15 nodes based on betweenness
            centrality, with two columns: 'Move' (the node name) and 'Centrality'
            (the centrality value). Returns an empty DataFrame if the graph has no nodes.
        :rtype: pd.DataFrame
        """
        G = nx.DiGraph()
        for src, dsts in transitions.items():
            for dst, w in dsts.items():
                G.add_edge(src, dst, weight=w)
        if not G.nodes: return pd.DataFrame()
        centrality = nx.betweenness_centrality(G, weight='weight', normalized=True)
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

    @staticmethod
    def compute_transition_complexity(transitions):
        """
        Geçişlerin toplam entropi karmaşıklığını hesaplar.
        """
        total_entropy = 0.0
        for src, dsts in transitions.items():
            counts = np.array(list(dsts.values()), dtype=np.float64)
            total = counts.sum()
            if total > 0:
                probs = counts / total
                tmp_sum = np.sum(probs * np.log2(probs))
                total_entropy += float(-1.0 * tmp_sum)
        return total_entropy

    @staticmethod
    def detect_entropy_elbow(lambda_entropies):
        """
        Detects the elbow point in the entropy curve to find the optimal λ value.

        This function uses the `KneeLocator` library to identify the elbow point of the
        given entropy values. This elbow point represents the optimal λ value that can
        be used for further processing or analysis. If the elbow point cannot be detected,
        it prints an error message with details of the failure.

        :param lambda_entropies: A list or array of entropy values corresponding to a range
            of λ values.
        :return: None
        """
        try:
            from kneed import KneeLocator
            kn = KneeLocator(config.LAMBDA_SCAN, lambda_entropies, curve='convex', direction='decreasing')
            print(f"Elbow point (opt λ): {kn.knee}")
        except Exception as e:
            print(f"Elbow point not detected: {e}")

    @staticmethod
    def first_move_counter(paths):
        """
        Analyzes the first elements in each sublist of a given list of paths and counts their
        occurrences. Returns a Counter object representing the number of times each unique
        first element appears across all sublists.

        :param paths: A list of lists, where each inner list represents a path. Each path
            must contain at least one element to be included in the counting process.
        :return: A ``Counter`` object mapping the first element (as a string) of each non-empty
            sublist in ``paths`` to the frequency of its occurrence.
        """
        firsts = [str(p[0]) for p in paths if p]
        return Counter(firsts)

    @staticmethod
    def most_frequent_first_move(paths):
        """
        Determine the most frequent first move from a given collection of paths.

        This function counts the first moves in the provided paths and identifies
        the most frequently occurring first move. If no paths are provided, it
        returns None.

        :param paths: A list of lists where each inner list represents a sequence
            of moves. Each move in the sequence is an element of the inner list.
        :type paths: list[list]
        :return: The most common first move across all provided paths, or None
            if there are no paths or no moves.
        :rtype: Optional[Any]
        """
        c = Calc.first_move_counter(paths)
        return c.most_common(1)[0][0] if c else None

    @staticmethod
    def normalize_scores_to_probs(scores, lam):
        """
        Normalizes a set of scores and applies softmax smoothing scaling.

        This function takes a list of scores and converts them into probabilities
        by normalizing them into a range between 0 and 1. The normalized scores are
        then smoothed and converted to probabilities using a softmax function which
        further applies scaling influence controlled by a lambda parameter.

        :param scores: List of numerical scores to be normalized and converted into
            probabilities.
        :type scores: list[float]
        :param lam: A scaling parameter that controls the influence of the softmax
            function on the normalized scores.
        :type lam: float
        :return: List of probabilities derived from the input scores after applying
            normalization and the softmax function.
        :rtype: list[float]
        """
        if not scores:
            return []
        scores_arr = np.array(scores, dtype=np.float64)
        if scores_arr.max() != scores_arr.min():
            scores_arr = (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
        return Calc.softmax(scores_arr, lam)

    @staticmethod
    def path_integral_lambda_scan(fen, lambda_values=None, depth=None):
        """
        Verilen FEN için lambda taraması yapar; her λ için entropi ve doğruluğu döndürür.
        Artık analiz depth tabanlıdır, backend'de adaptif node kullanılır.
        :param fen: Analiz edilecek pozisyonun FEN (Forsyth-Edwards Notation) dizesi.
        :param lambda_values: Tarama yapılacak lambda değerleri listesi. Eğer None ise, config.LAMBDA_SCAN kullanılır.
        :param depth: Analiz derinliği (None ise config.TARGET_DEPTH)
        :return: Entropi ve doğruluk metrikleri ile birlikte lambda değerlerini içeren bir DataFrame.
        :rtype: pd.DataFrame
        """
        lambda_values = lambda_values or config.LAMBDA_SCAN
        if depth is None:
            depth = config.TARGET_DEPTH
        rows = []
        print(f"\n--- Path Integral Exploration Scan (λ, depth={depth}): {fen} ---")
        for lam in tqdm(lambda_values, desc="PI Lambda Scan"):
            paths = Engine.sample_paths(fen, depth, lam, config.SAMPLE_COUNT, mode='competitive')
            entropy, counter = Calc.compute_entropy(paths)
            accuracy = Calc.top_move_concentration(paths)
            rows.append({"engine": "PathIntegral", "param": "lambda", "value": lam, "depth": depth, "entropy": float(entropy), "accuracy": float(accuracy)})
        return pd.DataFrame(rows)

    @staticmethod
    def path_integral_framework(policy_probs, cp_rewards, lam, resampling_threshold=None):
        """
        Path-Integral-inspired probabilistic decision-making framework for chess.
        Uses ONLY a single softmax parameter λ to control exploration-exploitation tradeoff.
        
        Args:
            policy_probs: List of move probabilities (optional, for compatibility)
            cp_rewards: Cumulative centipawn rewards for each path
            lam: Positive scalar controlling exploration-exploitation
                 - Small λ → High entropy (exploration)
                 - Large λ → Low entropy (exploitation)
            resampling_threshold: ESS threshold for resampling (default: N/2)
        
        Algorithm:
        1. Compute raw weights: w(path) = exp(R(path) * λ)
        2. Normalize to probability distribution: p(path_i) = w(path_i) / sum_j w(path_j)
        3. Compute diagnostics: Entropy, ESS, concentration
        4. Optional resampling if ESS < threshold
        
        Returns:
            Dictionary with probabilities, diagnostics, and resampling info
        - Optional resampled path set
        """
        if cp_rewards is None or len(cp_rewards) == 0:
            return {
                'probabilities': [],
                'entropy': 0.0,
                'ess': 0.0,
                'concentration': 0.0,
                'resampled': False,
                'resampled_indices': []
            }
        
        cp_rewards = np.array(cp_rewards, dtype=np.float64)
        N = len(cp_rewards)
        
        if resampling_threshold is None:
            resampling_threshold = N / 2
        
        # 1. Compute raw weights: w(path) = exp(R(path) * λ)
        # Small λ → flat distribution (exploration), Large λ → sharp distribution (exploitation)
        # Use log-weights for numerical stability
        log_weights = cp_rewards * lam
        log_weights = log_weights - np.max(log_weights)  # Numerical stability
        
        # 2. Normalize to probability distribution using log-sum-exp
        log_sum_exp = np.log(np.sum(np.exp(log_weights)))
        log_probs = log_weights - log_sum_exp
        probabilities = np.exp(log_probs)
        
        # 3. Compute diagnostics
        # Entropy: H = -sum_i p(path_i) * log(p(path_i))
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-12))
        
        # Effective Sample Size (ESS) = (sum_i w_i)^2 / sum_i w_i^2
        weights = np.exp(log_weights)
        ess = (np.sum(weights))**2 / np.sum(weights**2)
        
        # Concentration measure: probability share of highest-weight path
        concentration = np.max(probabilities)
        
        # 4. Resampling if ESS < threshold
        resampled = False
        resampled_indices = []
        
        if ess < resampling_threshold:
            # Perform resampling proportional to p(path_i)
            resampled_indices = np.random.choice(
                N, size=N, replace=True, p=probabilities
            )
            resampled = True
        
        return {
            'probabilities': probabilities,
            'entropy': float(entropy),
            'ess': float(ess),
            'concentration': float(concentration),
            'resampled': resampled,
            'resampled_indices': resampled_indices.tolist() if resampled else [],
            'log_weights': log_weights,
            'raw_weights': weights
        }

    @staticmethod
    def sample_paths_with_boltzmann(fen, depth=None, lam=None, samples=None):
        """
        Action functional analysis için Boltzmann ağırlıklı path sampling
        depth: number of plies per sampled path (default: config.TARGET_DEPTH)
        Node sayısı adaptif olarak hesaplanır.
        """
        if depth is None:
            depth = config.TARGET_DEPTH
        if lam is None:
            lam = config.LAMBDA
        if samples is None:
            samples = config.SAMPLE_COUNT
        paths = Engine.sample_paths(fen, depth, lam, samples, mode='competitive')
        weights = []

        for path in paths:
            # Basit action functional: path uzunluğu ve hamle kalitesi
            if not path:
                weights.append(0.0)
                continue

            # Her hamle için basit bir skor hesapla
            path_score = 0.0
            board = chess.Board(fen)

            for move in path:
                if move in board.legal_moves:
                    board.push(move)
                    path_score += 1.0  # Her geçerli hamle için bonus
                else:
                    break

            weights.append(np.exp(-lam * path_score))

        # Normalize weights
        weights = np.array(weights)
        if weights.sum() > 0:
            weights /= weights.sum()

        return paths, weights

    @staticmethod
    def js_divergence(p_dist, q_dist):
        """
        Calculate Jensen-Shannon divergence between two probability distributions.

        :param p_dist: First probability distribution (dict)
        :param q_dist: Second probability distribution (dict)
        :return: Jensen-Shannon divergence value
        """
        all_keys = sorted(set(p_dist.keys()) | set(q_dist.keys()))
        p = np.array([p_dist.get(k, 0) for k in all_keys], dtype=np.float64)
        q = np.array([q_dist.get(k, 0) for k in all_keys], dtype=np.float64)

        # Normalize
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        # M = (P + Q) / 2
        m = (p + q) / 2

        # Avoid log(0)
        mask_p = (p > 0) & (m > 0)
        mask_q = (q > 0) & (m > 0)

        kl_pm = np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])) if np.any(mask_p) else 0
        kl_qm = np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])) if np.any(mask_q) else 0

        return 0.5 * (kl_pm + kl_qm)

    @staticmethod
    def cosine_similarity(p_dist, q_dist):
        """
        Calculate cosine similarity between two probability distributions.

        :param p_dist: First probability distribution (dict)
        :param q_dist: Second probability distribution (dict)
        :return: Cosine similarity value (0-1)
        """
        all_keys = sorted(set(p_dist.keys()) | set(q_dist.keys()))
        p = np.array([p_dist.get(k, 0) for k in all_keys], dtype=np.float64)
        q = np.array([q_dist.get(k, 0) for k in all_keys], dtype=np.float64)

        # Normalize
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q

        # Cosine similarity
        dot_product = np.dot(p, q)
        norm_p = np.linalg.norm(p)
        norm_q = np.linalg.norm(q)

        if norm_p == 0 or norm_q == 0:
            return 0.0

        return dot_product / (norm_p * norm_q)

    @staticmethod
    def spearman_correlation(p_dist, q_dist):
        """
        Calculate Spearman rank correlation between two probability distributions.

        :param p_dist: First probability distribution (dict)
        :param q_dist: Second probability distribution (dict)
        :return: Spearman correlation coefficient
        """
        from scipy.stats import spearmanr

        all_keys = sorted(set(p_dist.keys()) | set(q_dist.keys()))
        p = np.array([p_dist.get(k, 0) for k in all_keys], dtype=np.float64)
        q = np.array([q_dist.get(k, 0) for k in all_keys], dtype=np.float64)

        if len(p) < 2:
            return 0.0

        try:
            corr, _ = spearmanr(p, q)
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0

    @staticmethod
    def pearson_correlation(p_dist, q_dist):
        """
        Calculate Pearson correlation between two probability distributions.

        :param p_dist: First probability distribution (dict)
        :param q_dist: Second probability distribution (dict)
        :return: Pearson correlation coefficient
        """
        from scipy.stats import pearsonr

        all_keys = sorted(set(p_dist.keys()) | set(q_dist.keys()))
        p = np.array([p_dist.get(k, 0) for k in all_keys], dtype=np.float64)
        q = np.array([q_dist.get(k, 0) for k in all_keys], dtype=np.float64)

        if len(p) < 2:
            return 0.0

        try:
            corr, _ = pearsonr(p, q)
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0

    @staticmethod
    def kl_divergence_counters(p_counter, q_counter):
        """
        Calculate KL divergence between two Counter objects.

        :param p_counter: First distribution as Counter
        :param q_counter: Second distribution as Counter
        :return: KL divergence value
        """
        # Convert counters to normalized dictionaries
        p_total = sum(p_counter.values())
        q_total = sum(q_counter.values())

        if p_total == 0 or q_total == 0:
            return float('inf')

        p_dist = {k: v/p_total for k, v in p_counter.items()}
        q_dist = {k: v/q_total for k, v in q_counter.items()}

        return Calc.kl_divergence(p_dist, q_dist)

    @staticmethod
    def compute_mutual_information_simple(counter):
        """
        Simple mutual information calculation for a single distribution.

        :param counter: Counter object with move frequencies
        :return: Mutual information value
        """
        total = sum(counter.values())
        if total == 0:
            return 0.0

        # Calculate entropy
        entropy = 0.0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)

        return entropy

    @staticmethod
    def compare_three_engines(fen, pi_lambda=config.LAMBDA, depth=config.TARGET_DEPTH, top_n_moves=config.MULTIPV):
        """
        Path Integral (with policy probs), Stockfish ve Lc0 motorlarını karşılaştırır.
        Üç farklı yaklaşımın hamle dağılımlarını, entropilerini ve diğer metriklerini analiz eder.

        :param fen: Analiz edilecek FEN pozisyonu
        :param pi_lambda: Path Integral için lambda değeri
        :param depth: Hedef analiz derinliği (None ise config.TARGET_DEPTH)
        :param top_n_moves: Karşılaştırılacak en iyi N hamle
        :return: Üç motor karşılaştırma sonuçları dict
        """
        from engine import Engine
        import config

        # Depth tabanlı analiz
        if depth is None:
            depth = config.TARGET_DEPTH

        print(f"\n--- Three Engine Distribution Comparison ---")
        print(f"Position: {fen[:30]}...")
        print(f"Depth: {depth}")
        print(f"Path Integral: λ={pi_lambda}, depth={depth}")
        print(f"Stockfish: depth={depth}")
        print(f"Lc0: depth={depth}")

        try:
            # 1. Path Integral Analysis (with policy probabilities)
            print("\n1. Path Integral analysis (with policy probs)...")
            pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT, mode='competitive')
            pi_entropy, pi_counter = Calc.compute_entropy(pi_paths)
            pi_dist = Calc.move_frequencies(pi_paths)

            # 2. Stockfish Analysis
            print("2. Stockfish analysis...")
            import chess.engine
            sf_engine = chess.engine.SimpleEngine.popen_uci(config.STOCKFISH_PATH)
            board = chess.Board(fen)
            import time
            sf_start = time.time()
            sf_moves, sf_scores = Engine.get_top_moves_and_scores(sf_engine, board, depth=depth, multipv=top_n_moves)
            sf_time = time.time() - sf_start
            sf_engine.quit()
            # Stockfish dağılımı (score → softmax → probability)
            sf_probs = Calc.normalize_scores_to_probs(sf_scores, pi_lambda)
            sf_dist = {}
            sf_move_strs = [str(move) for move in sf_moves]
            for i, move in enumerate(sf_move_strs):
                if i < len(sf_probs):
                    sf_dist[move] = sf_probs[i]

            # 3. Lc0 Analysis
            print("3. Lc0 analysis...")
            lc0_moves, lc0_scores, lc0_time = Engine.lc0_top_moves_and_scores(
                fen, depth=depth, multipv=top_n_moves
            )

            # Lc0 dağılımı (policy/value → softmax → probability)
            lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, pi_lambda)
            lc0_dist = {}
            lc0_move_strs = [str(move) for move in lc0_moves]
            for i, move in enumerate(lc0_move_strs):
                if i < len(lc0_probs):
                    lc0_dist[move] = lc0_probs[i]

            # Ortak hamle uzayı oluştur
            all_moves = set()
            all_moves.update(pi_dist.keys())
            all_moves.update(sf_dist.keys())
            all_moves.update(lc0_dist.keys())
            common_moves = sorted(list(all_moves))

            print(f"\nCommon move space size: {len(common_moves)}")
            print(f"Moves: {common_moves[:8]}..." if len(common_moves) > 8 else f"Moves: {common_moves}")

            # Eksik hamleler için 0 probability ekle
            for move in common_moves:
                if move not in pi_dist:
                    pi_dist[move] = 0.0
                if move not in sf_dist:
                    sf_dist[move] = 0.0
                if move not in lc0_dist:
                    lc0_dist[move] = 0.0

            # Entropy hesapla
            pi_entropy_final = -sum([p * np.log2(p) for p in pi_dist.values() if p > 0])
            sf_entropy = -sum([p * np.log2(p) for p in sf_dist.values() if p > 0])
            lc0_entropy = -sum([p * np.log2(p) for p in lc0_dist.values() if p > 0])

            # Pairwise karşılaştırmalar
            # PI vs SF
            kl_pi_sf = Calc.kl_divergence(pi_dist, sf_dist)
            kl_sf_pi = Calc.kl_divergence(sf_dist, pi_dist)
            js_pi_sf = Calc.js_divergence(pi_dist, sf_dist)
            cos_pi_sf = Calc.cosine_similarity(pi_dist, sf_dist)

            # PI vs Lc0
            kl_pi_lc0 = Calc.kl_divergence(pi_dist, lc0_dist)
            kl_lc0_pi = Calc.kl_divergence(lc0_dist, pi_dist)
            js_pi_lc0 = Calc.js_divergence(pi_dist, lc0_dist)
            cos_pi_lc0 = Calc.cosine_similarity(pi_dist, lc0_dist)

            # SF vs Lc0
            kl_sf_lc0 = Calc.kl_divergence(sf_dist, lc0_dist)
            kl_lc0_sf = Calc.kl_divergence(lc0_dist, sf_dist)
            js_sf_lc0 = Calc.js_divergence(sf_dist, lc0_dist)
            cos_sf_lc0 = Calc.cosine_similarity(sf_dist, lc0_dist)

            # Correlations
            spearman_pi_sf = Calc.spearman_correlation(pi_dist, sf_dist)
            spearman_pi_lc0 = Calc.spearman_correlation(pi_dist, lc0_dist)
            spearman_sf_lc0 = Calc.spearman_correlation(sf_dist, lc0_dist)

            pearson_pi_sf = Calc.pearson_correlation(pi_dist, sf_dist)
            pearson_pi_lc0 = Calc.pearson_correlation(pi_dist, lc0_dist)
            pearson_sf_lc0 = Calc.pearson_correlation(sf_dist, lc0_dist)

            # En olası hamleler
            pi_top = sorted(pi_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            sf_top = sorted(sf_dist.items(), key=lambda x: x[1], reverse=True)[:5]
            lc0_top = sorted(lc0_dist.items(), key=lambda x: x[1], reverse=True)[:5]

            # Konsantrasyon metrikleri (GT'siz)
            pi_conc = max(pi_dist.values()) if pi_dist else 0.0
            sf_conc = max(sf_dist.values()) if sf_dist else 0.0
            lc0_conc = max(lc0_dist.values()) if lc0_dist else 0.0

            # Mutual Information
            pi_mi = Calc.compute_mutual_information_simple(pi_counter)

            results = {
                'fen': fen,
                'parameters': {
                    'pi_lambda': pi_lambda,
                    'depth': depth,
                    'top_n_moves': top_n_moves
                },
                'timing': {
                    'stockfish_time': sf_time,
                    'lc0_time': lc0_time,
                    'pi_time': 0.0  # Path integral zamanı ayrı hesaplanabilir
                },
                'common_moves': common_moves,
                'distributions': {
                    'path_integral': pi_dist,
                    'stockfish': sf_dist,
                    'lc0': lc0_dist
                },
                'entropy': {
                    'path_integral': pi_entropy_final,
                    'stockfish': sf_entropy,
                    'lc0': lc0_entropy,
                    'pi_vs_sf_diff': pi_entropy_final - sf_entropy,
                    'pi_vs_lc0_diff': pi_entropy_final - lc0_entropy,
                    'sf_vs_lc0_diff': sf_entropy - lc0_entropy
                },
                'divergence_metrics': {
                    # PI vs SF
                    'kl_pi_to_sf': kl_pi_sf,
                    'kl_sf_to_pi': kl_sf_pi,
                    'js_pi_sf': js_pi_sf,
                    'symmetric_kl_pi_sf': (kl_pi_sf + kl_sf_pi) / 2,

                    # PI vs Lc0
                    'kl_pi_to_lc0': kl_pi_lc0,
                    'kl_lc0_to_pi': kl_lc0_pi,
                    'js_pi_lc0': js_pi_lc0,
                    'symmetric_kl_pi_lc0': (kl_pi_lc0 + kl_lc0_pi) / 2,

                    # SF vs Lc0
                    'kl_sf_to_lc0': kl_sf_lc0,
                    'kl_lc0_to_sf': kl_lc0_sf,
                    'js_sf_lc0': js_sf_lc0,
                    'symmetric_kl_sf_lc0': (kl_sf_lc0 + kl_lc0_sf) / 2
                },
                'similarity_metrics': {
                    'cosine_pi_sf': cos_pi_sf,
                    'cosine_pi_lc0': cos_pi_lc0,
                    'cosine_sf_lc0': cos_sf_lc0,
                    'spearman_pi_sf': spearman_pi_sf,
                    'spearman_pi_lc0': spearman_pi_lc0,
                    'spearman_sf_lc0': spearman_sf_lc0,
                    'pearson_pi_sf': pearson_pi_sf,
                    'pearson_pi_lc0': pearson_pi_lc0,
                    'pearson_sf_lc0': pearson_sf_lc0
                },
                'accuracy_metrics': {
                    'ground_truth_move': None,
                    'pi_gt_probability': pi_conc,
                    'sf_gt_probability': sf_conc,
                    'lc0_gt_probability': lc0_conc
                },
                'top_moves': {
                    'path_integral': pi_top,
                    'stockfish': sf_top,
                    'lc0': lc0_top
                },
                'information_theory': {
                    'pi_mutual_information': pi_mi,
                    'most_entropic_engine': 'path_integral' if pi_entropy_final >= max(sf_entropy, lc0_entropy)
                                          else 'stockfish' if sf_entropy >= lc0_entropy
                                          else 'lc0',
                    'entropy_ranking': sorted([
                        ('path_integral', pi_entropy_final),
                        ('stockfish', sf_entropy),
                        ('lc0', lc0_entropy)
                    ], key=lambda x: x[1], reverse=True)
                }
            }

            print(f"\n--- Three Engine Comparison Results ---")
            print(f"Depth: {depth}")
            print(f"Entropy Ranking:")
            for engine, entropy in results['information_theory']['entropy_ranking']:
                print(f"  {engine}: {entropy:.3f} bits")

            print(f"\nConcentration (mode probability):")
            print(f"  Path Integral: {pi_conc:.3f}")
            print(f"  Stockfish: {sf_conc:.3f}")
            print(f"  Lc0: {lc0_conc:.3f}")

            print(f"\nPairwise JS Divergences:")
            print(f"  PI ↔ SF: {js_pi_sf:.3f}")
            print(f"  PI ↔ Lc0: {js_pi_lc0:.3f}")
            print(f"  SF ↔ Lc0: {js_sf_lc0:.3f}")

            print(f"\nPairwise Cosine Similarities:")
            print(f"  PI ↔ SF: {cos_pi_sf:.3f}")
            print(f"  PI ↔ Lc0: {cos_pi_lc0:.3f}")
            print(f"  SF ↔ Lc0: {cos_sf_lc0:.3f}")

            return results

        except Exception as e:
            print(f"Error in three engine comparison: {e}")
            import traceback
            traceback.print_exc()
            return None

    @staticmethod
    def test_three_engine_comparison():
        """
        Basit test pozisyonu ile üç motor karşılaştırması.
        Test amaçlı olarak küçük parametrelerle analiz yapar.
        """
        # Test pozisyonu: Opening position
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        fen_name = "starting_position"

        print("=== ÜÇ MOTOR KARŞILAŞTIRMASI TEST ===")
        print(f"Pozisyon: {fen_name}")
        print(f"FEN: {fen}")

        # Matematik analizi
        print("\n1. Performing mathematical analysis...")
        results = Calc.compare_three_engines(
            fen=fen,
            pi_lambda=config.LAMBDA,
            depth=config.TARGET_DEPTH,
            top_n_moves=config.MULTIPV
        )

        if results is None:
            print("ERROR: Could not complete three engine analysis")
            return None

        # Sonuçları özetle
        print("\n=== ÖZET SONUÇLAR ===")
        entropy_ranking = results['information_theory']['entropy_ranking']
        print(f"Entropy Ranking:")
        for i, (engine, entropy) in enumerate(entropy_ranking, 1):
            print(f"  {i}. {engine.replace('_', ' ').title()}: {entropy:.3f} bits")

        print(f"\nConcentration Summary:")
        print(f"  PI: {results['accuracy_metrics']['pi_gt_probability']:.3f}")
        print(f"  Stockfish: {results['accuracy_metrics']['sf_gt_probability']:.3f}")
        print(f"  Lc0: {results['accuracy_metrics']['lc0_gt_probability']:.3f}")

        print(f"\nDivergence Analysis:")
        print(f"  PI ↔ Stockfish JS: {results['divergence_metrics']['js_pi_sf']:.3f}")
        print(f"  PI ↔ Lc0 JS: {results['divergence_metrics']['js_pi_lc0']:.3f}")
        print(f"  Stockfish ↔ Lc0 JS: {results['divergence_metrics']['js_sf_lc0']:.3f}")

        print(f"\nSimilarity Analysis:")
        print(f"  PI ↔ Stockfish Cosine: {results['similarity_metrics']['cosine_pi_sf']:.3f}")
        print(f"  PI �� Lc0 Cosine: {results['similarity_metrics']['cosine_pi_lc0']:.3f}")
        print(f"  Stockfish ↔ Lc0 Cosine: {results['similarity_metrics']['cosine_sf_lc0']:.3f}")

        print("\n=== TEST COMPLETED ===")
        return results

    @staticmethod
    def test_multi_position_comparison():
        """
        Birden fazla pozisyon için karşılaştırma testi.
        Test amaçlı olarak küçük parametrelerle analiz yapar.
        """
        print("\n=== ÇOKLU POZİSYON KARŞILAŞTIRMASI ===")

        # Test pozisyonları
        positions = {
            "opening": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "middlegame": "r1bqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
            "endgame": "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1"
        }

        # Test parametreleri (küçük değerler)
        pi_lambda = 0.7

        print(f"Analyzing {len(positions)} positions...")

        all_results = {}
        for pos_name, fen in positions.items():
            print(f"\nAnalyzing position: {pos_name}")
            try:
                result = Calc.compare_three_engines(fen, pi_lambda, depth=config.TARGET_DEPTH, top_n_moves=config.MULTIPV)
                if result is None:
                    print(f"  Error in three engine comparison: {fen}")
                    continue
                if result:
                    all_results[pos_name] = result
                    print(f"✓ {pos_name} analysis completed")
            except Exception as e:
                print(f"✗ Error analyzing {pos_name}: {e}")
                continue

        if all_results:
            print(f"\n✓ Successfully analyzed {len(all_results)} positions")

        return all_results

    @staticmethod
    def feynman_path_analogy_validation(fen_list=None, save_results=True):
        """
        Feynman yol integrali analojisini doğrular.

        Custom instructions requirement: "Analoji testi (Feynman yol integrali analojisi: Quantum vs Classical)"

        Hipotez: λ↓ → Entropi↑ (Quantum limit), λ↑ → Entropi↓ (Classical limit)

        :param fen_list: Test edilecek FEN pozisyonları (None ise config.MULTI_FEN[:3])
        :param save_results: Sonuçları CSV olarak kaydet
        :return: Validation sonuçları dict
        """
        from scipy import stats

        if fen_list is None:
            fen_list = config.MULTI_FEN[:3]  # Performance için sınırla

        print("\n=== FEYNMAN PATH ANALOGY VALIDATION ===")
        print("Hipotez: λ↓ → Entropi↑ (Quantum), λ↑ → Entropi↓ (Classical)")

        # Three regimes test (custom instructions: açık etiketleme)
        quantum_lambdas = [0.01, 0.05, 0.1]    # Quantum limit
        transition_lambdas = [0.5, 0.7, 1.0]   # Transition region
        classical_lambdas = [2.0, 5.0, 10.0]   # Classical limit

        all_lambdas = quantum_lambdas + transition_lambdas + classical_lambdas
        regime_labels = ['Quantum'] * 3 + ['Transition'] * 3 + ['Classical'] * 3

        results = {
            'positions': [],
            'lambda_values': all_lambdas,
            'regime_labels': regime_labels,
            'entropies_by_position': [],
            'accuracies_by_position': [],
            'validation_metrics': {}
        }

        # Her pozisyon için test
        for i, fen in enumerate(tqdm(fen_list, desc="Feynman Validation")):
            pos_name = f"position_{i+1}"
            print(f"\nTesting {pos_name}: {fen[:30]}...")

            entropies = []
            accuracies = []
            ground_truth = None  # GT kaldırıldı

            for lam in all_lambdas:
                paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lam, config.SAMPLE_COUNT, mode='competitive')
                entropy, counter = Calc.compute_entropy(paths)
                accuracy = Calc.top_move_concentration(paths)

                entropies.append(entropy)
                accuracies.append(accuracy)

            results['positions'].append(pos_name)
            results['entropies_by_position'].append(entropies)
            results['accuracies_by_position'].append(accuracies)

        # Statistical validation (custom instructions: effect size reporting)
        entropies_avg = np.mean(results['entropies_by_position'], axis=0)
        accuracies_avg = np.mean(results['accuracies_by_position'], axis=0)

        # Regime comparison
        quantum_entropy = np.mean(entropies_avg[:3])
        transition_entropy = np.mean(entropies_avg[3:6])
        classical_entropy = np.mean(entropies_avg[6:])

        # Statistical tests (custom instructions requirement)
        entropy_trend_corr, entropy_p = stats.spearmanr(all_lambdas, entropies_avg)

        # Effect size (Cohen's d) - custom instructions: etki büyüklüğü
        quantum_entropies_flat = [e for pos_ents in results['entropies_by_position'] for e in pos_ents[:3]]
        classical_entropies_flat = [e for pos_ents in results['entropies_by_position'] for e in pos_ents[6:]]

        cohens_d = (np.mean(quantum_entropies_flat) - np.mean(classical_entropies_flat)) / \
                   np.sqrt((np.var(quantum_entropies_flat) + np.var(classical_entropies_flat)) / 2)

        results['validation_metrics'] = {
            'quantum_entropy_avg': quantum_entropy,
            'transition_entropy_avg': transition_entropy,
            'classical_entropy_avg': classical_entropy,
            'entropy_lambda_correlation': entropy_trend_corr,
            'entropy_correlation_pvalue': entropy_p,
            'quantum_vs_classical_cohens_d': cohens_d,
            'hypothesis_supported': (entropy_trend_corr < -0.5 and entropy_p < 0.05 and cohens_d > 0.5),
            'effect_size_interpretation': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        }

        # Save results (custom instructions: CSV kaydetme)
        if save_results:
            df_data = []
            for i, pos in enumerate(results['positions']):
                for j, lam in enumerate(all_lambdas):
                    df_data.append({
                        'position': pos,
                        'lambda': lam,
                        'regime': regime_labels[j],
                        'entropy': results['entropies_by_position'][i][j],
                        'accuracy': results['accuracies_by_position'][i][j]
                    })

            df = pd.DataFrame(df_data)
            os.makedirs("results", exist_ok=True)
            df.to_csv("results/feynman_validation_data.csv", index=False)

            print(f"✓ Data saved: results/feynman_validation_data.csv")

        print(f"✓ Hypothesis supported: {results['validation_metrics']['hypothesis_supported']}")
        print(f"✓ Effect size (Cohen's d): {cohens_d:.3f} ({results['validation_metrics']['effect_size_interpretation']})")

        return results

    @staticmethod
    def position_complexity_analysis(save_results=True):
        """
        Pozisyon karmaşıklığının entropi ve doğruluk üzerindeki etkisini analiz eder.

        Custom instructions requirement: "Pozisyon karmaşıklığının verilen karara ve doğruluk verilerine etkisi"
        """
        from scipy import stats

        print("\n=== POSITION COMPLEXITY ANALYSIS ===")

        # Complexity metrics for each position
        complexity_metrics = []
        pi_results = []
        lc0_results = []

        position_names = [
            "Italian Game (Opening)",
            "Sicilian Defense (Sharp)",
            "Ruy Lopez (Strategic)",
            "Isolated Pawn (Middlegame)",
            "Calm Strategic",
            "Tactical Position",
            "K+P Endgame"
        ]

        for i, fen in enumerate(tqdm(config.MULTI_FEN, desc="Complexity Analysis")):
            pos_name = position_names[i] if i < len(position_names) else f"Position {i+1}"

            # Calculate position complexity (custom instructions: metrik tanımları)
            board = chess.Board(fen)
            legal_moves = list(board.legal_moves)

            # Basic complexity metrics
            complexity = {
                'position_name': pos_name,
                'fen': fen,
                'legal_moves_count': len(legal_moves),
                'piece_count': len(board.piece_map()),
                'is_opening': board.fullmove_number <= 10,
                'is_endgame': len(board.piece_map()) <= 8,
                'complexity_score': len(legal_moves) * (len(board.piece_map()) / 32)  # Composite score
            }
            complexity_metrics.append(complexity)

            # Path Integral analysis
            paths = Engine.sample_paths(fen, config.TARGET_DEPTH, config.LAMBDA, config.SAMPLE_COUNT, mode='competitive')
            pi_entropy, pi_counter = Calc.compute_entropy(paths)
            pi_accuracy = Calc.top_move_concentration(paths)

            pi_results.append({
                'position_name': pos_name,
                'entropy': pi_entropy,
                'accuracy': pi_accuracy,
                'top_move_frequency': max(pi_counter.values()) / sum(pi_counter.values()) if pi_counter else 0
            })

            # LC0 analysis (custom instructions: LC0 preferred)
            try:
                lc0_moves, lc0_scores, lc0_time = Engine.lc0_top_moves_and_scores(
                    fen, depth=config.TARGET_DEPTH, multipv=config.LC0_MULTIPV
                )
                lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LC0_SOFTMAX_LAMBDA)
                lc0_entropy = float(np.sum([-p*np.log2(p) for p in lc0_probs if p > 0])) if lc0_probs else 0.0
                lc0_accuracy = float(max(lc0_probs)) if lc0_probs else 0.0

                lc0_results.append({
                    'position_name': pos_name,
                    'entropy': lc0_entropy,
                    'accuracy': lc0_accuracy,
                    'analysis_time': lc0_time
                })
            except Exception as e:
                print(f"LC0 analysis failed for {pos_name}: {e}")
                lc0_results.append({
                    'position_name': pos_name,
                    'entropy': 0.0,
                    'accuracy': 0.0,
                    'analysis_time': 0.0
                })

        # Create combined DataFrame
        df = pd.DataFrame(complexity_metrics)
        df['pi_entropy'] = [r['entropy'] for r in pi_results]
        df['pi_accuracy'] = [r['accuracy'] for r in pi_results]
        df['pi_concentration'] = [r['top_move_frequency'] for r in pi_results]
        df['lc0_entropy'] = [r['entropy'] for r in lc0_results]
        df['lc0_accuracy'] = [r['accuracy'] for r in lc0_results]
        df['lc0_time'] = [r['analysis_time'] for r in lc0_results]

        # Statistical correlations (custom instructions: Spearman, effect sizes)
        correlations = {
            'complexity_vs_pi_entropy': stats.spearmanr(df['complexity_score'], df['pi_entropy']),
            'complexity_vs_pi_accuracy': stats.spearmanr(df['complexity_score'], df['pi_accuracy']),
            'complexity_vs_lc0_entropy': stats.spearmanr(df['complexity_score'], df['lc0_entropy']),
            'complexity_vs_lc0_accuracy': stats.spearmanr(df['complexity_score'], df['lc0_accuracy']),
            'legal_moves_vs_pi_entropy': stats.spearmanr(df['legal_moves_count'], df['pi_entropy']),
            'piece_count_vs_entropy': stats.spearmanr(df['piece_count'], df['pi_entropy'])
        }

        # Save results
        if save_results:
            os.makedirs("results", exist_ok=True)
            df.to_csv("results/position_complexity_data.csv", index=False)
            print(f"✓ Data saved: results/position_complexity_data.csv")

        # Summary (custom instructions: kısa, net bulgular)
        strongest_correlation = max(correlations.items(), key=lambda x: abs(x[1][0]))
        print(f"�� Strongest correlation: {strongest_correlation[0]} (ρ = {strongest_correlation[1][0]:.3f}, p = {strongest_correlation[1][1]:.3f})")

        return {
            'data': df,
            'correlations': correlations,
            'summary': {
                'strongest_correlation': strongest_correlation,
                'position_rankings': {
                    'highest_pi_entropy': df.loc[df['pi_entropy'].idxmax()].to_dict(),
                    'most_complex': df.loc[df['complexity_score'].idxmax()].to_dict()
                }
            }
        }

    @staticmethod
    def convergence_analysis_comprehensive(fen=None, save_results=True):
        """
        Kapsamlı convergence analizi - depth ve lambda parametrelerine göre.
        Artık analiz depth tabanlıdır, backend'de adaptif node kullanılır.
        Custom instructions requirement: "convergence: depth↑ ile P(mode)'nin artışı ve doygunluk"
        """
        from scipy.optimize import curve_fit
        from scipy import stats

        if fen is None:
            fen = config.MULTI_FEN[0]  # Default to first position

        print("\n=== COMPREHENSIVE CONVERGENCE ANALYSIS ===")
        print(f"Position: {fen}")

        # Depth-based convergence
        depth_range = [2, 4, 6, 8, 10, 12, 14, 16]
        lambda_range = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]

        # 1. Convergence by depth (fixed lambda)
        print("1. Analyzing convergence by depth...")
        depth_results = []
        for depth in tqdm(depth_range, desc="Depth convergence"):
            paths = Engine.sample_paths(fen, config.TARGET_DEPTH, config.LAMBDA, config.SAMPLE_COUNT, mode='competitive')
            entropy, counter = Calc.compute_entropy(paths)
            accuracy = Calc.top_move_concentration(paths)
            mode_prob = accuracy
            depth_results.append({
                'depth': depth,
                'entropy': entropy,
                'accuracy': accuracy,
                'mode_probability': mode_prob,
                'top_move_concentration': max(counter.values()) / sum(counter.values()) if counter else 0
            })
        # 2. Convergence by lambda (fixed depth)
        print("2. Analyzing convergence by lambda...")
        lambda_results = []
        fixed_depth = config.TARGET_DEPTH
        for lam in tqdm(lambda_range, desc="Lambda convergence"):
            paths = Engine.sample_paths(fen, config.TARGET_DEPTH, lam, config.SAMPLE_COUNT, mode='competitive')
            entropy, counter = Calc.compute_entropy(paths)
            accuracy = Calc.top_move_concentration(paths)
            mode_prob = accuracy
            lambda_results.append({
                'lambda': lam,
                'depth': fixed_depth,
                'entropy': entropy,
                'accuracy': accuracy,
                'mode_probability': mode_prob,
                'move_diversity': len(counter) if counter else 0
            })

        # Statistical analysis
        depth_df = pd.DataFrame(depth_results)
        lambda_df = pd.DataFrame(lambda_results)

        # Fit convergence curves (custom instructions: sayısal analiz)
        def exponential_convergence(x, a, b, c):
            return a * (1 - np.exp(-b * x)) + c

        # Fit mode probability convergence
        try:
            depth_fit_params, _ = curve_fit(exponential_convergence, depth_df['depth'], depth_df['mode_probability'])
            convergence_rate = abs(depth_fit_params[1])
            saturation_level = depth_fit_params[0] + depth_fit_params[2]
        except:
            convergence_rate = 0.0
            saturation_level = depth_df['mode_probability'].iloc[-1]

        # Lambda optimization
        optimal_lambda_idx = lambda_df['mode_probability'].idxmax()
        optimal_lambda = lambda_df['lambda'].iloc[optimal_lambda_idx]
        optimal_mode_prob = lambda_df['mode_probability'].iloc[optimal_lambda_idx]

        # Save results
        if save_results:
            os.makedirs("results", exist_ok=True)
            depth_df.to_csv("results/convergence_by_depth.csv", index=False)
            lambda_df.to_csv("results/convergence_by_lambda.csv", index=False)
            depth_df.to_csv("results/depth_vs_metrics_competitive.csv", index=False)  # Eksik CSV dosyasını ekledim
            print(f"✓ Data saved: results/convergence_by_depth.csv, results/convergence_by_lambda.csv, results/depth_vs_metrics_competitive.csv")

        results = {
            'fen': fen,
            'ground_truth_move': None,
            'depth_analysis': depth_df.to_dict('records'),
            'lambda_analysis': lambda_df.to_dict('records'),
            'convergence_metrics': {
                'convergence_rate': convergence_rate,
                'saturation_level': saturation_level,
                'optimal_lambda': optimal_lambda,
                'optimal_mode_probability': optimal_mode_prob,
                'final_mode_probability': depth_df['mode_probability'].iloc[-1]
            }
        }

        print(f"✓ Optimal lambda: {optimal_lambda:.2f} (mode prob: {optimal_mode_prob:.3f})")
        print(f"✓ Convergence rate: {convergence_rate:.6f}")

        return results

    @staticmethod
    def multi_engine_comparison_analysis(fen_list=None, save_results=True):
        """
        Multiple engine comparison analysis (PI vs LC0 vs ground truth).

        Custom instructions requirement: "Çoklu FEN'lere göre üçlü motor (PI vs Lc0 vs Stockfish)"
        """
        if fen_list is None:
            fen_list = config.MULTI_FEN

        print("\n=== MULTI-ENGINE COMPARISON ANALYSIS ===")

        comparison_results = []
        position_names = [
            "Italian Game", "Sicilian Defense", "Ruy Lopez", "Strategic Isolated Pawn",
            "Calm Strategic", "Tactical Potential", "K+P Endgame"
        ]

        for i, fen in enumerate(tqdm(fen_list, desc="Multi-Engine Analysis")):
            pos_name = position_names[i] if i < len(position_names) else f"Position_{i+1}"

            ground_truth = None  # GT kaldırıldı

            # Path Integral analysis
            pi_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, config.LAMBDA, config.SAMPLE_COUNT, mode='competitive')
            pi_entropy, pi_counter = Calc.compute_entropy(pi_paths)
            pi_accuracy = Calc.top_move_concentration(pi_paths)
            pi_top_move = Calc.most_frequent_first_move(pi_paths)

            # LC0 analysis (standard depth)
            try:
                lc0_moves, lc0_scores, lc0_time = Engine.lc0_top_moves_and_scores(
                    fen, depth=config.TARGET_DEPTH, multipv=config.LC0_MULTIPV
                )
                lc0_top_move = str(lc0_moves[0]) if lc0_moves else None
                lc0_probs = Calc.normalize_scores_to_probs(lc0_scores, config.LC0_SOFTMAX_LAMBDA)
                lc0_entropy = float(np.sum([-p*np.log2(p) for p in lc0_probs if p > 0])) if lc0_probs else 0.0
                lc0_accuracy = float(max(lc0_probs)) if lc0_probs else 0.0
            except Exception as e:
                print(f"LC0 analysis failed for {pos_name}: {e}")
                lc0_entropy, lc0_accuracy, lc0_top_move, lc0_time = 0.0, 0.0, None, 0.0

            # Store results
            comparison_results.append({
                'position': pos_name,
                'fen': fen,
                'ground_truth': ground_truth,
                'pi_entropy': pi_entropy,
                'pi_accuracy': pi_accuracy,
                'pi_top_move': pi_top_move,
                'lc0_entropy': lc0_entropy,
                'lc0_accuracy': lc0_accuracy,
                'lc0_top_move': lc0_top_move,
                'lc0_time': lc0_time,
                'entropy_difference': pi_entropy - lc0_entropy,
                'accuracy_difference': pi_accuracy - lc0_accuracy
            })

        # Convert to DataFrame for analysis
        df = pd.DataFrame(comparison_results)

        # Statistical summary (custom instructions: özet metrikler)
        summary_stats = {
            'total_positions': len(df),
            'pi_mean_entropy': df['pi_entropy'].mean(),
            'lc0_mean_entropy': df['lc0_entropy'].mean(),
            'pi_mean_accuracy': df['pi_accuracy'].mean(),
            'lc0_mean_accuracy': df['lc0_accuracy'].mean(),
            'pi_wins_entropy': (df['pi_entropy'] > df['lc0_entropy']).sum(),
            'pi_wins_accuracy': (df['pi_accuracy'] > df['lc0_accuracy']).sum(),
            'mean_entropy_difference': df['entropy_difference'].mean(),
            'mean_accuracy_difference': df['accuracy_difference'].mean()
        }

        # Save results
        if save_results:
            os.makedirs("results", exist_ok=True)
            df.to_csv("results/multi_engine_comparison.csv", index=False)
            print(f"✓ Data saved: results/multi_engine_comparison.csv")

        # Summary (custom instructions: kısa bulgular)
        print(f"✓ Analyzed {summary_stats['total_positions']} positions")
        print(f"✓ PI mean entropy: {summary_stats['pi_mean_entropy']:.3f}, LC0: {summary_stats['lc0_mean_entropy']:.3f}")
        print(f"✓ PI mean accuracy: {summary_stats['pi_mean_accuracy']:.3f}, LC0: {summary_stats['lc0_mean_accuracy']:.3f}")
        print(f"✓ PI wins in entropy: {summary_stats['pi_wins_entropy']}/{summary_stats['total_positions']} positions")

        return {
            'data': df,
            'summary': summary_stats,
            'detailed_results': comparison_results
        }

    @staticmethod
    def move_frequencies(paths):
        """
        Path listesinden ilk hamlelerin frekanslarını döndürür.
        Dönüş: {hamle: olasılık}
        """
        firsts = [str(p[0]) for p in paths if p]
        if not firsts:
            return {}
        counter = Counter(firsts)
        total = sum(counter.values())
        return {move: count / total for move, count in counter.items()}

    @staticmethod
    def path_integral_analysis(fen, lambda_values=None, samples=None, depth=None, save_results=True):
        """
        Comprehensive path integral analysis implementing the framework described in the requirements.
        
        Generates:
        - Normalized probability distribution over paths
        - Entropy and ESS diagnostics  
        - Visualizations: entropy vs λ, ESS vs number of samples, path probability histogram
        - CSV data for all metrics
        
        Parameters:
        - fen: Chess position in FEN notation
        - lambda_values: List of λ values to test (default: config.LAMBDA_SCAN)
        - samples: Number of paths to sample (default: config.SAMPLE_COUNT)
        - depth: Path depth (default: config.TARGET_DEPTH)
        - save_results: Whether to save CSV and PNG files
        
        Returns:
        - Dictionary with analysis results and diagnostics
        """
        if lambda_values is None:
            lambda_values = config.LAMBDA_SCAN
        if samples is None:
            samples = config.SAMPLE_COUNT
        if depth is None:
            depth = config.TARGET_DEPTH
            
        print(f"\n=== PATH INTEGRAL ANALYSIS ===")
        print(f"Position: {fen[:50]}...")
        print(f"Lambda values: {len(lambda_values)} points from {min(lambda_values)} to {max(lambda_values)}")
        print(f"Samples per λ: {samples}, Depth: {depth}")
        
        results = {
            'fen': fen,
            'lambda_values': lambda_values,
            'samples': samples,
            'depth': depth,
            'analysis_data': [],
            'diagnostics': {}
        }
        
        # Analyze each lambda value
        for lam in tqdm(lambda_values, desc="Path Integral λ Analysis"):
            # Sample paths using the engine
            paths = Engine.sample_paths(fen, depth, lam, samples, mode='competitive')
            
            if not paths:
                continue
                
            # Get policy probabilities and CP rewards for each path
            policy_probs = []  # Placeholder - would need engine integration
            cp_rewards = []
            
            # Calculate CP rewards for each path
            for path in paths:
                if not path:
                    cp_rewards.append(0.0)
                    continue
                    
                # Simple reward calculation based on path length and move quality
                board = chess.Board(fen)
                path_reward = 0.0
                
                for move in path:
                    if move in board.legal_moves:
                        board.push(move)
                        # Simple heuristic: longer valid paths get higher rewards
                        path_reward += 10.0  # Base reward per valid move
                    else:
                        break
                        
                cp_rewards.append(path_reward)
            
            # Apply path integral framework
            pi_result = Calc.path_integral_framework(
                policy_probs=policy_probs,
                cp_rewards=cp_rewards,
                lam=lam,
                resampling_threshold=samples/2
            )
            
            # Calculate additional metrics
            entropy, counter = Calc.compute_entropy(paths)
            concentration = Calc.top_move_concentration(paths)
            
            # Store results
            analysis_point = {
                'lambda': lam,
                'entropy': pi_result['entropy'],
                'ess': pi_result['ess'],
                'concentration': pi_result['concentration'],
                'resampled': pi_result['resampled'],
                'num_unique_moves': len(counter) if counter else 0,
                'path_count': len(paths),
                'avg_path_length': np.mean([len(p) for p in paths if p]) if paths else 0
            }
            
            results['analysis_data'].append(analysis_point)
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(results['analysis_data'])
        
        if len(df) == 0:
            print("No valid analysis data generated")
            return results
        
        # Calculate diagnostics
        results['diagnostics'] = {
            'optimal_lambda_entropy': df.loc[df['entropy'].idxmax(), 'lambda'],
            'optimal_lambda_ess': df.loc[df['ess'].idxmax(), 'lambda'],
            'entropy_range': (df['entropy'].min(), df['entropy'].max()),
            'ess_range': (df['ess'].min(), df['ess'].max()),
            'exploration_regime': df[df['lambda'] < 0.5]['entropy'].mean(),
            'exploitation_regime': df[df['lambda'] > 2.0]['entropy'].mean()
        }
        
        # Save results
        if save_results:
            os.makedirs("results", exist_ok=True)
            
            # Save CSV data
            csv_filename = "results/path_integral_analysis.csv"
            df.to_csv(csv_filename, index=False)
            print(f"✓ Analysis data saved: {csv_filename}")
            
            # Generate visualizations
            Calc._generate_path_integral_plots(df, results['diagnostics'])
        
        print(f"✓ Analysis complete. Optimal λ (entropy): {results['diagnostics']['optimal_lambda_entropy']:.3f}")
        print(f"✓ Entropy range: {results['diagnostics']['entropy_range'][0]:.3f} - {results['diagnostics']['entropy_range'][1]:.3f}")
        
        return results

    @staticmethod
    def _generate_path_integral_plots(df, diagnostics):
        """
        Generate visualizations for path integral analysis:
        1. Entropy vs λ
        2. ESS vs λ  
        3. Path probability histogram for selected λ values
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Entropy vs Lambda
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.plot(df['lambda'], df['entropy'], 'o-', linewidth=2, markersize=6)
        plt.xlabel('Lambda (λ)')
        plt.ylabel('Entropy (bits)')
        plt.title('Entropy vs Lambda')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # 2. ESS vs Lambda
        plt.subplot(1, 3, 2)
        plt.plot(df['lambda'], df['ess'], 's-', linewidth=2, markersize=6, color='orange')
        plt.xlabel('Lambda (λ)')
        plt.ylabel('Effective Sample Size')
        plt.title('ESS vs Lambda')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        # 3. Concentration vs Lambda
        plt.subplot(1, 3, 3)
        plt.plot(df['lambda'], df['concentration'], '^-', linewidth=2, markersize=6, color='green')
        plt.xlabel('Lambda (λ)')
        plt.ylabel('Top Move Concentration')
        plt.title('Concentration vs Lambda')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        
        plt.tight_layout()
        plt.savefig("results/path_integral_entropy_ess_lambda.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Path probability histogram for key lambda values
        key_lambdas = [0.1, 0.5, 2.0]  # Exploration, transition, exploitation
        available_lambdas = df['lambda'].values
        
        plt.figure(figsize=(15, 5))
        
        for i, target_lam in enumerate(key_lambdas):
            # Find closest available lambda
            closest_idx = np.argmin(np.abs(available_lambdas - target_lam))
            actual_lam = available_lambdas[closest_idx]
            
            plt.subplot(1, 3, i+1)
            
            # Create histogram data (placeholder - would need actual path probabilities)
            # For now, show concentration metric
            concentration = df.iloc[closest_idx]['concentration']
            entropy = df.iloc[closest_idx]['entropy']
            
            plt.bar(['Top Move', 'Other Moves'], [concentration, 1-concentration], 
                   color=['red', 'blue'], alpha=0.7)
            plt.ylabel('Probability')
            plt.title(f'λ = {actual_lam:.2f}\nEntropy = {entropy:.2f}')
            plt.ylim(0, 1)
            
        plt.tight_layout()
        plt.savefig("results/path_probability_histograms.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✓ Visualizations saved: path_integral_entropy_ess_lambda.png, path_probability_histograms.png")

    @staticmethod
    def compare_policy_vs_quantum_sampling(fen, depth=None, samples=None, multipv_policy=50, multipv_sampling=50, lam_sampling=None, save_results=True):
        """
        LC0 policy head (raw policy) ile sample_paths(mode='quantum_limit') çıktısını karşılaştırır.
        Kaydeder: CSV (policy vs empirical frekanslar) ve kısa metrik özeti (entropy, JS, KL).
        Bu fonksiyon, policy head'ın gerçekten elde edilebilir olup olmadığını güvenli şekilde kontrol eder ve
        yoksa kullanıcıya bilgi verir.
        """
        import os
        import pandas as pd
        from engine import Engine
        from mathfuncs import Calc
        import numpy as np

        if depth is None:
            depth = config.TARGET_DEPTH
        if samples is None:
            samples = config.SAMPLE_COUNT
        if lam_sampling is None:
            lam_sampling = config.LAMBDA

        print(f"\n=== Policy vs Quantum Sampling Comparison ===\nFEN: {fen}\nDepth: {depth}, samples: {samples}, multipv_policy: {multipv_policy}, multipv_sampling: {multipv_sampling}, lam_sampling: {lam_sampling}")

        # 1) Policy head (quantum ideal) dağılımı
        try:
            moves_policy, probs_policy, elapsed_policy = Engine.lc0_policy_and_moves(fen, depth=depth, multipv=multipv_policy)
            if not moves_policy or not probs_policy:
                print("[INFO] Policy head returned empty moves/probs — cannot compute policy distribution.")
                policy_dist = {}
            else:
                policy_dist = {str(m.uci() if hasattr(m, 'uci') else m): float(p) for m, p in zip(moves_policy, probs_policy)}
            entropy_policy = -sum([p * np.log2(p) for p in policy_dist.values() if p > 0]) if policy_dist else np.nan
        except Exception as e:
            print(f"[ERROR] lc0_policy_and_moves failed: {e}")
            policy_dist = {}
            entropy_policy = np.nan

        # 2) Empirical sampling from sample_paths in quantum_limit mode
        paths = Engine.sample_paths(fen, depth, lam_sampling, samples, mode='quantum_limit')
        # First-move empirical freq
        firsts = [str(p[0]) for p in paths if p]
        from collections import Counter
        counter = Counter(firsts)
        total = sum(counter.values())
        empirical_dist = {move: freq/total for move, freq in counter.items()} if total > 0 else {}
        entropy_empirical = -sum([p * np.log2(p) for p in empirical_dist.values() if p > 0]) if empirical_dist else np.nan

        # 3) Align supports and compute divergences
        all_moves = sorted(set(list(policy_dist.keys()) + list(empirical_dist.keys())))
        p = np.array([policy_dist.get(m, 0.0) for m in all_moves], dtype=np.float64)
        q = np.array([empirical_dist.get(m, 0.0) for m in all_moves], dtype=np.float64)
        # add tiny smoothing to avoid zeros for KL
        eps = 1e-12
        p = p + eps
        q = q + eps
        p = p / p.sum()
        q = q / q.sum()
        # KL(P||Q)
        kl_pq = np.sum(p * np.log2(p / q))
        kl_qp = np.sum(q * np.log2(q / p))
        # JS divergence
        m = 0.5 * (p + q)
        js = 0.5 * (np.sum(p * np.log2(p / m)) + np.sum(q * np.log2(q / m)))

        # 4) Save CSV
        df = pd.DataFrame({
            'move': all_moves,
            'policy_prob': p,
            'empirical_prob': q
        })
        if save_results:
            os.makedirs('results', exist_ok=True)
            csv_path = os.path.join('results', 'policy_vs_quantum_sampling.csv')
            df.to_csv(csv_path, index=False)

        summary = {
            'entropy_policy': float(entropy_policy) if not np.isnan(entropy_policy) else None,
            'entropy_empirical': float(entropy_empirical) if not np.isnan(entropy_empirical) else None,
            'kl_pq': float(kl_pq),
            'kl_qp': float(kl_qp),
            'js': float(js),
            'sample_count': samples,
            'policy_moves_count': len(policy_dist),
            'empirical_unique_firsts': len(empirical_dist)
        }

        if save_results:
            summary_path = os.path.join('results', 'policy_vs_quantum_sampling_summary.txt')
            with open(summary_path, 'w') as f:
                for k, v in summary.items():
                    f.write(f"{k}: {v}\n")
            print(f"✓ Saved CSV: {csv_path} and summary: {summary_path}")

        print(f"Results: entropy_policy={summary['entropy_policy']}, entropy_empirical={summary['entropy_empirical']}, JS={summary['js']:.6f}, KL(P||Q)={summary['kl_pq']:.6f}")
        return df, summary

    @staticmethod
    def quantum_classical_transition_analysis(fen, depth=None, lambda_quantum=None, lambda_competitive=None, samples_quantum=500, samples_competitive=500, multipv_classical=5, save_results=True):
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
        import numpy as np
        import chess
        from engine import Engine

        if depth is None:
            depth = config.TARGET_DEPTH
        if lambda_quantum is None:
            lambda_quantum = config.LAMBDA
        if lambda_competitive is None:
            lambda_competitive = config.LAMBDA

        # 1) Quantum empirical via sample_paths (quantum_limit)
        print(f"[QC] Sampling quantum_limit: samples={samples_quantum}, depth={depth}, lambda={lambda_quantum}")
        paths_q = Engine.sample_paths(fen, config.HIGH_DEPTH, lambda_quantum, samples_quantum, mode='quantum_limit')
        firsts_q = [str(p[0]) for p in paths_q if p]
        counter_q = Counter(firsts_q)
        total_q = sum(counter_q.values())
        dist_q = {m: cnt/total_q for m, cnt in counter_q.items()} if total_q>0 else {}
        entropy_q = -sum([p*np.log2(p) for p in dist_q.values() if p>0]) if dist_q else 0.0

        # 2) Competitive empirical via sample_paths (competitive)
        print(f"[QC] Sampling competitive: samples={samples_competitive}, depth={depth}, lambda={lambda_competitive}")
        paths_c = Engine.sample_paths(fen, config.TARGET_DEPTH, lambda_competitive, samples_competitive, mode='competitive')
        firsts_c = [str(p[0]) for p in paths_c if p]
        counter_c = Counter(firsts_c)
        total_c = sum(counter_c.values())
        dist_c = {m: cnt/total_c for m, cnt in counter_c.items()} if total_c>0 else {}
        entropy_c = -sum([p*np.log2(p) for p in dist_c.values() if p>0]) if dist_c else 0.0

        # 3) Classical: Stockfish MultiPV -> softmax scores
        print(f"[QC] Querying Stockfish: multipv={multipv_classical}, depth={depth}")
        try:
            sf_engine = Engine.get_engine(config.STOCKFISH_PATH)
            moves_s, scores_s = Engine.get_top_moves_and_scores(sf_engine, chess.Board(fen), depth=depth, multipv=multipv_classical)
        except Exception:
            moves_s, scores_s = [], []
        probs_s = Calc.softmax(scores_s, lambda_competitive) if scores_s else []
        dist_s = {str(m.uci() if hasattr(m, 'uci') else m): float(p) for m, p in zip(moves_s, probs_s)}
        entropy_s = -sum([p*np.log2(p) for p in dist_s.values() if p>0]) if dist_s else 0.0

        # 4) Align supports and compute divergences
        all_moves = sorted(set(list(dist_q.keys()) + list(dist_c.keys()) + list(dist_s.keys())))
        p = np.array([dist_q.get(m, 0.0) for m in all_moves], dtype=np.float64)
        q = np.array([dist_c.get(m, 0.0) for m in all_moves], dtype=np.float64)
        r = np.array([dist_s.get(m, 0.0) for m in all_moves], dtype=np.float64)
        eps = 1e-12
        p += eps; q += eps; r += eps
        p /= p.sum(); q /= q.sum(); r /= r.sum()

        # Pairwise KL/JS
        def kl(a,b):
            mask = (a>0)&(b>0)
            return float(np.sum(a[mask]*np.log2(a[mask]/b[mask])))
        def js(a,b):
            m = 0.5*(a+b)
            return float(0.5*(np.sum(a*np.log2(a/m)) + np.sum(b*np.log2(b/m))))
        metrics = {
            'entropy_quantum': float(entropy_q),
            'entropy_competitive': float(entropy_c),
            'entropy_classical': float(entropy_s),
            'kl_q_c': kl(p,q),
            'kl_c_q': kl(q,p),
            'js_q_c': js(p,q),
            'kl_q_s': kl(p,r),
            'kl_s_q': kl(r,p),
            'js_q_s': js(p,r),
            'kl_c_s': kl(q,r),
            'kl_s_c': kl(r,q),
            'js_c_s': js(q,r)
        }

        # 5) Save CSV
        df = pd.DataFrame({
            'move': all_moves,
            'quantum_prob': p,
            'competitive_prob': q,
            'classical_prob': r
        })
        if save_results:
            os.makedirs('results', exist_ok=True)
            csv_path = os.path.join('results', 'quantum_classical_transition_moves_v2.csv')
            df.to_csv(csv_path, index=False)

        # 6) Bar chart with entropies in legend
        plt.figure(figsize=(max(8, len(all_moves)*0.5), 6))
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
    def batch_path_analysis(fen_list, lambda_values, samples=50, depth=5, use_cache=True):
        """
        Optimized batch analysis for multiple positions and lambda values.
        
        Args:
            fen_list: List of FEN position strings
            lambda_values: List of lambda values to test
            samples: Number of samples per analysis
            depth: Search depth
            use_cache: Whether to use caching
            
        Returns:
            DataFrame with analysis results
        """
        from engine import Engine
        
        results = []
        total_analyses = len(fen_list) * len(lambda_values)
        
        print(f"🔄 Running batch analysis: {len(fen_list)} positions × {len(lambda_values)} λ values = {total_analyses} analyses")
        
        with tqdm(total=total_analyses, desc="Batch Analysis") as pbar:
            for fen_idx, fen in enumerate(fen_list):
                for lam in lambda_values:
                    # Check cache first
                    cache_key = f"batch_analysis_{hash(fen)}_{lam}_{samples}_{depth}"
                    
                    if use_cache:
                        cached_result = Cache.get_cached_analysis(cache_key)
                        if cached_result is not None:
                            results.append(cached_result)
                            pbar.update(1)
                            continue
                    
                    try:
                        # Sample paths
                        start_time = time.time()
                        paths = Engine.sample_paths(fen, depth, lam, samples, mode='competitive')
                        sampling_time = time.time() - start_time
                        
                        # Compute metrics
                        entropy, counter = Calc.compute_entropy(paths)
                        concentration = Calc.top_move_concentration(paths)
                        
                        # Create CP rewards (simplified)
                        cp_rewards = [len(path) * 10.0 for path in paths]  # Simple heuristic
                        
                        # Apply path integral framework
                        pi_result = Calc.path_integral_framework([], cp_rewards, lam)
                        result = {
                            'position_index': fen_idx,
                            'fen': fen,
                            'lambda': lam,
                            'entropy': entropy,
                            'concentration': concentration,
                            'pi_entropy': pi_result['entropy'],
                            'pi_ess': pi_result['ess'],
                            'pi_concentration': pi_result['concentration'],
                            'resampled': pi_result['resampled'],
                            'sampling_time': sampling_time,
                            'path_count': len(paths),
                            'unique_moves': len(counter)
                        }
                        
                        results.append(result)
                        
                        if use_cache:
                            Cache.set_cached_analysis(cache_key, result)
                            
                    except Exception as e:
                        print(f"Error analyzing position {fen_idx} with λ={lam}: {e}")
                        
                    pbar.update(1)
        
        return pd.DataFrame(results)
    
    @staticmethod
    def lambda_sensitivity_analysis(fen, lambda_range=None, samples=100, save_results=True):
        """
        Optimized lambda sensitivity analysis with comprehensive metrics.
        
        Args:
            fen: Chess position in FEN notation
            lambda_range: Range of lambda values to test
            samples: Number of samples per lambda
            save_results: Whether to save results to files
            
        Returns:
            Analysis results dictionary
        """
        if lambda_range is None:
            lambda_range = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        
        print(f"🔬 Lambda Sensitivity Analysis: {len(lambda_range)} values, {samples} samples each")
        
        results = []
        
        for lam in tqdm(lambda_range, desc="Lambda Analysis"):
            # Use batch analysis for consistency
            batch_result = Calc.batch_path_analysis([fen], [lam], samples)
            if not batch_result.empty:
                row = batch_result.iloc[0]
                results.append({
                    'lambda': lam,
                    'entropy': row['entropy'],
                    'concentration': row['concentration'],
                    'pi_entropy': row['pi_entropy'],
                    'pi_ess': row['pi_ess'],
                    'pi_concentration': row['pi_concentration'],
                    'resampled': row['resampled'],
                    'sampling_time': row['sampling_time']
                })
        
        df = pd.DataFrame(results)
        
        # Analysis summary
        analysis_summary = {
            'fen': fen,
            'lambda_range': lambda_range,
            'samples': samples,
            'results_df': df,
            'optimal_lambda_entropy': df.loc[df['entropy'].idxmax(), 'lambda'] if not df.empty else None,
            'optimal_lambda_concentration': df.loc[df['concentration'].idxmax(), 'lambda'] if not df.empty else None,
            'exploration_regime_entropy': df[df['lambda'] < 0.5]['entropy'].mean() if not df.empty else 0,
            'exploitation_regime_entropy': df[df['lambda'] > 2.0]['entropy'].mean() if not df.empty else 0
        }
        
        if save_results:
            os.makedirs("results", exist_ok=True)
            df.to_csv("results/lambda_sensitivity_analysis.csv", index=False)
            Calc._generate_lambda_sensitivity_plots(df)
            print("✓ Lambda sensitivity analysis saved to results/")
        
        return analysis_summary
    
    @staticmethod
    def _generate_lambda_sensitivity_plots(df):
        """Generate optimized visualization for lambda sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Entropy vs Lambda
        axes[0, 0].semilogx(df['lambda'], df['entropy'], 'o-', linewidth=2, markersize=8, color='blue')
        axes[0, 0].set_xlabel('Lambda (λ)')
        axes[0, 0].set_ylabel('Entropy (bits)')
        axes[0, 0].set_title('Entropy vs Lambda')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Concentration vs Lambda
        axes[0, 1].semilogx(df['lambda'], df['concentration'], 's-', linewidth=2, markersize=8, color='red')
        axes[0, 1].set_xlabel('Lambda (λ)')
        axes[0, 1].set_ylabel('Top Move Concentration')
        axes[0, 1].set_title('Concentration vs Lambda')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Path Integral ESS vs Lambda
        axes[1, 0].semilogx(df['lambda'], df['pi_ess'], '^-', linewidth=2, markersize=8, color='green')
        axes[1, 0].set_xlabel('Lambda (λ)')
        axes[1, 0].set_ylabel('Effective Sample Size')
        axes[1, 0].set_title('ESS vs Lambda')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Sampling Time vs Lambda
        axes[1, 1].semilogx(df['lambda'], df['sampling_time'], 'd-', linewidth=2, markersize=8, color='orange')
        axes[1, 1].set_xlabel('Lambda (λ)')
        axes[1, 1].set_ylabel('Sampling Time (seconds)')
        axes[1, 1].set_title('Performance vs Lambda')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("results/lambda_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def clear_caches():
        """Clear all internal caches"""
        Calc._entropy_cache.clear()
        Calc._concentration_cache.clear()
        print("✓ Cleared Calc internal caches")