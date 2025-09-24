import os
import time

import chess
import chess.pgn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

import config  # Direct module import for better performance
from engine import Engine
from mathfuncs import Calc


class Experiments:
    @staticmethod
    def path_integral_depth_scan_experiment(fen=config.FEN, lambda_values=None, depth_values=None,
                                           sample_count=config.SAMPLE_COUNT, save_results=True):
        """
        Core path-integral analysis experiment implementing the custom instructions requirements.
        Now GT-free: accuracy is reported as concentration (mode probability of the first move).

        Performs systematic lambda and depth scanning as specified in copilot-instructions.md:
        - Lambda scanning across specified values (default: 0.01, 0.05, 0.1, 0.2, 0.5)
        - Depth scanning for convergence analysis
        - Comprehensive metrics calculation (entropy, concentration, KL divergence, Top-N proxy)
        - Progress logging with tqdm bars
        - All outputs saved as PNG (non-interactive) and CSV

        :param fen: FEN position to analyze
        :param lambda_values: List of lambda values to scan (default: config.LAMBDA_SCAN subset)
        :param depth_values: List of depth values for analysis (default: [2,4,8,16])
        :param sample_count: Number of path samples per lambda/depth combination
        :param save_results: Whether to save results to files
        :return: Dictionary containing all analysis results
        """
        if lambda_values is None:
            lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5]

        if depth_values is None:
            depth_values = [2, 4, 8, 16]

        print(f"\n=== Path-Integral Depth Scan Analysis ===")
        print(f"FEN: {fen}")
        print(f"Lambda values: {lambda_values}")
        print(f"Depth values: {depth_values}")
        print(f"Samples per combination: {sample_count}")

        # Initialize results storage (GT removed)
        results = {
            'fen': fen,
            'lambda_values': lambda_values,
            'depth_values': depth_values,
            'sample_count': sample_count,
            'metrics': {},
            'raw_data': {}
        }

        # Lambda scanning with progress bar
        print(f"\nLambda scanning analysis...")
        lambda_metrics = {}

        for lam in tqdm(lambda_values, desc="Lambda scan"):
            start_time = time.time()
            paths = Engine.sample_paths(fen, depth_values[-1], lam, sample_count)  # En yüksek depth ile

            # Calculate metrics
            entropy, move_counter = Calc.compute_entropy(paths)
            concentration = Calc.top_move_concentration(paths)

            # KL divergence against uniform baseline on observed moves
            all_moves = list(move_counter.keys())
            total_samples = sum(move_counter.values())
            current_dist = {move: move_counter.get(move, 0) / max(total_samples, 1) for move in all_moves}
            if len(all_moves) > 0:
                ref_dist = {move: 1.0/len(all_moves) for move in all_moves}
                kl_divergence = Calc.kl_divergence(current_dist, ref_dist)
            else:
                kl_divergence = 0.0

            elapsed = time.time() - start_time
            avg_sample_time = elapsed / max(sample_count, 1)

            lambda_metrics[lam] = {
                'entropy': entropy,
                'accuracy': concentration,
                'top_n_accuracy': concentration,  # proxy (no GT)
                'kl_divergence': kl_divergence,
                'move_probabilities': move_counter,
                'move_counter': move_counter,
                'elapsed_time': elapsed,
                'avg_sample_time': avg_sample_time
            }

            print(f"λ={lam}: entropy={entropy:.3f}, concentration={concentration:.3f}, time={elapsed:.1f}s")

        results['metrics']['lambda_scan'] = lambda_metrics

        # Depth scanning for convergence analysis
        print(f"\nDepth convergence analysis...")
        depth_metrics = {}

        middle_lambda = lambda_values[len(lambda_values)//2]

        for depth in tqdm(depth_values, desc="Depth scan"):
            start_time = time.time()

            paths = Engine.sample_paths(fen, depth, middle_lambda, sample_count)
            entropy, move_probs = Calc.compute_entropy(paths)
            concentration = Calc.top_move_concentration(paths)

            elapsed = time.time() - start_time

            depth_metrics[depth] = {
                'entropy': entropy,
                'accuracy': concentration,
                'move_probabilities': move_probs,
                'elapsed_time': elapsed
            }

            print(f"depth={depth}: entropy={entropy:.3f}, concentration={concentration:.3f}, time={elapsed:.1f}s")

        results['metrics']['depth_scan'] = depth_metrics
        results['metrics']['depth_scan_lambda'] = middle_lambda

        if save_results:
            os.makedirs("results", exist_ok=True)

            lambda_df = pd.DataFrame.from_dict(
                {lam: {k: v for k, v in metrics.items() if k not in ['move_probabilities', 'move_counter']}
                 for lam, metrics in lambda_metrics.items()},
                orient='index'
            )
            lambda_df.to_csv("results/lambda_scan_metrics.csv")

            depth_df = pd.DataFrame.from_dict(
                {depth: {k: v for k, v in metrics.items() if k != 'move_probabilities'}
                 for depth, metrics in depth_metrics.items()},
                orient='index'
            )
            depth_df.to_csv("results/depth_scan_metrics.csv")

            Experiments._generate_comprehensive_depth_plots(results, fen)

            print("Results and plots saved to results/ directory")

        return results

    @staticmethod
    def _generate_comprehensive_depth_plots(results, fen):
        """
        Generate all required plots as specified in custom instructions (depth version).

        Creates:
        - Entropy-concentration vs λ plot (x-axis: 0.01, 0.05, 0.1, 0.2, 0.5...)
        - KL divergence vs λ plot
        - Heat map for λ × depth showing mode probability
        - Bar chart for move distributions with highest bar highlighted
        - Depth vs entropy convergence plot
        """
        lambda_metrics = results['metrics']['lambda_scan']
        depth_metrics = results['metrics']['depth_scan']
        lambda_values = results['lambda_values']
        depth_values = results['depth_values']

        # Ensure matplotlib uses non-interactive backend
        plt.switch_backend('Agg')

        # 1. Entropy-concentration vs λ plot
        entropies = [lambda_metrics[lam]['entropy'] for lam in lambda_values]
        accuracies = [lambda_metrics[lam]['accuracy'] for lam in lambda_values]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot entropy vs lambda
        ax1.plot(lambda_values, entropies, 'o-', color='blue', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (λ) - Softmax Temperature')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Entropy vs Lambda')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

        # Plot concentration vs lambda
        ax2.plot(lambda_values, accuracies, 's-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel('Lambda (λ) - Softmax Temperature')
        ax2.set_ylabel('Concentration (mode probability)')
        ax2.set_title('Concentration vs Lambda')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')

        plt.tight_layout()
        plt.savefig('results/entropy_accuracy_vs_lambda.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. KL divergence vs λ plot
        kl_divergences = [lambda_metrics[lam]['kl_divergence'] for lam in lambda_values]

        plt.figure(figsize=(10, 6))
        plt.plot(lambda_values, kl_divergences, 'o-', color='green', linewidth=2, markersize=8)
        plt.xlabel('Lambda (λ) - Softmax Temperature')
        plt.ylabel('KL Divergence from Uniform')
        plt.title('KL Divergence vs Lambda')
        plt.grid(True, alpha=0.3)
        plt.xscale('log')
        plt.savefig('results/kl_divergence_vs_lambda.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Heat map for λ × depth showing mode probability (using lambda scan as proxy)
        heat_data = []
        for lam in lambda_values:
            row = []
            for depth in depth_values:
                opt_prob = lambda_metrics[lam]['accuracy']
                row.append(opt_prob)
            heat_data.append(row)

        plt.figure(figsize=(12, 8))
        im = plt.imshow(heat_data, aspect='auto', cmap='RdYlBu_r', origin='lower')
        plt.colorbar(im, label='Mode Probability')
        plt.xlabel('Depth')
        plt.ylabel('Lambda (λ)')
        plt.title('Mode Probability Heat Map')

        # Set ticks and labels
        plt.xticks(range(len(depth_values)), [f'{d}' for d in depth_values])
        plt.yticks(range(len(lambda_values)), [f'{lam:.2f}' for lam in lambda_values])

        plt.savefig('results/lambda_depth_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 4. Bar chart for move distributions (highest bar highlighted)
        middle_idx = len(lambda_values) // 2
        middle_lam = lambda_values[middle_idx]
        move_counter = lambda_metrics[middle_lam]['move_counter']

        moves = list(move_counter.keys())
        counts = list(move_counter.values())
        colors = ['red' if i == int(np.argmax(counts)) else 'lightblue' for i in range(len(moves))]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(moves)), counts, color=colors, alpha=0.7)
        plt.xlabel('Moves')
        plt.ylabel('Frequency')
        plt.title(f'Move Distribution (λ={middle_lam}, mode highlighted in red)')
        plt.xticks(range(len(moves)), moves, rotation=45)
        plt.grid(True, alpha=0.3, axis='y')

        # Highlight the highest bar
        if counts:
            max_idx = int(np.argmax(counts))
            bars[max_idx].set_edgecolor('black')
            bars[max_idx].set_linewidth(2)

        plt.tight_layout()
        plt.savefig('results/move_distribution_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 5. Depth vs entropy and concentration convergence plot
        depth_list = list(depth_values)
        depth_entropies = [depth_metrics[depth]['entropy'] for depth in depth_list]
        depth_accuracies = [depth_metrics[depth]['accuracy'] for depth in depth_list]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Entropy convergence
        ax1.semilogx(depth_list, depth_entropies, 'o-', color='purple', linewidth=2, markersize=8)
        ax1.set_xlabel('Depth')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Entropy Convergence vs Depth')
        ax1.grid(True, alpha=0.3)

        # Concentration convergence
        ax2.semilogx(depth_list, depth_accuracies, 's-', color='orange', linewidth=2, markersize=8)
        ax2.set_xlabel('Depth')
        ax2.set_ylabel('Concentration (mode probability)')
        ax2.set_title('Concentration Convergence vs Depth')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/convergence_vs_depth.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("Generated all required plots as per custom instructions (depth version):")
        print("  - entropy_accuracy_vs_lambda.png")
        print("  - kl_divergence_vs_lambda.png")
        print("  - lambda_depth_heatmap.png")
        print("  - move_distribution_bar_chart.png")
        print("  - convergence_vs_depth.png")

    @staticmethod  
    def _generate_multi_position_plots(results_by_position):
        """Generate comparison plots across multiple positions."""
        plt.switch_backend('Agg')
        
        # Extract data for plotting
        positions = list(results_by_position.keys())
        
        # Position complexity vs concentration analysis
        complexities = []
        accuracies_by_lambda = {lam: [] for lam in [0.05, 0.2, 0.5]}
        
        for pos_key, pos_results in results_by_position.items():
            lambda_metrics = pos_results['metrics']['lambda_scan']
            
            # Use entropy as complexity measure
            avg_entropy = np.mean([lambda_metrics[lam]['entropy'] for lam in lambda_metrics.keys()])
            complexities.append(avg_entropy)
            
            # Collect concentration for each lambda
            for lam in accuracies_by_lambda.keys():
                if lam in lambda_metrics:
                    accuracies_by_lambda[lam].append(lambda_metrics[lam]['accuracy'])
        
        # Plot position complexity vs concentration
        plt.figure(figsize=(12, 8))
        for i, lam in enumerate(accuracies_by_lambda.keys()):
            plt.scatter(complexities, accuracies_by_lambda[lam], 
                       label=f'λ={lam}', s=100, alpha=0.7)
        
        plt.xlabel('Position Complexity (Average Entropy)')
        plt.ylabel('Concentration (mode probability)')
        plt.title('Position Complexity vs Concentration\n(Multiple FEN Positions)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('results/multi_fen_complexity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Generated multi-FEN analysis plot:")
        print("  - multi_fen_complexity_analysis.png")

    @staticmethod
    def dynamic_lambda_adaptation_experiment(fen):
        """
        Conducts a dynamic lambda adaptation experiment to evaluate an adaptive lambda strategy that varies
        exploration-exploitation balance based on position complexity. Outputs a comparative analysis with
        a fixed lambda strategy, including visualization and performance metrics.
        Lambda values are now smaller for more sensitive exploration-exploitation comparison.
        - Sabit λ: 0.12
        - Adaptif λ: 0.25 (basit), 0.12 (orta), 0.03 (karmaşık)
        """
        print(f"\n--- Dynamic Lambda Adaptation Experiment ---")

        board = chess.Board(fen)

        # Optimize edilmiş lambda değerleri
        elbow_lambda = 0.12

        def adaptive_lambda_strategy(board_position):
            legal_moves_count = len(list(board_position.legal_moves))
            if legal_moves_count < 20:
                return 0.25  # Yüksek sömürü
            elif legal_moves_count < 35:
                return elbow_lambda  # Denge
            else:
                return 0.03  # Yüksek keşif

        fixed_lambda = elbow_lambda
        adaptive_lambda = adaptive_lambda_strategy(board)

        print(f"Number of legal moves in a position: {len(list(board.legal_moves))}")
        print(f"Constant λ: {fixed_lambda}")
        print(f"Adaptive λ: {adaptive_lambda}")

        # Her iki stratejiyi test et
        fixed_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, fixed_lambda, config.SAMPLE_COUNT)
        adaptive_paths = Engine.sample_paths(fen, config.TARGET_DEPTH, adaptive_lambda, config.SAMPLE_COUNT)

        fixed_entropy, _ = Calc.compute_entropy(fixed_paths)
        adaptive_entropy, _ = Calc.compute_entropy(adaptive_paths)

        fixed_top_move = Calc.most_frequent_first_move(fixed_paths)
        adaptive_top_move = Calc.most_frequent_first_move(adaptive_paths)

        fixed_accuracy = Calc.top_move_concentration(fixed_paths)
        adaptive_accuracy = Calc.top_move_concentration(adaptive_paths)

        # Hata/log kontrolü
        if np.isnan(fixed_entropy) or np.isnan(adaptive_entropy):
            print("UYARI: Entropi hesaplanamadı! Yeterli çeşitlilik veya örneklem yok.")
        if np.isnan(fixed_accuracy) or np.isnan(adaptive_accuracy):
            print("UYARI: Konsantrasyon hesaplanamadı! Yeterli çeşitlilik veya örneklem yok.")
        if len(fixed_paths) == 0 or len(adaptive_paths) == 0:
            print("UYARI: Örneklem boş! Motor veya parametreleri kontrol edin.")

        results = {
            'fixed_lambda': fixed_lambda,
            'adaptive_lambda': adaptive_lambda,
            'fixed_entropy': fixed_entropy,
            'adaptive_entropy': adaptive_entropy,
            'fixed_accuracy': fixed_accuracy,
            'adaptive_accuracy': adaptive_accuracy,
            'fixed_top_move': fixed_top_move,
            'adaptive_top_move': adaptive_top_move,
            'improvement': adaptive_accuracy > fixed_accuracy
        }

        # Görselleştirme
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        strategies = [f'Sabit λ\n({fixed_lambda})', f'Adaptif λ\n({adaptive_lambda})']
        entropies = [fixed_entropy, adaptive_entropy]
        accuracies = [fixed_accuracy, adaptive_accuracy]

        ax1.bar(strategies, entropies, color=['lightcoral', 'lightgreen'], alpha=0.7)
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title('Lambda Adaptation Strategy: Exploratory Behavior')
        ax1.grid(True, alpha=0.3)

        ax2.bar(strategies, accuracies, color=['lightcoral', 'lightgreen'], alpha=0.7)
        ax2.set_ylabel('Concentration (mode probability)')
        ax2.set_title('Lambda Adaptation Strategy: Mode Probability')
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)

        improvement_text = "✓ IMPROVEMENT" if results['improvement'] else "✗ DETERIORATION"
        fig.suptitle(f'Dynamic Lambda Adaptation ({improvement_text})', fontsize=16)

        plt.tight_layout()
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/dynamic_lambda_adaptation.png", dpi=300)
        plt.close()

        return results

    @staticmethod
    def perfect_play_self_play_experiment(starting_fen=None, max_moves=100):
        """
        Simulates a self-play experience under "perfect play" constraints. This function models chess games between
        identical players (PI Player) with a perfect play parameter (lambda) set to a high value, ensuring optimal moves
        are prioritized. Games can either start from a given position or the standard initial chess position, with results
        saved in PGN format and logging throughout the simulation for analysis purposes.

        :param starting_fen: The starting position of the chess game in Forsyth–Edwards Notation (FEN). Defaults to the
            standard initial chess position if not provided.
        :type starting_fen: str or None
        :param max_moves: The maximum number of moves allowed in the simulation before it is stopped, to prevent infinite
            games. Defaults to 100.
        :type max_moves: int
        :return: A tuple containing:
            1. The result of the game as a string ("1-0" for White victory, "0-1" for Black victory,
               "1/2-1/2" for draw, or "*" for ongoing/indeterminate);
            2. The total number of moves made during the game;
            3. A list of dictionaries representing the game history with keys:
               - 'move_number': the move number.
        """
        print(f"\n--- Perfect Game Self-Play Experiment ---")

        if starting_fen is None:
            board = chess.Board()  # Standart başlangıç pozisyonu
        else:
            board = chess.Board(starting_fen)

        game = chess.pgn.Game()
        game.headers["Event"] = "PI vs PI Perfect Play Simulation"
        game.headers["Site"] = "Local"
        game.headers["White"] = "PI Player (λ=2.0)"
        game.headers["Black"] = "PI Player (λ=2.0)"
        node = game

        move_count = 0
        game_history = []

        # Yüksek lambda ile "mükemmel" oyun simüle et
        perfect_lambda = 2.0

        print(f"Starting position: {board.fen()}")
        print(f"Perfect game simulation (λ={perfect_lambda}) is begging...")

        while not board.is_game_over() and move_count < max_moves:
            current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
            print(f"Move {move_count + 1}: {current_player} is thinking...")

            paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, perfect_lambda, config.SAMPLE_COUNT)
            move_uci = Calc.most_frequent_first_move(paths)

            if move_uci:
                try:
                    move = chess.Move.from_uci(move_uci)
                    if move in board.legal_moves:
                        board.push(move)
                        node = node.add_variation(move)
                        game_history.append({
                            'move_number': move_count + 1,
                            'player': current_player,
                            'move': move_uci,
                            'fen_after': board.fen()
                        })
                        print(f"  Oynanan hamle: {move_uci}")
                        move_count += 1
                    else:
                        print(f"  Invalid move: {move_uci}, selecting random move...")
                        legal_moves = list(board.legal_moves)
                        if legal_moves:
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            break
                except Exception as e:
                    print(f"  Move processing error: {e}, selecting random move...")
                    legal_moves = list(board.legal_moves)
                    if legal_moves:
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                    else:
                        break
            else:
                print(f"  PI model could not find a move, random move is selected...")
                legal_moves = list(board.legal_moves)
                if legal_moves:
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1
                else:
                    break

        # Oyun sonucu
        game_result = board.result()
        game.headers["Result"] = game_result

        # PGN dosyasına kaydet
        pgn_filename = "results/perfect_play_self_play.pgn"
        os.makedirs("results", exist_ok=True)
        with open(pgn_filename, "w", encoding="utf-8") as f:
            f.write(str(game))

        print(f"\nGame over! Result: {game_result}")
        print(f"Total number of moves: {move_count}")
        print(f"The game save was saved to file '{pgn_filename}'.")

        # Oyun analizi
        result_analysis = {
            '1-0': 'White won',
            '0-1': 'Black won',
            '1/2-1/2': 'Draw',
            '*': 'Game in progress/Uncertain'
        }

        print(f"Game analysis: {result_analysis.get(game_result, 'Unknown result')}")

        # Basit istatistikler
        if game_history:
            avg_moves_per_player = move_count / 2
            print(f"Average moves per player: {avg_moves_per_player:.1f}")

        return game_result, move_count, game_history

    @staticmethod
    def pi_vs_lc0_match_experiment(starting_fen=None, max_moves=500, match_count=3, save_results=True):
        """
        PI vs Lc0 karşılaştırmalı deney: Best of 5 formatında, 5 maç oynulur.
        Her maçta PI ve Lc0 sırayla oynar, hamleler ilgili motorlardan alınır.
        Her maç PGN olarak kaydedilir, sonuçlar CSV ve TXT olarak özetlenir.
        Ayrıca her hamle için süre ve performans metrikleri ayrı bir CSV'ye kaydedilir.
        """
        print("\n--- PI vs Lc0 Best of 5 Karşılaştırmalı Deney ---")
        results = []
        perf_rows = []
        for match_idx in range(match_count):
            print(f"\nMaç {match_idx+1} başlıyor...")
            pi_is_white = (match_idx % 2 == 0)
            board = chess.Board() if starting_fen is None else chess.Board(starting_fen)
            game = chess.pgn.Game()
            game.headers["Event"] = f"PI vs Lc0 Match {match_idx+1}"
            game.headers["Site"] = "Local"
            game.headers["White"] = "PI Player" if pi_is_white else "Lc0 Player"
            game.headers["Black"] = "Lc0 Player" if pi_is_white else "PI Player"
            node = game
            move_count = 0
            pi_lambda = config.LAMBDA # Sabit lambda
            while not board.is_game_over() and move_count < max_moves:
                current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
                start_time = time.time()
                engine_time_sec = None
                pi_turn = (board.turn == chess.WHITE and pi_is_white) or (board.turn == chess.BLACK and not pi_is_white)
                if pi_turn:
                    # PI Player
                    paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT)
                    move_uci = Calc.most_frequent_first_move(paths)
                    engine_type = "PI"
                else:
                    # Lc0 Player
                    lc0_start = time.time()
                    lc0_opts = {"MultiPV": 1, "Temperature": 0.2, "Deterministic": False, "UseNoise": False}
                    moves, scores, _ = Engine.lc0_top_moves_and_scores(board.fen(), depth=config.TARGET_DEPTH*100, multipv=1, options=lc0_opts)
                    lc0_end = time.time()
                    if moves and scores:
                        best_idx = int(np.argmax(scores))
                        move_uci = moves[best_idx]
                    else:
                        move_uci = moves[0] if moves else None
                    engine_type = "Lc0"
                    engine_time_sec = lc0_end - lc0_start
                elapsed = time.time() - start_time

                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            legal_moves = list(board.legal_moves)
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                    except Exception as e:
                        legal_moves = list(board.legal_moves)
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                else:
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1
                # Performans kaydı
                perf_rows.append({
                    'match': match_idx+1,
                    'move_number': move_count,
                    'player': current_player,
                    'move': move_uci if move_uci else 'random',
                    'time_sec': elapsed,
                    'fen': board.fen(),
                    'engine': engine_type,
                    'pi_is_white': pi_is_white
                })
            game_result = board.result()
            game.headers["Result"] = game_result
            pgn_filename = f"results/pi_vs_lc0_match_{match_idx+1}.pgn"
            os.makedirs("results", exist_ok=True)
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(game))
            print(f"Maç {match_idx+1} bitti! Sonuç: {game_result}, Hamle sayısı: {move_count}")
            results.append({
                'match': match_idx+1,
                'result': game_result,
                'move_count': move_count,
                'pgn_file': pgn_filename
            })
        # Sonuçları CSV ve TXT olarak kaydet
        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_vs_lc0_match_results.csv", index=False)
            with open("results/pi_vs_lc0_match_summary.txt", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(f"Maç {r['match']}: Sonuç={r['result']}, Hamle={r['move_count']}, PGN={r['pgn_file']}\n")
            perf_df = pd.DataFrame(perf_rows)
            perf_df.to_csv("results/pi_vs_lc0_match_performance.csv", index=False)
            print("Tüm maçlar kaydedildi: results/pi_vs_lc0_match_results.csv, results/pi_vs_lc0_match_summary.txt ve hamle başı performans: results/pi_vs_lc0_match_performance.csv")
        return results

    @staticmethod
    def pi_vs_stockfish_match_experiment(starting_fen=None, max_moves=500, match_count=3, save_results=True):
        """
        PI vs Stockfish karşılaştırmalı deney: Best of 5 formatında, 5 maç oynulur.
        Her maçta PI ve Stockfish sırayla oynar, hamleler ilgili motorlardan alınır.
        Her maç PGN olarak kaydedilir, sonuçlar CSV ve TXT olarak özetlenir.
        Ayrıca her hamle için süre ve performans metrikleri ayrı bir CSV'ye kaydedilir.
        """
        print("\n--- PI vs Stockfish Best of 5 Karşılaştırmalı Deney ---")
        results = []
        perf_rows = []
        for match_idx in range(match_count):
            print(f"\nMaç {match_idx+1} başlıyor...")
            pi_is_white = (match_idx % 2 == 0)
            board = chess.Board() if starting_fen is None else chess.Board(starting_fen)
            game = chess.pgn.Game()
            game.headers["Event"] = f"PI vs Stockfish Match {match_idx+1}"
            game.headers["Site"] = "Local"
            game.headers["White"] = "PI Player" if pi_is_white else "Stockfish Player"
            game.headers["Black"] = "Stockfish Player" if pi_is_white else "PI Player"
            node = game
            move_count = 0
            pi_lambda = config.LAMBDA # Sabit lambda
            while not board.is_game_over() and move_count < max_moves:
                current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
                start_time = time.time()
                pi_turn = (board.turn == chess.WHITE and pi_is_white) or (board.turn == chess.BLACK and not pi_is_white)
                if pi_turn:
                    # PI Player
                    paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT)
                    move_uci = Calc.most_frequent_first_move(paths)
                    engine_type = "PI"
                else:
                    # Stockfish Player
                    moves, scores, _ = Engine.stockfish_top_moves_and_scores(board.fen(), depth=int(config.TARGET_DEPTH*4), multipv=1)
                    if moves and scores:
                        best_idx = int(np.argmax(scores))
                        move_uci = moves[best_idx]
                    else:
                        move_uci = moves[0] if moves else None
                    engine_type = "Stockfish"
                elapsed = time.time() - start_time

                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            legal_moves = list(board.legal_moves)
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                    except Exception as e:
                        legal_moves = list(board.legal_moves)
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                else:
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1
                # Performans kaydı
                perf_rows.append({
                    'match': match_idx+1,
                    'move_number': move_count,
                    'player': current_player,
                    'move': move_uci if move_uci else 'random',
                    'time_sec': elapsed,
                    'fen': board.fen(),
                    'engine': engine_type,
                    'pi_is_white': pi_is_white
                })
            game_result = board.result()
            game.headers["Result"] = game_result
            pgn_filename = f"results/pi_vs_stockfish_match_{match_idx+1}.pgn"
            os.makedirs("results", exist_ok=True)
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(game))
            print(f"Maç {match_idx+1} bitti! Sonuç: {game_result}, Hamle sayısı: {move_count}")
            results.append({
                'match': match_idx+1,
                'result': game_result,
                'move_count': move_count,
                'pgn_file': pgn_filename
            })
        # Sonuçları CSV ve TXT olarak kaydet
        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_vs_stockfish_match_results.csv", index=False)
            with open("results/pi_vs_stockfish_match_summary.txt", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(f"Maç {r['match']}: Sonuç={r['result']}, Hamle={r['move_count']}, PGN={r['pgn_file']}\n")
            perf_df = pd.DataFrame(perf_rows)
            perf_df.to_csv("results/pi_vs_stockfish_match_performance.csv", index=False)
            print("Tüm maçlar kaydedildi: results/pi_vs_stockfish_match_results.csv, results/pi_vs_stockfish_match_summary.txt ve hamle başı performans: results/pi_vs_stockfish_match_performance.csv")
        return results

    @staticmethod
    def pi_quantum_vs_stockfish_match_experiment(starting_fen=None, max_moves=500, match_count=3, save_results=True):
        """
        PI (quantum_limit) vs Stockfish karşılaştırmalı deney: Best of 5 formatında, 5 maç oynanır.
        Her maçta PI ve Stockfish sırayla oynar, hamleler ilgili motorlardan alınır.
        Her maç PGN olarak kaydedilir, sonuçlar CSV ve TXT olarak özetlenir.
        Ayrıca her hamle için süre ve performans metrikleri ayrı bir CSV'ye kaydedilir.
        """
        print("\n--- PI (quantum_limit) vs Stockfish Chess960 5 Varyantlı Deney ---")
        results = []
        perf_rows = []
        chess960_fens = list(config.CHESS960_VARIANTS.keys())[:match_count]
        for match_idx, fen in enumerate(chess960_fens):
            variant_info = config.CHESS960_VARIANTS[fen]
            pi_is_white = (match_idx % 2 == 0)
            board = chess.Board(fen)
            game = chess.pgn.Game()
            game.headers["Event"] = f"PI-QUANTUM vs Stockfish Chess960 Match {match_idx+1}"
            game.headers["Site"] = "Local"
            game.headers["Variant"] = f"Chess960 #{variant_info['variant']}"
            game.headers["VariantDesc"] = variant_info['desc']
            game.headers["White"] = "PI-QUANTUM Player" if pi_is_white else "Stockfish Player"
            game.headers["Black"] = "Stockfish Player" if pi_is_white else "PI-QUANTUM Player"
            node = game
            move_count = 0
            pi_lambda = config.LAMBDA
            while not board.is_game_over() and move_count < max_moves:
                current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
                start_time = time.time()
                pi_turn = (board.turn == chess.WHITE and pi_is_white) or (board.turn == chess.BLACK and not pi_is_white)
                if pi_turn:
                    # PI-QUANTUM Player: try mode arg, fallback if not supported
                    warning_mode = None
                    try:
                        paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT, mode='quantum_limit')
                        engine_type = "PI-QUANTUM"
                    except Exception as e:
                        # If something else went wrong, re-raise after recording a warning
                        paths = []
                        engine_type = "PI-QUANTUM (error)"
                        warning_mode = f"error_sampling:{e}"
                    move_uci = Calc.most_frequent_first_move(paths) if paths else None
                else:
                    # Stockfish Player
                    moves, scores, _ = Engine.stockfish_top_moves_and_scores(board.fen(), depth=int(config.STOCKFISH_DEPTH), multipv=1)
                    if moves and scores:
                        best_idx = int(np.argmax(scores))
                        move_uci = moves[best_idx]
                    else:
                        move_uci = moves[0] if moves else None
                    engine_type = "Stockfish"
                elapsed = time.time() - start_time

                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            legal_moves = list(board.legal_moves)
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                    except Exception as e:
                        legal_moves = list(board.legal_moves)
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                else:
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1
                # Performans kaydı
                perf_rows.append({
                    'match': match_idx+1,
                    'move_number': move_count,
                    'player': current_player,
                    'move': move_uci if move_uci else 'random',
                    'time_sec': elapsed,
                    'fen': board.fen(),
                    'engine': engine_type,
                    'pi_is_white': pi_is_white
                })
            game_result = board.result()
            game.headers["Result"] = game_result
            pgn_filename = f"results/pi_quantum_vs_stockfish_chess960_match_{match_idx+1}.pgn"
            os.makedirs("results", exist_ok=True)
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(game))
            print(f"Chess960 Maç {match_idx+1} bitti! Varyant: {variant_info['variant']} Sonuç: {game_result}, Hamle: {move_count}")
            results.append({
                'match': match_idx+1,
                'variant': variant_info['variant'],
                'variant_desc': variant_info['desc'],
                'result': game_result,
                'move_count': move_count,
                'pgn_file': pgn_filename,
                'pi_is_white': pi_is_white
            })

        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_quantum_vs_stockfish_chess960_results.csv", index=False)
            with open("results/pi_quantum_vs_stockfish_chess960_summary.txt", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(f"Maç {r['match']} (Varyant {r['variant']}): Sonuç={r['result']}, Hamle={r['move_count']}, PGN={r['pgn_file']}, PI beyaz mı: {r['pi_is_white']}\n")
            perf_df = pd.DataFrame(perf_rows)
            perf_df.to_csv("results/pi_quantum_vs_stockfish_chess960_performance.csv", index=False)
            print("Chess960 tüm maçlar kaydedildi: results/pi_quantum_vs_stockfish_chess960_results.csv, results/pi_quantum_vs_stockfish_chess960_summary.txt ve hamle başı performans: results/pi_quantum_vs_stockfish_chess960_performance.csv")

        return results

    @staticmethod
    def pi_vs_stockfish_chess960_experiment(starting_fen=None, max_moves=500, match_count=3, save_results=True):
        """
        PI (quantum_limit) vs Stockfish karşılaştırmalı deney: Best of 5 formatında, 5 maç oynanır.
        Her maçta PI ve Stockfish sırayla oynar, hamleler ilgili motorlardan alınır.
        Her maç PGN olarak kaydedilir, sonuçlar CSV ve TXT olarak özetlenir.
        Ayrıca her hamle için süre ve performans metrikleri ayrı bir CSV'ye kaydedilir.
        """
        print("\n--- PI (quantum_limit) vs Stockfish Chess960 5 Varyantlı Deney ---")
        results = []
        perf_rows = []
        chess960_fens = list(config.CHESS960_VARIANTS.keys())[:match_count]
        for match_idx, fen in enumerate(chess960_fens):
            variant_info = config.CHESS960_VARIANTS[fen]
            pi_is_white = (match_idx % 2 == 0)
            board = chess.Board(fen)
            game = chess.pgn.Game()
            game.headers["Event"] = f"PI-Competitive vs Stockfish Chess960 Match {match_idx+1}"
            game.headers["Site"] = "Local"
            game.headers["Variant"] = f"Chess960 #{variant_info['variant']}"
            game.headers["VariantDesc"] = variant_info['desc']
            game.headers["White"] = "PI-Competitive Player" if pi_is_white else "Stockfish Player"
            game.headers["Black"] = "Stockfish Player" if pi_is_white else "PI-Competitive Player"
            node = game
            move_count = 0
            pi_lambda = config.LAMBDA
            while not board.is_game_over() and move_count < max_moves:
                current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
                start_time = time.time()
                pi_turn = (board.turn == chess.WHITE and pi_is_white) or (board.turn == chess.BLACK and not pi_is_white)
                if pi_turn:
                    try:
                        paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT, mode='competitive')
                        engine_type = "PI-Competitive"
                    except Exception as e:
                        # If something else went wrong, re-raise after recording a warning
                        paths = []
                        engine_type = "PI-Competitive (error)"
                        warning_mode = f"error_sampling:{e}"
                    move_uci = Calc.most_frequent_first_move(paths) if paths else None
                else:
                    # Stockfish Player
                    moves, scores, _ = Engine.stockfish_top_moves_and_scores(board.fen(), depth=int(config.STOCKFISH_DEPTH), multipv=1)
                    if moves and scores:
                        best_idx = int(np.argmax(scores))
                        move_uci = moves[best_idx]
                    else:
                        move_uci = moves[0] if moves else None
                    engine_type = "Stockfish"
                elapsed = time.time() - start_time

                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            legal_moves = list(board.legal_moves)
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                    except Exception as e:
                        legal_moves = list(board.legal_moves)
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                else:
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1
                # Performans kaydı
                perf_rows.append({
                    'match': match_idx+1,
                    'move_number': move_count,
                    'player': current_player,
                    'move': move_uci if move_uci else 'random',
                    'time_sec': elapsed,
                    'fen': board.fen(),
                    'engine': engine_type,
                    'pi_is_white': pi_is_white
                })
            game_result = board.result()
            game.headers["Result"] = game_result
            pgn_filename = f"results/pi_competitive_vs_stockfish_chess960_match_{match_idx+1}.pgn"
            os.makedirs("results", exist_ok=True)
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(game))
            print(f"Chess960 Maç {match_idx+1} bitti! Varyant: {variant_info['variant']} Sonuç: {game_result}, Hamle: {move_count}")
            results.append({
                'match': match_idx+1,
                'variant': variant_info['variant'],
                'variant_desc': variant_info['desc'],
                'result': game_result,
                'move_count': move_count,
                'pgn_file': pgn_filename,
                'pi_is_white': pi_is_white
            })

        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_competitive_vs_stockfish_chess960_results.csv", index=False)
            with open("results/pi_competitive_vs_stockfish_chess960_summary.txt", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(f"Maç {r['match']} (Varyant {r['variant']}): Sonuç={r['result']}, Hamle={r['move_count']}, PGN={r['pgn_file']}, PI beyaz mı: {r['pi_is_white']}\n")
            perf_df = pd.DataFrame(perf_rows)
            perf_df.to_csv("results/pi_competitive_vs_stockfish_chess960_performance.csv", index=False)
            print("Chess960 tüm maçlar kaydedildi: results/pi_competitive_vs_stockfish_chess960_results.csv, results/pi_competitive_vs_stockfish_chess960_summary.txt ve hamle başı performans: results/pi_competitive_vs_stockfish_chess960_performance.csv")

        return results

    @staticmethod
    def _generate_comprehensive_depth_plots_with_variant(results, fen, variant_num, variant_desc):
        """
        Standart figür üretimini Chess960 varyant bilgisiyle başlıklandırır.
        """
        lambda_metrics = results['metrics']['lambda_scan']
        depth_metrics = results['metrics']['depth_scan']
        lambda_values = results['lambda_values']
        depth_values = results['depth_values']
        plt.switch_backend('Agg')
        entropies = [lambda_metrics[lam]['entropy'] for lam in lambda_values]
        accuracies = [lambda_metrics[lam]['accuracy'] for lam in lambda_values]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(lambda_values, entropies, 'o-', color='blue', linewidth=2, markersize=8)
        ax1.set_xlabel('Lambda (λ) - Softmax Temperature')
        ax1.set_ylabel('Entropy (bits)')
        ax1.set_title(f'Entropy vs Lambda\nVaryant: {variant_num} - {variant_desc}')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax2.plot(lambda_values, accuracies, 's-', color='red', linewidth=2, markersize=8)
        ax2.set_xlabel('Lambda (λ) - Softmax Temperature')
        ax2.set_ylabel('Concentration (mode probability)')
        ax2.set_title(f'Concentration vs Lambda\nVaryant: {variant_num} - {variant_desc}')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        plt.tight_layout()
        fname = f"results/entropy_accuracy_vs_lambda_variant_{variant_num}.png" if variant_num is not None else "results/entropy_accuracy_vs_lambda_standard.png"
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        plt.close()
        # Diğer figürler için benzer şekilde varyant bilgisini başlığa ekleyebilirsiniz.

    @staticmethod
    def pi_quantum_vs_lc0_chess960_experiment(match_count=5, max_moves=500, save_results=True):
        """
        Chess960 modunda PI vs Lc0: 5 farklı varyant, her maçta renkler değişerek oynanır.
        Varyant bilgisi ve renkler PGN/CSV'ye eklenir.
        """
        print("\n--- PI vs Lc0 Chess960 5 Varyantlı Deney ---")
        results = []
        perf_rows = []
        chess960_fens = list(config.CHESS960_VARIANTS.keys())[:match_count]
        for match_idx, fen in enumerate(chess960_fens):
            variant_info = config.CHESS960_VARIANTS[fen]
            # Renk değişimi: çift maçlarda PI siyah, tek maçlarda PI beyaz
            pi_is_white = (match_idx % 2 == 0)
            board = chess.Board(fen)
            game = chess.pgn.Game()
            game.headers["Event"] = f"PI vs Lc0 Chess960 Match {match_idx+1}"
            game.headers["Site"] = "Local"
            game.headers["Variant"] = f"Chess960 #{variant_info['variant']}"
            game.headers["VariantDesc"] = variant_info['desc']
            game.headers["White"] = "PI Player" if pi_is_white else "Lc0 Player"
            game.headers["Black"] = "Lc0 Player" if pi_is_white else "PI Player"
            node = game
            move_count = 0
            pi_lambda = config.LAMBDA
            while not board.is_game_over() and move_count < max_moves:
                current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
                start_time = time.time()
                engine_time_sec = None
                pi_turn = (board.turn == chess.WHITE and pi_is_white) or (board.turn == chess.BLACK and not pi_is_white)
                if pi_turn:
                    # PI-QUANTUM Player: try mode arg, fallback if not supported
                    warning_mode = None
                    try:
                        paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT, mode='quantum_limit')
                        engine_type = "PI-QUANTUM"
                    except Exception as e:
                        paths = []
                        engine_type = "PI-QUANTUM (error)"
                        warning_mode = f"error_sampling:{e}"
                    move_uci = Calc.most_frequent_first_move(paths) if paths else None
                else:
                    # Stockfish Player
                    moves, scores, _ = Engine.stockfish_top_moves_and_scores(board.fen(), depth=int(config.TARGET_DEPTH*4), multipv=1)
                    if moves and scores:
                        best_idx = int(np.argmax(scores))
                        move_uci = moves[best_idx]
                    else:
                        move_uci = moves[0] if moves else None
                    engine_type = "Stockfish"
                elapsed = time.time() - start_time

                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            legal_moves = list(board.legal_moves)
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                    except Exception as e:
                        legal_moves = list(board.legal_moves)
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                else:
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1

                perf_rows.append({
                    'match': match_idx+1,
                    'variant': variant_info['variant'],
                    'variant_desc': variant_info['desc'],
                    'move_number': move_count,
                    'player': current_player,
                    'move': move_uci if move_uci else 'random',
                    'time_sec': elapsed,
                    'fen': board.fen(),
                    'engine': engine_type,
                    'pi_is_white': pi_is_white
                })
            game_result = board.result()
            game.headers["Result"] = game_result
            pgn_filename = f"results/pi_vs_lc0_chess960_match_{match_idx+1}.pgn"
            os.makedirs("results", exist_ok=True)
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(game))
            print(f"Chess960 Maç {match_idx+1} bitti! Varyant: {variant_info['variant']} Sonuç: {game_result}, Hamle: {move_count}")
            results.append({
                'match': match_idx+1,
                'variant': variant_info['variant'],
                'variant_desc': variant_info['desc'],
                'result': game_result,
                'move_count': move_count,
                'pgn_file': pgn_filename,
                'pi_is_white': pi_is_white
            })
        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_vs_lc0_chess960_results.csv", index=False)
            with open("results/pi_vs_lc0_chess960_summary.txt", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(f"Maç {r['match']} (Varyant {r['variant']}): Sonuç={r['result']}, Hamle={r['move_count']}, PGN={r['pgn_file']}, PI beyaz mı: {r['pi_is_white']}\n")
            perf_df = pd.DataFrame(perf_rows)
            perf_df.to_csv("results/pi_vs_lc0_chess960_performance.csv", index=False)
            print("Chess960 tüm maçlar kaydedildi: results/pi_vs_lc0_chess960_results.csv, results/pi_vs_lc0_chess960_summary.txt ve hamle başı performans: results/pi_vs_lc0_chess960_performance.csv")
        return results

    @staticmethod
    def pi_quantum_vs_stockfish_chess960_experiment(match_count=5, max_moves=500, save_results=True):
        """
        Chess960 modunda PI vs Stockfish: 5 farklı varyant, her maçta renkler değişerek oynanır.
        Varyant bilgisi ve renkler PGN/CSV'ye eklenir.
        """
        print("\n--- PI vs Stockfish Chess960 5 Varyantlı Deney ---")
        results = []
        perf_rows = []
        chess960_fens = list(config.CHESS960_VARIANTS.keys())[:match_count]
        for match_idx, fen in enumerate(chess960_fens):
            variant_info = config.CHESS960_VARIANTS[fen]
            pi_is_white = (match_idx % 2 == 0)
            board = chess.Board(fen)
            game = chess.pgn.Game()
            game.headers["Event"] = f"PI vs Stockfish Chess960 Match {match_idx+1}"
            game.headers["Site"] = "Local"
            game.headers["Variant"] = f"Chess960 #{variant_info['variant']}"
            game.headers["VariantDesc"] = variant_info['desc']
            game.headers["White"] = "PI Player" if pi_is_white else "Stockfish Player"
            game.headers["Black"] = "Stockfish Player" if pi_is_white else "PI Player"
            node = game
            move_count = 0
            pi_lambda = config.LAMBDA
            while not board.is_game_over() and move_count < max_moves:
                current_player = "WHITE" if board.turn == chess.WHITE else "BLACK"
                start_time = time.time()
                pi_turn = (board.turn == chess.WHITE and pi_is_white) or (board.turn == chess.BLACK and not pi_is_white)
                if pi_turn:
                    # PI-QUANTUM Player: try mode arg, fallback if not supported
                    warning_mode = None
                    try:
                        paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT, mode='quantum_limit')
                        engine_type = "PI-QUANTUM"
                    except TypeError:
                        # Older Engine.sample_paths may not accept mode; fallback without it
                        paths = Engine.sample_paths(board.fen(), config.TARGET_DEPTH, pi_lambda, config.SAMPLE_COUNT)
                        engine_type = "PI-QUANTUM (fallback)"
                        warning_mode = "mode_not_supported_fallback_used"
                    except Exception as e:
                        # If something else went wrong, re-raise after recording a warning
                        paths = []
                        engine_type = "PI-QUANTUM (error)"
                        warning_mode = f"error_sampling:{e}"
                    move_uci = Calc.most_frequent_first_move(paths) if paths else None
                else:
                    # Stockfish Player
                    moves, scores, _ = Engine.stockfish_top_moves_and_scores(board.fen(), depth=int(config.TARGET_DEPTH*4), multipv=1)
                    if moves and scores:
                        best_idx = int(np.argmax(scores))
                        move_uci = moves[best_idx]
                    else:
                        move_uci = moves[0] if moves else None
                    engine_type = "Stockfish"
                elapsed = time.time() - start_time

                if move_uci:
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move in board.legal_moves:
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                        else:
                            legal_moves = list(board.legal_moves)
                            move = np.random.choice(legal_moves)
                            board.push(move)
                            node = node.add_variation(move)
                            move_count += 1
                    except Exception as e:
                        legal_moves = list(board.legal_moves)
                        move = np.random.choice(legal_moves)
                        board.push(move)
                        node = node.add_variation(move)
                        move_count += 1
                else:
                    legal_moves = list(board.legal_moves)
                    move = np.random.choice(legal_moves)
                    board.push(move)
                    node = node.add_variation(move)
                    move_count += 1
                # Performans kaydı
                perf_rows.append({
                    'match': match_idx+1,
                    'variant': variant_info['variant'],
                    'variant_desc': variant_info['desc'],
                    'move_number': move_count,
                    'player': current_player,
                    'move': move_uci if move_uci else 'random',
                    'time_sec': elapsed,
                    'fen': board.fen(),
                    'engine': engine_type,
                    'pi_is_white': pi_is_white
                })
            game_result = board.result()
            game.headers["Result"] = game_result
            pgn_filename = f"results/pi_vs_stockfish_chess960_match_{match_idx+1}.pgn"
            os.makedirs("results", exist_ok=True)
            with open(pgn_filename, "w", encoding="utf-8") as f:
                f.write(str(game))
            print(f"Chess960 Maç {match_idx+1} bitti! Varyant: {variant_info['variant']} Sonuç: {game_result}, Hamle: {move_count}")
            results.append({
                'match': match_idx+1,
                'variant': variant_info['variant'],
                'variant_desc': variant_info['desc'],
                'result': game_result,
                'move_count': move_count,
                'pgn_file': pgn_filename,
                'pi_is_white': pi_is_white
            })
        if save_results:
            df = pd.DataFrame(results)
            df.to_csv("results/pi_vs_stockfish_chess960_results.csv", index=False)
            with open("results/pi_vs_stockfish_chess960_summary.txt", "w", encoding="utf-8") as f:
                for r in results:
                    f.write(f"Maç {r['match']} (Varyant {r['variant']}): Sonuç={r['result']}, Hamle={r['move_count']}, PGN={r['pgn_file']}, PI beyaz mı: {r['pi_is_white']}\n")
            perf_df = pd.DataFrame(perf_rows)
            perf_df.to_csv("results/pi_vs_stockfish_chess960_performance.csv", index=False)
            print("Chess960 tüm maçlar kaydedildi: results/pi_vs_stockfish_chess960_results.csv, results/pi_vs_stockfish_chess960_summary.txt ve hamle başı performans: results/pi_vs_stockfish_chess960_performance.csv")
        return results

    @staticmethod
    def chess960_variant_entropy_concentration_analysis():
        """
        Her Chess960 varyantı için açılış ve orta oyun entropi/konsantrasyon karşılaştırması.
        Sonuçlar: CSV ve figür.
        """
        import config
        results = []
        # Açılış ve orta oyun FEN'leri
        opening_fens = list(config.CHESS960_VARIANTS.keys())
        midgame_fens = list(config.CHESS960_MIDGAME_VARIANTS.keys())
        for fen_type, fen_list, variant_dict in [
            ("Açılış", opening_fens, config.CHESS960_VARIANTS),
            ("OrtaOyun", midgame_fens, config.CHESS960_MIDGAME_VARIANTS)
        ]:
            for fen in fen_list:
                variant_info = variant_dict[fen]
                res = Experiments.path_integral_depth_scan_experiment(
                    fen=fen,
                    lambda_values=[0.05, 0.2, 0.5],
                    depth_values=[8, 12, 16],
                    sample_count=30,
                    save_results=False
                )
                for lam, metrics in res["metrics"]["lambda_scan"].items():
                    results.append({
                        "fen_type": fen_type,
                        "variant": variant_info["variant"],
                        "variant_desc": variant_info["desc"],
                        "lambda": lam,
                        "entropy": metrics["entropy"],
                        "accuracy": metrics["accuracy"]
                    })
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/chess960_variant_entropy_concentration.csv", index=False)
        # Figür: varyant bazlı barplot
        import seaborn as sns
        plt.switch_backend('Agg')
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x="variant", y="entropy", hue="fen_type", ci=None)
        plt.title("Chess960 Varyant Bazlı Entropi Karşılaştırması (Açılış/Orta Oyun)")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Entropi (bits)")
        plt.legend(title="Pozisyon Tipi")
        plt.tight_layout()
        plt.savefig("results/chess960_variant_entropy_barplot.png", dpi=300)
        plt.close()
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x="variant", y="accuracy", hue="fen_type", ci=None)
        plt.title("Chess960 Varyant Bazlı Konsantrasyon Karşılaştırması (Açılış/Orta Oyun)")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Konsantrasyon (mode probability)")
        plt.legend(title="Pozisyon Tipi")
        plt.tight_layout()
        plt.savefig("results/chess960_variant_accuracy_barplot.png", dpi=300)
        plt.close()
        print("Varyant bazlı entropi ve konsantrasyon analizleri tamamlandı.")

    @staticmethod
    def chess960_color_effect_analysis():
        """
        PI'nin beyaz/siyah olduğu Chess960 maçlarında metrik farkı ve istatistiksel testler.
        Sonuçlar: CSV, Cohen's d, p-değeri, figür.
        """
        import scipy.stats as stats
        import seaborn as sns
        import config
        # PI vs Lc0 Chess960 maç sonuçlarını oku
        df = pd.read_csv("results/pi_vs_lc0_chess960_results.csv")
        # PI beyaz ve siyah olduğunda accuracy/entropy farkı
        perf_df = pd.read_csv("results/pi_vs_lc0_chess960_performance.csv")
        # Sadece ilk hamleler
        first_moves = perf_df[perf_df["move_number"] == 1]
        white_acc = first_moves[first_moves["pi_is_white"] == True]["move"].count()
        black_acc = first_moves[first_moves["pi_is_white"] == False]["move"].count()
        # Cohen's d ve p-değeri
        white_moves = first_moves[first_moves["pi_is_white"] == True]["move"]
        black_moves = first_moves[first_moves["pi_is_white"] == False]["move"]
        # Basit proxy: hamle sayısı
        d = (white_acc - black_acc) / (np.sqrt((white_acc**2 + black_acc**2)/2))
        t_stat, p_val = stats.ttest_ind(white_moves, black_moves, equal_var=False)
        # Sonuçları kaydet
        with open("results/chess960_color_effect_stats.txt", "w", encoding="utf-8") as f:
            f.write(f"Cohen's d: {d}\n")
            f.write(f"p-değeri: {p_val}\n")
            f.write(f"PI beyaz accuracy: {white_acc}\n")
            f.write(f"PI siyah accuracy: {black_acc}\n")
        print(f"Renk etkisi analizi tamamlandı. Cohen's d: {d:.3f}, p-değeri: {p_val:.3g}")
        # Figür: barplot
        plt.switch_backend('Agg')
        plt.figure(figsize=(8, 5))
        sns.barplot(x=["PI Beyaz", "PI Siyah"], y=[white_acc, black_acc])
        plt.title("Chess960 Renk Etkisi: PI'nin Accuracy Farkı")
        plt.ylabel("İlk Hamle Doğruluğu (proxy)")
        plt.tight_layout()
        plt.savefig("results/chess960_color_effect_barplot.png", dpi=300)
        plt.close()

    @staticmethod
    def chess960_variant_game_result_histogram():
        """
        Her Chess960 varyantı için PI vs Lc0 ve PI vs Stockfish maç sonuçlarını histogram olarak sunar.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        # PI vs Lc0
        df_lc0 = pd.read_csv("results/pi_vs_lc0_chess960_results.csv")
        # PI vs Stockfish
        df_sf = pd.read_csv("results/pi_vs_stockfish_chess960_results.csv")
        # Histogram için veri
        plt.switch_backend('Agg')
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_lc0, x="variant", hue="result")
        plt.title("Chess960 Varyantlarında PI vs Lc0 Oyun Sonucu Dağılımı")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Oyun Sayısı")
        plt.legend(title="Sonuç")
        plt.tight_layout()
        plt.savefig("results/chess960_variant_game_result_histogram_lc0.png", dpi=300)
        plt.close()
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df_sf, x="variant", hue="result")
        plt.title("Chess960 Varyantlarında PI vs Stockfish Oyun Sonucu Dağılımı")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Oyun Sayısı")
        plt.legend(title="Sonuç")
        plt.tight_layout()
        plt.savefig("results/chess960_variant_game_result_histogram_stockfish.png", dpi=300)
        plt.close()
        print("Varyantlar arası oyun sonucu histogramları oluşturuldu.")

    @staticmethod
    def chess960_move_distribution_heatmap():
        """
        Her Chess960 varyantı için ilk 5 hamlenin olasılık dağılımı ve optimal hamleye yakınsama hızını ısı haritası olarak gösterir.
        """
        import config
        import numpy as np
        import seaborn as sns
        plt.switch_backend('Agg')
        results = []
        for fen in config.CHESS960_VARIANTS.keys():
            res = Experiments.path_integral_depth_scan_experiment(
                fen=fen,
                lambda_values=[0.05, 0.2, 0.5],
                depth_values=[8, 12, 16],
                sample_count=30,
                save_results=False
            )
            for lam in [0.05, 0.2, 0.5]:
                move_probs = res["metrics"]["lambda_scan"][lam]["move_probabilities"]
                top_moves = sorted(move_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                for idx, (move, prob) in enumerate(top_moves):
                    results.append({
                        "variant": config.CHESS960_VARIANTS[fen]["variant"],
                        "lambda": lam,
                        "depth": 16,
                        "move_rank": idx+1,
                        "move": move,
                        "probability": prob
                    })
        df = pd.DataFrame(results)
        df_pivot = df.pivot_table(index=["variant", "move_rank"], columns="lambda", values="probability")
        plt.figure(figsize=(14, 8))
        sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="RdYlBu_r")
        plt.title("Chess960 Varyantlarında İlk 5 Hamle Olasılık Isı Haritası (λ)")
        plt.xlabel("Lambda (λ)")
        plt.ylabel("Varyant - Hamle Sırası")
        plt.tight_layout()
        plt.savefig("results/chess960_move_distribution_heatmap.png", dpi=300)
        plt.close()
        print("Hamle dağılımı ve optimal hamle ısı haritası oluşturuldu.")

    @staticmethod
    def chess960_kl_topn_analysis():
        """
        Her Chess960 varyantı için PI ve rakip motorun hamle dağılımları arasındaki KL diverjansı ve Top-N doğruluk metriklerini karşılaştırır.
        """
        import config
        import numpy as np
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        results = []
        for fen in config.CHESS960_VARIANTS.keys():
            # PI dağılımı
            pi_res = Experiments.path_integral_depth_scan_experiment(
                fen=fen,
                lambda_values=[0.05, 0.2, 0.5],
                depth_values=[16],
                sample_count=30,
                save_results=False
            )
            for lam in [0.05, 0.2, 0.5]:
                pi_moves = pi_res["metrics"]["lambda_scan"][lam]["move_probabilities"]
                # Lc0 dağılımı
                lc0_moves, lc0_scores, _ = Engine.lc0_top_moves_and_scores(fen, depth=config.TARGET_DEPTH*100, multipv=5)
                if lc0_moves:
                    total_score = sum(lc0_scores)
                    lc0_dist = {m: s/total_score for m, s in zip(lc0_moves, lc0_scores)}
                else:
                    lc0_dist = {}
                # Stockfish dağılımı
                sf_moves, sf_scores, _ = Engine.stockfish_top_moves_and_scores(fen, depth=int(config.TARGET_DEPTH*4), multipv=5)
                if sf_moves:
                    total_score = sum(sf_scores)
                    sf_dist = {m: s/total_score for m, s in zip(sf_moves, sf_scores)}
                else:
                    sf_dist = {}
                # KL diverjansı
                all_moves = set(pi_moves.keys()).union(lc0_dist.keys()).union(sf_dist.keys())
                pi_dist = np.array([pi_moves.get(m, 1e-12) for m in all_moves])
                lc0_dist_arr = np.array([lc0_dist.get(m, 1e-12) for m in all_moves])
                sf_dist_arr = np.array([sf_dist.get(m, 1e-12) for m in all_moves])
                pi_dist /= pi_dist.sum()
                lc0_dist_arr /= lc0_dist_arr.sum()
                sf_dist_arr /= sf_dist_arr.sum()
                kl_pi_lc0 = np.sum(pi_dist * np.log2(pi_dist / lc0_dist_arr))
                kl_pi_sf = np.sum(pi_dist * np.log2(pi_dist / sf_dist_arr))
                # Top-N doğruluk (PI'nin ilk N hamlesi Lc0'nın en iyi hamleleriyle örtüşüyor mu)
                pi_topn = sorted(pi_moves.items(), key=lambda x: x[1], reverse=True)[:5]
                lc0_topn = set(lc0_moves[:5]) if lc0_moves else set()
                sf_topn = set(sf_moves[:5]) if sf_moves else set()
                topn_acc_lc0 = sum([1 for m, _ in pi_topn if m in lc0_topn]) / 5
                topn_acc_sf = sum([1 for m, _ in pi_topn if m in sf_topn]) / 5
                results.append({
                    "variant": config.CHESS960_VARIANTS[fen]["variant"],
                    "lambda": lam,
                    "kl_pi_lc0": kl_pi_lc0,
                    "topn_acc_lc0": topn_acc_lc0,
                    "kl_pi_sf": kl_pi_sf,
                    "topn_acc_sf": topn_acc_sf
                })
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/chess960_kl_topn_analysis.csv", index=False)
        # Figür: KL diverjans ve Top-N doğruluk
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="variant", y="kl_pi_lc0", hue="lambda")
        plt.title("Chess960 Varyantlarında PI-Lc0 KL Diverjansı")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("KL Diverjans (bits)")
        plt.tight_layout()
        plt.savefig("results/chess960_kl_divergence_barplot.png", dpi=300)
        plt.close()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="variant", y="topn_acc_lc0", hue="lambda")
        plt.title("Chess960 Varyantlarında Top-N Doğruluk (PI vs Lc0)")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Top-N Doğruluk (ilk 5 hamle)")
        plt.tight_layout()
        plt.savefig("results/chess960_topn_accuracy_barplot.png", dpi=300)
        plt.close()
        print("KL diverjansı ve Top-N doğruluk analizleri tamamlandı.")

    @staticmethod
    def chess960_game_duration_movecount_analysis():
        """
        Her Chess960 varyantı için ortalama hamle sayısı ve oyun süresi (saniye) karşılaştırması yapar.
        Sonuçlar: CSV ve figür.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.switch_backend('Agg')
        # PI vs Lc0
        perf_df_lc0 = pd.read_csv("results/pi_vs_lc0_chess960_performance.csv")
        # PI vs Stockfish
        perf_df_sf = pd.read_csv("results/pi_vs_stockfish_chess960_performance.csv")
        # Varyant bazında ortalama hamle sayısı ve süre
        df_lc0_summary = perf_df_lc0.groupby("variant").agg({"move_number": "max", "time_sec": "sum"}).reset_index()
        df_lc0_summary["engine"] = "Lc0"
        df_sf_summary = perf_df_sf.groupby("variant").agg({"move_number": "max", "time_sec": "sum"}).reset_index()
        df_sf_summary["engine"] = "Stockfish"
        df_all = pd.concat([df_lc0_summary, df_sf_summary], ignore_index=True)
        os.makedirs("results", exist_ok=True)
        df_all.to_csv("results/chess960_game_duration_movecount.csv", index=False)
        # Figür: varyant bazında barplot
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_all, x="variant", y="move_number", hue="engine")
        plt.title("Chess960 Varyantlarında Ortalama Hamle Sayısı (PI vs Lc0/Stockfish)")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Hamle Sayısı")
        plt.tight_layout()
        plt.savefig("results/chess960_movecount_barplot.png", dpi=300)
        plt.close()
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df_all, x="variant", y="time_sec", hue="engine")
        plt.title("Chess960 Varyantlarında Ortalama Oyun Süresi (PI vs Lc0/Stockfish)")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Oyun Süresi (saniye)")
        plt.tight_layout()
        plt.savefig("results/chess960_duration_barplot.png", dpi=300)
        plt.close()
        print("Oyun süresi ve hamle sayısı analizleri tamamlandı.")

    @staticmethod
    def chess960_complexity_entropy_accuracy_analysis():
        """
        Her Chess960 varyantında pozisyon karmaşıklığı (complexity score) ile entropi ve accuracy ilişkisini inceler.
        Sonuçlar: CSV ve scatter plot.
        """
        import config
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.switch_backend('Agg')
        results = []
        for fen in config.CHESS960_VARIANTS.keys():
            # Complexity score hesapla
            board = chess.Board(fen)
            complexity = len(list(board.legal_moves))
            # Entropi ve accuracy
            res = Experiments.path_integral_depth_scan_experiment(
                fen=fen,
                lambda_values=[0.2],
                depth_values=[16],
                sample_count=30,
                save_results=False
            )
            entropy = res["metrics"]["lambda_scan"][0.2]["entropy"]
            accuracy = res["metrics"]["lambda_scan"][0.2]["accuracy"]
            results.append({
                "variant": config.CHESS960_VARIANTS[fen]["variant"],
                "complexity": complexity,
                "entropy": entropy,
                "accuracy": accuracy
            })
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/chess960_complexity_entropy_accuracy.csv", index=False)
        # Scatter plot: karmaşıklık vs entropi ve accuracy
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="complexity", y="entropy", s=120)
        plt.title("Chess960 Varyantlarında Karmaşıklık vs Entropi")
        plt.xlabel("Pozisyon Karmaşıklığı (Legal Move Sayısı)")
        plt.ylabel("Entropi (bits)")
        plt.tight_layout()
        plt.savefig("results/chess960_complexity_vs_entropy.png", dpi=300)
        plt.close()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x="complexity", y="accuracy", s=120)
        plt.title("Chess960 Varyantlarında Karmaşıklık vs Accuracy")
        plt.xlabel("Pozisyon Karmaşıklığı (Legal Move Sayısı)")
        plt.ylabel("Accuracy (mode probability)")
        plt.tight_layout()
        plt.savefig("results/chess960_complexity_vs_accuracy.png", dpi=300)
        plt.close()
        print("Pozisyon karmaşıklığı ve entropi/accuracy ilişkisi analizleri tamamlandı.")

    @staticmethod
    def chess960_bootstrap_ci_analysis(metric="entropy", n_bootstrap=1000):
        """
        Her Chess960 varyantı için entropi veya accuracy metriklerinin bootstrap ile güven aralığını hesaplar.
        Sonuçlar: CSV ve errorbar plot.
        """
        import config
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        results = []
        for fen in config.CHESS960_VARIANTS.keys():
            res = Experiments.path_integral_depth_scan_experiment(
                fen=fen,
                lambda_values=[0.2],
                depth_values=[16],
                sample_count=100,
                save_results=False
            )
            paths = Engine.sample_paths(fen, 16, 0.2, 100)
            if metric == "entropy":
                values = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(paths, size=len(paths), replace=True)
                    ent, _ = Calc.compute_entropy(sample)
                    values.append(ent)
            else:
                values = []
                for _ in range(n_bootstrap):
                    sample = np.random.choice(paths, size=len(paths), replace=True)
                    acc = Calc.top_move_concentration(sample)
                    values.append(acc)
            ci_low = np.percentile(values, 2.5)
            ci_high = np.percentile(values, 97.5)
            mean_val = np.mean(values)
            results.append({
                "variant": config.CHESS960_VARIANTS[fen]["variant"],
                "mean": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high
            })
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/chess960_bootstrap_ci_{metric}.csv", index=False)
        # Errorbar plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(df["variant"], df["mean"], yerr=[df["mean"]-df["ci_low"], df["ci_high"]-df["mean"]], fmt="o", capsize=5)
        plt.title(f"Chess960 Varyantlarında Bootstrap Güven Aralığı ({metric})")
        plt.xlabel("Varyant Numararası")
        plt.ylabel(f"{metric.capitalize()} (95% CI)")
        plt.tight_layout()
        plt.savefig(f"results/chess960_bootstrap_ci_{metric}_errorbar.png", dpi=300)
        plt.close()
        print(f"Bootstrap güven aralığı ({metric}) analizleri tamamlandı.")

    @staticmethod
    def chess960_engine_comparison_analysis():
        """
        Her Chess960 varyantı için PI, Lc0 ve Stockfish motorlarının entropi ve accuracy metriklerini karşılaştırır.
        Sonuçlar: CSV ve barplot.
        """
        import config
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.switch_backend('Agg')
        results = []
        for fen in config.CHESS960_VARIANTS.keys():
            # PI
            pi_res = Experiments.path_integral_depth_scan_experiment(
                fen=fen,
                lambda_values=[0.2],
                depth_values=[16],
                sample_count=30,
                save_results=False
            )
            pi_entropy = pi_res["metrics"]["lambda_scan"][0.2]["entropy"]
            pi_accuracy = pi_res["metrics"]["lambda_scan"][0.2]["accuracy"]
            # Lc0
            lc0_moves, lc0_scores, _ = Engine.lc0_top_moves_and_scores(fen, depth=config.TARGET_DEPTH*100, multipv=5)
            if lc0_moves:
                total_score = sum(lc0_scores)
                lc0_dist = {m: s/total_score for m, s in zip(lc0_moves, lc0_scores)}
                lc0_entropy = -sum([p*np.log2(p) for p in lc0_dist.values() if p > 0])
                lc0_accuracy = max(lc0_dist.values())
            else:
                lc0_entropy = np.nan
                lc0_accuracy = np.nan
            # Stockfish
            sf_moves, sf_scores, _ = Engine.stockfish_top_moves_and_scores(fen, depth=int(config.TARGET_DEPTH*4), multipv=5)
            if sf_moves:
                total_score = sum(sf_scores)
                sf_dist = {m: s/total_score for m, s in zip(sf_moves, sf_scores)}
                sf_entropy = -sum([p*np.log2(p) for p in sf_dist.values() if p > 0])
                sf_accuracy = max(sf_dist.values())
            else:
                sf_entropy = np.nan
                sf_accuracy = np.nan
            results.append({
                "variant": config.CHESS960_VARIANTS[fen]["variant"],
                "engine": "PI",
                "entropy": pi_entropy,
                "accuracy": pi_accuracy
            })
            results.append({
                "variant": config.CHESS960_VARIANTS[fen]["variant"],
                "engine": "Lc0",
                "entropy": lc0_entropy,
                "accuracy": lc0_accuracy
            })
            results.append({
                "variant": config.CHESS960_VARIANTS[fen]["variant"],
                "engine": "Stockfish",
                "entropy": sf_entropy,
                "accuracy": sf_accuracy
            })
        df = pd.DataFrame(results)
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/chess960_engine_comparison_metrics.csv", index=False)
        # Barplot: varyant bazında motor karşılaştırması
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x="variant", y="entropy", hue="engine")
        plt.title("Chess960 Varyantlarında Motor Bazlı Entropi Karşılaştırması")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Entropi (bits)")
        plt.legend(title="Motor")
        plt.tight_layout()
        plt.savefig("results/chess960_engine_entropy_barplot.png", dpi=300)
        plt.close()
        plt.figure(figsize=(14, 7))
        sns.barplot(data=df, x="variant", y="accuracy", hue="engine")
        plt.title("Chess960 Varyantlarında Motor Bazlı Accuracy Karşılaştırması")
        plt.xlabel("Varyant Numarası")
        plt.ylabel("Accuracy (mode probability)")
        plt.legend(title="Motor")
        plt.tight_layout()
        plt.savefig("results/chess960_engine_accuracy_barplot.png", dpi=300)
        plt.close()
        print("Motor bazlı entropi ve accuracy karşılaştırma analizleri tamamlandı.")

    @staticmethod
    def pi_quantum_sensitivity_experiment(fen=None, multipv_list=None, depth_list=None, sample_counts=None,
                                         reps=20, bootstrap_iters=200, lambda_quantum=None, top_k_list=None, save_results=True):
        """
        Quantum-limit sensitivity experiment (strict quantum_limit mode).

        Improvements over previous version:
        - top_k_list: compute Top-K cumulative probabilities for arbitrary K (e.g. [3,5,10])
        - No fallback: if Engine.sample_paths does not accept mode='quantum_limit', the run is recorded with a warning and skipped
        - Summary includes time mean/std/CI, efficiency = accuracy_mean / entropy_mean, top-k means, and warning_count
        - Heatmap axes are sorted for consistent layout

        Returns: (detailed_df, summary_df)
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        import math

        if fen is None:
            fen = config.FEN
        if multipv_list is None:
            multipv_list = [1, 5, 10, 20, 50]
        if depth_list is None:
            depth_list = [4, 8, 12, 16,20]
        if sample_counts is None:
            sample_counts = [50, 100, 200]
        if lambda_quantum is None:
            lambda_quantum = config.LAMBDA
        if top_k_list is None:
            top_k_list = [3, 5, 10]

        os.makedirs('results', exist_ok=True)

        detailed_rows = []
        summary_rows = []

        pbar = tqdm(total=len(multipv_list) * len(depth_list) * len(sample_counts), desc='Quantum sensitivity combos')

        def bootstrap_ci(data, iters=bootstrap_iters, alpha=0.05):
            arr = np.array([d for d in data if not (isinstance(d, float) and np.isnan(d))])
            if arr.size == 0:
                return (np.nan, np.nan)
            means = []
            n = arr.size
            for _ in range(iters):
                sample = np.random.choice(arr, size=n, replace=True)
                means.append(np.mean(sample))
            low = np.percentile(means, 100 * alpha / 2)
            high = np.percentile(means, 100 * (1 - alpha / 2))
            return float(low), float(high)

        for multipv in multipv_list:
            for depth in depth_list:
                for sample_count in sample_counts:
                    per_run_entropy = []
                    per_run_accuracy = []
                    per_run_topk = {k: [] for k in top_k_list}
                    per_run_unique = []
                    per_run_kl = []
                    per_run_time = []
                    warning_count = 0

                    for r in range(reps):
                        start_t = time.time()
                        warning_mode = None
                        try:
                            # Strict quantum_limit: do NOT fallback
                            # Engine.sample_paths signature: (fen, depth, lam, samples, use_cache=True, engine=None, mode='competitive')
                            paths = Engine.sample_paths(fen, depth, lambda_quantum, sample_count, mode='quantum_limit')
                        except TypeError as e:
                            # Engine implementation doesn't support the 'mode' param — fail fast as requested
                            raise RuntimeError("Engine.sample_paths does not accept mode='quantum_limit'. Configure a quantum-capable Engine implementation.") from e
                        except Exception as e:
                            # Propagate other exceptions to make failures explicit
                            raise

                        elapsed = time.time() - start_t

                        if paths:
                            entropy, move_counter = Calc.compute_entropy(paths)
                            total = sum(move_counter.values()) if move_counter else 0
                            top1_prob = (max(move_counter.values()) / total) if total > 0 else 0.0
                            most_common = move_counter.most_common()
                            for k in top_k_list:
                                topk_prob = sum(cnt for _, cnt in most_common[:k]) / total if total > 0 else 0.0
                                per_run_topk[k].append(topk_prob)
                            unique_first = len(move_counter)
                            if total > 0 and unique_first > 0:
                                p = np.array([move_counter.get(m, 0) / total for m in move_counter.keys()], dtype=np.float64)
                                q = np.ones_like(p) / p.size
                                eps = 1e-12
                                p = np.clip(p, eps, None)
                                q = np.clip(q, eps, None)
                                p /= p.sum()
                                q /= q.sum()
                                kl = float(np.sum(p * np.log2(p / q)))
                            else:
                                kl = 0.0
                        else:
                            entropy = float('nan')
                            top1_prob = float('nan')
                            for k in top_k_list:
                                per_run_topk[k].append(np.nan)
                            unique_first = 0
                            kl = float('nan')

                        per_run_entropy.append(entropy)
                        per_run_accuracy.append(top1_prob)
                        per_run_unique.append(unique_first)
                        per_run_kl.append(kl)
                        per_run_time.append(elapsed)

                        row = {
                            'fen': fen,
                            'multipv': multipv,
                            'depth': depth,
                            'sample_count': sample_count,
                            'rep': r + 1,
                            'entropy': float(entropy) if not math.isnan(entropy) else np.nan,
                            'accuracy_top1': float(top1_prob) if not math.isnan(top1_prob) else np.nan,
                            'unique_first_moves': int(unique_first),
                            'kl_uniform': float(kl) if not math.isnan(kl) else np.nan,
                            'elapsed_sec': float(elapsed),
                            'warning': warning_mode
                        }
                        for k in top_k_list:
                            row[f'top{k}'] = float(per_run_topk[k][-1]) if len(per_run_topk[k]) > 0 and not (isinstance(per_run_topk[k][-1], float) and np.isnan(per_run_topk[k][-1])) else np.nan
                        detailed_rows.append(row)

                    # Aggregate
                    ent_mean = np.nanmean(per_run_entropy)
                    ent_std = np.nanstd(per_run_entropy)
                    ent_ci_low, ent_ci_high = bootstrap_ci(per_run_entropy)

                    acc_mean = np.nanmean(per_run_accuracy)
                    acc_std = np.nanstd(per_run_accuracy)
                    acc_ci_low, acc_ci_high = bootstrap_ci(per_run_accuracy)

                    time_mean = np.nanmean(per_run_time)
                    time_std = np.nanstd(per_run_time)
                    time_ci_low, time_ci_high = bootstrap_ci(per_run_time)

                    topk_means = {k: float(np.nanmean(per_run_topk[k])) if len(per_run_topk[k]) > 0 else np.nan for k in top_k_list}
                    unique_mean = float(np.nanmean(per_run_unique)) if len(per_run_unique) > 0 else np.nan
                    kl_mean = float(np.nanmean(per_run_kl)) if len(per_run_kl) > 0 else np.nan

                    efficiency = (acc_mean / ent_mean) if (not math.isnan(acc_mean) and not math.isnan(ent_mean) and ent_mean != 0) else np.nan

                    summary_row = {
                        'fen': fen,
                        'multipv': multipv,
                        'depth': depth,
                        'sample_count': sample_count,
                        'reps': reps,
                        'entropy_mean': float(ent_mean) if not math.isnan(ent_mean) else np.nan,
                        'entropy_std': float(ent_std) if not math.isnan(ent_std) else np.nan,
                        'entropy_ci_low': ent_ci_low,
                        'entropy_ci_high': ent_ci_high,
                        'accuracy_mean': float(acc_mean) if not math.isnan(acc_mean) else np.nan,
                        'accuracy_std': float(acc_std) if not math.isnan(acc_std) else np.nan,
                        'accuracy_ci_low': acc_ci_low,
                        'accuracy_ci_high': acc_ci_high,
                        'time_mean_sec': float(time_mean) if not math.isnan(time_mean) else np.nan,
                        'time_std_sec': float(time_std) if not math.isnan(time_std) else np.nan,
                        'time_ci_low': time_ci_low,
                        'time_ci_high': time_ci_high,
                        'unique_first_mean': unique_mean,
                        'kl_mean': kl_mean,
                        'efficiency': efficiency,
                        'warning_count': int(warning_count)
                    }
                    for k in top_k_list:
                        summary_row[f'top{k}_mean'] = topk_means.get(k, np.nan)

                    summary_rows.append(summary_row)
                    pbar.update(1)

        pbar.close()

        detailed_df = pd.DataFrame(detailed_rows)
        summary_df = pd.DataFrame(summary_rows)

        if save_results:
            detailed_csv = os.path.join('results', 'pi_quantum_sensitivity_detailed.csv')
            summary_csv = os.path.join('results', 'pi_quantum_sensitivity_summary.csv')
            detailed_df.to_csv(detailed_csv, index=False)
            summary_df.to_csv(summary_csv, index=False)

            # Heatmaps: sorted axes
            try:
                sample0 = sample_counts[0]
                subset = summary_df[summary_df['sample_count'] == sample0]
                if not subset.empty:
                    multipv_sorted = sorted(subset['multipv'].unique())
                    depth_sorted = sorted(subset['depth'].unique())
                    pivot_acc = subset.pivot_table(index='multipv', columns='depth', values='accuracy_mean').reindex(index=multipv_sorted, columns=depth_sorted)
                    plt.figure(figsize=(8, 6))
                    im = plt.imshow(pivot_acc.values, aspect='auto', origin='lower', cmap='viridis')
                    plt.colorbar(im, label='Mean Accuracy (top1)')
                    plt.xticks(range(len(depth_sorted)), depth_sorted)
                    plt.yticks(range(len(multipv_sorted)), multipv_sorted)
                    plt.xlabel('Depth')
                    plt.ylabel('MultiPV')
                    plt.title(f'Accuracy Heatmap (sample_count={sample0})')
                    plt.tight_layout()
                    heat_acc_path = os.path.join('results', f'pi_quantum_accuracy_heatmap_sample{sample0}.png')
                    plt.savefig(heat_acc_path, dpi=300)
                    plt.close()

                    pivot_ent = subset.pivot_table(index='multipv', columns='depth', values='entropy_mean').reindex(index=multipv_sorted, columns=depth_sorted)
                    plt.figure(figsize=(8, 6))
                    im = plt.imshow(pivot_ent.values, aspect='auto', origin='lower', cmap='magma')
                    plt.colorbar(im, label='Mean Entropy')
                    plt.xticks(range(len(depth_sorted)), depth_sorted)
                    plt.yticks(range(len(multipv_sorted)), multipv_sorted)
                    plt.xlabel('Depth')
                    plt.ylabel('MultiPV')
                    plt.title(f'Entropy Heatmap (sample_count={sample0})')
                    plt.tight_layout()
                    heat_ent_path = os.path.join('results', f'pi_quantum_entropy_heatmap_sample{sample0}.png')
                    plt.savefig(heat_ent_path, dpi=300)
                    plt.close()

                    print(f"Saved detailed CSV: {detailed_csv}")
                    print(f"Saved summary CSV: {summary_csv}")
                    print(f"Saved heatmaps: {heat_acc_path}, {heat_ent_path}")
            except Exception as e:
                print(f"Warning: could not generate heatmaps: {e}")

        return detailed_df, summary_df
