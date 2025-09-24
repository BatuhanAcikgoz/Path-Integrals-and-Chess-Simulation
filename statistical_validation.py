"""
İstatistiksel analiz ve hipotez testleri modülü
Bu modül akademik rigor için gerekli istatistiksel testleri sağlar.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings

class StatisticalValidator:
    """Akademik çalışma için istatistiksel validasyon sınıfı"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results_log = []

    def test_path_integral_convergence(self, lambda_values: List[float],
                                     entropy_values: List[float]) -> Dict[str, Any]:
        """Path integral convergence hipotezini test et"""
        print("\n🔬 Testing Path Integral Convergence Hypothesis...")

        # H0: Lambda artışı ile entropy arasında monoton ilişki yok
        # H1: Lambda artışı ile entropy arasında monoton azalış var

        tau, p_value = stats.kendalltau(lambda_values, entropy_values)

        result = {
            'test_name': 'Kendall Tau Convergence Test',
            'tau': tau,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': abs(tau),
            'interpretation': self._interpret_correlation(tau, p_value)
        }

        self.results_log.append(result)
        print(f"   Kendall's τ = {tau:.4f}, p = {p_value:.6f}")
        print(f"   {'✓ Significant' if result['is_significant'] else '✗ Not significant'}")

        return result

    def test_quantum_classical_transition(self, low_lambda_data: List[float],
                                        high_lambda_data: List[float]) -> Dict[str, Any]:
        """Quantum-classical transition hipotezini test et"""
        print("\n🔬 Testing Quantum-Classical Transition Hypothesis...")

        # Mann-Whitney U test (non-parametric)
        statistic, p_value = mannwhitneyu(low_lambda_data, high_lambda_data,
                                         alternative='two-sided')

        # Effect size (r = Z / sqrt(N))
        n1, n2 = len(low_lambda_data), len(high_lambda_data)
        z_score = stats.norm.ppf(p_value/2)
        effect_size = abs(z_score) / np.sqrt(n1 + n2)

        result = {
            'test_name': 'Mann-Whitney U Test (Quantum vs Classical)',
            'statistic': statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': effect_size,
            'mean_difference': np.mean(high_lambda_data) - np.mean(low_lambda_data),
            'interpretation': self._interpret_transition_test(p_value, effect_size)
        }

        self.results_log.append(result)
        print(f"   U-statistic = {statistic:.2f}, p = {p_value:.6f}")
        print(f"   Effect size r = {effect_size:.4f}")
        print(f"   {'✓ Significant transition' if result['is_significant'] else '✗ No significant transition'}")

        return result

    def test_position_complexity_correlation(self, complexity_scores: List[float],
                                           performance_metrics: List[float]) -> Dict[str, Any]:
        """Pozisyon karmaşıklığı ile performans korelasyonunu test et"""
        print("\n🔬 Testing Position Complexity-Performance Correlation...")

        # Pearson ve Spearman korelasyonları
        pearson_r, pearson_p = stats.pearsonr(complexity_scores, performance_metrics)
        spearman_rho, spearman_p = stats.spearmanr(complexity_scores, performance_metrics)

        result = {
            'test_name': 'Complexity-Performance Correlation',
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_rho': spearman_rho,
            'spearman_p': spearman_p,
            'is_pearson_significant': pearson_p < self.alpha,
            'is_spearman_significant': spearman_p < self.alpha,
            'interpretation': self._interpret_correlation_test(pearson_r, pearson_p, spearman_rho, spearman_p)
        }

        self.results_log.append(result)
        print(f"   Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
        print(f"   Spearman ρ = {spearman_rho:.4f}, p = {spearman_p:.6f}")

        return result

    def bonferroni_correction(self, p_values: List[float]) -> List[float]:
        """Bonferroni multiple comparison correction"""
        n_tests = len(p_values)
        corrected_alpha = self.alpha / n_tests
        corrected_p_values = [p * n_tests for p in p_values]

        print(f"\n📊 Bonferroni Correction Applied:")
        print(f"   Original α = {self.alpha}")
        print(f"   Corrected α = {corrected_alpha:.6f} (for {n_tests} tests)")

        return corrected_p_values

    def power_analysis(self, effect_size: float, sample_size: int,
                      test_type: str = 'two-sample') -> Dict[str, float]:
        """Statistical power analysis"""
        if test_type == 'two-sample':
            # Cohen's conventions: small=0.2, medium=0.5, large=0.8
            power = stats.ttest_ind_from_stats(
                mean1=0, std1=1, nobs1=sample_size//2,
                mean2=effect_size, std2=1, nobs2=sample_size//2
            ).pvalue
            power = 1 - power
        else:
            power = 0.8  # Default assumption

        result = {
            'effect_size': effect_size,
            'sample_size': sample_size,
            'power': power,
            'adequate_power': power >= 0.8
        }

        return result

    def generate_statistical_report(self) -> str:
        """Comprehensive statistical report"""
        report = "\n" + "="*60 + "\n"
        report += "STATISTICAL VALIDATION REPORT\n"
        report += "="*60 + "\n\n"

        significant_tests = [r for r in self.results_log if r.get('is_significant', False)]
        total_tests = len(self.results_log)

        report += f"Total Statistical Tests Conducted: {total_tests}\n"
        report += f"Significant Results: {len(significant_tests)}\n"
        report += f"Significance Rate: {len(significant_tests)/total_tests*100:.1f}%\n\n"

        for i, result in enumerate(self.results_log, 1):
            report += f"{i}. {result['test_name']}:\n"
            report += f"   p-value: {result.get('p_value', 'N/A'):.6f}\n"
            report += f"   Significant: {'Yes' if result.get('is_significant') else 'No'}\n"
            report += f"   Interpretation: {result.get('interpretation', 'N/A')}\n\n"

        # Multiple comparison warning
        if total_tests > 1:
            report += "⚠️  Multiple Comparison Note:\n"
            report += f"   With {total_tests} tests, consider Bonferroni correction.\n"
            report += f"   Adjusted α = {self.alpha/total_tests:.6f}\n\n"

        return report

    def _interpret_correlation(self, tau: float, p_value: float) -> str:
        """Interpret correlation results"""
        if p_value >= self.alpha:
            return "No significant monotonic relationship detected"

        if abs(tau) < 0.1:
            return "Significant but very weak monotonic relationship"
        elif abs(tau) < 0.3:
            return "Significant weak monotonic relationship"
        elif abs(tau) < 0.5:
            return "Significant moderate monotonic relationship"
        else:
            return "Significant strong monotonic relationship"

    def _interpret_transition_test(self, p_value: float, effect_size: float) -> str:
        """Interpret transition test results"""
        if p_value >= self.alpha:
            return "No significant difference between quantum and classical regimes"

        if effect_size < 0.1:
            return "Significant but small effect size transition"
        elif effect_size < 0.3:
            return "Significant medium effect size transition"
        else:
            return "Significant large effect size transition"

    def _interpret_correlation_test(self, pearson_r: float, pearson_p: float,
                                   spearman_rho: float, spearman_p: float) -> str:
        """Interpret correlation test results"""
        interpretations = []

        if pearson_p < self.alpha:
            interpretations.append(f"Significant linear correlation (r={pearson_r:.3f})")

        if spearman_p < self.alpha:
            interpretations.append(f"Significant monotonic correlation (ρ={spearman_rho:.3f})")

        if not interpretations:
            return "No significant correlations detected"

        return "; ".join(interpretations)

def create_statistical_plots(validator: StatisticalValidator, save_dir: str = "results"):
    """
    Generates statistical validation plots using the results in validator.results_log.
    - P-value: Shows the significance level of statistical tests performed between different metrics (e.g., entropy, accuracy) and parameters (lambda, nodes).
    - Effect size: Indicates the practical importance of the observed difference (e.g., Cohen's d, Tau, r).
    - validator.results_log contains a summary of all tests performed (test name, p-value, effect size, confidence interval, etc.).
    - Now also logs: engine type, mode, lambda, depth, multipv, sample_count, and sample info for full reproducibility.
    """

    # Parametre ve örneklem özetini oluştur
    # Varsayılan: validator nesnesinde veya validator.results_log[0] içinde parametreler varsa al
    param_info = validator.results_log[0] if validator.results_log else {}
    engine = param_info.get('engine', 'unknown')
    mode = param_info.get('mode', 'unknown')
    lambda_val = param_info.get('lambda', 'N/A')
    depth = param_info.get('depth', 'N/A')
    multipv = param_info.get('multipv', 'N/A')
    sample_count = param_info.get('sample_count', 'N/A')
    sample_desc = param_info.get('sample_desc', 'N/A')
    param_summary = f"Motor: {engine}, Mod: {mode}, λ={lambda_val}, depth={depth}, multipv={multipv}, örneklem={sample_count}, örnek: {sample_desc}"

    # Save validator.results_log to a file (write summary of each test)
    with open(f"{save_dir}/statistical_test_log.txt", "w", encoding="utf-8") as f:
        f.write(f"# Parametreler: {param_summary}\n")
        f.write("# validator.results_log: Summary results of statistical tests performed\n")
        for r in validator.results_log:
            f.write(f"Test: {r.get('test_name', 'N/A')}, p-value: {r.get('p_value', 'N/A')}, effect size: {r.get('effect_size', 'N/A')}, extra info: {r}\n")

    # 1. P-value distribution
    # P-value shows the significance rate and overall reliability of the tests performed.
    p_values = [r.get('p_value') for r in validator.results_log if r.get('p_value') is not None]

    if p_values:
        plt.figure(figsize=(10, 6))
        plt.hist(p_values, bins=20, alpha=0.7, edgecolor='black')
        plt.axvline(validator.alpha, color='red', linestyle='--', label=f'α = {validator.alpha}')
        plt.xlabel('P-values')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of P-values from Statistical Tests\n{param_summary}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_p_value_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Effect sizes
    # Effect size shows the practical importance and strength of the observed difference in the tests performed.
    effect_sizes = [r.get('effect_size') for r in validator.results_log if r.get('effect_size') is not None]
    test_names = [r.get('test_name', f'Test {i}') for i, r in enumerate(validator.results_log)
                  if r.get('effect_size') is not None]

    if effect_sizes:
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(effect_sizes)), effect_sizes, alpha=0.7)
        plt.axhline(0.1, color='red', linestyle='--', alpha=0.5, label='Small effect')
        plt.axhline(0.3, color='orange', linestyle='--', alpha=0.5, label='Medium effect')
        plt.axhline(0.5, color='green', linestyle='--', alpha=0.5, label='Large effect')
        plt.xlabel('Statistical Tests')
        plt.ylabel('Effect Size')
        plt.title(f'Effect Sizes of Statistical Tests\n{param_summary}')
        plt.xticks(range(len(test_names)), test_names, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_effect_sizes.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. CSV meta satırı
    # Eğer bir CSV çıktı üretiliyorsa, ilk satıra param_summary eklenmeli
    # (Burada örnek kod, gerçek CSV üretimi fonksiyonunda eklenmeli)
    # with open(f"{save_dir}/statistical_results.csv", "w", encoding="utf-8") as f:
    #     f.write(f"# {param_summary}\n")
    #     ...

# Integration with main analysis
def integrate_statistical_validation():
    """Main integration function for statistical validation"""
    print("\n🔬 STATISTICAL VALIDATION MODULE")
    print("=" * 50)

    validator = StatisticalValidator(alpha=0.05)

    # This will be called from main.py with actual data
    return validator
