"""
İstatistiksel analiz ve hipotez testleri modülü
Bu modül akademik rigor için gerekli istatistiksel testleri sağlar.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, ks_2samp, chi2_contingency
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from typing import Dict, List, Tuple, Any
import warnings
import os

class StatisticalValidator:
    """Akademik çalışma için istatistiksel validasyon sınıfı"""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results_log = []
        self.effect_sizes = []
        self.confidence_intervals = []
        self.power_analyses = []

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

    def test_normality(self, data: List[float], test_name: str = "Shapiro-Wilk") -> Dict[str, Any]:
        """Test for normality using Shapiro-Wilk or Kolmogorov-Smirnov tests"""
        print(f"\n🔬 Testing Normality: {test_name}...")
        
        if test_name == "Shapiro-Wilk":
            statistic, p_value = stats.shapiro(data)
        elif test_name == "Kolmogorov-Smirnov":
            statistic, p_value = stats.kstest(data, 'norm')
        else:
            raise ValueError("Unsupported normality test")
        
        result = {
            'test_name': f'Normality Test ({test_name})',
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': p_value > self.alpha,
            'interpretation': f"Data {'appears' if p_value > self.alpha else 'does not appear'} to be normally distributed"
        }
        
        self.results_log.append(result)
        print(f"   {test_name} statistic = {statistic:.4f}, p = {p_value:.6f}")
        print(f"   {'✓ Normal distribution' if result['is_normal'] else '✗ Non-normal distribution'}")
        
        return result

    def test_homoscedasticity(self, *groups) -> Dict[str, Any]:
        """Test for equal variances using Levene's test"""
        print(f"\n🔬 Testing Homoscedasticity (Equal Variances)...")
        
        statistic, p_value = stats.levene(*groups)
        
        result = {
            'test_name': 'Levene Test for Equal Variances',
            'statistic': statistic,
            'p_value': p_value,
            'is_homoscedastic': p_value > self.alpha,
            'interpretation': f"Variances {'appear' if p_value > self.alpha else 'do not appear'} to be equal"
        }
        
        self.results_log.append(result)
        print(f"   Levene statistic = {statistic:.4f}, p = {p_value:.6f}")
        print(f"   {'✓ Equal variances' if result['is_homoscedastic'] else '✗ Unequal variances'}")
        
        return result

    def test_independence(self, data1: List[float], data2: List[float]) -> Dict[str, Any]:
        """Test for independence using chi-square test"""
        print(f"\n🔬 Testing Independence...")
        
        # Create contingency table (simplified approach)
        median1, median2 = np.median(data1), np.median(data2)
        
        # Categorize data as above/below median
        cat1 = [1 if x > median1 else 0 for x in data1]
        cat2 = [1 if x > median2 else 0 for x in data2]
        
        # Create contingency table
        contingency = pd.crosstab(cat1, cat2)
        
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        result = {
            'test_name': 'Chi-square Test of Independence',
            'chi2_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'is_independent': p_value > self.alpha,
            'interpretation': f"Variables {'appear' if p_value > self.alpha else 'do not appear'} to be independent"
        }
        
        self.results_log.append(result)
        print(f"   χ² = {chi2:.4f}, p = {p_value:.6f}, df = {dof}")
        print(f"   {'✓ Independent' if result['is_independent'] else '✗ Dependent'}")
        
        return result

    def bootstrap_confidence_interval(self, data: List[float], statistic_func=np.mean, 
                                    confidence_level: float = 0.95, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """Calculate bootstrap confidence interval for any statistic"""
        print(f"\n🔬 Bootstrap Confidence Interval Analysis...")
        
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))
        
        alpha_level = 1 - confidence_level
        lower_percentile = (alpha_level / 2) * 100
        upper_percentile = (1 - alpha_level / 2) * 100
        
        ci_lower = np.percentile(bootstrap_stats, lower_percentile)
        ci_upper = np.percentile(bootstrap_stats, upper_percentile)
        
        original_stat = statistic_func(data)
        
        result = {
            'test_name': f'Bootstrap CI ({statistic_func.__name__})',
            'original_statistic': original_stat,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
            'ci_width': ci_upper - ci_lower,
            'interpretation': f"{confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]"
        }
        
        self.confidence_intervals.append(result)
        print(f"   Original {statistic_func.__name__}: {original_stat:.4f}")
        print(f"   {confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
        
        return result

    def effect_size_analysis(self, group1: List[float], group2: List[float], 
                           effect_type: str = "cohen_d") -> Dict[str, Any]:
        """Calculate various effect size measures"""
        print(f"\n🔬 Effect Size Analysis: {effect_type}...")
        
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)
        
        if effect_type == "cohen_d":
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            effect_size = (mean1 - mean2) / pooled_std
            
            # Interpret Cohen's d
            if abs(effect_size) < 0.2:
                interpretation = "Negligible effect"
            elif abs(effect_size) < 0.5:
                interpretation = "Small effect"
            elif abs(effect_size) < 0.8:
                interpretation = "Medium effect"
            else:
                interpretation = "Large effect"
                
        elif effect_type == "glass_delta":
            effect_size = (mean1 - mean2) / std2
            interpretation = f"Glass's Δ = {effect_size:.3f}"
            
        elif effect_type == "hedges_g":
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            correction_factor = 1 - (3 / (4 * (n1 + n2) - 9))
            effect_size = ((mean1 - mean2) / pooled_std) * correction_factor
            interpretation = f"Hedges' g = {effect_size:.3f}"
            
        else:
            raise ValueError("Unsupported effect size type")
        
        result = {
            'test_name': f'Effect Size ({effect_type})',
            'effect_size': effect_size,
            'effect_type': effect_type,
            'group1_mean': mean1,
            'group2_mean': mean2,
            'group1_std': std1,
            'group2_std': std2,
            'interpretation': interpretation
        }
        
        self.effect_sizes.append(result)
        print(f"   {effect_type}: {effect_size:.4f}")
        print(f"   Interpretation: {interpretation}")
        
        return result

    def comprehensive_anova(self, *groups, group_names: List[str] = None) -> Dict[str, Any]:
        """Comprehensive one-way ANOVA with post-hoc tests"""
        print(f"\n🔬 Comprehensive ANOVA Analysis...")
        
        if group_names is None:
            group_names = [f"Group_{i+1}" for i in range(len(groups))]
        
        # One-way ANOVA
        f_statistic, p_value = stats.f_oneway(*groups)
        
        # Effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_total = sum((x - grand_mean)**2 for x in all_data)
        eta_squared = ss_between / ss_total
        
        result = {
            'test_name': 'One-way ANOVA',
            'f_statistic': f_statistic,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'eta_squared': eta_squared,
            'interpretation': self._interpret_anova(p_value, eta_squared)
        }
        
        # Post-hoc tests if significant
        if result['is_significant'] and len(groups) > 2:
            print("   Performing post-hoc pairwise comparisons...")
            pairwise_results = []
            
            for i in range(len(groups)):
                for j in range(i+1, len(groups)):
                    t_stat, p_val = stats.ttest_ind(groups[i], groups[j])
                    pairwise_results.append({
                        'comparison': f"{group_names[i]} vs {group_names[j]}",
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'is_significant': p_val < (self.alpha / len(groups))  # Bonferroni correction
                    })
            
            result['post_hoc'] = pairwise_results
        
        self.results_log.append(result)
        print(f"   F({len(groups)-1}, {len(all_data)-len(groups)}) = {f_statistic:.4f}, p = {p_value:.6f}")
        print(f"   η² = {eta_squared:.4f}")
        print(f"   {'✓ Significant' if result['is_significant'] else '✗ Not significant'}")
        
        return result

    def bayesian_t_test(self, group1: List[float], group2: List[float]) -> Dict[str, Any]:
        """Bayesian t-test using BIC approximation"""
        print(f"\n🔬 Bayesian T-test Analysis...")
        
        # Classical t-test
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        # BIC approximation for Bayes Factor
        n1, n2 = len(group1), len(group2)
        n = n1 + n2
        
        # BIC for null model (no difference)
        bic_null = n * np.log(np.var(np.concatenate([group1, group2]), ddof=1))
        
        # BIC for alternative model (difference exists)
        pooled_var = ((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n-2)
        bic_alt = n * np.log(pooled_var) + np.log(n)
        
        # Bayes Factor (BF10)
        bayes_factor = np.exp((bic_null - bic_alt) / 2)
        
        # Interpret Bayes Factor
        if bayes_factor > 10:
            bf_interpretation = "Strong evidence for H1"
        elif bayes_factor > 3:
            bf_interpretation = "Moderate evidence for H1"
        elif bayes_factor > 1:
            bf_interpretation = "Weak evidence for H1"
        elif bayes_factor > 0.33:
            bf_interpretation = "Weak evidence for H0"
        elif bayes_factor > 0.1:
            bf_interpretation = "Moderate evidence for H0"
        else:
            bf_interpretation = "Strong evidence for H0"
        
        result = {
            'test_name': 'Bayesian T-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'bayes_factor': bayes_factor,
            'log_bayes_factor': np.log10(bayes_factor),
            'interpretation': bf_interpretation
        }
        
        self.results_log.append(result)
        print(f"   t = {t_stat:.4f}, p = {p_value:.6f}")
        print(f"   BF₁₀ = {bayes_factor:.4f}")
        print(f"   Interpretation: {bf_interpretation}")
        
        return result

    def robust_statistics_analysis(self, data: List[float]) -> Dict[str, Any]:
        """Comprehensive robust statistics analysis"""
        print(f"\n🔬 Robust Statistics Analysis...")
        
        # Basic statistics
        mean_val = np.mean(data)
        median_val = np.median(data)
        std_val = np.std(data, ddof=1)
        
        # Robust statistics
        mad = stats.median_abs_deviation(data)  # Median Absolute Deviation
        iqr = stats.iqr(data)  # Interquartile Range
        
        # Trimmed mean (10% trimmed)
        trimmed_mean = stats.trim_mean(data, 0.1)
        
        # NumPy array'e dönüştür
        data_np = np.array(data, dtype=float).flatten()
        # Winsorized statistics
        winsorized_data = stats.mstats.winsorize(data_np, limits=(0.05, 0.05))
        winsorized_mean = np.mean(winsorized_data)
        winsorized_std = np.std(winsorized_data, ddof=1)
        
        # Outlier detection using IQR method
        q1, q3 = np.percentile(data_np, [25, 75])
        iqr_val = q3 - q1
        lower_bound = q1 - 1.5 * iqr_val
        upper_bound = q3 + 1.5 * iqr_val
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        
        result = {
            'test_name': 'Robust Statistics Analysis',
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'mad': mad,
            'iqr': iqr,
            'trimmed_mean': trimmed_mean,
            'winsorized_mean': winsorized_mean,
            'winsorized_std': winsorized_std,
            'outliers_count': len(outliers),
            'outliers_percentage': len(outliers) / len(data) * 100,
            'outliers': outliers[:10] if len(outliers) > 10 else outliers  # Limit to first 10
        }
        
        self.results_log.append(result)
        print(f"   Mean: {mean_val:.4f}, Median: {median_val:.4f}")
        print(f"   MAD: {mad:.4f}, IQR: {iqr:.4f}")
        print(f"   Outliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)")
        
        return result

    def generate_statistical_report(self) -> str:
        """Comprehensive statistical report"""
        report = "\n" + "="*60 + "\n"
        report += "COMPREHENSIVE STATISTICAL VALIDATION REPORT\n"
        report += "="*60 + "\n\n"

        significant_tests = [r for r in self.results_log if r.get('is_significant', False)]
        total_tests = len(self.results_log)

        report += f"Total Statistical Tests Conducted: {total_tests}\n"
        report += f"Significant Results: {len(significant_tests)}\n"
        report += f"Significance Rate: {len(significant_tests)/total_tests*100:.1f}%\n\n"

        # Main test results
        report += "MAIN TEST RESULTS:\n"
        report += "-" * 30 + "\n"
        for i, result in enumerate(self.results_log, 1):
            report += f"{i}. {result['test_name']}:\n"
            pval = result.get('p_value', 'N/A')
            if isinstance(pval, float):
                report += f"   p-value: {pval:.6f}\n"
            else:
                report += f"   p-value: {pval}\n"
            report += f"   Significant: {'Yes' if result.get('is_significant') else 'No'}\n"
            report += f"   Interpretation: {result.get('interpretation', 'N/A')}\n\n"

        # Effect sizes summary
        if self.effect_sizes:
            report += "EFFECT SIZES SUMMARY:\n"
            report += "-" * 30 + "\n"
            for effect in self.effect_sizes:
                report += f"   {effect['effect_type']}: {effect['effect_size']:.4f} ({effect['interpretation']})\n"
            report += "\n"

        # Confidence intervals summary
        if self.confidence_intervals:
            report += "CONFIDENCE INTERVALS SUMMARY:\n"
            report += "-" * 30 + "\n"
            for ci in self.confidence_intervals:
                report += f"   {ci['test_name']}: {ci['interpretation']}\n"
            report += "\n"

        # Multiple comparison warning
        if total_tests > 1:
            report += "⚠️  MULTIPLE COMPARISON CORRECTION:\n"
            report += f"   With {total_tests} tests, consider Bonferroni correction.\n"
            report += f"   Adjusted α = {self.alpha/total_tests:.6f}\n\n"

        # Recommendations
        report += "STATISTICAL RECOMMENDATIONS:\n"
        report += "-" * 30 + "\n"
        report += self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> str:
        """Generate statistical recommendations based on results"""
        recommendations = []
        
        # Check for multiple testing
        if len(self.results_log) > 5:
            recommendations.append("• Consider using False Discovery Rate (FDR) correction for multiple testing")
        
        # Check for effect sizes
        if not self.effect_sizes:
            recommendations.append("• Report effect sizes alongside p-values for practical significance")
        
        # Check for confidence intervals
        if not self.confidence_intervals:
            recommendations.append("• Include confidence intervals to show precision of estimates")
        
        # Check for normality violations
        normality_tests = [r for r in self.results_log if 'Normality' in r.get('test_name', '')]
        if any(not r.get('is_normal', True) for r in normality_tests):
            recommendations.append("• Consider non-parametric alternatives due to non-normal distributions")
        
        # Check for homoscedasticity violations
        variance_tests = [r for r in self.results_log if 'Variance' in r.get('test_name', '')]
        if any(not r.get('is_homoscedastic', True) for r in variance_tests):
            recommendations.append("• Use Welch's t-test or robust methods due to unequal variances")
        
        if not recommendations:
            recommendations.append("• Statistical assumptions appear to be met")
            recommendations.append("• Results are statistically sound")
        
        return "\n".join(recommendations) + "\n\n"

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

    def _interpret_anova(self, p_value: float, eta_squared: float) -> str:
        """Interpret ANOVA results"""
        if p_value >= self.alpha:
            return "No significant differences between groups"
        
        if eta_squared < 0.01:
            effect_size = "negligible"
        elif eta_squared < 0.06:
            effect_size = "small"
        elif eta_squared < 0.14:
            effect_size = "medium"
        else:
            effect_size = "large"
        
        return f"Significant group differences with {effect_size} effect size"

    def log_dataset_info(self, dataset_name: str, data: list, params: dict = None):
        """Kullanılan veri seti ve parametreleri logla"""
        info = {
            'dataset_name': dataset_name,
            'n': len(data),
            'mean': float(np.mean(data)) if len(data) > 0 else None,
            'std': float(np.std(data)) if len(data) > 0 else None,
            'params': params if params else {}
        }
        self.results_log.append({'dataset_info': info})
        print(f"[DATASET] {dataset_name} | n={info['n']} | mean={info['mean']:.4f} | std={info['std']:.4f} | params={info['params']}")

    def export_results_to_csv(self, csv_path: str = "results/statistical_test_results.csv"):
        """Tüm test sonuçlarını ve veri seti özetini CSV olarak kaydet"""
        import pandas as pd
        rows = []
        for r in self.results_log:
            if 'test_name' in r:
                row = {
                    'test_name': r.get('test_name', ''),
                    'dataset_name': r.get('dataset_info', {}).get('dataset_name', ''),
                    'n': r.get('dataset_info', {}).get('n', ''),
                    'params': r.get('dataset_info', {}).get('params', ''),
                    'p_value': r.get('p_value', ''),
                    'effect_size': r.get('effect_size', ''),
                    'ci_lower': r.get('ci_lower', ''),
                    'ci_upper': r.get('ci_upper', ''),
                    'interpretation': r.get('interpretation', '')
                }
                rows.append(row)
            elif 'dataset_info' in r:
                # Sadece veri seti özetini ekle
                row = {
                    'test_name': 'DATASET_INFO',
                    'dataset_name': r['dataset_info'].get('dataset_name', ''),
                    'n': r['dataset_info'].get('n', ''),
                    'params': r['dataset_info'].get('params', ''),
                    'p_value': '',
                    'effect_size': '',
                    'ci_lower': '',
                    'ci_upper': '',
                    'interpretation': ''
                }
                rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[CSV] Tüm analiz sonuçları ve veri seti özetleri kaydedildi: {csv_path}")

def create_statistical_plots(validator: StatisticalValidator, save_dir: str = "results"):
    """
    Generates comprehensive statistical validation plots and analyses.
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract parameter information
    param_info = validator.results_log[0] if validator.results_log else {}
    engine = param_info.get('engine', 'unknown')
    mode = param_info.get('mode', 'unknown')
    lambda_val = param_info.get('lambda', 'N/A')
    depth = param_info.get('depth', 'N/A')
    multipv = param_info.get('multipv', 'N/A')
    sample_count = param_info.get('sample_count', 'N/A')
    param_summary = f"Engine: {engine}, Mode: {mode}, λ={lambda_val}, depth={depth}, multipv={multipv}, samples={sample_count}"

    # Save comprehensive test log
    with open(f"{save_dir}/statistical_test_log.txt", "w", encoding="utf-8") as f:
        f.write(f"# Parameters: {param_summary}\n")
        f.write("# Comprehensive Statistical Test Results\n")
        f.write("="*60 + "\n")
        for i, r in enumerate(validator.results_log, 1):
            f.write(f"\n{i}. {r.get('test_name', 'N/A')}\n")
            f.write(f"   p-value: {r.get('p_value', 'N/A')}\n")
            f.write(f"   Effect size: {r.get('effect_size', 'N/A')}\n")
            f.write(f"   Significant: {r.get('is_significant', 'N/A')}\n")
            f.write(f"   Interpretation: {r.get('interpretation', 'N/A')}\n")

    # 1. Comprehensive P-value Analysis
    p_values = [r.get('p_value') for r in validator.results_log if r.get('p_value') is not None]
    
    if p_values:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # P-value histogram
        ax1.hist(p_values, bins=20, alpha=0.7, edgecolor='black', color='skyblue')
        ax1.axvline(validator.alpha, color='red', linestyle='--', linewidth=2, label=f'α = {validator.alpha}')
        ax1.set_xlabel('P-values')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of P-values')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # P-value vs Test Index
        ax2.plot(range(len(p_values)), p_values, 'o-', markersize=8, linewidth=2)
        ax2.axhline(validator.alpha, color='red', linestyle='--', linewidth=2, label=f'α = {validator.alpha}')
        ax2.set_xlabel('Test Index')
        ax2.set_ylabel('P-value')
        ax2.set_title('P-values by Test Order')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot for p-values (should be uniform under null)
        from scipy import stats
        stats.probplot(p_values, dist="uniform", plot=ax3)
        ax3.set_title('Q-Q Plot: P-values vs Uniform Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Cumulative distribution of p-values
        sorted_p = np.sort(p_values)
        ax4.plot(sorted_p, np.arange(1, len(sorted_p)+1)/len(sorted_p), 'b-', linewidth=2, label='Observed')
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Expected (Uniform)')
        ax4.set_xlabel('P-value')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution of P-values')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Comprehensive P-value Analysis\n{param_summary}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_p_value_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 2. Effect Size Analysis
    effect_sizes = [r.get('effect_size') for r in validator.results_log if r.get('effect_size') is not None]
    test_names = [r.get('test_name', f'Test {i}') for i, r in enumerate(validator.results_log)
                  if r.get('effect_size') is not None]

    if effect_sizes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Effect size bar chart
        colors = ['red' if abs(es) >= 0.8 else 'orange' if abs(es) >= 0.5 else 'yellow' if abs(es) >= 0.2 else 'lightblue' 
                 for es in effect_sizes]
        bars = ax1.bar(range(len(effect_sizes)), effect_sizes, alpha=0.8, color=colors)
        
        # Add reference lines
        ax1.axhline(0.2, color='green', linestyle='--', alpha=0.7, label='Small effect (0.2)')
        ax1.axhline(0.5, color='orange', linestyle='--', alpha=0.7, label='Medium effect (0.5)')
        ax1.axhline(0.8, color='red', linestyle='--', alpha=0.7, label='Large effect (0.8)')
        ax1.axhline(-0.2, color='green', linestyle='--', alpha=0.7)
        ax1.axhline(-0.5, color='orange', linestyle='--', alpha=0.7)
        ax1.axhline(-0.8, color='red', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Statistical Tests')
        ax1.set_ylabel('Effect Size')
        ax1.set_title('Effect Sizes by Test')
        ax1.set_xticks(range(len(test_names)))
        ax1.set_xticklabels(test_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Effect size distribution
        ax2.hist(effect_sizes, bins=15, alpha=0.7, edgecolor='black', color='lightcoral')
        ax2.axvline(np.mean(effect_sizes), color='blue', linestyle='-', linewidth=2, label=f'Mean: {np.mean(effect_sizes):.3f}')
        ax2.axvline(np.median(effect_sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(effect_sizes):.3f}')
        ax2.set_xlabel('Effect Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Effect Sizes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Effect Size Analysis\n{param_summary}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_effect_sizes_comprehensive.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Confidence Intervals Visualization
    if validator.confidence_intervals:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ci_names = [ci['test_name'] for ci in validator.confidence_intervals]
        ci_lowers = [ci['ci_lower'] for ci in validator.confidence_intervals]
        ci_uppers = [ci['ci_upper'] for ci in validator.confidence_intervals]
        ci_means = [ci['original_statistic'] for ci in validator.confidence_intervals]
        
        # Confidence intervals plot
        y_pos = np.arange(len(ci_names))
        ax1.errorbar(ci_means, y_pos, 
                    xerr=[np.array(ci_means) - np.array(ci_lowers), 
                          np.array(ci_uppers) - np.array(ci_means)],
                    fmt='o', capsize=5, capthick=2, markersize=8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(ci_names)
        ax1.set_xlabel('Statistic Value')
        ax1.set_title('Confidence Intervals')
        ax1.grid(True, alpha=0.3)
        
        # CI widths
        ci_widths = [ci['ci_width'] for ci in validator.confidence_intervals]
        ax2.bar(range(len(ci_widths)), ci_widths, alpha=0.8, color='lightgreen')
        ax2.set_xlabel('Confidence Intervals')
        ax2.set_ylabel('CI Width')
        ax2.set_title('Confidence Interval Widths')
        ax2.set_xticks(range(len(ci_names)))
        ax2.set_xticklabels(ci_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Confidence Interval Analysis\n{param_summary}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_confidence_intervals.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 4. Test Results Summary Heatmap
    if len(validator.results_log) > 1:
        # Create summary matrix
        test_names_short = [r.get('test_name', f'Test {i}')[:20] for i, r in enumerate(validator.results_log)]
        metrics = ['p_value', 'effect_size', 'is_significant']
        
        summary_matrix = np.zeros((len(test_names_short), len(metrics)))
        for i, result in enumerate(validator.results_log):
            summary_matrix[i, 0] = result.get('p_value', np.nan) if result.get('p_value') is not None else np.nan
            summary_matrix[i, 1] = result.get('effect_size', np.nan) if result.get('effect_size') is not None else np.nan
            summary_matrix[i, 2] = 1 if result.get('is_significant', False) else 0
        
        plt.figure(figsize=(10, max(6, len(test_names_short) * 0.5)))
        
        # Handle NaN values for visualization
        summary_matrix_vis = np.nan_to_num(summary_matrix, nan=0)
        
        im = plt.imshow(summary_matrix_vis, cmap='RdYlBu_r', aspect='auto')
        plt.colorbar(im, label='Value')
        plt.xlabel('Metrics')
        plt.ylabel('Statistical Tests')
        plt.title(f'Statistical Test Results Summary\n{param_summary}')
        plt.xticks(range(len(metrics)), metrics)
        plt.yticks(range(len(test_names_short)), test_names_short)
        
        # Add text annotations
        for i in range(len(test_names_short)):
            for j in range(len(metrics)):
                if not np.isnan(summary_matrix[i, j]):
                    text = f'{summary_matrix[i, j]:.3f}' if j < 2 else ('Sig' if summary_matrix[i, j] else 'NS')
                    plt.text(j, i, text, ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_results_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    # 5. Statistical Power Analysis Visualization
    if validator.power_analyses:
        power_data = validator.power_analyses
        
        plt.figure(figsize=(12, 8))
        
        effect_sizes_power = [p['effect_size'] for p in power_data]
        powers = [p['power'] for p in power_data]
        sample_sizes = [p['sample_size'] for p in power_data]
        
        # Power vs Effect Size
        plt.subplot(2, 2, 1)
        plt.scatter(effect_sizes_power, powers, s=60, alpha=0.7, c=sample_sizes, cmap='viridis')
        plt.colorbar(label='Sample Size')
        plt.axhline(0.8, color='red', linestyle='--', label='Adequate Power (0.8)')
        plt.xlabel('Effect Size')
        plt.ylabel('Statistical Power')
        plt.title('Power vs Effect Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Power distribution
        plt.subplot(2, 2, 2)
        plt.hist(powers, bins=15, alpha=0.7, edgecolor='black')
        plt.axvline(0.8, color='red', linestyle='--', label='Adequate Power')
        plt.xlabel('Statistical Power')
        plt.ylabel('Frequency')
        plt.title('Distribution of Statistical Power')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.suptitle(f'Statistical Power Analysis\n{param_summary}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/statistical_power_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    print(f"   📊 Comprehensive statistical plots generated in {save_dir}/")
    print(f"   📄 Statistical test log saved: {save_dir}/statistical_test_log.txt")

# Integration with main analysis
def integrate_statistical_validation():
    """Main integration function for statistical validation"""
    print("\n🔬 STATISTICAL VALIDATION MODULE")
    print("=" * 50)

    validator = StatisticalValidator(alpha=0.05)

    # This will be called from main.py with actual data
    return validator