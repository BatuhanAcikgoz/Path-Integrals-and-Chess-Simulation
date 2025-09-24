import os

# Global constants - module level for better performance
DEFAULT_STOCKFISH_PATH = r"C:\stockfish\stockfish-windows-x86-64-avx2.exe"
DEFAULT_LC0_PATH = r"C:\lc0-v0.32.0-windows-gpu-nvidia-cuda11\lc0.exe"

# Module-level variables for the best performance
STOCKFISH_PATH = os.getenv("STOCKFISH_PATH", DEFAULT_STOCKFISH_PATH)
LC0_PATH = os.getenv("LC0_PATH", DEFAULT_LC0_PATH)

# ------------------------------
# Deney veri kümeleri ve FEN seçimi
# ------------------------------
MULTI_FEN = [
    # Openings
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4",  # Italian Game (Calm)
    "rnbqkb1r/pp2pp1p/3p1np1/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",  # Sicilian Defense (Sharp)
    "r1bqkbnr/1ppp1ppp/p1n5/4p3/B3P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 1 4",  # Ruy Lopez (Strategic)
    # Mid Game
    "2r3k1/pp2rppp/3p1n2/q2P4/3p1Q2/2N5/PPP2PPP/1K1RR3 w - - 0 19",  # Strategic (Isolated Pawn)
    "r3r1k1/pp1n1ppp/2p2q2/3p1b2/3P4/2N1PN2/PP3PPP/R2Q1RK1 w - - 0 13",  # Calm Strategic
    "r1b2rk1/pp3pbp/1n1p1np1/2pP4/4P3/2N2N1P/PP2BPP1/R1Bq1RK1 w - - 0 12",  # Tactical potential
    "8/8/4k3/8/4P3/8/4K3/8 w - - 0 1",  # Endgame (K+P vs K)
    # Chess960
    "rbqknrnb/pppppppp/8/8/8/8/PPPPPPPP/RBQKNRNB w KQkq - 0 1",  # Chess960 #0
    "brnqknrb/pppppppp/8/8/8/8/PPPPPPPP/BRNQKNRB w KQkq - 0 1",  # Chess960 #100
    "nbqkrbnr/pppppppp/8/8/8/8/PPPPPPPP/NBQKRBNR w KQkq - 0 1",  # Chess960 #300
    "brnkqrbn/pppppppp/8/8/8/8/PPPPPPPP/BRNKQRBN w KQkq - 0 1",  # Chess960 #700
    "rbnkqbrn/pppppppp/8/8/8/8/PPPPPPPP/RBNKQBRN w KQkq - 0 1",  # Chess960 #900
]

CHESS960_VARIANTS = {
    "rbqknrnb/pppppppp/8/8/8/8/PPPPPPPP/RBQKNRNB w KQkq - 0 1": {"variant": 0, "desc": "Çift fil farklı renk, vezir kenarda (Chess960 #0)"},
    "brnqknrb/pppppppp/8/8/8/8/PPPPPPPP/BRNQKNRB w KQkq - 0 1": {"variant": 100, "desc": "Fil ve vezir bitişik, atlar kenarda (Chess960 #100)"},
    "nbqkrbnr/pppppppp/8/8/8/8/PPPPPPPP/NBQKRBNR w KQkq - 0 1": {"variant": 300, "desc": "Vezir ve şah ortada, atlar kenarda (Chess960 #300)"},
    "brnkqrbn/pppppppp/8/8/8/8/PPPPPPPP/BRNKQRBN w KQkq - 0 1": {"variant": 700, "desc": "Şah ve vezir bitişik, filler farklı renk (Chess960 #700)"},
    "rbnkqbrn/pppppppp/8/8/8/8/PPPPPPPP/RBNKQBRN w KQkq - 0 1": {"variant": 900, "desc": "Şah ortada, vezir kenarda (Chess960 #900)"},
}

CHESS960_MIDGAME_FEN = [
    "rbqknrnb/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/RBQKNRNB w KQkq - 10 10",  # Chess960 #0 orta oyun
    "brnqknrb/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/BRNQKNRB w KQkq - 10 10",  # Chess960 #100 orta oyun
    "nbqkrbnr/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/NBQKRBNR w KQkq - 10 10",  # Chess960 #300 orta oyun
    "brnkqrbn/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/BRNKQRBN w KQkq - 10 10",  # Chess960 #700 orta oyun
    "rbnkqbrn/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/RBNKQBRN w KQkq - 10 10",  # Chess960 #900 orta oyun
]

CHESS960_MIDGAME_VARIANTS = {
    "rbqknrnb/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/RBQKNRNB w KQkq - 10 10": {"variant": 0, "desc": "Chess960 #0 orta oyun, taşlar gelişmiş"},
    "brnqknrb/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/BRNQKNRB w KQkq - 10 10": {"variant": 100, "desc": "Chess960 #100 orta oyun, taşlar gelişmiş"},
    "nbqkrbnr/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/NBQKRBNR w KQkq - 10 10": {"variant": 300, "desc": "Chess960 #300 orta oyun, taşlar gelişmiş"},
    "brnkqrbn/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/BRNKQRBN w KQkq - 10 10": {"variant": 700, "desc": "Chess960 #700 orta oyun, taşlar gelişmiş"},
    "rbnkqbrn/ppp2ppp/3p4/4p3/2B1P3/2N2N2/PPP2PPP/RBNKQBRN w KQkq - 10 10": {"variant": 900, "desc": "Chess960 #900 orta oyun, taşlar gelişmiş"},
}

HORIZON_EFFECT_FENS = {
    "Queen_Sac_Trap": "q6k/5p1p/5P2/8/8/8/8/K7 w - - 0 1",
    "Pawn_Breakthrough": "8/k7/3p4/p2P1p2/P2P1P2/8/8/K7 w - - 0 1"
}

FEN = MULTI_FEN[0]
SAMPLE_COUNT = 30

# ------------------------------
# Düğüm (nodes) bütçeleri
# ------------------------------
# Uzun analiz/GT için y��ksek bütçe (RTX 4060 ile uygun)
LC0_NODES = 10000

# ------------------------------
# DEPTH-BASED ADAPTIF NODES SİSTEMİ
# ------------------------------
# Adil karşılaştırma için sabit depth parametresi
TARGET_DEPTH = 5

# Pozisyon karmaşıklığına göre adaptif nodes hesaplama
ADAPTIVE_NODES_ENABLED = True
BASE_NODES_PER_DEPTH = 5000  # Her depth seviyesi için temel nodes

# Nodes sın��rları
MIN_NODES_PER_PLY = 100
MAX_NODES_PER_PLY = 1000000

# ------------------------------
# Diğer deney parametreleri
# ------------------------------
MULTIPV = 5
LAMBDA = 0.2
TOP_N = 5
LAMBDA_SCAN = [0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.18, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0]
SAMPLE_SIZES = [50, 100, 200, 500]
DEPTH_SCAN = [2, 4, 6, 8, 10, 12, 14, 16, 20, 30]
LC0_SOFTMAX_LAMBDA = 0.7
LC0_CPUCT_GRID = [0.01, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.18, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 5.0, 10.0]
LC0_MULTIPV = 5
LC0_TEMPERATURE = 0.7
LC0_CPUCT = 1.0
STOCKFISH_DEPTH = 30

# Performance optimization settings
SKIP_EXISTING_RESULTS = True  # Mevcut sonuçları atla
MEMORY_CLEANUP_INTERVAL = 2   # Her N pozisyonda bellek temizliği
MAX_MEMORY_USAGE_MB = 20480    # Maksimum bellek kullanımı (MB)

# Global state variables
worker_engine = None
worker_fen = None
worker_nodes = None
worker_lam = None

# Global caches
_ENGINES = {}
_SAMPLE_PATHS_CACHE: dict[tuple, list] = {}
_ANALYSIS_CACHE: dict[str, dict] = {}
_COMPLEXITY_CACHE: dict[str, str] = {}
_TOP_MOVES_CACHE = {}

class Config:
    """Singleton config class for backward compatibility"""
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if Config._initialized:
            return
        # Direct reference to module variables for performance
        self.STOCKFISH_PATH = STOCKFISH_PATH
        self.LC0_PATH = LC0_PATH
        self.MULTI_FEN = MULTI_FEN
        self.HORIZON_EFFECT_FENS = HORIZON_EFFECT_FENS
        self.FEN = FEN
        self.SAMPLE_COUNT = SAMPLE_COUNT
        self.LC0_NODES = LC0_NODES
        self.STOCKFISH_DEPTH = STOCKFISH_DEPTH
        self.MULTIPV = MULTIPV
        self.LAMBDA = LAMBDA
        self.TOP_N = TOP_N
        self.LAMBDA_SCAN = LAMBDA_SCAN
        self.SAMPLE_SIZES = SAMPLE_SIZES
        self.LC0_SOFTMAX_LAMBDA = LC0_SOFTMAX_LAMBDA
        self.LC0_CPUCT_GRID = LC0_CPUCT_GRID
        self.LC0_MULTIPV = LC0_MULTIPV
        self.LC0_TEMPERATURE = LC0_TEMPERATURE
        self.LC0_CPUCT = LC0_CPUCT
        self.SKIP_EXISTING_RESULTS = SKIP_EXISTING_RESULTS
        self.MEMORY_CLEANUP_INTERVAL = MEMORY_CLEANUP_INTERVAL
        self.MAX_MEMORY_USAGE_MB = MAX_MEMORY_USAGE_MB
        self.TARGET_DEPTH = TARGET_DEPTH
        Config._initialized = True

# Direct access functions for maximum performance
def get_stockfish_path():
    return STOCKFISH_PATH

def get_lc0_path():
    return LC0_PATH

def get_engines():
    return _ENGINES

def get_sample_cache():
    return _SAMPLE_PATHS_CACHE

def get_analysis_cache():
    return _ANALYSIS_CACHE

# ------------------------------
# DEPTH-BASED ADAPTIF NODES FONKSİYONLARI
# ------------------------------

def calculate_adaptive_nodes(depth):
    """
    Depth ve pozisyon karmaşıklığına göre adaptif nodes hesaplar.

    :param depth: Hedef analiz derinliği
    :param fen: Pozisyon FEN stringi (karmaşıklık hesabı için)
    :param complexity_override: Manuel karmaşıklık kategorisi
    :return: Hesaplanan nodes sayısı
    """
    if not ADAPTIVE_NODES_ENABLED:
        return LC0_NODES

    complexity = 'medium'  # Varsayılan

    # Base nodes hesapla
    base_nodes = BASE_NODES_PER_DEPTH * depth

    # Karmaşıklık çarpanını uygula
    multiplier = 1.3
    adaptive_nodes = int(base_nodes * multiplier)

    # Sınırları uygula
    adaptive_nodes = max(MIN_NODES_PER_PLY, min(MAX_NODES_PER_PLY, adaptive_nodes))

    return adaptive_nodes

def get_depth_equivalent_nodes(depth):
    """
    Stockfish depth'e eşdeğer LC0 nodes hesaplar.
    Adil karşılaştırma için kullanılır.

    :param depth: Stockfish depth değeri
    :param fen: Pozisyon FEN stringi
    :return: Eşdeğer nodes sayısı
    """
    return calculate_adaptive_nodes(depth)

# ------------------------------
# MOD SEÇİMİ (competitive / quantum_limit)
# ------------------------------
MODE = os.getenv("PI_MODE", "competitive")  # "competitive" veya "quantum_limit"

# Quantum limit için özel parametreler
HIGH_MULTIPV = 20
HIGH_DEPTH = 20

# Competitive mod için varsayılanlar
COMPETITIVE_MULTIPV = 5
COMPETITIVE_DEPTH = 5