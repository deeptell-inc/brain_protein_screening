#!/usr/bin/env python3
"""
Brain Protein Quantum-Active Site Screener
==========================================
PDB構造から局所的な量子活性部位（クロモフォア・金属中心・π共役系）を抽出し、
GFN2-xTB + sTDA近似で遷移双極子モーメント(TDM)をスクリーニングするパイプライン。

xtb が電子励起状態を直接計算できないため、以下の戦略を取る:
  1. GFN2-xTB で基底状態の電子構造（軌道エネルギー、双極子モーメント）
  2. --vipea で IP/EA（Koopmans近似）
  3. --hess で IR強度 → 振動遷移双極子モーメント
  4. 軌道エネルギーから簡易CIS推定（HOMO-LUMO遷移の振動子強度近似）
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
import numpy as np

# --- Biopython for PDB parsing ---
from Bio.PDB import PDBParser, NeighborSearch, Selection

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

# 脳内タンパク質で重要な金属イオン
METAL_IONS = {
    "FE", "FE2", "FE3", "CU", "CU1", "CU2",
    "ZN", "MN", "MG", "CO", "NI", "MO",
}

# クロモフォア/補因子の残基名
CHROMOPHORE_RESIDUES = {
    "RET",  # レチナール (ロドプシン)
    "FAD", "FMN", "FLV",  # フラビン (クリプトクロム)
    "HEM", "HEC", "HEA", "HEB",  # ヘム/ポルフィリン
    "NAD", "NAP", "NDP", "NAI",  # NADH/NADPH
    "PQQ",  # ピロロキノリンキノン
    "TPQ",  # トパキノン
    "LLP", "PLP",  # ピリドキサールリン酸 (PLP)
    "BCL", "CLA", "CHL",  # クロロフィル
    "BPH", "PHO",  # バクテリオフェオフィチン
    "SF4", "FES",  # 鉄硫黄クラスター
}

# π共役系が大きい芳香族アミノ酸
AROMATIC_RESIDUES = {"TRP", "TYR", "PHE", "HIS"}

# 座標抽出の半径 (Å)
METAL_CUTOFF = 4.5   # 金属中心の第2配位圏
CHROMO_CUTOFF = 3.5  # クロモフォア周辺残基
PI_STACK_CUTOFF = 4.0  # π-πスタッキング検出

# TDM閾値 (Debye)
ELECTRONIC_TDM_THRESHOLD = 1.5
VIBRATIONAL_TDM_THRESHOLD = 0.15  # 振動遷移は小さい

# ===========================================================================
# 磁気活性核の物性データ
# ===========================================================================

# {元素記号: [(質量数, 核スピンI, 天然存在比, 磁気回転比γ (MHz/T), 核磁気モーメントμ/μ_N)]}
MAGNETIC_NUCLEI = {
    "H":  [(1,   0.5, 0.9999, 42.577, 2.7928)],
    "C":  [(13,  0.5, 0.0107, 10.708, 0.7024)],
    "N":  [(14,  1.0, 0.9963,  3.077, 0.4038),
            (15,  0.5, 0.0036, -4.316, -0.2832)],
    "O":  [(17,  2.5, 0.0004, -5.774, -1.8938)],
    "P":  [(31,  0.5, 1.0000, 17.235, 1.1317)],
    "S":  [(33,  1.5, 0.0076,  3.266, 0.6433)],
    "FE": [(57,  0.5, 0.0212,  1.382, 0.0906)],
    "CU": [(63,  1.5, 0.6917, 11.319, 2.2273),
            (65,  1.5, 0.3083, 12.103, 2.3817)],
    "ZN": [(67,  2.5, 0.0410,  2.669, 0.8755)],
    "MN": [(55,  2.5, 1.0000, 10.576, 3.4687)],
    "MG": [(25,  2.5, 0.1000, -2.608, -0.8554)],
    "CO": [(59,  3.5, 1.0000, 10.077, 4.627)],
    "NI": [(61,  1.5, 0.0114, -3.811, -0.7500)],
    "MO": [(95,  2.5, 0.1592, -2.787, -0.9142),
            (97,  2.5, 0.0955, -2.847, -0.9335)],
}

# 原子スピン軌道結合定数 ξ (cm⁻¹) — 1電子SOC
ATOMIC_SOC_CONSTANTS = {
    "H": 0.24, "C": 29, "N": 42, "O": 79, "S": 382,
    "P": 230, "CL": 587, "BR": 2460,
    "FE": 460, "CU": 830, "ZN": 390,
    "MN": 300, "CO": 533, "NI": 630, "MO": 750,
    "MG": 40,
}

# McConnell関係 A_H = Q * ρ_π (MHz) — πラジカルの¹H超微細結合
MCCONNELL_Q_MHZ = -72.0  # McConnell定数 Q (MHz)

# ===========================================================================
# デコヒーレンス/緩和計算用の物理定数
# ===========================================================================
HBAR_EV_S = 6.582119569e-16      # ℏ (eV·s)
HBAR_J_S = 1.0545718e-34          # ℏ (J·s)
KB_EV = 8.617333262e-5            # k_B (eV/K)
C_CM_S = 2.99792458e10            # 光速 (cm/s)
EV_TO_CM = 8065.54                # 1 eV → cm⁻¹
MHZ_TO_RAD_S = 2.0 * np.pi * 1e6 # MHz → rad/s
BODY_TEMP_K = 310.0               # 生体温度 (K)

# 電磁気定数
MU_BOHR = 9.2740100783e-24        # Bohr磁子 (J/T)
MU_NUCLEAR = 5.0507837461e-27     # 核磁子 (J/T)
G_ELECTRON = 2.00231930436256     # 電子g因子
EARTH_FIELD_T = 5.0e-5            # 地磁気 (~50 μT)
XBAND_FIELD_T = 0.34              # X-band ESR磁場 (T)
EV_PER_J = 6.241509074e18         # 1 J → eV


# ===========================================================================
# Data structures
# ===========================================================================

@dataclass
class QuantumSite:
    """量子活性部位の記述子"""
    site_id: str
    site_type: str            # "metal_center" | "chromophore" | "pi_stack"
    residues: list             # 含まれる残基名のリスト
    n_atoms: int = 0
    charge: int = 0
    multiplicity: int = 1
    # xTB計算結果
    homo_lumo_gap_eV: float = 0.0
    ground_dipole_D: float = 0.0
    ip_eV: float = 0.0
    ea_eV: float = 0.0
    # 推定された電子遷移TDM (CIS近似)
    estimated_electronic_tdm_D: float = 0.0
    oscillator_strength: float = 0.0
    # 振動遷移TDM
    max_vib_tdm_D: float = 0.0
    max_vib_freq_cm: float = 0.0
    vib_modes: list = field(default_factory=list)
    # ── 核スピン ──
    nuclear_spin_inventory: list = field(default_factory=list)
    total_magnetic_nuclei: int = 0
    dominant_spin_nuclei: str = ""   # 最も豊富な磁気活性核
    # ── 電子スピン ──
    singlet_triplet_gap_eV: float = 0.0  # S-T gap (正=Sが安定)
    radical_cation_ip_eV: float = 0.0    # 垂直IP
    radical_anion_ea_eV: float = 0.0     # 垂直EA
    somo_energy_eV: float = 0.0          # SOMO軌道エネルギー
    # ── 超微細結合 ──
    estimated_hfc_MHz: list = field(default_factory=list)   # [{atom, element, A_iso_MHz}]
    max_hfc_MHz: float = 0.0
    # ── スピン軌道結合 ──
    effective_soc_cm: float = 0.0        # 有効SOC定数 (cm⁻¹)
    soc_classification: str = ""         # "weak" / "moderate" / "strong"
    # ── QM/MM メタデータ ──
    embedding_method: str = "vacuum"   # "vacuum" | "oniom"
    n_mm_atoms: int = 0
    mm_cutoff_A: float = 0.0
    oniom_total_energy_Eh: float = 0.0
    # ── アンサンブル統計 (MD) ──
    n_frames: int = 1
    homo_lumo_gap_std: float = 0.0
    estimated_electronic_tdm_std: float = 0.0
    singlet_triplet_gap_std: float = 0.0
    max_hfc_std: float = 0.0
    # ── 真空 vs 埋め込み比較 ──
    vacuum_homo_lumo_gap_eV: float = 0.0
    vacuum_ground_dipole_D: float = 0.0
    embedding_shift_gap_eV: float = 0.0
    embedding_shift_dipole_D: float = 0.0
    # ── デコヒーレンス/緩和時間 ──
    T2_star_spin_ns: float = 0.0           # HFC分布からのスピン脱位相時間
    k_isc_per_s: float = 0.0              # ISC速度定数 (Fermi黄金則)
    tau_isc_ns: float = 0.0               # ISC寿命 (= 1/k_ISC)
    tau_rad_ns: float = 0.0               # 輻射寿命 (Strickler-Berg)
    tau_vib_fastest_ps: float = 0.0       # 最速振動緩和モード
    T2_el_fs: float = 0.0                 # 電子純粋脱位相 (Kubo静的極限)
    k_et_per_s: float = 0.0               # Marcus電子移動速度
    T1e_spin_lattice_us: float = 0.0      # スピン格子緩和 (BPP)
    lambda_reorg_eV: float = 0.0          # 再配向エネルギー
    coherence_bottleneck: str = ""         # 最短時間スケール名
    coherence_class: str = ""              # "coherent" / "partial" / "classical"
    # ── Zeeman分裂 & 磁気遷移双極子 ──
    zeeman_electron_earth_MHz: float = 0.0    # 電子Zeeman分裂 @地磁気 (MHz)
    zeeman_electron_xband_GHz: float = 0.0    # 電子Zeeman分裂 @X-band (GHz)
    zeeman_P31_earth_Hz: float = 0.0          # ³¹P Zeeman分裂 @地磁気 (Hz)
    zeeman_P31_xband_kHz: float = 0.0         # ³¹P Zeeman分裂 @X-band (kHz)
    magnetic_tdm_electron_muB: float = 0.0    # 電子スピン遷移の磁気TDM (μ_B)
    magnetic_tdm_P31_muN: float = 0.0         # ³¹P核スピン遷移の磁気TDM (μ_N)
    electric_tdm_spin_D: float = 0.0          # SOC誘起 S→T 電気遷移双極子 (D)
    T2_dipolar_ns: float = 0.0                # 双極子-双極子横緩和 (ns)
    T2_total_ns: float = 0.0                  # 合計横緩和 T₂ (ns)
    n_P_atoms: int = 0                        # リン原子数
    # ── 核スピン緩和 ──
    T1_P31_s: float = 0.0                     # ³¹P 核スピン格子緩和 T₁ (s)
    T2_P31_ms: float = 0.0                    # ³¹P 核横緩和 T₂ (ms)
    T1_H1_s: float = 0.0                      # ¹H 核スピン格子緩和 T₁ (s)
    T2_H1_ms: float = 0.0                     # ¹H 核横緩和 T₂ (ms)
    T1_N14_ms: float = 0.0                    # ¹⁴N 核四極子緩和 T₁ (ms)
    PRE_T1_P31_ms: float = 0.0               # ³¹P 常磁性緩和促進 T₁ (ms, ラジカル存在時)
    PRE_T2_P31_us: float = 0.0               # ³¹P 常磁性緩和促進 T₂ (μs, ラジカル存在時)
    nuclear_coherence_time_ms: float = 0.0    # 核スピンコヒーレンス時間 (ms)
    # ── 電子T₂ᵉ完全版 ──
    T2e_g_aniso_ns: float = 0.0              # g-tensor異方性による T₂ (ns)
    T2e_hfc_mod_ns: float = 0.0              # HFC変調による T₂ (ns)
    T2e_spin_orbit_ns: float = 0.0           # SOC誘起 T₂ (ns)
    T2e_total_ns: float = 0.0                # 電子T₂ᵉ合計 (ns)
    # ── 結合スピン系 (e⁻-³¹P) ──
    coupled_level_energies_MHz: list = field(default_factory=list)  # 4準位エネルギー
    coupled_EPR_freq_MHz: list = field(default_factory=list)        # EPR遷移周波数
    coupled_NMR_freq_MHz: list = field(default_factory=list)        # NMR遷移周波数
    coupled_forbidden_freq_MHz: list = field(default_factory=list)  # 禁制遷移周波数
    coupled_tdm_EPR_muB: float = 0.0        # EPR遷移 磁気TDM (μ_B)
    coupled_tdm_NMR_muN: float = 0.0        # NMR遷移 磁気TDM (μ_N)
    coupled_tdm_forbidden_muB: float = 0.0  # 禁制遷移 磁気TDM (μ_B)
    W0_cross_relax_per_s: float = 0.0       # W₀ 零量子交差緩和 (s⁻¹)
    W2_cross_relax_per_s: float = 0.0       # W₂ 二量子交差緩和 (s⁻¹)
    T1x_cross_relax_us: float = 0.0         # 交差緩和時間 T₁ₓ (μs)
    coupled_T2_electron_ns: float = 0.0     # 結合系での電子T₂ (ns)
    coupled_T2_nuclear_us: float = 0.0      # 結合系での核T₂ (μs)
    # ── 脳内環境での核スピン (地磁気50μT, 310K) ──
    brain_P31_T1_s: float = 0.0             # ³¹P T₁ @地磁気 (s)
    brain_P31_T2_s: float = 0.0             # ³¹P T₂ @地磁気 (s)
    brain_H1_T1_s: float = 0.0              # ¹H T₁ @地磁気 (s)
    brain_H1_T2_s: float = 0.0              # ¹H T₂ @地磁気 (s)
    brain_PRE_P31_T1_ms: float = 0.0        # ³¹P PRE T₁ @地磁気 (ms)
    brain_PRE_P31_T2_ms: float = 0.0        # ³¹P PRE T₂ @地磁気 (ms)
    brain_paramag_T1_s: float = 0.0         # 常磁性不純物によるT₁ (s)
    brain_water_exchange_ms: float = 0.0    # 水プロトン交換寿命 (ms)
    brain_O2_T1_s: float = 0.0              # 溶存O₂によるT₁短縮 (s)
    brain_P31_T2_effective_ms: float = 0.0  # ³¹P 実効T₂ (全効果込み) (ms)
    brain_H1_T2_effective_ms: float = 0.0   # ¹H 実効T₂ (全効果込み) (ms)
    brain_nuclear_coherence_regime: str = "" # extreme narrowing / intermediate
    # ── 脳内T₂ᵉ・結合スピン系 @地磁気 ──
    brain_T2e_g_aniso_ns: float = 0.0       # T₂(g-ani) @50μT (ns)
    brain_T2e_total_ns: float = 0.0         # T₂ᵉ(total) @50μT (ns)
    brain_W0_per_s: float = 0.0             # W₀ @50μT (s⁻¹)
    brain_W2_per_s: float = 0.0             # W₂ @50μT (s⁻¹)
    brain_T1x_us: float = 0.0              # 交差緩和時間 @50μT (μs)
    brain_coupled_T2e_ns: float = 0.0       # 結合系T₂(e⁻) @50μT (ns)
    brain_coupled_T2n_us: float = 0.0       # 結合系T₂(³¹P) @50μT (μs)
    # ── PDB実測距離 ──
    measured_eP_distance_A: float = 0.0     # PDBから計測した e⁻-P 最短距離 (Å)
    measured_eP_distances_A: list = field(default_factory=list)  # 全P原子への距離 (Å)
    radical_center_type: str = ""           # ラジカル中心の種類
    # ステータス
    calc_success: bool = False
    error_msg: str = ""


# ===========================================================================
# 1. PDB解析 & フラグメント抽出
# ===========================================================================

class ProteinSiteExtractor:
    """PDB構造から量子活性部位フラグメントを抽出"""

    def __init__(self, pdb_path: str):
        parser = PDBParser(QUIET=True)
        self.structure = parser.get_structure("protein", pdb_path)
        self.model = self.structure[0]
        self._all_atoms = list(self.model.get_atoms())
        self._ns = NeighborSearch(self._all_atoms)

    def find_metal_centers(self) -> list[dict]:
        """金属イオンとその配位圏を検出"""
        sites = []
        for atom in self._all_atoms:
            resname = atom.get_parent().get_resname().strip()
            if resname in METAL_IONS or atom.element.strip().upper() in {
                "FE", "CU", "ZN", "MN", "MG", "CO", "NI", "MO"
            }:
                neighbors = self._ns.search(atom.get_vector().get_array(), METAL_CUTOFF)
                residues = list({a.get_parent() for a in neighbors})
                sites.append({
                    "type": "metal_center",
                    "center_atom": atom,
                    "metal_element": atom.element.strip(),
                    "residues": residues,
                    "atoms": neighbors,
                })
        return sites

    def find_chromophores(self) -> list[dict]:
        """クロモフォア/補因子残基を検出"""
        sites = []
        seen = set()
        for chain in self.model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in CHROMOPHORE_RESIDUES:
                    key = (chain.id, residue.id)
                    if key in seen:
                        continue
                    seen.add(key)
                    # クロモフォア本体 + 周辺残基
                    center = residue.center_of_mass(geometric=True)
                    neighbors = self._ns.search(center, CHROMO_CUTOFF)
                    extra_residues = list({a.get_parent() for a in neighbors})
                    # クロモフォア残基自体の全原子を確実に含める
                    chromo_atoms = list(residue.get_atoms())
                    all_atoms = list(set(neighbors) | set(chromo_atoms))
                    sites.append({
                        "type": "chromophore",
                        "core_residue": resname,
                        "residues": extra_residues,
                        "atoms": all_atoms,
                    })
        return sites

    def find_pi_stacks(self, min_cluster_size: int = 2) -> list[dict]:
        """芳香族アミノ酸のπ-πスタッキングクラスターを検出"""
        aromatic_residues = []
        for chain in self.model:
            for residue in chain:
                if residue.get_resname().strip() in AROMATIC_RESIDUES:
                    aromatic_residues.append(residue)

        if not aromatic_residues:
            return []

        # クラスタリング: 重心間距離で隣接行列を構築
        centers = []
        for res in aromatic_residues:
            try:
                c = res.center_of_mass(geometric=True)
                centers.append(c)
            except Exception:
                # 重心計算失敗時はスキップ
                centers.append(None)

        # 簡易クラスタリング（Union-Find）
        n = len(aromatic_residues)
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb

        for i in range(n):
            if centers[i] is None:
                continue
            for j in range(i + 1, n):
                if centers[j] is None:
                    continue
                dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
                if dist < PI_STACK_CUTOFF:
                    union(i, j)

        # クラスターを集約
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        sites = []
        for members in clusters.values():
            if len(members) < min_cluster_size:
                continue
            cluster_residues = [aromatic_residues[i] for i in members]
            cluster_atoms = []
            for res in cluster_residues:
                cluster_atoms.extend(res.get_atoms())
            sites.append({
                "type": "pi_stack",
                "residues": cluster_residues,
                "atoms": cluster_atoms,
                "cluster_size": len(members),
            })
        return sites

    def measure_P_chromophore_distances(self, site: dict) -> dict:
        """PDB座標からP原子とラジカル中心の距離を計測。

        ラジカル中心の推定:
        - FAD/FMN: N5原子 (イソアロキサジン環の中心窒素)
        - PLP/LLP: C4A原子 (ピリジン環の共役中心)
        - NAD/NAP: NC4原子 (ニコチンアミドC4)
        - その他: HOMO密度最大の芳香族中心

        Returns: {"radical_center": coords, "P_atoms": [(elem, coords, dist)],
                  "min_dist_A": float, "radical_type": str}
        """
        # --- ラジカル中心を特定 ---
        radical_center = None
        radical_type = "unknown"

        # サイトの残基名を取得
        resnames = set()
        for a in site.get("atoms", []):
            try:
                resnames.add(a.get_parent().get_resname().strip())
            except:
                pass

        # FAD/FMN: N5原子を探す
        if resnames & {"FAD", "FMN", "RBF", "FLV"}:
            radical_type = "isoalloxazine_N5"
            for a in site["atoms"]:
                rn = a.get_parent().get_resname().strip()
                if rn in ("FAD", "FMN", "RBF", "FLV"):
                    # N5 or N5A (FADのイソアロキサジンN5)
                    if a.get_name().strip() in ("N5", "N5A", "N10", "N10A"):
                        radical_center = a.get_vector().get_array()
                        if "N5" in a.get_name().strip():
                            break  # N5を優先

        # PLP/LLP: C4A原子
        if radical_center is None and resnames & {"PLP", "LLP", "PXP", "PMP"}:
            radical_type = "pyridine_C4A"
            for a in site["atoms"]:
                rn = a.get_parent().get_resname().strip()
                if rn in ("PLP", "LLP", "PXP", "PMP"):
                    if a.get_name().strip() in ("C4A", "C4", "C3"):
                        radical_center = a.get_vector().get_array()
                        break

        # NAD/NADP: ニコチンアミドC4
        if radical_center is None and resnames & {"NAD", "NAP", "NAI", "NDP", "NAH"}:
            radical_type = "nicotinamide_C4"
            for a in site["atoms"]:
                rn = a.get_parent().get_resname().strip()
                if rn in ("NAD", "NAP", "NAI", "NDP", "NAH"):
                    if a.get_name().strip() in ("NC4", "C4N", "C4B"):
                        radical_center = a.get_vector().get_array()
                        break

        # フォールバック: 芳香族原子の重心
        if radical_center is None:
            aromatic_atoms = [a for a in site["atoms"]
                              if a.element.strip() in ("C", "N") and
                              a.get_parent().get_resname().strip() in
                              ("TRP", "TYR", "PHE", "HIS", "FAD", "FMN",
                               "NAD", "NAP", "PLP", "LLP")]
            if aromatic_atoms:
                radical_type = "aromatic_centroid"
                coords = np.array([a.get_vector().get_array() for a in aromatic_atoms])
                radical_center = coords.mean(axis=0)

        if radical_center is None:
            return {"radical_center": None, "P_atoms": [], "min_dist_A": 0.0,
                    "radical_type": radical_type}

        # --- P原子を収集して距離計測 ---
        P_atoms = []
        # サイト内のP原子
        for a in site["atoms"]:
            if a.element.strip().upper() == "P":
                p_coord = a.get_vector().get_array()
                dist = np.linalg.norm(p_coord - radical_center)
                P_atoms.append({
                    "residue": a.get_parent().get_resname().strip(),
                    "atom_name": a.get_name().strip(),
                    "distance_A": float(dist),
                })

        # サイト外（全タンパク質中）のP原子も検索（20Å以内）
        for a in self._all_atoms:
            if a.element.strip().upper() == "P":
                p_coord = a.get_vector().get_array()
                dist = np.linalg.norm(p_coord - radical_center)
                if dist < 20.0:
                    entry = {
                        "residue": a.get_parent().get_resname().strip(),
                        "atom_name": a.get_name().strip(),
                        "distance_A": float(dist),
                    }
                    # 重複排除
                    if not any(abs(p["distance_A"] - dist) < 0.1 for p in P_atoms):
                        P_atoms.append(entry)

        P_atoms.sort(key=lambda x: x["distance_A"])
        min_dist = P_atoms[0]["distance_A"] if P_atoms else 0.0

        return {
            "radical_center": radical_center,
            "P_atoms": P_atoms,
            "min_dist_A": min_dist,
            "radical_type": radical_type,
        }

    def extract_all_sites(self) -> list[dict]:
        """全ての量子活性部位を検出"""
        sites = []
        metals = self.find_metal_centers()
        chromos = self.find_chromophores()
        pi_stacks = self.find_pi_stacks()

        log.info(f"検出: 金属中心 {len(metals)}, クロモフォア {len(chromos)}, "
                 f"π-stack {len(pi_stacks)}")

        sites.extend(metals)
        sites.extend(chromos)
        sites.extend(pi_stacks)
        return sites


# ===========================================================================
# 1b. タンパク質準備 (pdbfixer + OpenMM) — QM/MM用
# ===========================================================================

ANG2BOHR = 1.88972612463

class ProteinPreparer:
    """pdbfixer で PDB を修復し、OpenMM + amber14 でパラメタライズ。
    MM 領域の部分電荷を提供する。
    """

    def __init__(self, pdb_path: str):
        from pdbfixer import PDBFixer
        from openmm.app import PDBFile
        import io

        log.info("ProteinPreparer: PDB修復中 (pdbfixer)...")
        self.fixer = PDBFixer(filename=pdb_path)
        self.fixer.findMissingResidues()
        self.fixer.findMissingAtoms()
        self.fixer.addMissingAtoms()
        self.fixer.addMissingHydrogens(pH=7.0)

        self.topology = self.fixer.topology
        self.positions = self.fixer.positions  # openmm.unit Quantity
        self.system = None
        self._charges = None  # list[float] indexed by OpenMM atom index
        self._unparameterized_residues = set()

        # 原子マップ: (chain_id, res_seq, atom_name) → OpenMM index
        self._atom_map = {}
        self._build_atom_map()

        n_atoms = self.topology.getNumAtoms()
        n_res = self.topology.getNumResidues()
        log.info(f"ProteinPreparer: 修復完了 ({n_atoms} atoms, {n_res} residues)")

    def _build_atom_map(self):
        """BioPython式キー → OpenMM atom index のマッピングを構築"""
        for atom in self.topology.atoms():
            res = atom.residue
            chain_id = res.chain.id
            res_seq = res.id  # OpenMM の residue.id は文字列 "123" 等
            key = (chain_id, res_seq, atom.name)
            self._atom_map[key] = atom.index

    def parameterize(self) -> bool:
        """amber14 で全タンパク質をパラメタライズ。
        非標準残基の失敗は警告のみ（MM領域から除外）。
        """
        from openmm.app import ForceField, Modeller
        from openmm import NonbondedForce

        log.info("ProteinPreparer: OpenMM パラメタライズ中 (amber14)...")

        # 非標準残基を特定
        standard_res = {
            "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
            "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
            "THR", "TRP", "TYR", "VAL",
            "HID", "HIE", "HIP",  # AMBER His variants
            "ACE", "NME",  # caps
        }
        # 水も非標準扱い（OpenMMのテンプレートマッチで問題になる場合あり）
        water_res = {"HOH", "WAT", "TIP", "SOL"}
        nonstandard = set()
        nonstandard_atom_indices = set()
        for res in self.topology.residues():
            if res.name not in standard_res and res.name not in water_res:
                nonstandard.add(res.name)
                for atom in res.atoms():
                    nonstandard_atom_indices.add(atom.index)
            elif res.name in water_res:
                # 水分子も除去（テンプレートマッチ問題回避）
                for atom in res.atoms():
                    nonstandard_atom_indices.add(atom.index)
        self._unparameterized_residues = nonstandard | water_res

        if nonstandard:
            log.warning(f"  非標準残基 (MM除外): {nonstandard}")

        # 非標準残基 + 水を除去した Modeller でパラメタライズ
        modeller = Modeller(self.topology, self.positions)
        to_delete = [r for r in modeller.topology.residues()
                     if r.name in nonstandard or r.name in water_res]
        if to_delete:
            modeller.delete(to_delete)

        try:
            ff = ForceField("amber14-all.xml")
            self.system = ff.createSystem(modeller.topology)
        except Exception as e:
            log.error(f"  パラメタライズ失敗: {e}")
            return False

        # 部分電荷を抽出
        self._charges = [0.0] * self.topology.getNumAtoms()
        for force in self.system.getForces():
            if isinstance(force, NonbondedForce):
                # modeller atom index → full topology index の対応
                mod_idx = 0
                for atom in self.topology.atoms():
                    if atom.index not in nonstandard_atom_indices:
                        if mod_idx < force.getNumParticles():
                            charge, _, _ = force.getParticleParameters(mod_idx)
                            self._charges[atom.index] = charge._value
                            mod_idx += 1
                break

        n_charged = sum(1 for c in self._charges if abs(c) > 0.001)
        log.info(f"  パラメタライズ完了: {n_charged} atoms with charges")
        return True

    def get_charges(self) -> list[float]:
        """全原子の部分電荷リスト (e 単位)"""
        return self._charges or [0.0] * self.topology.getNumAtoms()

    def get_positions_angstrom(self) -> np.ndarray:
        """現在の座標 (Nx3, Å)"""
        import openmm.unit as u
        return np.array(self.positions.value_in_unit(u.angstrom))

    def set_positions(self, positions_angstrom: np.ndarray):
        """座標を更新 (MDスナップショット用)"""
        import openmm.unit as u
        self.positions = positions_angstrom * u.angstrom

    def get_atom_map(self) -> dict:
        return self._atom_map

    def is_unparameterized(self, res_name: str) -> bool:
        return res_name in self._unparameterized_residues


# ===========================================================================
# 1c. QM/MM 領域分割 & 入力ファイル生成
# ===========================================================================

class QMMMPartitioner:
    """量子活性部位を QM/MM に分割し、ONIOM用入力ファイルを生成。

    xtb --oniom gfn2:gfnff は全系XYZファイル + inner region原子インデックスを必要とする。
    inner region = QM原子、outer region = 周辺MM原子。
    """

    def __init__(self, site_atoms: list, all_protein_atoms: list,
                 mm_cutoff_A: float = 12.0):
        """
        Args:
            site_atoms: BioPython Atom objects for the QM (inner) region
            all_protein_atoms: 全タンパク質の BioPython Atom list
            mm_cutoff_A: outer 領域のカットオフ半径 (Å)
        """
        self.site_atoms = site_atoms
        self.all_atoms = all_protein_atoms
        self.mm_cutoff = mm_cutoff_A

        # QM 原子の座標集合（高速ルックアップ用）
        self._qm_coord_set = set()
        for a in site_atoms:
            c = a.get_vector().get_array()
            self._qm_coord_set.add((round(c[0], 3), round(c[1], 3), round(c[2], 3)))

    def partition(self) -> dict:
        """QM/MM 原子を分割し、ONIOM用データを返す。

        Returns:
            {
                "inner_atoms": [...],  # QM原子 (BioPython Atom)
                "outer_atoms": [...],  # MM原子 (BioPython Atom)
                "n_inner": int,
                "n_outer": int,
            }
        """
        # QM 原子の重心を計算
        qm_coords = [a.get_vector().get_array() for a in self.site_atoms]
        qm_center = np.mean(qm_coords, axis=0)

        # outer 原子: QM以外で mm_cutoff 以内
        outer_atoms = []
        for atom in self.all_atoms:
            c = atom.get_vector().get_array()
            key = (round(c[0], 3), round(c[1], 3), round(c[2], 3))
            if key in self._qm_coord_set:
                continue  # inner 原子は除外

            dist = np.linalg.norm(c - qm_center)
            if dist <= self.mm_cutoff:
                outer_atoms.append(atom)

        return {
            "inner_atoms": list(self.site_atoms),
            "outer_atoms": outer_atoms,
            "n_inner": len(self.site_atoms),
            "n_outer": len(outer_atoms),
        }

    def write_oniom_xyz(self, partition: dict, workdir: str,
                        xyz_str_inner: str = None) -> tuple[str, str]:
        """ONIOM用の全系XYZファイルとinner atom indexリストを書き出す。

        inner原子にはH付加済みのXYZ（xyz_str_inner）を使い、
        outer原子はPDB座標から直接取得する。

        Args:
            partition: partition()の返り値
            workdir: 作業ディレクトリ
            xyz_str_inner: H付加済みのinner region XYZ文字列

        Returns:
            (full_xyz_path, inner_indices_str)
            inner_indices_str: "1,2,3,..." (1-indexed for xtb)
        """
        workdir = Path(workdir)

        # inner原子のXYZ行をパース
        inner_lines = []
        if xyz_str_inner:
            raw = xyz_str_inner.strip().split("\n")
            n_inner = int(raw[0])
            for line in raw[2:2 + n_inner]:
                parts = line.split()
                if len(parts) >= 4:
                    inner_lines.append(
                        f" {parts[0]:<2s}  {float(parts[1]):12.6f}"
                        f"  {float(parts[2]):12.6f}  {float(parts[3]):12.6f}")
        else:
            # フォールバック: BioPython原子から直接
            seen = set()
            for atom in partition["inner_atoms"]:
                c = atom.get_vector().get_array()
                key = (atom.element.strip(), round(c[0], 3),
                       round(c[1], 3), round(c[2], 3))
                if key in seen:
                    continue
                seen.add(key)
                elem = atom.element.strip() or atom.get_name().strip()[0]
                inner_lines.append(
                    f" {elem:<2s}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}")

        # outer原子のXYZ行
        outer_lines = []
        seen_outer = set()
        for atom in partition["outer_atoms"]:
            c = atom.get_vector().get_array()
            elem = atom.element.strip()
            if not elem:
                elem = atom.get_name().strip()[0]
            key = (elem, round(c[0], 3), round(c[1], 3), round(c[2], 3))
            if key in seen_outer:
                continue
            seen_outer.add(key)
            outer_lines.append(
                f" {elem:<2s}  {c[0]:12.6f}  {c[1]:12.6f}  {c[2]:12.6f}")

        # 全系XYZ: inner原子が先、outer原子が後
        n_total = len(inner_lines) + len(outer_lines)
        all_lines = inner_lines + outer_lines
        xyz_content = f"{n_total}\nONIOM full system\n" + "\n".join(all_lines) + "\n"

        full_xyz_path = workdir / "oniom_full.xyz"
        full_xyz_path.write_text(xyz_content)

        # inner atom indices (1-indexed, xtb convention)
        inner_indices = list(range(1, len(inner_lines) + 1))
        inner_indices_str = ",".join(str(i) for i in inner_indices)

        log.info(f"    ONIOM XYZ: {len(inner_lines)} inner + "
                 f"{len(outer_lines)} outer = {n_total} total atoms")

        return str(full_xyz_path), inner_indices_str


# ===========================================================================
# 1d. MD サンプラー (OpenMM)
# ===========================================================================

class MDSampler:
    """OpenMM による短時間 MD で構造アンサンブルを生成"""

    def __init__(self, preparer: ProteinPreparer,
                 temperature_K: float = 300.0,
                 timestep_fs: float = 2.0):
        self.preparer = preparer
        self.temperature_K = temperature_K
        self.timestep_fs = timestep_fs
        self.snapshots: list[np.ndarray] = []

    def run(self, n_steps: int = 50000,
            snapshot_interval: int = 5000) -> list[np.ndarray]:
        """NVT MD を実行してスナップショットを返す。

        Args:
            n_steps: プロダクションステップ数 (default: 50000 = 100ps @ 2fs)
            snapshot_interval: スナップショット間隔 (default: 5000 = 10ps)

        Returns:
            list of np.ndarray (Nx3, Å)
        """
        from openmm import LangevinMiddleIntegrator, Platform
        from openmm.app import Simulation
        import openmm.unit as u
        import openmm

        if self.preparer.system is None:
            log.error("MDSampler: system がパラメタライズされていません")
            return []

        log.info(f"MDSampler: {n_steps} steps ({n_steps * self.timestep_fs / 1000:.1f} ps), "
                 f"T={self.temperature_K} K")

        integrator = LangevinMiddleIntegrator(
            self.temperature_K * u.kelvin,
            1.0 / u.picosecond,
            self.timestep_fs * u.femtosecond,
        )

        # 非標準残基に位置拘束を付加
        force = openmm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        force.addGlobalParameter("k", 1000.0 * u.kilojoules_per_mole / u.nanometer ** 2)
        force.addPerParticleParameter("x0")
        force.addPerParticleParameter("y0")
        force.addPerParticleParameter("z0")

        positions_nm = self.preparer.get_positions_angstrom() / 10.0  # Å → nm
        unparameterized = self.preparer._unparameterized_residues
        for atom in self.preparer.topology.atoms():
            if atom.residue.name in unparameterized:
                pos = positions_nm[atom.index]
                force.addParticle(atom.index, [pos[0], pos[1], pos[2]])

        system_copy = openmm.XmlSerializer.deserialize(
            openmm.XmlSerializer.serialize(self.preparer.system))
        system_copy.addForce(force)

        # Simulation 構築
        platform = Platform.getPlatformByName("CPU")
        sim = Simulation(self.preparer.topology, system_copy,
                         integrator, platform)
        sim.context.setPositions(self.preparer.positions)

        # エネルギー最小化
        log.info("  エネルギー最小化中...")
        sim.minimizeEnergy(maxIterations=1000)

        # 平衡化 (5000 steps)
        log.info("  平衡化中 (10 ps)...")
        sim.step(5000)

        # プロダクション
        log.info(f"  プロダクション MD ({n_steps} steps)...")
        self.snapshots = []
        n_snapshots_taken = 0
        for step in range(0, n_steps, snapshot_interval):
            sim.step(snapshot_interval)
            state = sim.context.getState(getPositions=True)
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
            self.snapshots.append(np.array(pos_nm) * 10.0)  # nm → Å
            n_snapshots_taken += 1

        log.info(f"  MD完了: {n_snapshots_taken} スナップショット取得")
        return self.snapshots


# ===========================================================================
# 2. フラグメント → XYZ変換
# ===========================================================================

def atoms_to_xyz(atoms: list, charge: int = 0) -> str:
    """BioPython atomリストからXYZ形式の文字列を生成"""
    # 重複除去（座標ベースで）
    seen = set()
    lines = []
    for atom in atoms:
        coord = atom.get_vector().get_array()
        key = (atom.element.strip(), round(coord[0], 3),
               round(coord[1], 3), round(coord[2], 3))
        if key in seen:
            continue
        seen.add(key)
        elem = atom.element.strip()
        if not elem:
            elem = atom.get_name().strip()[0]
        lines.append(f" {elem:<2s}  {coord[0]:12.6f}  {coord[1]:12.6f}  {coord[2]:12.6f}")

    n_atoms = len(lines)
    xyz = f"{n_atoms}\n\n" + "\n".join(lines) + "\n"
    return xyz


def add_hydrogens_obabel(xyz_str: str) -> str:
    """OpenBabelを使ってXYZフラグメントに水素を付加"""
    try:
        from openbabel import openbabel as ob
    except ImportError:
        log.warning("OpenBabel not available, skipping H addition")
        return xyz_str

    conv = ob.OBConversion()
    conv.SetInAndOutFormats("xyz", "xyz")

    mol = ob.OBMol()
    conv.ReadString(mol, xyz_str)

    # 結合を推定してから水素付加
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    mol.AddHydrogens()

    result = conv.WriteString(mol)
    return result.strip() + "\n"


def write_fragment_pdb(atoms: list, pdb_path: str):
    """BioPython atomリストからPDBファイルを書き出す"""
    from Bio.PDB import PDBIO
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain

    struct = Structure("frag")
    model = Model(0)
    chain = Chain("A")
    seen_res = set()
    for atom in atoms:
        res = atom.get_parent()
        res_key = (res.get_resname(), res.id)
        if res_key not in seen_res:
            seen_res.add(res_key)
            # Deep copy the residue
            from Bio.PDB.Residue import Residue
            new_res = Residue(res.id, res.get_resname(), res.get_segid())
            for a in res.get_atoms():
                new_res.add(a.copy())
            chain.add(new_res)
    model.add(chain)
    struct.add(model)

    io = PDBIO()
    io.set_structure(struct)
    io.save(pdb_path)


def add_hydrogens_pdb(atoms: list) -> str:
    """PDB形式経由でOpenBabelによる水素付加 → XYZ文字列を返す"""
    try:
        from openbabel import openbabel as ob
    except ImportError:
        log.warning("OpenBabel not available, skipping H addition")
        return atoms_to_xyz(atoms)

    import tempfile
    # フラグメントをPDB形式で書き出し
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as f:
        pdb_tmp = f.name
    write_fragment_pdb(atoms, pdb_tmp)

    # PDB → OpenBabel → H付加 → XYZ
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("pdb", "xyz")

    mol = ob.OBMol()
    conv.ReadFile(mol, pdb_tmp)
    mol.AddHydrogens()

    conv_out = ob.OBConversion()
    conv_out.SetOutFormat("xyz")
    result = conv_out.WriteString(mol)

    os.unlink(pdb_tmp)

    n_heavy = sum(1 for a in atoms if a.element.strip() != "H")
    n_total = int(result.split("\n")[0])
    log.info(f"  H付加: {n_heavy} → {n_total} atoms (+{n_total - n_heavy} H)")

    return result.strip() + "\n"


def estimate_charge(site: dict) -> int:
    """フラグメントの総電荷を推定（簡易）"""
    charge = 0
    metal = site.get("metal_element", "")
    # 金属イオンの典型的な電荷
    metal_charges = {
        "FE": 2, "CU": 2, "ZN": 2, "MN": 2, "MG": 2,
        "CO": 2, "NI": 2, "MO": 4,
    }
    if metal:
        charge += metal_charges.get(metal.upper(), 2)
    return charge


def estimate_multiplicity(site: dict) -> int:
    """スピン多重度を推定（簡易）"""
    metal = site.get("metal_element", "")
    # 開殻金属の典型的な多重度
    metal_mult = {
        "FE": 5,   # Fe(II) high-spin d6 → S=2, 2S+1=5
        "CU": 2,   # Cu(II) d9 → S=1/2, 2S+1=2
        "MN": 6,   # Mn(II) d5 → S=5/2, 2S+1=6
        "CO": 4,   # Co(II) d7 → S=3/2, 2S+1=4
        "NI": 3,   # Ni(II) d8 → S=1, 2S+1=3
    }
    return metal_mult.get(metal.upper(), 1)


# ===========================================================================
# 3. xTB計算エンジン
# ===========================================================================

class XTBRunner:
    """xTB計算の実行とパース"""

    def __init__(self, xtb_cmd: str = "xtb", n_cores: int = 1):
        self.xtb_cmd = xtb_cmd
        self.n_cores = n_cores
        # xTBの存在確認
        result = subprocess.run([xtb_cmd, "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"xtb not found: {xtb_cmd}")

    def _clean_workdir(self, workdir: str):
        """前回のxTB中間ファイルを削除"""
        for fname in ["xtbrestart", "charges", "wbo", "xtbtopo.mol",
                       ".xtboptok", "g98.out", "vibspectrum", "hessian"]:
            p = os.path.join(workdir, fname)
            if os.path.exists(p):
                os.remove(p)

    def run_single_point(self, xyz_path: str, charge: int = 0,
                         uhf: int = 0, workdir: str = ".",
                         extra_args: list[str] = None) -> dict:
        """GFN2-xTB 単一点計算 + 双極子"""
        self._clean_workdir(workdir)
        # Use basename since cwd is set to workdir
        xyz_basename = os.path.basename(xyz_path)
        cmd = [
            self.xtb_cmd, xyz_basename,
            "--gfn", "2",
            "--sp",
            "--chrg", str(charge),
            "--uhf", str(uhf),
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_cores)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=workdir, timeout=300, env=env,
        )
        return self._parse_sp_output(result.stdout, result.stderr, result.returncode)

    def run_hessian(self, xyz_path: str, charge: int = 0,
                    uhf: int = 0, workdir: str = ".",
                    extra_args: list[str] = None) -> dict:
        """GFN2-xTB 振動解析 (Hessian) → IR強度"""
        self._clean_workdir(workdir)
        xyz_basename = os.path.basename(xyz_path)
        cmd = [
            self.xtb_cmd, xyz_basename,
            "--gfn", "2",
            "--hess",
            "--chrg", str(charge),
            "--uhf", str(uhf),
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_cores)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=workdir, timeout=600, env=env,
        )
        return self._parse_hess_output(result.stdout, result.stderr,
                                       result.returncode, workdir)

    def run_sp_json(self, xyz_path: str, charge: int = 0,
                    uhf: int = 0, workdir: str = ".",
                    extra_args: list[str] = None) -> dict:
        """GFN2-xTB 単一点計算 (JSON出力)"""
        self._clean_workdir(workdir)
        # JSONファイルも掃除
        json_out = os.path.join(workdir, "xtbout.json")
        if os.path.exists(json_out):
            os.remove(json_out)

        xyz_basename = os.path.basename(xyz_path)
        cmd = [
            self.xtb_cmd, xyz_basename,
            "--gfn", "2", "--sp", "--json",
            "--chrg", str(charge),
            "--uhf", str(uhf),
        ]
        if extra_args:
            cmd.extend(extra_args)
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_cores)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=workdir, timeout=300, env=env,
        )
        data = {"success": result.returncode == 0}
        if result.returncode == 0 and os.path.exists(json_out):
            import json as _json
            with open(json_out) as f:
                data.update(_json.load(f))
        else:
            data["error"] = result.stderr[-500:] if result.stderr else ""
        return data

    def run_singlet_triplet(self, xyz_path: str, charge: int = 0,
                            workdir: str = ".",
                            extra_args: list[str] = None) -> dict:
        """一重項-三重項ギャップを計算 (ΔE_ST)"""
        # 一重項 (UHF=0)
        sp_s = self.run_sp_json(xyz_path, charge=charge, uhf=0,
                                workdir=workdir, extra_args=extra_args)
        # 三重項 (UHF=2)
        sp_t = self.run_sp_json(xyz_path, charge=charge, uhf=2,
                                workdir=workdir, extra_args=extra_args)

        result = {"success": sp_s.get("success") and sp_t.get("success")}
        if result["success"]:
            E_s = sp_s.get("total energy", 0.0)
            E_t = sp_t.get("total energy", 0.0)
            # 正の値 = 一重項が安定 (通常)
            result["delta_E_ST_eV"] = (E_t - E_s) * 27.2114
            result["E_singlet_Eh"] = E_s
            result["E_triplet_Eh"] = E_t
        return result

    def run_radical_energies(self, xyz_path: str, charge: int = 0,
                             workdir: str = ".",
                             extra_args: list[str] = None) -> dict:
        """中性→ラジカルカチオン/アニオンの垂直エネルギー差を計算"""
        # 中性 (closed-shell)
        sp_n = self.run_sp_json(xyz_path, charge=charge, uhf=0,
                                workdir=workdir, extra_args=extra_args)
        # ラジカルカチオン (charge+1, doublet)
        sp_cat = self.run_sp_json(xyz_path, charge=charge + 1, uhf=1,
                                  workdir=workdir, extra_args=extra_args)
        # ラジカルアニオン (charge-1, doublet)
        sp_an = self.run_sp_json(xyz_path, charge=charge - 1, uhf=1,
                                 workdir=workdir, extra_args=extra_args)

        result = {"success": sp_n.get("success", False)}
        E_n = sp_n.get("total energy", 0.0)

        if sp_cat.get("success"):
            E_cat = sp_cat.get("total energy", 0.0)
            result["vertical_ip_eV"] = (E_cat - E_n) * 27.2114
            result["cation_charges"] = sp_cat.get("partial charges", [])
        else:
            result["vertical_ip_eV"] = 0.0

        if sp_an.get("success"):
            E_an = sp_an.get("total energy", 0.0)
            result["vertical_ea_eV"] = (E_n - E_an) * 27.2114
            result["anion_charges"] = sp_an.get("partial charges", [])
        else:
            result["vertical_ea_eV"] = 0.0

        # SOMO をラジカルカチオン（奇数電子）から取得
        if sp_cat.get("success"):
            orb_e = sp_cat.get("orbital energies / eV", [])
            occ = sp_cat.get("fractional occupation", [])
            # SOMO = 占有数が1の軌道（最後の占有軌道）
            for e, o in zip(orb_e, occ):
                if abs(o - 1.0) < 0.1:
                    result["somo_eV"] = e
                    break
            # SOMOが見つからない場合: HOMO-LUMO境界を使う
            if "somo_eV" not in result:
                for i in range(len(occ) - 1):
                    if occ[i] > 0.5 and occ[i + 1] < 0.5:
                        result["somo_eV"] = orb_e[i]
                        break
            result["radical_charges"] = sp_cat.get("partial charges", [])
        elif sp_an.get("success"):
            result["radical_charges"] = sp_an.get("partial charges", [])
        return result

    # --- ONIOM メソッド ---

    def run_oniom_sp(self, xyz_path: str, inner_indices: str,
                     charge: int = 0, uhf: int = 0,
                     workdir: str = ".") -> dict:
        """ONIOM (gfn2:gfnff) 単一点計算"""
        self._clean_workdir(workdir)
        xyz_basename = os.path.basename(xyz_path)
        cmd = [
            self.xtb_cmd, xyz_basename,
            "--gfn", "2", "--sp",
            "--chrg", str(charge),
            "--uhf", str(uhf),
            "--oniom", "gfn2:gfnff", inner_indices,
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_cores)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=workdir, timeout=600, env=env,
        )
        return self._parse_oniom_output(result.stdout, result.stderr,
                                        result.returncode)

    def run_oniom_singlet_triplet(self, xyz_path: str, inner_indices: str,
                                  charge: int = 0,
                                  workdir: str = ".") -> dict:
        """ONIOM 一重項-三重項ギャップ"""
        sp_s = self.run_oniom_sp(xyz_path, inner_indices,
                                 charge=charge, uhf=0, workdir=workdir)
        sp_t = self.run_oniom_sp(xyz_path, inner_indices,
                                 charge=charge, uhf=2, workdir=workdir)

        result = {"success": sp_s.get("success") and sp_t.get("success")}
        if result["success"]:
            E_s = sp_s.get("oniom_total_Eh", 0.0)
            E_t = sp_t.get("oniom_total_Eh", 0.0)
            result["delta_E_ST_eV"] = (E_t - E_s) * 27.2114
            result["E_singlet_Eh"] = E_s
            result["E_triplet_Eh"] = E_t
        return result

    def run_oniom_radical_energies(self, xyz_path: str, inner_indices: str,
                                   charge: int = 0,
                                   workdir: str = ".") -> dict:
        """ONIOM ラジカルカチオン/アニオンのエネルギー"""
        sp_n = self.run_oniom_sp(xyz_path, inner_indices,
                                 charge=charge, uhf=0, workdir=workdir)
        sp_cat = self.run_oniom_sp(xyz_path, inner_indices,
                                   charge=charge + 1, uhf=1, workdir=workdir)
        sp_an = self.run_oniom_sp(xyz_path, inner_indices,
                                  charge=charge - 1, uhf=1, workdir=workdir)

        result = {"success": sp_n.get("success", False)}
        E_n = sp_n.get("oniom_total_Eh", 0.0)

        if sp_cat.get("success"):
            E_cat = sp_cat.get("oniom_total_Eh", 0.0)
            result["vertical_ip_eV"] = (E_cat - E_n) * 27.2114
        else:
            result["vertical_ip_eV"] = 0.0

        if sp_an.get("success"):
            E_an = sp_an.get("oniom_total_Eh", 0.0)
            result["vertical_ea_eV"] = (E_n - E_an) * 27.2114
        else:
            result["vertical_ea_eV"] = 0.0

        # SOMO: ONIOM doesn't produce JSON, so skip SOMO detection
        result["somo_eV"] = 0.0
        return result

    def run_oniom_hessian(self, xyz_path: str, inner_indices: str,
                          charge: int = 0, uhf: int = 0,
                          workdir: str = ".") -> dict:
        """ONIOM (gfn2:gfnff) Hessian計算"""
        self._clean_workdir(workdir)
        xyz_basename = os.path.basename(xyz_path)
        cmd = [
            self.xtb_cmd, xyz_basename,
            "--gfn", "2", "--hess",
            "--chrg", str(charge),
            "--uhf", str(uhf),
            "--oniom", "gfn2:gfnff", inner_indices,
        ]
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(self.n_cores)
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=workdir, timeout=600, env=env,
        )
        return self._parse_hess_output(result.stdout, result.stderr,
                                       result.returncode, workdir)

    def _parse_oniom_output(self, stdout: str, stderr: str, rc: int) -> dict:
        """ONIOM出力をパース。

        ONIOM出力形式:
        - 3つの SUMMARY ブロック (inner-high, inner-low, outer)
        - 最後のSUMMARYブロックにHOMO-LUMO gap
        - ONIOM TOTAL ENERGY が最終エネルギー
        """
        data = {
            "success": rc == 0,
            "homo_lumo_gap_eV": 0.0,
            "dipole_D": 0.0,
            "oniom_total_Eh": 0.0,
            "total_energy_Eh": 0.0,
            "inner_energies_Eh": [],
            "error": "" if rc == 0 else stderr[-500:],
        }
        if rc != 0:
            return data

        for line in stdout.split("\n"):
            line_s = line.strip()
            # ONIOM total energy
            if "ONIOM TOTAL ENERGY" in line:
                try:
                    # | ONIOM TOTAL ENERGY   -5.445546324982 Eh   |
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if p == "ENERGY":
                            data["oniom_total_Eh"] = float(parts[i + 1])
                            data["total_energy_Eh"] = data["oniom_total_Eh"]
                            break
                except (ValueError, IndexError):
                    pass
            # HOMO-LUMO gap (appears in last SUMMARY block = inner high level)
            elif "HOMO-LUMO gap" in line and "eV" in line:
                try:
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if p == "gap":
                            data["homo_lumo_gap_eV"] = float(parts[i + 1])
                            break
                except (ValueError, IndexError):
                    pass
            # total energy lines (collect all 3)
            elif "total energy" in line_s and "Eh" in line_s and "::" in line_s:
                try:
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if p == "energy":
                            data["inner_energies_Eh"].append(float(parts[i + 1]))
                            break
                except (ValueError, IndexError):
                    pass

        return data

    # --- パーサー ---

    def _parse_sp_output(self, stdout: str, stderr: str, rc: int) -> dict:
        """単一点計算の出力をパース"""
        data = {
            "success": rc == 0,
            "homo_lumo_gap_eV": 0.0,
            "dipole_D": 0.0,
            "homo_eV": 0.0,
            "lumo_eV": 0.0,
            "ip_eV": 0.0,
            "ea_eV": 0.0,
            "total_energy_Eh": 0.0,
            "error": "" if rc == 0 else stderr[-500:],
        }
        for line in stdout.split("\n"):
            line_s = line.strip()
            if "HOMO-LUMO gap" in line and "eV" in line:
                try:
                    # :: HOMO-LUMO gap    9.330 eV ::
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if p == "gap":
                            data["homo_lumo_gap_eV"] = float(parts[i + 1])
                            break
                except (ValueError, IndexError):
                    pass
            elif "tot (Debye)" in line or "total (Debye)" in line:
                try:
                    # format: "   full:  -5.737  3.523  -2.567  18.314"
                    # or  "... total (Debye):  2.838"
                    if "total (Debye):" in line:
                        data["dipole_D"] = float(line_s.split("total (Debye):")[-1].strip())
                    else:
                        data["dipole_D"] = float(line_s.split()[-1])
                except ValueError:
                    pass
            elif line_s.startswith("full:"):
                try:
                    parts = line_s.split()
                    data["dipole_D"] = float(parts[-1])
                except (ValueError, IndexError):
                    pass
            elif "(HOMO)" in line:
                try:
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if "(HOMO)" in p:
                            data["homo_eV"] = float(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass
            elif "(LUMO)" in line:
                try:
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if "(LUMO)" in p:
                            data["lumo_eV"] = float(parts[i - 1])
                            break
                except (ValueError, IndexError):
                    pass
            elif "delta SCC IP" in line:
                try:
                    data["ip_eV"] = float(line_s.split()[-1])
                except ValueError:
                    pass
            elif "delta SCC EA" in line:
                try:
                    data["ea_eV"] = float(line_s.split()[-1])
                except ValueError:
                    pass
            elif "| TOTAL ENERGY" in line:
                try:
                    parts = line_s.split()
                    for i, p in enumerate(parts):
                        if p == "ENERGY":
                            data["total_energy_Eh"] = float(parts[i + 1])
                            break
                except (ValueError, IndexError):
                    pass
        return data

    def _parse_hess_output(self, stdout: str, stderr: str,
                           rc: int, workdir: str) -> dict:
        """Hessian計算の出力をパース → IR強度 & 振動TDM"""
        data = {
            "success": rc == 0,
            "modes": [],     # [{"freq_cm": float, "ir_intensity": float, "vib_tdm_D": float}]
            "error": "" if rc == 0 else stderr[-500:],
        }
        if rc != 0:
            return data

        # g98.out (Gaussian形式) からIR強度と振動数をパース
        g98_path = os.path.join(workdir, "g98.out")
        if os.path.exists(g98_path):
            data["modes"] = self._parse_g98(g98_path)

        return data

    def _parse_g98(self, g98_path: str) -> list[dict]:
        """Gaussian98形式の振動出力をパース"""
        modes = []
        with open(g98_path) as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            if "Frequencies --" in lines[i]:
                freqs = [float(x) for x in lines[i].split()[2:]]
                # IR Inten行を探す
                j = i + 1
                ir_intensities = []
                while j < len(lines) and "IR Inten" not in lines[j]:
                    j += 1
                if j < len(lines) and "IR Inten" in lines[j]:
                    ir_intensities = [float(x) for x in lines[j].split()[3:]]

                for k, freq in enumerate(freqs):
                    ir_int = ir_intensities[k] if k < len(ir_intensities) else 0.0
                    # IR強度 (km/mol) → 振動遷移双極子モーメント (Debye)
                    # IR intensity = (N_A * π) / (3 * c^2) * |μ_if|^2
                    # |μ_if|^2 (D^2) = IR_int (km/mol) * 3 * c^2 / (N_A * π)
                    # 簡易変換: |μ_if| (D) ≈ 0.01276 * sqrt(IR_int (km/mol))
                    vib_tdm = 0.01276 * np.sqrt(abs(ir_int))
                    modes.append({
                        "freq_cm": freq,
                        "ir_intensity_km_mol": ir_int,
                        "vib_tdm_D": vib_tdm,
                    })
                i = j + 1
            else:
                i += 1
        return modes


# ===========================================================================
# 4. 簡易CIS推定 (電子遷移TDM)
# ===========================================================================

def estimate_electronic_tdm(sp_data: dict, n_atoms: int) -> tuple[float, float]:
    """
    HOMO→LUMO遷移の遷移双極子モーメントを簡易推定。

    Thomas-Reiche-Kuhn (TRK) 総和則と
    単一励起近似を用いた upper-bound 推定:
      f ≈ (2/3) * ΔE * |⟨r⟩|²
    ここで |⟨r⟩| は分子サイズの特性長スケール。

    より実用的には、xTBの双極子モーメントと分極率から:
      μ_if ≈ sqrt(α * ΔE_gap) (原子単位)
    を使う（Unsöld近似の変形）。
    """
    gap_eV = sp_data.get("homo_lumo_gap_eV", 0.0)
    dipole_D = sp_data.get("dipole_D", 0.0)

    if gap_eV <= 0:
        return 0.0, 0.0

    # 方法1: 分子サイズからの推定
    # 特性長 ≈ n_atoms^(1/3) * 1.5 Å (典型的な結合長)
    char_length_A = (n_atoms ** (1.0 / 3.0)) * 1.5
    char_length_au = char_length_A / 0.529177  # Bohr

    # f_osc ≈ (2/3) * ΔE(Hartree) * |⟨r⟩|² (a.u.)
    gap_Eh = gap_eV / 27.2114
    f_osc = (2.0 / 3.0) * gap_Eh * char_length_au ** 2

    # 遷移双極子モーメント:  f = (2/3) * ΔE * |μ_if|²  (a.u.)
    # → |μ_if|² = f / ((2/3) * ΔE)  = |⟨r⟩|² (a.u.)
    mu_if_au = char_length_au
    mu_if_D = mu_if_au * 2.5418  # 1 a.u. = 2.5418 Debye

    # これは上限推定なので0.3倍に補正（経験的）
    mu_if_D *= 0.3
    f_osc *= 0.09  # 0.3^2

    return mu_if_D, f_osc


# ===========================================================================
# 5. 核スピン・電子スピン解析
# ===========================================================================

def analyze_nuclear_spins(xyz_str: str) -> dict:
    """
    XYZフラグメント中の磁気活性核をインベントリする。

    返り値:
        {
          "inventory": [{"element": str, "mass_number": int, "spin_I": float,
                         "abundance": float, "gamma_MHz_T": float, "count": int}],
          "total_magnetic": int,
          "dominant": str,
          "effective_spin_density": float,  # 磁気活性核の面密度的指標
        }
    """
    # XYZから元素リストを取得
    lines = xyz_str.strip().split("\n")
    elements = []
    for line in lines[2:]:  # skip n_atoms and comment
        parts = line.split()
        if len(parts) >= 4:
            elements.append(parts[0].strip().upper())

    # 元素ごとにカウント
    from collections import Counter
    elem_counts = Counter(elements)

    inventory = []
    total_magnetic = 0

    for elem, count in elem_counts.items():
        elem_key = elem.upper()
        if elem_key in MAGNETIC_NUCLEI:
            for mass, spin_I, abundance, gamma, mu in MAGNETIC_NUCLEI[elem_key]:
                # 期待される磁気活性核の数
                n_active = count * abundance
                inventory.append({
                    "element": elem,
                    "mass_number": mass,
                    "spin_I": spin_I,
                    "abundance": abundance,
                    "gamma_MHz_T": gamma,
                    "mu_nuclear_magneton": mu,
                    "total_atoms": count,
                    "expected_active": round(n_active, 2),
                })
                total_magnetic += count  # 全原子数（少なくとも確率的に寄与）

    # ドミナントな核（期待数が最大のもの）
    dominant = ""
    if inventory:
        best = max(inventory, key=lambda x: x["expected_active"])
        dominant = f"{best['mass_number']}{best['element']}(I={best['spin_I']})"

    return {
        "inventory": inventory,
        "total_magnetic": total_magnetic,
        "dominant": dominant,
    }


def estimate_soc(xyz_str: str, site_type: str) -> tuple[float, str]:
    """
    有効スピン軌道結合(SOC)定数を推定。

    重原子の原子SOC定数の重み付き平均:
      ξ_eff ≈ Σ_A (ξ_A * w_A) / Σ w_A
    ここで w_A はスピン密度（近似的には不対電子が局在する原子に重み）。

    簡易版: 最大のξを持つ原子のSOC定数を返す（heavy atom effectの上限推定）。
    """
    lines = xyz_str.strip().split("\n")
    elements = []
    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= 4:
            elements.append(parts[0].strip().upper())

    from collections import Counter
    elem_counts = Counter(elements)

    # 重み付きSOC推定
    # 方法: 各原子のSOC定数を二乗平均（SOCは二乗で効くため）
    soc_sum_sq = 0.0
    n_heavy = 0  # 非水素原子

    max_soc = 0.0
    max_soc_elem = ""

    for elem, count in elem_counts.items():
        if elem == "H":
            continue
        n_heavy += count
        soc = ATOMIC_SOC_CONSTANTS.get(elem, 0.0)
        soc_sum_sq += count * soc ** 2
        if soc > max_soc:
            max_soc = soc
            max_soc_elem = elem

    # 有効SOC: RMS値（不対電子が全重原子に等分に分布と仮定）
    if n_heavy > 0:
        soc_eff = np.sqrt(soc_sum_sq / n_heavy)
    else:
        soc_eff = 0.0

    # 金属中心の場合はmax_socを使う（スピン密度が金属に集中）
    if site_type == "metal_center" and max_soc > 0:
        soc_eff = max_soc

    # 分類
    if soc_eff < 50:
        classification = "weak"
    elif soc_eff < 300:
        classification = "moderate"
    else:
        classification = "strong"

    return soc_eff, classification


def estimate_hyperfine_couplings(xyz_str: str, sp_data: dict,
                                  site_type: str,
                                  pdb_P_distances: list = None) -> list[dict]:
    """
    超微細結合定数(HFC)を推定。

    方法:
    1. πラジカル系: McConnell関係 A_H = Q * ρ_π
       - ρ_πはMulliken電荷から近似推定
    2. 金属中心: 原子HFC定数のスケーリング
    3. ¹⁴N, ³¹P: 経験的な範囲から推定

    xTBからはMulliken電荷が得られるので、それを使って
    電荷分布 → スピン密度分布の近似的な関係を利用。
    """
    lines = xyz_str.strip().split("\n")
    n_atoms = int(lines[0])
    atom_info = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) >= 4:
            atom_info.append({
                "element": parts[0].strip().upper(),
                "x": float(parts[1]),
                "y": float(parts[2]),
                "z": float(parts[3]),
            })

    charges = sp_data.get("partial_charges", [])
    if not charges or len(charges) != len(atom_info):
        # 電荷データがない場合、均等分布を仮定
        charges = [0.0] * len(atom_info)

    hfcs = []

    # π共役系の炭素数を数える
    n_conjugated_C = sum(1 for a in atom_info if a["element"] == "C")
    n_N = sum(1 for a in atom_info if a["element"] == "N")

    for i, atom in enumerate(atom_info):
        elem = atom["element"]
        q = charges[i] if i < len(charges) else 0.0

        if elem == "H":
            # McConnell関係: A_H = Q * ρ_π（隣接炭素のπスピン密度）
            # 簡易: ρ_π ≈ 1/N_conj（均等分布仮定）
            if n_conjugated_C > 0:
                rho_pi = 1.0 / max(n_conjugated_C, 1)
                A_iso = abs(MCCONNELL_Q_MHZ * rho_pi)
            else:
                A_iso = 0.0
            if A_iso > 1.0:  # 1 MHz以上のみ記録
                hfcs.append({"atom_idx": i, "element": "¹H",
                             "A_iso_MHz": round(A_iso, 2)})

        elif elem == "N":
            # ¹⁴N HFC: 典型的にはπラジカルで0-50 MHz
            # フラビンラジカルのN5, N10は~50 MHz
            if site_type == "chromophore":
                # クロモフォア中のN → 比較的大きなHFC
                rho_N = 1.0 / max(n_N, 1) * 0.3  # Nへのスピン密度割合
                A_iso_14N = rho_N * 150.0  # ¹⁴N原子HFC × スピン密度
            else:
                rho_N = 1.0 / max(n_N + n_conjugated_C, 1)
                A_iso_14N = rho_N * 80.0
            if A_iso_14N > 1.0:
                hfcs.append({"atom_idx": i, "element": "¹⁴N",
                             "A_iso_MHz": round(A_iso_14N, 2)})

        elif elem == "P":
            # ³¹P HFC: 補因子構造に依存。PDB実測距離を優先使用。
            #
            # 実験値:
            #   FAD FMN-P  (N5→P: ~7Å, 8結合) → 200 MHz  (Weber 1998 ENDOR)
            #   TPP        (C2→P: ~7.5Å, 8結合) → 0.51 MHz (Ragsdale 2008)
            # → 結合数では予測不能。共役経路が支配的。
            #
            # 距離ベース分類 (PDB実測 e⁻-P 距離を使用):
            #   r < 4 Å → PLP型 (C5-O-P直結): A = 100 MHz (50-300 range)
            #   4 < r < 9 Å → FAD FMN-P型: A = 200 MHz (実験値)
            #   r > 9 Å → 遠隔P: A = 2 MHz

            # PDB実測距離を使用 (P原子インデックスでマッチ)
            pdb_r = None
            if pdb_P_distances:
                # P原子のXYZ座標でPDB距離リストとマッチ
                px, py, pz = atom["x"], atom["y"], atom["z"]
                for pd in pdb_P_distances:
                    # PDBの各P原子との最近接照合 (2Å以内なら同一原子)
                    if "coords" in pd:
                        dx = px - pd["coords"][0]
                        dy = py - pd["coords"][1]
                        dz = pz - pd["coords"][2]
                        if np.sqrt(dx*dx + dy*dy + dz*dz) < 2.0:
                            pdb_r = pd["distance_A"]
                            break
                if pdb_r is None and pdb_P_distances:
                    # 座標マッチ失敗時: P原子順番でフォールバック
                    p_idx_in_list = sum(1 for h in hfcs if "P" in str(h.get("element", "")))
                    if p_idx_in_list < len(pdb_P_distances):
                        pdb_r = pdb_P_distances[p_idx_in_list]["distance_A"]

            # PDB距離がない場合はフラグメント内のN距離で代用
            if pdb_r is None:
                px, py, pz = atom["x"], atom["y"], atom["z"]
                min_r_N = float("inf")
                for j, a2 in enumerate(atom_info):
                    if a2["element"] == "N":
                        r = np.sqrt((px - a2["x"])**2 + (py - a2["y"])**2
                                    + (pz - a2["z"])**2)
                        min_r_N = min(min_r_N, r)
                pdb_r = min_r_N

            # HFC推定: 補因子タイプ + 結合トポロジーに基づく
            # through-bond HFCはthrough-space距離と異なる
            #
            # 補因子判別: 残基名と原子数で推定
            is_FAD = n_N >= 8  # FAD: 9N (イソアロキサジン4N + アデニン5N)
            is_NADPH = 6 <= n_N < 8  # NADPH: 7N
            is_PLP = pdb_r < 5.0 if pdb_r < 100 else False

            if is_PLP:
                # PLP: P直結 (C5-O-P, 3結合, through-bond共役)
                # ENDOR信号検出済み (Aminomutase studies) だが定量値未報告
                A_iso_31P = 100.0   # 推定中央値 (不確実性: 50-300 MHz)
                hfc_class = "direct_bond"
            elif is_FAD:
                # FAD: FMN-P vs AMP-P の区別
                # FMN-P: イソアロキサジンN10→リビチル→リン酸 (8σ結合)
                #   → A = 200 MHz (Weber 1998 ENDOR実験値)
                # AMP-P: FMN-P→O→P (さらに4結合先)
                #   → A ~ 2 MHz (遠隔、共役なし)
                # 判別: PDB内でP原子が2個あれば、closer one = FMN-P
                if pdb_P_distances and len(pdb_P_distances) >= 2:
                    sorted_dists = sorted(
                        [p["distance_A"] for p in pdb_P_distances])
                    if pdb_r <= sorted_dists[0] + 0.5:
                        A_iso_31P = 200.0  # FMN-P (proximal)
                        hfc_class = "FAD_FMN_P"
                    else:
                        A_iso_31P = 2.0    # AMP-P (remote)
                        hfc_class = "FAD_AMP_P"
                else:
                    # P距離情報不足 → 保守的に200 MHz (FMN-Pと仮定)
                    A_iso_31P = 200.0
                    hfc_class = "FAD_assumed_FMN"
            elif is_NADPH:
                # NADPH: ニコチンアミド環のラジカル中心からPまでの経路が
                # フラビンと異なる (共役がない → HFCは小さい)
                # NMN-P (近位): ~20 MHz, AMP-P (遠位): ~1 MHz
                # 2'-P (NADPH固有): ~5 MHz
                if pdb_P_distances and len(pdb_P_distances) >= 2:
                    sorted_dists = sorted(
                        [p["distance_A"] for p in pdb_P_distances])
                    if pdb_r <= sorted_dists[0] + 0.5:
                        A_iso_31P = 20.0   # NMN-P (最近位)
                        hfc_class = "NADPH_NMN_P"
                    elif len(sorted_dists) >= 3 and pdb_r <= sorted_dists[1] + 0.5:
                        A_iso_31P = 5.0    # 2'-P (中位)
                        hfc_class = "NADPH_2P"
                    else:
                        A_iso_31P = 1.0    # AMP-P (遠位)
                        hfc_class = "NADPH_AMP_P"
                else:
                    A_iso_31P = 20.0
                    hfc_class = "NADPH_assumed"
            else:
                # その他: 距離ベースフォールバック
                if pdb_r < 4.0:
                    A_iso_31P = 100.0
                    hfc_class = "direct_bond"
                elif pdb_r < 9.0:
                    A_iso_31P = 50.0
                    hfc_class = "proximal"
                else:
                    A_iso_31P = 2.0
                    hfc_class = "remote"

            hfcs.append({"atom_idx": i, "element": "³¹P",
                         "A_iso_MHz": round(A_iso_31P, 2),
                         "hfc_class": hfc_class,
                         "r_eP_A": round(pdb_r, 1)})

        elif elem in ("FE", "CU", "MN", "CO", "NI"):
            # 遷移金属: 非常に大きなHFC (数百〜数千MHz)
            metal_hfc = {
                "FE": 30.0,   # ⁵⁷Fe, 低いγのため小さめ
                "CU": 600.0,  # ⁶³Cu, d9系で大きい
                "MN": 250.0,  # ⁵⁵Mn
                "CO": 400.0,  # ⁵⁹Co
                "NI": 100.0,  # ⁶¹Ni
            }
            A_iso = metal_hfc.get(elem, 100.0)
            hfcs.append({"atom_idx": i, "element": f"metal-{elem}",
                         "A_iso_MHz": round(A_iso, 2)})

    return hfcs


# ===========================================================================
# 5b. デコヒーレンス/緩和時間解析
# ===========================================================================

class DecoherenceAnalyzer:
    """計算済みのxTB/ONIOM結果からデコヒーレンス/緩和時間を推定。

    全てポスト処理 — 新たな量子化学計算は行わない。
    主要な緩和チャネル:
      1. スピン脱位相 T₂* (HFC分布)
      2. 項間交差 k_ISC (SOC + Marcus FCWD)
      3. 輻射寿命 τ_rad (Strickler-Berg)
      4. 振動緩和 τ_vib (エネルギーギャップ則)
      5. 電子脱位相 T₂_el (Kubo, MD揺らぎ)
      6. 電子移動 k_ET (Marcus理論)
      7. スピン格子緩和 T₁ᵉ (BPP理論)
    """

    # HFC element label → nuclear spin I
    _SPIN_I_MAP = {
        "¹H": 0.5, "¹³C": 0.5, "¹⁴N": 1.0, "¹⁵N": 0.5,
        "¹⁷O": 2.5, "³¹P": 0.5, "³³S": 1.5,
        "metal-FE": 0.5, "metal-CU": 1.5, "metal-MN": 2.5,
        "metal-CO": 3.5, "metal-NI": 1.5, "metal-MO": 2.5,
        "metal-ZN": 2.5,
    }

    def __init__(self, temperature_K: float = BODY_TEMP_K):
        self.T = temperature_K
        self.kT_eV = KB_EV * temperature_K

    @staticmethod
    def _get_eP_distance_m(qs) -> float:
        """e⁻-P距離をPDB実測値またはフォールバック値から取得 (m単位)."""
        if qs.measured_eP_distance_A > 0:
            return qs.measured_eP_distance_A * 1e-10
        # フォールバック: 残基名から推定
        if "PLP" in str(qs.residues) or "LLP" in str(qs.residues):
            return 3.5e-10
        return 7.0e-10

    # ----- 1. スピン脱位相 T₂* -----

    def compute_T2_star_spin(self, qs) -> None:
        """HFC分布からのスピン脱位相時間。

        1/T₂* = √( Σᵢ Aᵢ²·Iᵢ(Iᵢ+1)/3 )  [rad/s]
        Aᵢ: 超微細結合定数 (rad/s), Iᵢ: 結合核スピン量子数
        """
        hfcs = qs.estimated_hfc_MHz
        if not hfcs:
            return

        sum_sq = 0.0
        for h in hfcs:
            A_rad = h["A_iso_MHz"] * MHZ_TO_RAD_S
            I = self._SPIN_I_MAP.get(h["element"], 0.5)
            sum_sq += A_rad ** 2 * I * (I + 1) / 3.0

        if sum_sq > 0:
            inv_T2 = np.sqrt(sum_sq)  # rad/s
            qs.T2_star_spin_ns = 1e9 / inv_T2  # s → ns

    # ----- 2. 項間交差速度 k_ISC -----

    def _estimate_lambda(self, qs) -> float:
        """再配向エネルギーλを推定 (eV)。
        優先順位: ONIOM gap shift → MD gap variance → fallback 0.1 eV
        """
        # (1) ONIOM gap shift
        if abs(qs.embedding_shift_gap_eV) > 0.01:
            lam = abs(qs.embedding_shift_gap_eV) / 2.0
            if 0.01 < lam < 3.0:
                return lam

        # (2) MD ensemble variance: λ = σ²/(2kT)
        if qs.homo_lumo_gap_std > 0.001:
            lam = qs.homo_lumo_gap_std ** 2 / (2.0 * self.kT_eV)
            if 0.01 < lam < 3.0:
                return lam

        # (3) fallback
        if qs.site_type == "metal_center":
            return 0.5  # 金属中心は大きなλ
        return 0.1  # 有機系

    def compute_k_isc(self, qs) -> None:
        """Fermi黄金則 + Marcus FCWD による ISC 速度定数。

        k_ISC = (2π/ℏ) ξ_eff² × FCWD
        FCWD = (4πλkT)^(-1/2) exp(-(ΔE_ST - λ)²/(4λkT))
        """
        xi_cm = qs.effective_soc_cm
        delta_EST = abs(qs.singlet_triplet_gap_eV)
        if xi_cm < 0.01 or delta_EST < 1e-6:
            return

        lam = self._estimate_lambda(qs)
        qs.lambda_reorg_eV = lam

        # SOC matrix element in eV
        xi_eV = xi_cm / EV_TO_CM

        # Franck-Condon weighted density of states (1/eV)
        denom = 4.0 * lam * self.kT_eV
        if denom < 1e-10:
            return
        exponent = -(delta_EST - lam) ** 2 / denom
        if exponent < -100:
            return  # underflow guard
        fcwd = 1.0 / np.sqrt(np.pi * denom) * np.exp(exponent)

        # k_ISC = (2π/ℏ) |V_SOC|² × FCWD  [s⁻¹]
        # V_SOC in eV, FCWD in 1/eV → k in 1/s
        k_isc = (2.0 * np.pi / HBAR_EV_S) * xi_eV ** 2 * fcwd
        qs.k_isc_per_s = k_isc
        if k_isc > 0:
            qs.tau_isc_ns = 1e9 / k_isc

    # ----- 3. 輻射寿命 τ_rad -----

    def compute_tau_rad(self, qs) -> None:
        """Strickler-Berg の式による輻射寿命。

        1/τ_rad = 2.88×10⁻⁹ n² ν̃² f_osc   [s⁻¹]
        n = 1.4 (タンパク質中の屈折率)
        """
        f_osc = qs.oscillator_strength
        gap_eV = qs.homo_lumo_gap_eV
        if f_osc < 1e-8 or gap_eV < 0.01:
            return

        n_refr = 1.4  # タンパク質内部の屈折率
        nu_cm = gap_eV * EV_TO_CM  # cm⁻¹

        # 1/τ_rad [s⁻¹]
        k_rad = 2.88e-9 * n_refr ** 2 * nu_cm ** 2 * f_osc
        if k_rad > 0:
            qs.tau_rad_ns = 1e9 / k_rad

    # ----- 4. 振動緩和 τ_vib -----

    def compute_tau_vib(self, qs) -> None:
        """エネルギーギャップ則による非輻射振動緩和速度。

        k_nr = A × exp(-γ × (ΔE/ℏω_max - 1))
        A ~ 10¹³ s⁻¹, γ ~ ln(ΔE/(ℏω_max × S)) ≈ 1-3

        ΔE は真空のHOMO-LUMO gap を使用（ONIOM gapは環境シフトを含むため
        内部振動緩和には真空値が適切）。
        """
        if not qs.vib_modes:
            return

        # 真空gapを使用（環境効果は内部振動緩和には無関係）
        gap_eV = qs.vacuum_homo_lumo_gap_eV if qs.vacuum_homo_lumo_gap_eV > 0.1 \
            else qs.homo_lumo_gap_eV
        if gap_eV < 0.1:
            return

        delta_E_cm = gap_eV * EV_TO_CM
        A_prefactor = 1e13  # s⁻¹

        # 最高振動モードの周波数を取得
        max_freq = max(m.get("freq_cm", 0) for m in qs.vib_modes)
        if max_freq < 50:
            return

        # エネルギーギャップ則: p = ΔE/ℏω_max (受け入れモードの量子数)
        p = delta_E_cm / max_freq
        if p < 1:
            # ΔE < ℏω_max: 直接1量子過程 → 非常に速い
            qs.tau_vib_fastest_ps = 0.1  # ~100 fs
            return

        # γ ≈ ln(p/S) - 1, S = Huang-Rhys因子 ≈ 0.5-2 for organic
        gamma = max(1.0, np.log(p) - 0.3)

        # k_nr = A × exp(-γ × (p - 1))
        exponent = -gamma * (p - 1)
        if exponent < -50:
            return  # 極めて遅い → 無視 (> 10¹⁵ ps)

        k_nr = A_prefactor * np.exp(exponent)
        if k_nr > 0:
            tau_ps = 1e12 / k_nr
            # 物理的な範囲にクランプ (0.01 ps〜10⁹ ps = 1 ms)
            qs.tau_vib_fastest_ps = max(0.01, min(tau_ps, 1e9))

    # ----- 5. 電子純粋脱位相 T₂_el -----

    def compute_T2_el(self, qs) -> None:
        """Kubo静的極限: T₂_el = ℏ / σ_gap

        σ_gap = MD ensemble での HOMO-LUMO gap の標準偏差
        """
        sigma = qs.homo_lumo_gap_std
        if sigma < 1e-6:
            return
        # T₂ = ℏ/σ in seconds → femtoseconds
        T2_s = HBAR_EV_S / sigma
        qs.T2_el_fs = T2_s * 1e15

    # ----- 6. Marcus電子移動速度 k_ET -----

    def compute_k_et(self, qs) -> None:
        """Marcus理論による電子移動速度。

        k_ET = (2π/ℏ)|H_DA|² (4πλkT)^(-1/2) exp(-(ΔG°+λ)²/(4λkT))
        H_DA ≈ 0.01 eV (タンパク質中の典型値)
        """
        lam = qs.lambda_reorg_eV
        if lam < 0.001:
            lam = self._estimate_lambda(qs)
            qs.lambda_reorg_eV = lam

        H_DA = 0.01  # eV, タンパク質中の電子結合の典型値

        # ΔG° 推定: ラジカルカチオン生成自由エネルギー
        if qs.radical_cation_ip_eV > 0 and qs.radical_anion_ea_eV > 0:
            delta_G = -(qs.radical_cation_ip_eV - qs.radical_anion_ea_eV) / 2.0
        else:
            delta_G = 0.0  # activationless

        denom = 4.0 * lam * self.kT_eV
        if denom < 1e-10:
            return
        exponent = -(delta_G + lam) ** 2 / denom
        if exponent < -100:
            return
        fcwd = 1.0 / np.sqrt(np.pi * denom) * np.exp(exponent)

        k_et = (2.0 * np.pi / HBAR_EV_S) * H_DA ** 2 * fcwd
        qs.k_et_per_s = k_et

    # ----- 7. スピン格子緩和 T₁ᵉ -----

    def compute_T1e(self, qs) -> None:
        """BPP (Bloembergen-Purcell-Pound) 理論。

        1/T₁ = A_eff² τ_c / (1 + ω₀²τ_c²)
        A_eff: 最大HFC (rad/s)
        τ_c: 相関時間 (最低振動モードから推定)
        ω₀: 電子Larmor周波数 (X-band ESR ~9.4 GHz)
        """
        if qs.max_hfc_MHz < 0.1:
            return

        A_eff = qs.max_hfc_MHz * MHZ_TO_RAD_S  # rad/s

        # 相関時間: 最低振動モードの逆数
        if qs.vib_modes:
            freqs = [m["freq_cm"] for m in qs.vib_modes if m.get("freq_cm", 0) > 50]
            if freqs:
                freq_min_cm = min(freqs)
                tau_c = 1.0 / (2.0 * np.pi * freq_min_cm * C_CM_S)  # seconds
            else:
                tau_c = 1e-12  # 1 ps fallback
        else:
            tau_c = 1e-12  # 1 ps fallback

        # 電子 Larmor 周波数 (X-band ESR, ~9.4 GHz)
        omega_0 = 2.0 * np.pi * 9.4e9  # rad/s

        # BPP formula
        inv_T1 = A_eff ** 2 * tau_c / (1.0 + omega_0 ** 2 * tau_c ** 2)
        if inv_T1 > 0:
            qs.T1e_spin_lattice_us = 1e6 / inv_T1  # s → μs

    # ----- 8. Zeeman分裂 -----

    def compute_zeeman(self, qs) -> None:
        """電子スピンと³¹P核スピンのZeeman分裂を計算。

        ΔE_electron = g_e μ_B B
        ΔE_nuclear  = γ_N ℏ B  (= γ_N B in frequency units)
        """
        # 電子スピン Zeeman
        # ΔE = g μ_B B → 周波数 = g μ_B B / h = g × 14.0025 MHz/T × B
        freq_per_T = G_ELECTRON * 14.0025e6  # Hz/T (= μ_B/h)
        qs.zeeman_electron_earth_MHz = freq_per_T * EARTH_FIELD_T * 1e-6
        qs.zeeman_electron_xband_GHz = freq_per_T * XBAND_FIELD_T * 1e-9

        # ³¹P Zeeman (γ = 17.235 MHz/T)
        gamma_P31 = 17.235e6  # Hz/T
        qs.zeeman_P31_earth_Hz = gamma_P31 * EARTH_FIELD_T
        qs.zeeman_P31_xband_kHz = gamma_P31 * XBAND_FIELD_T * 1e-3

        # P原子数をカウント
        n_P = 0
        for entry in qs.nuclear_spin_inventory:
            if isinstance(entry, dict) and entry.get("element", "").upper() == "P":
                n_P += entry.get("total_atoms", entry.get("count", 0))
        qs.n_P_atoms = n_P

    # ----- 9. 磁気遷移双極子モーメント -----

    def compute_magnetic_tdm(self, qs) -> None:
        """スピン遷移の磁気遷移双極子モーメントを計算。

        電子スピン: μ_mag = g_e μ_B √(S(S+1))  for S=1/2
        ³¹P核スピン: μ_mag = γ ℏ √(I(I+1))  for I=1/2
        SOC誘起 S→T 電気TDM: |μ_el(S→T)| ≈ ξ_eff/ΔE_ST × |μ_el(S→S)|
        """
        # 電子スピン磁気TDM (μ_B単位)
        S = 0.5
        qs.magnetic_tdm_electron_muB = G_ELECTRON * np.sqrt(S * (S + 1))

        # ³¹P 核スピン磁気TDM (μ_N単位)
        I_P31 = 0.5
        mu_P31 = 1.1317  # μ/μ_N for ³¹P
        qs.magnetic_tdm_P31_muN = mu_P31 * np.sqrt(I_P31 * (I_P31 + 1))

        # SOC誘起 S→T 電気遷移双極子モーメント
        # |μ(S→T)| ≈ (ξ_SOC / ΔE_ST) × |μ(S₀→S₁)|
        xi_cm = qs.effective_soc_cm
        delta_st = abs(qs.singlet_triplet_gap_eV)
        mu_ss = qs.estimated_electronic_tdm_D
        if xi_cm > 0.1 and delta_st > 0.01 and mu_ss > 0.01:
            xi_eV = xi_cm / EV_TO_CM
            qs.electric_tdm_spin_D = (xi_eV / delta_st) * mu_ss

    # ----- 10. 双極子-双極子横緩和 T₂(dipolar) -----

    def compute_T2_dipolar(self, qs) -> None:
        """電子スピン双極子-双極子横緩和。

        ラジカルペアにおけるスピン間双極子結合:
        1/T₂(dd) = (μ₀/4π)² g⁴μ_B⁴ S(S+1) / (r⁶) × τ_c × (3 + 5/(1+ω₀²τ_c²))
        r: ラジカルペア間距離 (~15 Å for cryptochrome, ~10 Å for PLP)
        τ_c: 回転相関時間 (~10 ns for protein)
        """
        # ラジカルペア間距離の推定
        if qs.site_type == "chromophore":
            r_m = 15e-10  # 15 Å (Trpトライアド距離)
        elif qs.site_type == "pi_stack":
            r_m = 4e-10   # 4 Å (π-stack)
        else:
            r_m = 10e-10  # 10 Å default

        # 回転相関時間: ~10 ns for protein in solution
        tau_c = 10e-9  # 10 ns

        S = 0.5
        mu0_4pi = 1e-7  # μ₀/(4π) in T·m/A
        omega_0 = 2 * np.pi * 9.4e9  # X-band

        # 双極子結合定数
        D_factor = (mu0_4pi * G_ELECTRON ** 2 * MU_BOHR ** 2 / (r_m ** 3)) ** 2
        # スペクトル密度
        J0 = tau_c
        J_omega = tau_c / (1.0 + omega_0 ** 2 * tau_c ** 2)

        # 1/T₂(dd) ∝ D² × S(S+1) × (3J(0) + 5J(ω₀)) / (15ℏ²)
        # 簡略化: Solomon方程式
        inv_T2_dd = (D_factor * S * (S + 1) / HBAR_J_S ** 2) * (3 * J0 + 5 * J_omega) / 15.0

        if inv_T2_dd > 0:
            qs.T2_dipolar_ns = 1e9 / inv_T2_dd

    # ----- 11. 合計横緩和 T₂ -----

    def compute_T2_total(self, qs) -> None:
        """合計横緩和: 1/T₂ = 1/T₂*(HFC) + 1/T₂(dipolar)"""
        inv_T2 = 0.0
        if qs.T2_star_spin_ns > 0:
            inv_T2 += 1.0 / qs.T2_star_spin_ns
        if qs.T2_dipolar_ns > 0:
            inv_T2 += 1.0 / qs.T2_dipolar_ns
        if inv_T2 > 0:
            qs.T2_total_ns = 1.0 / inv_T2

    # ----- 12. 核スピン緩和 -----

    def compute_nuclear_relaxation(self, qs) -> None:
        """核スピン (³¹P, ¹H, ¹⁴N) の T₁, T₂ を BPP 理論で計算。

        核スピン-核スピン双極子結合による緩和:
        1/T₁ = K [J(ω_I - ω_S) + 3J(ω_I) + 6J(ω_I + ω_S)]
        1/T₂ = K/2 [4J(0) + J(ω_I - ω_S) + 3J(ω_I) + 6J(ω_S) + 6J(ω_I + ω_S)]
        K = (2/15)(μ₀/4π)² γ_I² γ_S² ℏ² I(I+1) / r⁶

        主要緩和源:
        - ³¹P: ¹H-³¹P 双極子結合 (r ~2.5 Å in PLP/FAD)
        - ¹H:  ¹H-¹H 双極子結合 (r ~1.8 Å geminal)
        - ¹⁴N: 四極子緩和 (I=1, Q ≠ 0)
        """
        mu0_4pi = 1e-7  # T·m/A
        tau_c = 10e-9    # タンパク質回転相関時間 ~10 ns

        # スペクトル密度関数 J(ω) = τ_c / (1 + ω²τ_c²)
        def J(omega):
            return tau_c / (1.0 + omega ** 2 * tau_c ** 2)

        # --- ³¹P T₁, T₂ ---
        gamma_P = 17.235e6 * 2 * np.pi  # rad/s/T
        gamma_H = 42.577e6 * 2 * np.pi  # rad/s/T
        B0 = 11.7  # T (500 MHz NMR, 典型的タンパク質NMR磁場)

        omega_P = gamma_P / (2 * np.pi) * B0 * 2 * np.pi  # ³¹P Larmor (rad/s)
        omega_H = gamma_H / (2 * np.pi) * B0 * 2 * np.pi  # ¹H Larmor (rad/s)

        # ¹H-³¹P 双極子結合 (P-H距離 ~2.5 Å in ribose phosphate)
        r_PH = 2.5e-10  # m
        I_H = 0.5
        K_PH = (2.0 / 15.0) * (mu0_4pi ** 2) * (gamma_P ** 2) * (gamma_H ** 2) \
            * (HBAR_J_S ** 2) * I_H * (I_H + 1) / (r_PH ** 6)

        # 近接¹Hの数 (PLP: ~3, FAD: ~5)
        if qs.n_P_atoms > 0:
            n_nearby_H = 3 if "PLP" in str(qs.residues) or "LLP" in str(qs.residues) else 5

            inv_T1_P = n_nearby_H * K_PH * (
                J(omega_H - omega_P) + 3 * J(omega_P) + 6 * J(omega_H + omega_P))
            inv_T2_P = n_nearby_H * K_PH / 2.0 * (
                4 * J(0) + J(omega_H - omega_P) + 3 * J(omega_P)
                + 6 * J(omega_H) + 6 * J(omega_H + omega_P))

            # CSA 寄与 (³¹P CSA ~120 ppm in phosphodiesters)
            delta_sigma_P = 120e-6  # ppm → dimensionless
            inv_T1_CSA = (2.0 / 15.0) * omega_P ** 2 * delta_sigma_P ** 2 * (
                J(omega_P))
            inv_T2_CSA = (1.0 / 45.0) * omega_P ** 2 * delta_sigma_P ** 2 * (
                4 * J(0) + 3 * J(omega_P))

            inv_T1_P_total = inv_T1_P + inv_T1_CSA
            inv_T2_P_total = inv_T2_P + inv_T2_CSA

            if inv_T1_P_total > 0:
                qs.T1_P31_s = 1.0 / inv_T1_P_total
            if inv_T2_P_total > 0:
                qs.T2_P31_ms = 1e3 / inv_T2_P_total

        # --- ¹H T₁, T₂ ---
        # ¹H-¹H 双極子 (geminal H-H ~1.8 Å)
        r_HH = 1.8e-10
        K_HH = (2.0 / 15.0) * (mu0_4pi ** 2) * (gamma_H ** 4) \
            * (HBAR_J_S ** 2) * I_H * (I_H + 1) / (r_HH ** 6)
        n_nearby_H_H = 2  # geminal pair

        inv_T1_H = n_nearby_H_H * K_HH * (
            J(0) + 3 * J(omega_H) + 6 * J(2 * omega_H))
        inv_T2_H = n_nearby_H_H * K_HH / 2.0 * (
            5 * J(0) + 9 * J(omega_H) + 6 * J(2 * omega_H))

        if inv_T1_H > 0:
            qs.T1_H1_s = 1.0 / inv_T1_H
        if inv_T2_H > 0:
            qs.T2_H1_ms = 1e3 / inv_T2_H

        # --- ¹⁴N 四極子緩和 ---
        # 1/T₁(Q) = (3/40)(2I+3)/(I²(2I-1)) × (e²qQ/ℏ)² × (1+η²/3) × τ_c
        # ¹⁴N: I=1, e²qQ/ℏ ~3 MHz (typical peptide bond)
        e2qQ_N14 = 3.0e6 * 2 * np.pi  # rad/s
        eta_N14 = 0.3  # 非対称パラメータ
        I_N = 1.0
        prefactor_Q = (3.0 / 40.0) * (2 * I_N + 3) / (I_N ** 2 * (2 * I_N - 1))
        inv_T1_N = prefactor_Q * e2qQ_N14 ** 2 * (1.0 + eta_N14 ** 2 / 3.0) * tau_c
        if inv_T1_N > 0:
            qs.T1_N14_ms = 1e3 / inv_T1_N

    # ----- 13. 常磁性緩和促進 (PRE) -----

    def compute_PRE(self, qs) -> None:
        """ラジカル存在時の³¹P核スピンへの常磁性緩和促進 (PRE)。

        Solomon-Bloembergen方程式:
        1/T₁(PRE) = (2/15)(μ₀/4π)² γ_I² g² μ_B² S(S+1) / r⁶ ×
                     [3J(ω_I) + 7J(ω_S)]
        r: 電子-核間距離
        """
        if qs.n_P_atoms == 0:
            return

        mu0_4pi = 1e-7
        gamma_P = 17.235e6 * 2 * np.pi  # rad/s/T
        S_e = 0.5
        tau_c = 10e-9  # ns
        B0 = 11.7  # T

        omega_I = gamma_P / (2 * np.pi) * B0 * 2 * np.pi
        omega_S = G_ELECTRON * MU_BOHR / HBAR_J_S * B0  # electron Larmor

        def J(omega):
            return tau_c / (1.0 + omega ** 2 * tau_c ** 2)

        r_eP = self._get_eP_distance_m(qs)

        K_PRE = (2.0 / 15.0) * (mu0_4pi ** 2) * (gamma_P ** 2) \
            * (G_ELECTRON ** 2) * (MU_BOHR ** 2) * S_e * (S_e + 1) / (r_eP ** 6)

        inv_T1_PRE = K_PRE * (3 * J(omega_I) + 7 * J(omega_S))
        inv_T2_PRE = K_PRE / 2.0 * (4 * J(0) + 3 * J(omega_I) + 13 * J(omega_S))

        if inv_T1_PRE > 0:
            qs.PRE_T1_P31_ms = 1e3 / inv_T1_PRE
        if inv_T2_PRE > 0:
            qs.PRE_T2_P31_us = 1e6 / inv_T2_PRE

    # ----- 14. 核スピンコヒーレンス時間 -----

    def compute_nuclear_coherence(self, qs) -> None:
        """核スピンの実効コヒーレンス時間を決定。

        反磁性状態: T₂(nuclear) = T₂(P31 or H1)
        ラジカル状態: T₂(nuclear) ≈ PRE_T₂ (常磁性効果が支配的)
        """
        # 反磁性状態での核スピンコヒーレンス
        if qs.n_P_atoms > 0 and qs.T2_P31_ms > 0:
            coh_diamag = qs.T2_P31_ms  # ms
        elif qs.T2_H1_ms > 0:
            coh_diamag = qs.T2_H1_ms
        else:
            coh_diamag = 0.0

        # ラジカル存在時のPRE T₂（より短い）
        if qs.PRE_T2_P31_us > 0:
            coh_paramag = qs.PRE_T2_P31_us / 1e3  # μs → ms
        else:
            coh_paramag = coh_diamag

        # 核スピンコヒーレンス = 反磁性/常磁性の短い方
        if coh_diamag > 0 and coh_paramag > 0:
            qs.nuclear_coherence_time_ms = min(coh_diamag, coh_paramag)
        elif coh_diamag > 0:
            qs.nuclear_coherence_time_ms = coh_diamag

    # ----- 15. 脳内環境での核スピン緩和 -----

    def compute_brain_nuclear_spins(self, qs) -> None:
        """脳内環境 (B₀=50μT, T=310K) での核スピン緩和を計算。

        地磁気では ω₀τ_c ≪ 1 (extreme narrowing) となり:
        - T₁ = T₂ (高磁場とは根本的に異なる)
        - CSA寄与はω₀²に比例 → 無視できるほど小さい
        - 双極子緩和が支配的

        脳内の追加効果:
        1. 溶存O₂ (常磁性, ~0.1 mM in brain tissue)
        2. 常磁性金属イオン (Fe³⁺ ~0.04 mM in grey matter)
        3. 水プロトン化学交換 (τ_ex ~1-10 ms)
        4. タンパク質表面水の高速交換
        """
        mu0_4pi = 1e-7
        tau_c = 10e-9  # タンパク質回転相関時間

        B0 = EARTH_FIELD_T  # 50 μT

        gamma_P_Hz = 17.235e6
        gamma_H_Hz = 42.577e6
        gamma_P = gamma_P_Hz * 2 * np.pi
        gamma_H = gamma_H_Hz * 2 * np.pi

        omega_P = gamma_P_Hz * B0 * 2 * np.pi  # ³¹P Larmor @50μT (rad/s)
        omega_H = gamma_H_Hz * B0 * 2 * np.pi  # ¹H Larmor @50μT (rad/s)

        # ω₀τ_c の確認
        omega_H_tau = omega_H * tau_c
        # ¹H: ω₀ = 2π × 42.577e6 × 50e-6 = 2π × 2129 Hz
        # ω₀τ_c = 2π × 2129 × 10e-9 = 1.34e-4 ≪ 1 → extreme narrowing
        qs.brain_nuclear_coherence_regime = "extreme narrowing" if omega_H_tau < 0.01 else "intermediate"

        I_H = 0.5

        def J(omega):
            return tau_c / (1.0 + omega ** 2 * tau_c ** 2)

        # =========================================================
        # (A) ³¹P 緩和 @地磁気
        # =========================================================
        if qs.n_P_atoms > 0:
            r_PH = 2.5e-10
            K_PH = (2.0 / 15.0) * (mu0_4pi ** 2) * (gamma_P ** 2) * (gamma_H ** 2) \
                * (HBAR_J_S ** 2) * I_H * (I_H + 1) / (r_PH ** 6)

            n_nearby_H = 3 if "PLP" in str(qs.residues) or "LLP" in str(qs.residues) else 5

            # 地磁気での双極子緩和
            inv_T1_P = n_nearby_H * K_PH * (
                J(omega_H - omega_P) + 3 * J(omega_P) + 6 * J(omega_H + omega_P))
            inv_T2_P = n_nearby_H * K_PH / 2.0 * (
                4 * J(0) + J(omega_H - omega_P) + 3 * J(omega_P)
                + 6 * J(omega_H) + 6 * J(omega_H + omega_P))

            # CSA @地磁気 → ω₀² ∝ B₀² → (50μT/11.7T)² = 1.8×10⁻¹¹ → 無視
            # CSA寄与は実質ゼロ

            if inv_T1_P > 0:
                qs.brain_P31_T1_s = 1.0 / inv_T1_P
            if inv_T2_P > 0:
                qs.brain_P31_T2_s = 1.0 / inv_T2_P
            # extreme narrowing: T₁ ≈ T₂ (確認)

            # --- PRE @地磁気 ---
            r_eP = self._get_eP_distance_m(qs)

            S_e = 0.5
            omega_S = G_ELECTRON * MU_BOHR * B0 / HBAR_J_S  # electron Larmor @50μT

            K_PRE = (2.0 / 15.0) * (mu0_4pi ** 2) * (gamma_P ** 2) \
                * (G_ELECTRON ** 2) * (MU_BOHR ** 2) * S_e * (S_e + 1) / (r_eP ** 6)

            # Solomon-Bloembergen @地磁気
            # 地磁気では ω_S ~8.8e6 rad/s, ω_Sτ_c ~0.088 ≈ 0.1 → nearly extreme narrowing
            inv_T1_PRE = K_PRE * (3 * J(omega_P) + 7 * J(omega_S))
            inv_T2_PRE = K_PRE / 2.0 * (4 * J(0) + 3 * J(omega_P) + 13 * J(omega_S))

            if inv_T1_PRE > 0:
                qs.brain_PRE_P31_T1_ms = 1e3 / inv_T1_PRE
            if inv_T2_PRE > 0:
                qs.brain_PRE_P31_T2_ms = 1e3 / inv_T2_PRE

        # =========================================================
        # (B) ¹H 緩和 @地磁気
        # =========================================================
        r_HH = 1.8e-10
        K_HH = (2.0 / 15.0) * (mu0_4pi ** 2) * (gamma_H ** 4) \
            * (HBAR_J_S ** 2) * I_H * (I_H + 1) / (r_HH ** 6)
        n_nearby_H_H = 2

        inv_T1_H = n_nearby_H_H * K_HH * (
            J(0) + 3 * J(omega_H) + 6 * J(2 * omega_H))
        inv_T2_H = n_nearby_H_H * K_HH / 2.0 * (
            5 * J(0) + 9 * J(omega_H) + 6 * J(2 * omega_H))

        if inv_T1_H > 0:
            qs.brain_H1_T1_s = 1.0 / inv_T1_H
        if inv_T2_H > 0:
            qs.brain_H1_T2_s = 1.0 / inv_T2_H

        # =========================================================
        # (C) 脳内常磁性不純物
        # =========================================================
        # 灰白質: [Fe³⁺] ~0.04 mM, [Cu²⁺] ~0.015 mM
        # 溶存 O₂ ~0.1 mM (pO₂ ~40 mmHg in brain tissue)
        # outer-sphere PRE: 1/T₁(para) = C × [para] × γ²
        # C ≈ 0.3 mM⁻¹s⁻¹ for Fe³⁺ at low field (outer sphere)

        # Fe³⁺ (S=5/2, τ_c ~10⁻¹⁰ s for aqua ion)
        conc_Fe = 0.04e-3  # mol/L = 0.04 mM
        # outer-sphere relaxivity: r₁ ~0.7 mM⁻¹s⁻¹ for low-field Fe³⁺
        r1_Fe = 0.7  # mM⁻¹ s⁻¹
        inv_T1_Fe = r1_Fe * conc_Fe * 1e3  # s⁻¹

        # O₂ (S=1, τ_rot ~5 ps)
        conc_O2 = 0.1e-3  # mol/L
        r1_O2 = 0.2  # mM⁻¹ s⁻¹ (low field)
        inv_T1_O2 = r1_O2 * conc_O2 * 1e3

        if inv_T1_Fe + inv_T1_O2 > 0:
            qs.brain_paramag_T1_s = 1.0 / (inv_T1_Fe + inv_T1_O2)
        if inv_T1_O2 > 0:
            qs.brain_O2_T1_s = 1.0 / inv_T1_O2

        # =========================================================
        # (D) 水プロトン化学交換
        # =========================================================
        # タンパク質表面のNH, OH基は水と化学交換
        # 交換速度: k_ex ~100-1000 s⁻¹ → τ_ex ~1-10 ms
        # PLPのシッフ塩基NH: k_ex ~500 s⁻¹
        # FADのリボースOH: k_ex ~200 s⁻¹
        if "PLP" in str(qs.residues) or "LLP" in str(qs.residues):
            k_ex = 500.0  # s⁻¹
        else:
            k_ex = 200.0  # s⁻¹
        qs.brain_water_exchange_ms = 1e3 / k_ex

        # =========================================================
        # (E) 実効T₂の統合
        # =========================================================
        # 1/T₂_eff = 1/T₂(dipolar) + 1/T₂(PRE) + 1/T₁(paramag) + 1/T₂(exchange)

        # ³¹P 実効T₂
        inv_T2_P_eff = 0.0
        if qs.brain_P31_T2_s > 0:
            inv_T2_P_eff += 1.0 / qs.brain_P31_T2_s
        if qs.brain_PRE_P31_T2_ms > 0:
            inv_T2_P_eff += 1.0 / (qs.brain_PRE_P31_T2_ms * 1e-3)
        if qs.brain_paramag_T1_s > 0:
            inv_T2_P_eff += 1.0 / qs.brain_paramag_T1_s  # paramag impurities
        if inv_T2_P_eff > 0:
            qs.brain_P31_T2_effective_ms = 1e3 / inv_T2_P_eff

        # ¹H 実効T₂
        inv_T2_H_eff = 0.0
        if qs.brain_H1_T2_s > 0:
            inv_T2_H_eff += 1.0 / qs.brain_H1_T2_s
        if qs.brain_paramag_T1_s > 0:
            inv_T2_H_eff += 1.0 / qs.brain_paramag_T1_s
        inv_T2_H_eff += k_ex  # 化学交換による脱位相
        if inv_T2_H_eff > 0:
            qs.brain_H1_T2_effective_ms = 1e3 / inv_T2_H_eff

        # =========================================================
        # (F) 脳内T₂ᵉ @地磁気
        # =========================================================
        # g-anisotropy: 1/T₂(g) = (1/5) Δg² ω₀² τ_c
        # 地磁気 ω₀ = 2π × 1.40 MHz vs X-band 2π × 9.4 GHz
        # 比率: (1.40e6/9.4e9)² = 2.2×10⁻⁸ → T₂(g-ani)は10⁸倍長い → 事実上∞
        xi_cm = qs.effective_soc_cm
        gap_eV = qs.homo_lumo_gap_eV
        if xi_cm > 0.1 and gap_eV > 0.01:
            xi_eV = xi_cm / EV_TO_CM
            delta_g = xi_eV / gap_eV
        else:
            delta_g = 0.003

        omega_e_earth = G_ELECTRON * MU_BOHR * B0 / HBAR_J_S  # ~8.8e6 rad/s
        inv_T2_g_earth = (1.0 / 5.0) * delta_g ** 2 * omega_e_earth ** 2 * tau_c
        if inv_T2_g_earth > 0:
            qs.brain_T2e_g_aniso_ns = 1e9 / inv_T2_g_earth

        # T₂ᵉ(total) @地磁気: HFC項は磁場非依存、g-ani項のみ変化
        inv_T2e_earth = 0.0
        if qs.T2_star_spin_ns > 0:
            inv_T2e_earth += 1.0 / qs.T2_star_spin_ns
        if qs.T2_dipolar_ns > 0:
            inv_T2e_earth += 1.0 / qs.T2_dipolar_ns
        if qs.brain_T2e_g_aniso_ns > 0:
            inv_T2e_earth += 1.0 / qs.brain_T2e_g_aniso_ns
        if qs.T2e_hfc_mod_ns > 0:
            inv_T2e_earth += 1.0 / qs.T2e_hfc_mod_ns
        # SOC項も磁場非依存
        if qs.T2e_spin_orbit_ns > 0:
            inv_T2e_earth += 1.0 / qs.T2e_spin_orbit_ns
        if inv_T2e_earth > 0:
            qs.brain_T2e_total_ns = 1.0 / inv_T2e_earth

        # =========================================================
        # (G) 結合スピン系 交差緩和 @地磁気
        # =========================================================
        if qs.n_P_atoms > 0:
            r_eP = self._get_eP_distance_m(qs)

            gamma_P_rad = 2 * np.pi * gamma_P_Hz
            d2 = (mu0_4pi ** 2) * (gamma_P_rad ** 2) * (G_ELECTRON ** 2) \
                * (MU_BOHR ** 2) * (HBAR_J_S ** 2) / (r_eP ** 6)

            # 地磁気での周波数
            omega_S_earth = omega_e_earth
            omega_I_earth = omega_P  # already computed above

            # Solomon方程式 @地磁気
            # extreme narrowing: J(ω) ≈ τ_c for all ω
            W0_earth = (1.0 / 20.0) * d2 * 6 * J(omega_S_earth - omega_I_earth) / HBAR_J_S ** 2
            W2_earth = (3.0 / 10.0) * d2 * J(omega_S_earth + omega_I_earth) / HBAR_J_S ** 2
            W1S_earth = (3.0 / 40.0) * d2 * 3 * J(omega_S_earth) / HBAR_J_S ** 2
            W1I_earth = (1.0 / 40.0) * d2 * 3 * J(omega_I_earth) / HBAR_J_S ** 2

            qs.brain_W0_per_s = W0_earth
            qs.brain_W2_per_s = W2_earth

            W_sum_earth = W0_earth + W2_earth
            if W_sum_earth > 0:
                qs.brain_T1x_us = 1e6 / W_sum_earth

            # 結合系T₂(e⁻) @地磁気
            inv_T2e_coupled_earth = 0.0
            if qs.brain_T2e_total_ns > 0:
                inv_T2e_coupled_earth = 1.0 / (qs.brain_T2e_total_ns * 1e-9)
            inv_T2e_coupled_earth += W1I_earth + W0_earth + W2_earth
            if inv_T2e_coupled_earth > 0:
                qs.brain_coupled_T2e_ns = 1e9 / inv_T2e_coupled_earth

            # 結合系T₂(³¹P) @地磁気
            inv_T2I_coupled_earth = 0.0
            if qs.brain_PRE_P31_T2_ms > 0:
                inv_T2I_coupled_earth = 1.0 / (qs.brain_PRE_P31_T2_ms * 1e-3)
            elif qs.brain_P31_T2_s > 0:
                inv_T2I_coupled_earth = 1.0 / qs.brain_P31_T2_s
            inv_T2I_coupled_earth += W1S_earth + W0_earth + W2_earth
            if inv_T2I_coupled_earth > 0:
                qs.brain_coupled_T2n_us = 1e6 / inv_T2I_coupled_earth

    # ----- 16. 電子T₂ᵉ完全版 -----

    def compute_T2e_full(self, qs) -> None:
        """電子横緩和 T₂ᵉ の3つの追加機構を計算。

        1. g-tensor異方性: 1/T₂(g) = (1/5) Δg² ω₀² τ_c
        2. HFC変調 (secular): 1/T₂(A) = (1/12) ΣA_i² τ_c I_i(I_i+1)
        3. SOC誘起 (Orbach-like): 1/T₂(SOC) ∝ ξ²/(ΔE)² × 1/τ_c

        τ_c: 回転相関時間 (~10 ns for protein)
        """
        tau_c = 10e-9  # s

        # --- (a) g-tensor異方性 ---
        # 有機ラジカル: Δg ≈ ξ_SOC / ΔE(n→π*)
        # 典型値: flavin radical Δg ~0.004, PLP Δg ~0.003
        xi_cm = qs.effective_soc_cm
        gap_eV = qs.homo_lumo_gap_eV
        if xi_cm > 0.1 and gap_eV > 0.01:
            xi_eV = xi_cm / EV_TO_CM
            delta_g = xi_eV / gap_eV  # Δg ≈ ξ/ΔE
        else:
            delta_g = 0.003  # 典型的有機ラジカル

        # X-band: ω₀ = 2π × 9.4 GHz
        omega_0 = 2 * np.pi * 9.4e9  # rad/s
        inv_T2_g = (1.0 / 5.0) * delta_g ** 2 * omega_0 ** 2 * tau_c
        if inv_T2_g > 0:
            qs.T2e_g_aniso_ns = 1e9 / inv_T2_g

        # --- (b) HFC変調 (secular contribution) ---
        # 1/T₂(A) = (1/12) Σ A_i² I_i(I_i+1) τ_c
        # A in rad/s
        sum_A2I = 0.0
        for hfc_entry in qs.estimated_hfc_MHz:
            if isinstance(hfc_entry, dict):
                A_MHz = abs(hfc_entry.get("A_iso_MHz", 0))
                elem = hfc_entry.get("element", "H")
                # 核スピンIを取得
                if elem in MAGNETIC_NUCLEI:
                    I_val = MAGNETIC_NUCLEI[elem][0][1]  # 最も豊富な同位体
                else:
                    I_val = 0.5
                A_rad_s = A_MHz * MHZ_TO_RAD_S
                sum_A2I += A_rad_s ** 2 * I_val * (I_val + 1)

        if sum_A2I > 0:
            inv_T2_A = (1.0 / 12.0) * sum_A2I * tau_c
            if inv_T2_A > 0:
                qs.T2e_hfc_mod_ns = 1e9 / inv_T2_A

        # --- (c) SOC誘起 (spin-orbit T₂) ---
        # Orbach-like: 1/T₂(SOC) ≈ ξ⁴/(ΔE²) × kT/ℏ × τ_c
        if xi_cm > 1.0 and gap_eV > 0.01:
            xi_eV = xi_cm / EV_TO_CM
            inv_T2_soc = (xi_eV ** 4 / gap_eV ** 2) * (self.kT_eV / HBAR_EV_S) * tau_c
            if inv_T2_soc > 0:
                qs.T2e_spin_orbit_ns = 1e9 / inv_T2_soc

        # --- 合計 T₂ᵉ ---
        inv_T2e = 0.0
        if qs.T2_star_spin_ns > 0:
            inv_T2e += 1.0 / qs.T2_star_spin_ns    # HFC inhomogeneous
        if qs.T2_dipolar_ns > 0:
            inv_T2e += 1.0 / qs.T2_dipolar_ns       # e-e dipolar
        if qs.T2e_g_aniso_ns > 0:
            inv_T2e += 1.0 / qs.T2e_g_aniso_ns      # g-anisotropy
        if qs.T2e_hfc_mod_ns > 0:
            inv_T2e += 1.0 / qs.T2e_hfc_mod_ns      # HFC modulation
        if qs.T2e_spin_orbit_ns > 0:
            inv_T2e += 1.0 / qs.T2e_spin_orbit_ns   # SOC
        if inv_T2e > 0:
            qs.T2e_total_ns = 1.0 / inv_T2e

    # ----- 16. 結合スピン系 (e⁻-³¹P) -----

    def compute_coupled_spin_system(self, qs) -> None:
        """結合電子-³¹P核スピン系の4準位構造、遷移、交差緩和を計算。

        ハミルトニアン (高磁場極限):
        H = ω_S S_z + ω_I I_z + A S_z I_z + B(S+I- + S-I+)/2

        4準位: |αα⟩, |αβ⟩, |βα⟩, |ββ⟩
        ω_S: 電子Larmor, ω_I: ³¹P Larmor, A: HFC (isotropic)
        """
        if qs.n_P_atoms == 0:
            return

        # ³¹P HFC: 最大のHFCを持つ³¹Pを使用（補因子別推定値）
        A_MHz = 0.0
        for hfc_entry in qs.estimated_hfc_MHz:
            if isinstance(hfc_entry, dict) and "P" in hfc_entry.get("element", ""):
                A_cand = abs(hfc_entry.get("A_iso_MHz", 0))
                A_MHz = max(A_MHz, A_cand)
        if A_MHz < 1.0:
            A_MHz = 100.0  # フォールバック

        tau_c = 10e-9  # s

        # --- 地磁気でのレベル構造 (B = 50 μT) ---
        B = EARTH_FIELD_T
        omega_S = G_ELECTRON * MU_BOHR * B / HBAR_J_S  # electron Larmor (rad/s)
        gamma_P_Hz = 17.235e6  # Hz/T
        omega_I = 2 * np.pi * gamma_P_Hz * B  # ³¹P Larmor (rad/s)
        A_rad = A_MHz * MHZ_TO_RAD_S

        # 4準位エネルギー (MHz) — 高磁場近似
        # E = ω_S m_S + ω_I m_I + A m_S m_I (in angular frequency)
        # |αα⟩ m_S=+1/2, m_I=+1/2: E = +ω_S/2 + ω_I/2 + A/4
        # |αβ⟩ m_S=+1/2, m_I=-1/2: E = +ω_S/2 - ω_I/2 - A/4
        # |βα⟩ m_S=-1/2, m_I=+1/2: E = -ω_S/2 + ω_I/2 - A/4
        # |ββ⟩ m_S=-1/2, m_I=-1/2: E = -ω_S/2 - ω_I/2 + A/4
        E_aa = (+omega_S / 2 + omega_I / 2 + A_rad / 4) / MHZ_TO_RAD_S
        E_ab = (+omega_S / 2 - omega_I / 2 - A_rad / 4) / MHZ_TO_RAD_S
        E_ba = (-omega_S / 2 + omega_I / 2 - A_rad / 4) / MHZ_TO_RAD_S
        E_bb = (-omega_S / 2 - omega_I / 2 + A_rad / 4) / MHZ_TO_RAD_S

        qs.coupled_level_energies_MHz = [
            {"state": "|αα⟩", "E_MHz": round(E_aa, 4)},
            {"state": "|αβ⟩", "E_MHz": round(E_ab, 4)},
            {"state": "|βα⟩", "E_MHz": round(E_ba, 4)},
            {"state": "|ββ⟩", "E_MHz": round(E_bb, 4)},
        ]

        # --- 遷移周波数 ---
        # EPR (Δm_S=±1, Δm_I=0):
        # |αα⟩→|βα⟩: ω_S + A/2, |αβ⟩→|ββ⟩: ω_S - A/2
        epr1 = abs(E_aa - E_ba)  # MHz
        epr2 = abs(E_ab - E_bb)  # MHz
        qs.coupled_EPR_freq_MHz = [
            {"transition": "|αα⟩→|βα⟩", "freq_MHz": round(epr1, 4)},
            {"transition": "|αβ⟩→|ββ⟩", "freq_MHz": round(epr2, 4)},
        ]

        # NMR (Δm_S=0, Δm_I=±1):
        # |αα⟩→|αβ⟩: ω_I + A/2, |βα⟩→|ββ⟩: ω_I - A/2
        nmr1 = abs(E_aa - E_ab)  # MHz
        nmr2 = abs(E_ba - E_bb)  # MHz
        qs.coupled_NMR_freq_MHz = [
            {"transition": "|αα⟩→|αβ⟩", "freq_MHz": round(nmr1, 4)},
            {"transition": "|βα⟩→|ββ⟩", "freq_MHz": round(nmr2, 4)},
        ]

        # Forbidden (Δm_S=±1, Δm_I=±1):
        # |αα⟩→|ββ⟩ (double quantum): ω_S + ω_I
        # |αβ⟩→|βα⟩ (zero quantum): ω_S - ω_I
        dq = abs(E_aa - E_bb)  # MHz
        zq = abs(E_ab - E_ba)  # MHz
        qs.coupled_forbidden_freq_MHz = [
            {"transition": "|αα⟩→|ββ⟩ (DQ)", "freq_MHz": round(dq, 4)},
            {"transition": "|αβ⟩→|βα⟩ (ZQ)", "freq_MHz": round(zq, 4)},
        ]

        # --- 遷移双極子モーメント ---
        # 許容EPR遷移: μ = g μ_B √(S(S+1)) — 変化なし
        qs.coupled_tdm_EPR_muB = G_ELECTRON * np.sqrt(0.5 * 1.5)

        # 許容NMR遷移: μ = γ ℏ √(I(I+1))
        mu_P31 = 1.1317  # μ/μ_N
        qs.coupled_tdm_NMR_muN = mu_P31 * np.sqrt(0.5 * 1.5)

        # 禁制遷移: 状態混合による
        # |αβ⟩ と |βα⟩ がHFCで混合 → 混合角 tan(2θ) = A/(ω_S - ω_I)
        # 禁制TDM = sin(θ) × g μ_B √(S(S+1))
        # 地磁気では A (200 MHz) >> ω_S (1.4 MHz) → 強結合極限 (θ→π/4)
        omega_diff = abs(omega_S - omega_I)
        if omega_diff > 0:
            tan_2theta = A_rad / (2 * omega_diff)
            # θ = arctan(tan_2theta)/2, sin(θ) → 1/√2 in strong coupling
            theta = np.arctan(tan_2theta) / 2.0
            qs.coupled_tdm_forbidden_muB = np.sin(theta) * G_ELECTRON * np.sqrt(0.75)

        # --- 交差緩和速度 (Solomon方程式) ---
        # W₀ (zero-quantum, flip-flop): ∝ dipolar × J(ω_S - ω_I)
        # W₂ (double-quantum, flip-flip): ∝ dipolar × J(ω_S + ω_I)
        mu0_4pi = 1e-7
        r_eP = self._get_eP_distance_m(qs)

        gamma_P_rad = 2 * np.pi * gamma_P_Hz  # rad/(s·T)
        # 双極子結合因子 d² = (μ₀/4π)² γ_I² g² μ_B² ℏ² / r⁶
        d2 = (mu0_4pi ** 2) * (gamma_P_rad ** 2) * (G_ELECTRON ** 2) \
            * (MU_BOHR ** 2) * (HBAR_J_S ** 2) / (r_eP ** 6)

        # X-bandでの周波数
        omega_S_xband = 2 * np.pi * 9.4e9
        omega_I_xband = 2 * np.pi * gamma_P_Hz * XBAND_FIELD_T

        def J(omega):
            return tau_c / (1.0 + omega ** 2 * tau_c ** 2)

        # Solomon方程式の速度定数
        # W₀ = (1/20) d² [6J(ω_S - ω_I)]  (IS spin pair, I=S=1/2)
        # W₂ = (3/10) d² [J(ω_S + ω_I)]
        # W₁S = (3/40) d² [3J(ω_S)]
        # W₁I = (1/40) d² [3J(ω_I)]
        W0 = (1.0 / 20.0) * d2 * 6 * J(omega_S_xband - omega_I_xband) / HBAR_J_S ** 2
        W2 = (3.0 / 10.0) * d2 * J(omega_S_xband + omega_I_xband) / HBAR_J_S ** 2
        W1S = (3.0 / 40.0) * d2 * 3 * J(omega_S_xband) / HBAR_J_S ** 2
        W1I = (1.0 / 40.0) * d2 * 3 * J(omega_I_xband) / HBAR_J_S ** 2

        qs.W0_cross_relax_per_s = W0
        qs.W2_cross_relax_per_s = W2

        # 交差緩和時間 T₁ₓ = 1/(W₀ + W₂)  (交差緩和の実効速度)
        W_sum = W0 + W2
        if W_sum > 0:
            qs.T1x_cross_relax_us = 1e6 / W_sum

        # 結合系でのT₂
        # 電子: 1/T₂(e, coupled) = 1/T₂ᵉ + W₁I + W₀ + W₂
        inv_T2e_coupled = 0.0
        if qs.T2e_total_ns > 0:
            inv_T2e_coupled = 1.0 / (qs.T2e_total_ns * 1e-9)
        inv_T2e_coupled += W1I + W0 + W2
        if inv_T2e_coupled > 0:
            qs.coupled_T2_electron_ns = 1e9 / inv_T2e_coupled

        # 核: 1/T₂(I, coupled) = 1/T₂(I, diamag) + PRE + W₁S + W₀ + W₂
        inv_T2I_coupled = 0.0
        if qs.PRE_T2_P31_us > 0:
            inv_T2I_coupled = 1.0 / (qs.PRE_T2_P31_us * 1e-6)
        elif qs.T2_P31_ms > 0:
            inv_T2I_coupled = 1.0 / (qs.T2_P31_ms * 1e-3)
        inv_T2I_coupled += W1S + W0 + W2
        if inv_T2I_coupled > 0:
            qs.coupled_T2_nuclear_us = 1e6 / inv_T2I_coupled

    # ----- 17. コヒーレンス分類 -----

    def classify_coherence(self, qs) -> None:
        """全緩和時間を統合してコヒーレンスボトルネックを特定。"""
        timescales = {}

        # 全時間を秒単位に統一
        if qs.T2_star_spin_ns > 0:
            timescales["T₂*(spin/HFC)"] = qs.T2_star_spin_ns * 1e-9
        if qs.tau_isc_ns > 0:
            timescales["τ_ISC(SOC)"] = qs.tau_isc_ns * 1e-9
        if qs.tau_rad_ns > 0:
            timescales["τ_rad(輻射)"] = qs.tau_rad_ns * 1e-9
        if qs.tau_vib_fastest_ps > 0:
            timescales["τ_vib(振動)"] = qs.tau_vib_fastest_ps * 1e-12
        if qs.T2_el_fs > 0:
            timescales["T₂(電子)"] = qs.T2_el_fs * 1e-15
        if qs.T1e_spin_lattice_us > 0:
            timescales["T₁ᵉ(スピン格子)"] = qs.T1e_spin_lattice_us * 1e-6

        if not timescales:
            return

        # 最短の時間スケールがボトルネック
        bottleneck_name = min(timescales, key=timescales.get)
        bottleneck_time = timescales[bottleneck_name]
        qs.coherence_bottleneck = bottleneck_name

        # 分類
        if bottleneck_time > 1e-6:
            qs.coherence_class = "coherent"
        elif bottleneck_time > 1e-9:
            qs.coherence_class = "partial"
        else:
            qs.coherence_class = "classical"

    # ----- 統合メソッド -----

    def analyze(self, qs) -> None:
        """全デコヒーレンス/緩和解析を順次実行。"""
        methods = [
            self.compute_T2_star_spin,
            self.compute_k_isc,
            self.compute_tau_rad,
            self.compute_tau_vib,
            self.compute_T2_el,
            self.compute_k_et,
            self.compute_T1e,
            self.compute_zeeman,
            self.compute_magnetic_tdm,
            self.compute_T2_dipolar,
            self.compute_T2_total,
            self.compute_nuclear_relaxation,
            self.compute_PRE,
            self.compute_nuclear_coherence,
            self.compute_brain_nuclear_spins,
            self.compute_T2e_full,
            self.compute_coupled_spin_system,
            self.classify_coherence,
        ]
        for method in methods:
            try:
                method(qs)
            except Exception as e:
                log.warning(f"  デコヒーレンス解析 {method.__name__} 失敗: {e}")


# ===========================================================================
# 6. メインパイプライン
# ===========================================================================

class QuantumSiteScreener:
    """量子活性部位スクリーニングパイプライン"""

    def __init__(self, pdb_path: str, output_dir: str = "screening_results",
                 n_cores: int = 1, embedding: str = "none",
                 do_md: bool = False, md_steps: int = 50000,
                 md_snapshots: int = 10, mm_cutoff: float = 12.0,
                 do_decoherence: bool = True):
        self.pdb_path = pdb_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.xtb = XTBRunner(n_cores=n_cores)
        self.extractor = ProteinSiteExtractor(pdb_path)
        self.results: list[QuantumSite] = []

        # QM/MM & MD 設定
        self.embedding = embedding   # "none" | "oniom"
        self.do_md = do_md
        self.md_steps = md_steps
        self.md_snapshots = md_snapshots
        self.mm_cutoff = mm_cutoff

        # デコヒーレンス解析
        self.do_decoherence = do_decoherence
        self._decoherence_analyzer = DecoherenceAnalyzer() if do_decoherence else None

        # QM/MM 用のオブジェクト（必要な場合のみ初期化）
        self.preparer: Optional[ProteinPreparer] = None
        self.md_sampler: Optional[MDSampler] = None

    def _init_qmmm(self):
        """QM/MM 用のタンパク質準備 (MD使用時のみOpenMMが必要)"""
        if self.do_md:
            log.info("\n" + "=" * 60)
            log.info("MD 準備 (OpenMM)")
            log.info("=" * 60)
            self.preparer = ProteinPreparer(self.pdb_path)
            if not self.preparer.parameterize():
                log.error("パラメタライズ失敗 → MDなしにフォールバック")
                self.do_md = False
                self.preparer = None
                return
            self.md_sampler = MDSampler(self.preparer)
        else:
            log.info("\n" + "=" * 60)
            log.info("ONIOM QM/MM 準備 (MD なし)")
            log.info("=" * 60)

    def _compute_single_frame(self, xyz_path: str, xyz_str: str,
                              site: dict, charge: int, uhf: int,
                              workdir: str, do_spin: bool,
                              do_hessian: bool, max_atoms_for_hess: int,
                              extra_args: list[str] = None,
                              pdb_P_distances: list = None) -> dict:
        """1フレームの全xTB計算を実行して結果辞書を返す。"""
        n_atoms = int(xyz_str.split("\n")[0])
        result = {}

        # --- 単一点計算 ---
        try:
            sp = self.xtb.run_single_point(
                xyz_path, charge=charge, uhf=uhf,
                workdir=workdir, extra_args=extra_args)
        except subprocess.TimeoutExpired:
            return {"error": "SP計算タイムアウト"}

        if not sp["success"]:
            return {"error": f"SP計算失敗: {sp.get('error', '')[:200]}"}

        result["homo_lumo_gap_eV"] = sp["homo_lumo_gap_eV"]
        result["dipole_D"] = sp["dipole_D"]
        result["ip_eV"] = sp.get("ip_eV", 0.0)
        result["ea_eV"] = sp.get("ea_eV", 0.0)

        # --- 電子遷移TDM ---
        tdm, f_osc = estimate_electronic_tdm(sp, n_atoms)
        result["electronic_tdm_D"] = tdm
        result["oscillator_strength"] = f_osc

        # --- スピン解析 ---
        if do_spin:
            # S-T gap
            try:
                st = self.xtb.run_singlet_triplet(
                    xyz_path, charge=charge, workdir=workdir,
                    extra_args=extra_args)
                result["singlet_triplet_gap_eV"] = (
                    st["delta_E_ST_eV"] if st.get("success") else 0.0)
            except subprocess.TimeoutExpired:
                result["singlet_triplet_gap_eV"] = 0.0

            # ラジカル
            try:
                rad = self.xtb.run_radical_energies(
                    xyz_path, charge=charge, workdir=workdir,
                    extra_args=extra_args)
                if rad.get("success"):
                    result["radical_ip_eV"] = rad.get("vertical_ip_eV", 0.0)
                    result["radical_ea_eV"] = rad.get("vertical_ea_eV", 0.0)
                    result["somo_eV"] = rad.get("somo_eV", 0.0)

                    sp_for_hfc = {"partial_charges": rad.get("radical_charges", [])}
                    hfcs = estimate_hyperfine_couplings(
                        xyz_str, sp_for_hfc, site["type"],
                        pdb_P_distances=pdb_P_distances)
                    result["hfcs"] = hfcs
                    result["max_hfc_MHz"] = (
                        max(h["A_iso_MHz"] for h in hfcs) if hfcs else 0.0)
            except subprocess.TimeoutExpired:
                pass

        # --- Hessian ---
        if do_hessian and n_atoms <= max_atoms_for_hess:
            try:
                hess = self.xtb.run_hessian(
                    xyz_path, charge=charge, uhf=uhf,
                    workdir=workdir, extra_args=extra_args)
            except subprocess.TimeoutExpired:
                hess = {"success": False, "modes": []}

            if hess["success"] and hess["modes"]:
                real_modes = [m for m in hess["modes"] if m["freq_cm"] > 50]
                if real_modes:
                    best = max(real_modes, key=lambda m: m["vib_tdm_D"])
                    result["max_vib_tdm_D"] = best["vib_tdm_D"]
                    result["max_vib_freq_cm"] = best["freq_cm"]
                    result["vib_modes"] = sorted(
                        real_modes, key=lambda m: -m["vib_tdm_D"])[:5]

        result["success"] = True
        return result

    def _compute_oniom_frame(self, oniom_xyz_path: str, inner_indices: str,
                             xyz_str_inner: str, site: dict,
                             charge: int, uhf: int,
                             workdir: str, do_spin: bool,
                             do_hessian: bool, max_atoms_for_hess: int,
                             vacuum_result: dict = None) -> dict:
        """1フレームのONIOM計算を実行して結果辞書を返す。

        Note: xtb ONIOM は --uhf を inner region に正しく適用しないため、
        スピン解析 (S-T gap, ラジカルエネルギー) は真空計算の結果を流用する。
        ONIOM で取得するのは基底状態の HOMO-LUMO gap と ONIOM total energy。
        """
        n_atoms = int(xyz_str_inner.split("\n")[0])
        result = {}

        # --- ONIOM 単一点計算 ---
        try:
            sp = self.xtb.run_oniom_sp(
                oniom_xyz_path, inner_indices,
                charge=charge, uhf=uhf, workdir=workdir)
        except subprocess.TimeoutExpired:
            return {"error": "ONIOM SP計算タイムアウト"}

        if not sp["success"]:
            return {"error": f"ONIOM SP計算失敗: {sp.get('error', '')[:200]}"}

        result["homo_lumo_gap_eV"] = sp["homo_lumo_gap_eV"]
        result["dipole_D"] = sp.get("dipole_D", 0.0)
        result["oniom_total_Eh"] = sp["oniom_total_Eh"]

        # --- 電子遷移TDM ---
        tdm, f_osc = estimate_electronic_tdm(sp, n_atoms)
        result["electronic_tdm_D"] = tdm
        result["oscillator_strength"] = f_osc

        # --- スピン解析: 真空計算の結果を流用 ---
        # xtb ONIOM は --uhf をフルシステムに適用するため、
        # inner region の S-T gap / ラジカルは真空結果を使用
        if do_spin and vacuum_result:
            result["singlet_triplet_gap_eV"] = vacuum_result.get(
                "singlet_triplet_gap_eV", 0.0)
            result["radical_ip_eV"] = vacuum_result.get("radical_ip_eV", 0.0)
            result["radical_ea_eV"] = vacuum_result.get("radical_ea_eV", 0.0)
            result["somo_eV"] = vacuum_result.get("somo_eV", 0.0)
            result["hfcs"] = vacuum_result.get("hfcs", [])
            result["max_hfc_MHz"] = vacuum_result.get("max_hfc_MHz", 0.0)

        # --- Hessian ---
        if do_hessian and n_atoms <= max_atoms_for_hess:
            try:
                hess = self.xtb.run_oniom_hessian(
                    oniom_xyz_path, inner_indices,
                    charge=charge, uhf=uhf, workdir=workdir)
            except subprocess.TimeoutExpired:
                hess = {"success": False, "modes": []}

            if hess["success"] and hess["modes"]:
                real_modes = [m for m in hess["modes"] if m["freq_cm"] > 50]
                if real_modes:
                    best = max(real_modes, key=lambda m: m["vib_tdm_D"])
                    result["max_vib_tdm_D"] = best["vib_tdm_D"]
                    result["max_vib_freq_cm"] = best["freq_cm"]
                    result["vib_modes"] = sorted(
                        real_modes, key=lambda m: -m["vib_tdm_D"])[:5]

        result["success"] = True
        return result

    @staticmethod
    def _aggregate_frames(frame_results: list[dict]) -> dict:
        """複数フレームの計算結果を集約（平均 ± 標準偏差）。"""
        good = [f for f in frame_results if f.get("success")]
        if not good:
            return {"success": False}

        agg = {"n_frames": len(good), "success": True}

        # 数値フィールドを集約
        numeric_keys = [
            "homo_lumo_gap_eV", "dipole_D", "electronic_tdm_D",
            "oscillator_strength", "singlet_triplet_gap_eV",
            "radical_ip_eV", "radical_ea_eV", "somo_eV", "max_hfc_MHz",
            "max_vib_tdm_D", "max_vib_freq_cm",
        ]
        for key in numeric_keys:
            vals = [f[key] for f in good if key in f and f[key] != 0.0]
            if vals:
                agg[f"{key}_mean"] = float(np.mean(vals))
                agg[f"{key}_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0
            else:
                agg[f"{key}_mean"] = 0.0
                agg[f"{key}_std"] = 0.0

        # 最初のフレームのリストデータを保持
        first = good[0]
        for key in ["hfcs", "vib_modes"]:
            if key in first:
                agg[key] = first[key]

        return agg

    def run(self, do_hessian: bool = True,
            max_atoms_for_hess: int = 150,
            do_spin: bool = True) -> list[QuantumSite]:
        """全パイプラインを実行"""

        # --- QM/MM 初期化 ---
        use_embedding = self.embedding != "none"
        if use_embedding or self.do_md:
            self._init_qmmm()

        # --- MD スナップショット生成 ---
        snapshots = None
        if self.do_md and self.md_sampler:
            interval = max(1, self.md_steps // self.md_snapshots)
            snapshots = self.md_sampler.run(
                n_steps=self.md_steps, snapshot_interval=interval)
        n_frames = len(snapshots) if snapshots else 1

        # --- サイト抽出 ---
        sites = self.extractor.extract_all_sites()
        log.info(f"\n合計 {len(sites)} サイトを処理します"
                 f" (embedding={self.embedding}, frames={n_frames})")

        for idx, site in enumerate(sites):
            site_id = f"{site['type']}_{idx:03d}"
            log.info(f"\n{'='*60}")
            log.info(f"サイト {site_id} ({site['type']})")

            qs = QuantumSite(
                site_id=site_id,
                site_type=site["type"],
                residues=[r.get_resname().strip() for r in site["residues"]],
            )

            # --- XYZ生成 ---
            charge = estimate_charge(site)
            mult = estimate_multiplicity(site)
            uhf = mult - 1
            xyz_str = add_hydrogens_pdb(site["atoms"])
            qs.n_atoms = int(xyz_str.split("\n")[0])
            qs.charge = charge
            qs.multiplicity = mult

            log.info(f"  原子数: {qs.n_atoms} (H付加後), 電荷: {charge}, "
                     f"多重度: {mult}, 残基: {qs.residues[:5]}...")

            # --- P-ラジカル中心距離の自動計測 ---
            try:
                dist_info = self.extractor.measure_P_chromophore_distances(site)
                if dist_info["min_dist_A"] > 0:
                    qs.measured_eP_distance_A = dist_info["min_dist_A"]
                    qs.measured_eP_distances_A = dist_info["P_atoms"]
                    qs.radical_center_type = dist_info["radical_type"]
                    log.info(f"  PDB実測 e⁻-P距離: {dist_info['min_dist_A']:.1f} Å "
                             f"({dist_info['radical_type']}, "
                             f"{len(dist_info['P_atoms'])}個のP原子)")
                    for p in dist_info["P_atoms"]:
                        log.info(f"    {p['residue']}/{p['atom_name']}: {p['distance_A']:.1f} Å")
            except Exception as e:
                log.warning(f"  P距離計測失敗: {e}")

            if qs.n_atoms < 3:
                qs.error_msg = "原子数が少なすぎる"
                self.results.append(qs)
                continue

            # --- 作業ディレクトリ ---
            site_dir = self.output_dir / site_id
            site_dir.mkdir(exist_ok=True)
            xyz_path = site_dir / "fragment.xyz"
            xyz_path.write_text(xyz_str)

            # --- 核スピン & SOC (構造非依存 → 1回だけ) ---
            if do_spin:
                nuc = analyze_nuclear_spins(xyz_str)
                qs.nuclear_spin_inventory = nuc["inventory"]
                qs.total_magnetic_nuclei = nuc["total_magnetic"]
                qs.dominant_spin_nuclei = nuc["dominant"]

                soc_val, soc_class = estimate_soc(xyz_str, site["type"])
                qs.effective_soc_cm = soc_val
                qs.soc_classification = soc_class

            # =================================================
            # フレームループ: 真空 or QM/MM × N スナップショット
            # =================================================
            frame_results = []

            # --- (A) 真空計算（常に1回実行 → baseline） ---
            log.info("  [真空] GFN2-xTB 計算中...")
            pdb_P_dists = getattr(qs, "measured_eP_distances_A", None)
            vacuum_result = self._compute_single_frame(
                str(xyz_path), xyz_str, site, charge, uhf,
                str(site_dir), do_spin, do_hessian, max_atoms_for_hess,
                pdb_P_distances=pdb_P_dists)

            if not vacuum_result.get("success"):
                qs.error_msg = vacuum_result.get("error", "真空計算失敗")
                self.results.append(qs)
                continue

            # 真空の結果を保存
            qs.vacuum_homo_lumo_gap_eV = vacuum_result["homo_lumo_gap_eV"]
            qs.vacuum_ground_dipole_D = vacuum_result["dipole_D"]

            log.info(f"  [真空] Gap={vacuum_result['homo_lumo_gap_eV']:.2f} eV, "
                     f"TDM={vacuum_result['electronic_tdm_D']:.3f} D")

            # --- (B) ONIOM QM/MM (gfn2:gfnff) ---
            if use_embedding:
                partitioner = QMMMPartitioner(
                    site["atoms"], self.extractor._all_atoms,
                    self.mm_cutoff)
                partition = partitioner.partition()

                # ONIOM XYZファイル生成
                oniom_xyz_path, inner_indices = partitioner.write_oniom_xyz(
                    partition, str(site_dir), xyz_str_inner=xyz_str)

                log.info(f"  [ONIOM] {partition['n_inner']} inner + "
                         f"{partition['n_outer']} outer atoms, 計算中...")

                fr = self._compute_oniom_frame(
                    oniom_xyz_path, inner_indices,
                    xyz_str, site, charge, uhf,
                    str(site_dir), do_spin, do_hessian, max_atoms_for_hess,
                    vacuum_result=vacuum_result)

                if fr.get("success"):
                    log.info(f"  [ONIOM] Gap={fr['homo_lumo_gap_eV']:.2f} eV, "
                             f"E_ONIOM={fr.get('oniom_total_Eh', 0.0):.6f} Eh")
                    frame_results.append(fr)

                # 集約（1フレームでも _aggregate_frames を使う）
                agg = self._aggregate_frames(frame_results)
                qs.embedding_method = "oniom"
                qs.n_mm_atoms = partition["n_outer"]
                qs.mm_cutoff_A = self.mm_cutoff
                qs.n_frames = agg.get("n_frames", 1)
                qs.oniom_total_energy_Eh = fr.get("oniom_total_Eh", 0.0)

                # ONIOM の平均値を主要フィールドに格納
                qs.homo_lumo_gap_eV = agg.get("homo_lumo_gap_eV_mean", 0.0)
                qs.ground_dipole_D = agg.get("dipole_D_mean", 0.0)
                qs.estimated_electronic_tdm_D = agg.get("electronic_tdm_D_mean", 0.0)
                qs.oscillator_strength = agg.get("oscillator_strength_mean", 0.0)
                qs.singlet_triplet_gap_eV = agg.get("singlet_triplet_gap_eV_mean", 0.0)
                qs.radical_cation_ip_eV = agg.get("radical_ip_eV_mean", 0.0)
                qs.radical_anion_ea_eV = agg.get("radical_ea_eV_mean", 0.0)
                qs.somo_energy_eV = agg.get("somo_eV_mean", 0.0)
                qs.max_hfc_MHz = agg.get("max_hfc_MHz_mean", 0.0)
                qs.max_vib_tdm_D = agg.get("max_vib_tdm_D_mean", 0.0)
                qs.max_vib_freq_cm = agg.get("max_vib_freq_cm_mean", 0.0)
                qs.estimated_hfc_MHz = agg.get("hfcs", [])
                qs.vib_modes = agg.get("vib_modes", [])

                # 標準偏差
                qs.homo_lumo_gap_std = agg.get("homo_lumo_gap_eV_std", 0.0)
                qs.estimated_electronic_tdm_std = agg.get("electronic_tdm_D_std", 0.0)
                qs.singlet_triplet_gap_std = agg.get("singlet_triplet_gap_eV_std", 0.0)
                qs.max_hfc_std = agg.get("max_hfc_MHz_std", 0.0)

                # 真空 vs ONIOM シフト
                qs.embedding_shift_gap_eV = (
                    qs.homo_lumo_gap_eV - qs.vacuum_homo_lumo_gap_eV)
                qs.embedding_shift_dipole_D = (
                    qs.ground_dipole_D - qs.vacuum_ground_dipole_D)

            else:
                # --- 真空のみ → 真空結果を直接格納 ---
                qs.embedding_method = "vacuum"
                qs.homo_lumo_gap_eV = vacuum_result["homo_lumo_gap_eV"]
                qs.ground_dipole_D = vacuum_result["dipole_D"]
                qs.ip_eV = vacuum_result.get("ip_eV", 0.0)
                qs.ea_eV = vacuum_result.get("ea_eV", 0.0)
                qs.estimated_electronic_tdm_D = vacuum_result["electronic_tdm_D"]
                qs.oscillator_strength = vacuum_result["oscillator_strength"]
                qs.singlet_triplet_gap_eV = vacuum_result.get(
                    "singlet_triplet_gap_eV", 0.0)
                qs.radical_cation_ip_eV = vacuum_result.get("radical_ip_eV", 0.0)
                qs.radical_anion_ea_eV = vacuum_result.get("radical_ea_eV", 0.0)
                qs.somo_energy_eV = vacuum_result.get("somo_eV", 0.0)
                qs.estimated_hfc_MHz = vacuum_result.get("hfcs", [])
                qs.max_hfc_MHz = vacuum_result.get("max_hfc_MHz", 0.0)
                qs.max_vib_tdm_D = vacuum_result.get("max_vib_tdm_D", 0.0)
                qs.max_vib_freq_cm = vacuum_result.get("max_vib_freq_cm", 0.0)
                qs.vib_modes = vacuum_result.get("vib_modes", [])

            qs.calc_success = True

            # --- デコヒーレンス/緩和解析 ---
            if self._decoherence_analyzer:
                try:
                    log.info("  [Decoherence] 緩和時間推定中...")
                    self._decoherence_analyzer.analyze(qs)
                    if qs.coherence_class:
                        log.info(f"  [Decoherence] 分類: {qs.coherence_class}, "
                                 f"ボトルネック: {qs.coherence_bottleneck}")
                except Exception as e:
                    log.warning(f"  デコヒーレンス解析失敗: {e}")

            self.results.append(qs)

        self._save_results()
        self._print_summary()
        return self.results

    def _save_results(self):
        """結果をJSONで保存"""
        out = [asdict(r) for r in self.results]
        json_path = self.output_dir / "screening_results.json"
        json_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        log.info(f"\n結果を {json_path} に保存しました")

    def _print_summary(self):
        """スクリーニング結果のサマリーを表示"""
        print("\n" + "=" * 72)
        print("  QUANTUM-ACTIVE SITE SCREENING SUMMARY")
        print("=" * 72)

        success = [r for r in self.results if r.calc_success]
        failed = [r for r in self.results if not r.calc_success]

        print(f"\n  計算成功: {len(success)} / {len(self.results)} サイト")
        if failed:
            print(f"  失敗: {', '.join(r.site_id for r in failed)}")

        if not success:
            print("  有効な結果がありません")
            return

        # --- 電子遷移TDMランキング ---
        print(f"\n{'─'*72}")
        print("  ■ 電子遷移TDM ランキング (簡易CIS推定)")
        print(f"{'─'*72}")
        print(f"  {'Rank':<5} {'Site ID':<25} {'Type':<15} "
              f"{'TDM(D)':<10} {'f_osc':<10} {'Gap(eV)':<10}")
        print(f"  {'─'*70}")
        e_ranked = sorted(success, key=lambda r: -r.estimated_electronic_tdm_D)
        for i, r in enumerate(e_ranked[:20]):
            flag = " ★" if r.estimated_electronic_tdm_D > ELECTRONIC_TDM_THRESHOLD else ""
            print(f"  {i+1:<5} {r.site_id:<25} {r.site_type:<15} "
                  f"{r.estimated_electronic_tdm_D:<10.3f} "
                  f"{r.oscillator_strength:<10.4f} "
                  f"{r.homo_lumo_gap_eV:<10.2f}{flag}")

        # --- 振動遷移TDMランキング ---
        vib_success = [r for r in success if r.max_vib_tdm_D > 0]
        if vib_success:
            print(f"\n{'─'*72}")
            print("  ■ 振動遷移TDM ランキング (GFN2-xTB Hessian)")
            print(f"{'─'*72}")
            print(f"  {'Rank':<5} {'Site ID':<25} {'Type':<15} "
                  f"{'TDM(D)':<10} {'Freq(cm⁻¹)':<12}")
            print(f"  {'─'*70}")
            v_ranked = sorted(vib_success, key=lambda r: -r.max_vib_tdm_D)
            for i, r in enumerate(v_ranked[:20]):
                flag = " ★" if r.max_vib_tdm_D > VIBRATIONAL_TDM_THRESHOLD else ""
                print(f"  {i+1:<5} {r.site_id:<25} {r.site_type:<15} "
                      f"{r.max_vib_tdm_D:<10.4f} "
                      f"{r.max_vib_freq_cm:<12.1f}{flag}")

        # --- 核スピン・電子スピン ---
        spin_success = [r for r in success if r.total_magnetic_nuclei > 0]
        if spin_success:
            print(f"\n{'─'*72}")
            print("  ■ 核スピン インベントリ")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<25} {'Type':<15} "
                  f"{'Dom.Nucleus':<15} {'Mag.Atoms':<10} "
                  f"{'SOC(cm⁻¹)':<12} {'SOC class':<10}")
            print(f"  {'─'*70}")
            for r in spin_success:
                print(f"  {r.site_id:<25} {r.site_type:<15} "
                      f"{r.dominant_spin_nuclei:<15} "
                      f"{r.total_magnetic_nuclei:<10} "
                      f"{r.effective_soc_cm:<12.1f} "
                      f"{r.soc_classification:<10}")

        st_success = [r for r in success if r.singlet_triplet_gap_eV != 0]
        if st_success:
            print(f"\n{'─'*72}")
            print("  ■ 電子スピン解析 (S-T gap, ラジカル安定性, HFC)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<25} {'ΔE(S-T)/eV':<12} "
                  f"{'IP/eV':<10} {'EA/eV':<10} "
                  f"{'SOMO/eV':<10} {'maxHFC/MHz':<12}")
            print(f"  {'─'*70}")
            for r in sorted(st_success,
                             key=lambda x: abs(x.singlet_triplet_gap_eV)):
                print(f"  {r.site_id:<25} "
                      f"{r.singlet_triplet_gap_eV:<12.4f} "
                      f"{r.radical_cation_ip_eV:<10.2f} "
                      f"{r.radical_anion_ea_eV:<10.2f} "
                      f"{r.somo_energy_eV:<10.2f} "
                      f"{r.max_hfc_MHz:<12.1f}")

            # ラジカルペア機構の候補評価
            print(f"\n  ■ ラジカルペア機構 (RPM) 候補評価")
            print(f"  {'─'*70}")
            for r in st_success:
                rpm_score = 0
                reasons = []
                # 小さなS-Tギャップ → RPM有利
                if abs(r.singlet_triplet_gap_eV) < 0.1:
                    rpm_score += 3
                    reasons.append(f"小S-Tgap({r.singlet_triplet_gap_eV:.4f}eV)")
                elif abs(r.singlet_triplet_gap_eV) < 0.5:
                    rpm_score += 1
                # 弱いSOC → スピン選択則維持
                if r.soc_classification == "weak":
                    rpm_score += 2
                    reasons.append("弱SOC(スピン保存)")
                elif r.soc_classification == "moderate":
                    rpm_score += 1
                # 大きなHFC → 核スピン-電子スピン結合
                if r.max_hfc_MHz > 10:
                    rpm_score += 2
                    reasons.append(f"大HFC({r.max_hfc_MHz:.0f}MHz)")
                # 低いIP → 電子移動ラジカル生成容易
                if r.radical_cation_ip_eV < 8:
                    rpm_score += 1
                    reasons.append(f"低IP({r.radical_cation_ip_eV:.1f}eV)")

                stars = "★" * min(rpm_score, 5)
                reason_str = ", ".join(reasons) if reasons else "—"
                print(f"  {r.site_id:<25} RPMスコア: {rpm_score}/8 {stars}")
                print(f"    根拠: {reason_str}")

        # --- QM/MM 環境効果 ---
        emb_sites = [r for r in success if r.embedding_method != "vacuum"]
        if emb_sites:
            print(f"\n{'─'*72}")
            print("  ■ タンパク質環境効果 (ONIOM gfn2:gfnff vs 真空)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<25} {'Gap真空':<10} {'GapONIOM':<10} "
                  f"{'ΔGap':<10} {'Outer':<10} {'Frames':<8}")
            print(f"  {'─'*70}")
            for r in emb_sites:
                delta = r.embedding_shift_gap_eV
                sign = "+" if delta >= 0 else ""
                std_str = f"±{r.homo_lumo_gap_std:.2f}" if r.n_frames > 1 else ""
                print(f"  {r.site_id:<25} "
                      f"{r.vacuum_homo_lumo_gap_eV:<10.2f} "
                      f"{r.homo_lumo_gap_eV:<10.2f} "
                      f"{sign}{delta:<9.2f} "
                      f"{r.n_mm_atoms:<10} "
                      f"{r.n_frames:<8}")
                if std_str:
                    print(f"    (Gap: {r.homo_lumo_gap_eV:.2f}{std_str}, "
                          f"TDM: {r.estimated_electronic_tdm_D:.2f}"
                          f"±{r.estimated_electronic_tdm_std:.2f}, "
                          f"ΔE_ST: {r.singlet_triplet_gap_eV:.3f}"
                          f"±{r.singlet_triplet_gap_std:.3f})")

        # --- デコヒーレンス/緩和時間 ---
        decoh_sites = [r for r in success if r.coherence_class]
        if decoh_sites:
            print(f"\n{'─'*72}")
            print("  ■ デコヒーレンス/緩和時間スケール")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'T₂*(spin)':<12} {'τ_ISC':<12} "
                  f"{'τ_rad':<12} {'τ_vib':<12} {'T₁ᵉ':<12} {'分類':<12}")
            print(f"  {'─'*70}")
            for r in decoh_sites:
                def _fmt_time(val, unit):
                    if val <= 0:
                        return "—"
                    # 自動スケーリング
                    if unit == "ns" and val >= 1e6:
                        return f"{val/1e6:.1f}ms"
                    elif unit == "ns" and val >= 1e3:
                        return f"{val/1e3:.1f}μs"
                    elif unit == "ps" and val >= 1e6:
                        return f"{val/1e6:.1f}μs"
                    elif unit == "ps" and val >= 1e3:
                        return f"{val/1e3:.1f}ns"
                    elif unit == "μs" and val >= 1e3:
                        return f"{val/1e3:.1f}ms"
                    elif val >= 100:
                        return f"{val:.0f}{unit}"
                    elif val >= 1:
                        return f"{val:.1f}{unit}"
                    else:
                        return f"{val:.2f}{unit}"
                t2s = _fmt_time(r.T2_star_spin_ns, "ns")
                tisc = _fmt_time(r.tau_isc_ns, "ns")
                trad = _fmt_time(r.tau_rad_ns, "ns")
                tvib = _fmt_time(r.tau_vib_fastest_ps, "ps")
                t1e = _fmt_time(r.T1e_spin_lattice_us, "μs")
                print(f"  {r.site_id:<22} {t2s:<12} {tisc:<12} "
                      f"{trad:<12} {tvib:<12} {t1e:<12} "
                      f"{r.coherence_class:<12}")

            print(f"\n  ■ コヒーレンスボトルネック解析")
            print(f"  {'─'*70}")
            for r in decoh_sites:
                if r.coherence_class == "coherent":
                    marker = "★"
                elif r.coherence_class == "partial":
                    marker = "◆"
                else:
                    marker = "○"
                lam_str = (f", λ_reorg={r.lambda_reorg_eV:.3f}eV"
                           if r.lambda_reorg_eV > 0 else "")
                print(f"  {marker} {r.site_id:<22} "
                      f"ボトルネック: {r.coherence_bottleneck}{lam_str}")

        # --- 横緩和 & Zeeman & 磁気TDM ---
        zeeman_sites = [r for r in success if r.zeeman_electron_earth_MHz > 0]
        if zeeman_sites:
            print(f"\n{'─'*72}")
            print("  ■ 横緩和 T₂ (transverse relaxation)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'T₂*(HFC)':<12} {'T₂(dipolar)':<12} "
                  f"{'T₂(total)':<12} {'ΔE_ST(eV)':<12}")
            print(f"  {'─'*70}")
            for r in zeeman_sites:
                def _fmt(val, unit):
                    if val <= 0: return "—"
                    if unit == "ns" and val >= 1e6: return f"{val/1e6:.1f}ms"
                    if unit == "ns" and val >= 1e3: return f"{val/1e3:.1f}μs"
                    if val >= 100: return f"{val:.0f}{unit}"
                    if val >= 1: return f"{val:.1f}{unit}"
                    return f"{val:.2f}{unit}"
                t2s = _fmt(r.T2_star_spin_ns, "ns")
                t2d = _fmt(r.T2_dipolar_ns, "ns")
                t2t = _fmt(r.T2_total_ns, "ns")
                print(f"  {r.site_id:<22} {t2s:<12} {t2d:<12} "
                      f"{t2t:<12} {r.singlet_triplet_gap_eV:<12.3f}")

            print(f"\n{'─'*72}")
            print("  ■ Zeeman分裂 (スピン遷移準位差)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'e⁻@地磁気':<14} {'e⁻@X-band':<14} "
                  f"{'³¹P@地磁気':<14} {'³¹P@X-band':<14} {'P原子数':<8}")
            print(f"  {'─'*70}")
            for r in zeeman_sites:
                print(f"  {r.site_id:<22} "
                      f"{r.zeeman_electron_earth_MHz:.4f}MHz  "
                      f"{r.zeeman_electron_xband_GHz:.3f}GHz  "
                      f"{r.zeeman_P31_earth_Hz:.1f}Hz    "
                      f"{r.zeeman_P31_xband_kHz:.1f}kHz   "
                      f"{r.n_P_atoms}")

            print(f"\n{'─'*72}")
            print("  ■ 遷移双極子モーメント (電子・磁気・スピン)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'電子TDM(D)':<14} {'f_osc':<10} "
                  f"{'磁気TDM(μB)':<14} {'³¹P磁気(μN)':<14} "
                  f"{'S→T電気(D)':<14}")
            print(f"  {'─'*70}")
            for r in zeeman_sites:
                st_tdm = f"{r.electric_tdm_spin_D:.4f}" if r.electric_tdm_spin_D > 0 else "—"
                print(f"  {r.site_id:<22} "
                      f"{r.estimated_electronic_tdm_D:<14.3f} "
                      f"{r.oscillator_strength:<10.4f} "
                      f"{r.magnetic_tdm_electron_muB:<14.4f} "
                      f"{r.magnetic_tdm_P31_muN:<14.4f} "
                      f"{st_tdm:<14}")

        # --- 核スピン緩和 ---
        nuc_sites = [r for r in success if r.T1_H1_s > 0]
        if nuc_sites:
            print(f"\n{'─'*72}")
            print("  ■ 核スピン緩和 (BPP理論, B₀=11.7T)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'¹H T₁(s)':<10} {'¹H T₂(ms)':<12} "
                  f"{'³¹P T₁(s)':<12} {'³¹P T₂(ms)':<12} "
                  f"{'¹⁴N T₁(ms)':<12} {'P原子数':<8}")
            print(f"  {'─'*70}")
            for r in nuc_sites:
                p_t1 = f"{r.T1_P31_s:.2f}" if r.T1_P31_s > 0 else "—"
                p_t2 = f"{r.T2_P31_ms:.1f}" if r.T2_P31_ms > 0 else "—"
                if r.T1_N14_ms > 0:
                    n_t1 = f"{r.T1_N14_ms:.3f}" if r.T1_N14_ms < 1 else f"{r.T1_N14_ms:.1f}"
                else:
                    n_t1 = "—"
                print(f"  {r.site_id:<22} {r.T1_H1_s:<10.2f} {r.T2_H1_ms:<12.1f} "
                      f"{p_t1:<12} {p_t2:<12} {n_t1:<12} {r.n_P_atoms:<8}")

            # PRE
            pre_sites = [r for r in success if r.PRE_T1_P31_ms > 0]
            if pre_sites:
                print(f"\n  ■ 常磁性緩和促進 PRE (ラジカル存在時, ³¹P)")
                print(f"  {'─'*70}")
                print(f"  {'Site ID':<22} {'PRE T₁(ms)':<14} {'PRE T₂(μs)':<14} "
                      f"{'核コヒーレンス':<16} {'e⁻-P距離(Å)':<12}")
                print(f"  {'─'*70}")
                for r in pre_sites:
                    coh = f"{r.nuclear_coherence_time_ms:.2f}ms" if r.nuclear_coherence_time_ms > 0 else "—"
                    r_est = "3.5" if "PLP" in str(r.residues) or "LLP" in str(r.residues) else "7.0"
                    print(f"  {r.site_id:<22} {r.PRE_T1_P31_ms:<14.2f} "
                          f"{r.PRE_T2_P31_us:<14.2f} {coh:<16} {r_est:<12}")

        # --- 脳内環境での核スピン ---
        brain_sites = [r for r in success if r.brain_P31_T2_s > 0 or r.brain_H1_T2_s > 0]
        if brain_sites:
            print(f"\n{'─'*72}")
            print("  ■ 脳内環境での核スピン緩和 (B₀=50μT, T=310K)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'regime':<18} {'³¹P T₁=T₂':<12} "
                  f"{'PRE T₂(³¹P)':<12} {'¹H T₁=T₂':<10} "
                  f"{'O₂+Fe T₁':<10} {'³¹P T₂eff':<12} {'¹H T₂eff':<10}")
            print(f"  {'─'*100}")
            for r in brain_sites:
                regime = r.brain_nuclear_coherence_regime or "—"
                p_t1 = f"{r.brain_P31_T1_s:.1f}s" if r.brain_P31_T1_s > 0 else "—"
                if r.brain_PRE_P31_T2_ms > 0:
                    if r.brain_PRE_P31_T2_ms >= 1:
                        pre = f"{r.brain_PRE_P31_T2_ms:.1f}ms"
                    elif r.brain_PRE_P31_T2_ms >= 0.001:
                        pre = f"{r.brain_PRE_P31_T2_ms*1e3:.1f}μs"
                    else:
                        pre = f"{r.brain_PRE_P31_T2_ms*1e6:.1f}ns"
                else:
                    pre = "—"
                h_t1 = f"{r.brain_H1_T1_s:.3f}s" if r.brain_H1_T1_s > 0 else "—"
                para = f"{r.brain_paramag_T1_s:.1f}s" if r.brain_paramag_T1_s > 0 else "—"
                if r.brain_P31_T2_effective_ms > 0:
                    if r.brain_P31_T2_effective_ms >= 1:
                        p_eff = f"{r.brain_P31_T2_effective_ms:.1f}ms"
                    elif r.brain_P31_T2_effective_ms >= 0.001:
                        p_eff = f"{r.brain_P31_T2_effective_ms*1e3:.1f}μs"
                    else:
                        p_eff = f"{r.brain_P31_T2_effective_ms*1e6:.1f}ns"
                else:
                    p_eff = "—"
                h_eff = f"{r.brain_H1_T2_effective_ms:.2f}ms" if r.brain_H1_T2_effective_ms > 0 else "—"
                print(f"  {r.site_id:<22} {regime:<18} {p_t1:<12} "
                      f"{pre:<12} {h_t1:<10} {para:<10} {p_eff:<12} {h_eff:<10}")

        # --- 脳内T₂ᵉ・結合系 ---
        brain_t2e = [r for r in success if r.brain_T2e_total_ns > 0]
        if brain_t2e:
            print(f"\n{'─'*72}")
            print("  ■ 脳内 T₂ᵉ・結合系 @地磁気 (50μT, 310K)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'T₂ᵉ@Xband':<11} {'T₂ᵉ@50μT':<11} "
                  f"{'T₂(g)@50μT':<12} {'W₀@50μT':<12} {'W₀@Xband':<12} "
                  f"{'T₂(e,cpl)':<10} {'T₂(P,cpl)':<10}")
            print(f"  {'─'*100}")
            for r in brain_t2e:
                def _ns(v):
                    if v <= 0: return "—"
                    if v >= 1e6: return f"{v/1e6:.1f}ms"
                    if v >= 1e3: return f"{v/1e3:.1f}μs"
                    return f"{v:.2f}ns"
                def _us(v):
                    if v <= 0: return "—"
                    if v >= 1e3: return f"{v/1e3:.1f}ms"
                    return f"{v:.1f}μs"
                xband = _ns(r.T2e_total_ns)
                earth = _ns(r.brain_T2e_total_ns)
                gani = _ns(r.brain_T2e_g_aniso_ns) if r.brain_T2e_g_aniso_ns > 0 else "∞"
                w0e = f"{r.brain_W0_per_s:.2e}" if r.brain_W0_per_s > 0 else "—"
                w0x = f"{r.W0_cross_relax_per_s:.2e}" if r.W0_cross_relax_per_s > 0 else "—"
                cpl_e = _ns(r.brain_coupled_T2e_ns)
                cpl_n = _us(r.brain_coupled_T2n_us)
                print(f"  {r.site_id:<22} {xband:<11} {earth:<11} {gani:<12} "
                      f"{w0e:<12} {w0x:<12} {cpl_e:<10} {cpl_n:<10}")

        # --- 電子T₂ᵉ完全版 ---
        t2e_sites = [r for r in success if r.T2e_total_ns > 0]
        if t2e_sites:
            print(f"\n{'─'*72}")
            print("  ■ 電子T₂ᵉ完全版 (全寄与)")
            print(f"{'─'*72}")
            print(f"  {'Site ID':<22} {'T₂*(HFC)':<10} {'T₂(dd)':<10} "
                  f"{'T₂(g-ani)':<10} {'T₂(A-mod)':<10} {'T₂(SOC)':<10} "
                  f"{'T₂ᵉ(total)':<12}")
            print(f"  {'─'*70}")
            for r in t2e_sites:
                def _f(v, u):
                    if v <= 0: return "—"
                    if u == "ns" and v >= 1e6: return f"{v/1e6:.1f}ms"
                    if u == "ns" and v >= 1e3: return f"{v/1e3:.1f}μs"
                    if v >= 100: return f"{v:.0f}{u}"
                    if v >= 1: return f"{v:.1f}{u}"
                    return f"{v:.2f}{u}"
                print(f"  {r.site_id:<22} {_f(r.T2_star_spin_ns,'ns'):<10} "
                      f"{_f(r.T2_dipolar_ns,'ns'):<10} "
                      f"{_f(r.T2e_g_aniso_ns,'ns'):<10} "
                      f"{_f(r.T2e_hfc_mod_ns,'ns'):<10} "
                      f"{_f(r.T2e_spin_orbit_ns,'ns'):<10} "
                      f"{_f(r.T2e_total_ns,'ns'):<12}")

        # --- 結合スピン系 ---
        coupled_sites = [r for r in success if r.coupled_level_energies_MHz]
        if coupled_sites:
            print(f"\n{'─'*72}")
            print("  ■ 結合スピン系 e⁻-³¹P (地磁気50μT)")
            print(f"{'─'*72}")
            for r in coupled_sites:
                print(f"  {r.site_id}:")
                # 準位
                for lev in r.coupled_level_energies_MHz:
                    print(f"    {lev['state']}: {lev['E_MHz']:.4f} MHz")
                # 遷移
                print(f"    EPR遷移:")
                for t in r.coupled_EPR_freq_MHz:
                    print(f"      {t['transition']}: {t['freq_MHz']:.4f} MHz")
                print(f"    NMR遷移:")
                for t in r.coupled_NMR_freq_MHz:
                    print(f"      {t['transition']}: {t['freq_MHz']:.4f} MHz")
                print(f"    禁制遷移:")
                for t in r.coupled_forbidden_freq_MHz:
                    print(f"      {t['transition']}: {t['freq_MHz']:.4f} MHz")
                # TDM
                print(f"    EPR TDM: {r.coupled_tdm_EPR_muB:.4f} μ_B")
                print(f"    NMR TDM: {r.coupled_tdm_NMR_muN:.4f} μ_N")
                fb = f"{r.coupled_tdm_forbidden_muB:.6f}" if r.coupled_tdm_forbidden_muB > 0 else "—"
                print(f"    禁制TDM: {fb} μ_B")
                # 交差緩和
                print(f"    W₀(ZQ): {r.W0_cross_relax_per_s:.3e} s⁻¹")
                print(f"    W₂(DQ): {r.W2_cross_relax_per_s:.3e} s⁻¹")
                if r.T1x_cross_relax_us > 0:
                    if r.T1x_cross_relax_us >= 1e6:
                        print(f"    T₁ₓ(交差): {r.T1x_cross_relax_us/1e6:.2f} s")
                    elif r.T1x_cross_relax_us >= 1e3:
                        print(f"    T₁ₓ(交差): {r.T1x_cross_relax_us/1e3:.2f} ms")
                    else:
                        print(f"    T₁ₓ(交差): {r.T1x_cross_relax_us:.2f} μs")
                # 結合系T₂
                if r.coupled_T2_electron_ns > 0:
                    print(f"    結合系 T₂(e⁻): {r.coupled_T2_electron_ns:.2f} ns")
                if r.coupled_T2_nuclear_us > 0:
                    print(f"    結合系 T₂(³¹P): {r.coupled_T2_nuclear_us:.2f} μs")

        # --- 候補サイト ---
        candidates = [r for r in success
                      if r.estimated_electronic_tdm_D > ELECTRONIC_TDM_THRESHOLD
                      or r.max_vib_tdm_D > VIBRATIONAL_TDM_THRESHOLD]
        print(f"\n{'─'*72}")
        print(f"  ■ 閾値を超えたサイト: {len(candidates)} 個")
        print(f"    電子TDM > {ELECTRONIC_TDM_THRESHOLD} D: "
              f"{sum(1 for r in success if r.estimated_electronic_tdm_D > ELECTRONIC_TDM_THRESHOLD)}")
        print(f"    振動TDM > {VIBRATIONAL_TDM_THRESHOLD} D: "
              f"{sum(1 for r in success if r.max_vib_tdm_D > VIBRATIONAL_TDM_THRESHOLD)}")
        print("=" * 72)


# ===========================================================================
# 6. CLI
# ===========================================================================

def download_pdb(pdb_id: str, output_dir: str = ".") -> str:
    """RCSB PDBからPDBファイルをダウンロード"""
    from Bio.PDB import PDBList
    pdbl = PDBList()
    path = pdbl.retrieve_pdb_file(pdb_id, pdir=output_dir, file_format="pdb")
    return path


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Brain protein quantum-active site screener (GFN2-xTB)",
    )
    parser.add_argument("pdb", help="PDBファイルパス or PDB ID (e.g., 4I6G)")
    parser.add_argument("-o", "--output", default="screening_results",
                        help="出力ディレクトリ")
    parser.add_argument("-j", "--cores", type=int, default=1,
                        help="並列コア数")
    parser.add_argument("--no-hess", action="store_true",
                        help="Hessian計算をスキップ")
    parser.add_argument("--no-spin", action="store_true",
                        help="核スピン・電子スピン解析をスキップ")
    parser.add_argument("--max-hess-atoms", type=int, default=150,
                        help="Hessian計算の最大原子数")
    # QM/MM & MD オプション
    parser.add_argument("--embedding", choices=["none", "oniom"],
                        default="none",
                        help="ONIOM QM/MM (gfn2:gfnff) 環境効果 (default: none)")
    parser.add_argument("--md", action="store_true",
                        help="OpenMM による MD 構造サンプリングを有効化")
    parser.add_argument("--md-steps", type=int, default=50000,
                        help="MD 総ステップ数 (default: 50000 = 100ps)")
    parser.add_argument("--md-snapshots", type=int, default=10,
                        help="MD スナップショット数 (default: 10)")
    parser.add_argument("--mm-cutoff", type=float, default=12.0,
                        help="MM 領域のカットオフ半径 Å (default: 12.0)")
    parser.add_argument("--no-decoherence", action="store_true",
                        help="デコヒーレンス/緩和時間解析をスキップ")
    args = parser.parse_args()

    # PDB IDの場合はダウンロード
    pdb_path = args.pdb
    if len(pdb_path) == 4 and pdb_path.isalnum():
        log.info(f"PDB ID {pdb_path} をダウンロード中...")
        pdb_path = download_pdb(pdb_path, args.output)
        log.info(f"ダウンロード完了: {pdb_path}")

    if not os.path.exists(pdb_path):
        log.error(f"ファイルが見つかりません: {pdb_path}")
        sys.exit(1)

    screener = QuantumSiteScreener(
        pdb_path=pdb_path,
        output_dir=args.output,
        n_cores=args.cores,
        embedding=args.embedding,
        do_md=args.md,
        md_steps=args.md_steps,
        md_snapshots=args.md_snapshots,
        mm_cutoff=args.mm_cutoff,
        do_decoherence=not args.no_decoherence,
    )
    screener.run(
        do_hessian=not args.no_hess,
        max_atoms_for_hess=args.max_hess_atoms,
        do_spin=not args.no_spin,
    )


if __name__ == "__main__":
    main()
