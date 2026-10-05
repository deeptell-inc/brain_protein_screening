# 改稿版 構成案（magnetobio v4）

投稿先の既定：J. Phys. Chem. B（未確定。テンプレートの差し替えは最後）

## 仮題

**A thermally formed flavin radical pair that forms fast enough to matter cannot sense the geomagnetic field**

（副案）Catalytic competence limits the lifetime of flavoenzyme radical pairs and forbids weak-field magnetosensitivity

## 中心の主張（一文）

暗所で働くフラビン酵素のラジカル対は、触媒回転に見合う速さで熱的に生成するなら、詳細釣り合いにより 1 ps 以内に再結合する。その交換結合と寿命は同じ電子結合 $t$ で結ばれていて独立に選べない。さらに、接触距離の双極子結合だけで弱磁場応答は $\le 2\times10^{-6}\,\%$ に抑えられる。光駆動のラジカル対（photolyase、クリプトクロム）がこの制約を免れるのは、光子が駆動力 Δ を支払うからである。

## 査読指摘との対応

| 指摘 | 対応する節 |
|---|---|
| R1：酵素特異的か一般論かを明確に | 導入：一般論（J–τ の結びつき、暗所と光駆動の対比）＋ MAO/DAO での較正 |
| R1：パラメータを較正し、根拠を示す | §2（DFT の $t$、λ、超微細結合）、§3（photolyase の実測値） |
| R1：ラジカル対ができないなら計算は無意味では | §4：詳細釣り合いで「生成するなら短寿命、長寿命なら生成しない」の二択に閉じる |
| R2-1：J の見積もり | §2.2：超交換 $|J|=2t^2/\Delta$。Efimova の $J_0$ は 3.5 Å に外挿しない理由も書く |
| R2-2：寿命 1–10 ps は下限では？ 100 ns は？ | §3・§4：実測 70–120 ps（光駆動）。熱的生成では ≲1 ps。100 ns は触媒関与と両立しない |
| R2-3：超微細結合の詳細 | 表 1（核、I、$a_{\rm iso}$、出典。フラビンは Lee 2014、パートナーは DFT） |
| R2-4：T₂ᵉ = 1 ns の根拠 | 廃止。Kattnig 2016 に基づく µs オーダー。結論は T₂ᵉ に依存しない（コヒーレント極限でも null） |
| R2-5：J と D の規約 | §1：$J_{\rm gap}=E_T-E_S=-2J_{\rm EH}$、$D$ は T±–T₀ 分裂幅（Efimova の定義）。本文の式とコードの両方で明記 |

## 節の構成

1. **Introduction**：弱磁場効果の一般的な上限は既知（Binhi 2025、Efimova 2008）。未解決なのは「酵素の化学がラジカル対の J と τ を独立に許すか」。先行性（Weiss 2003：2J と V、Efimova 2008：共通の β）を明示したうえで、本論文の寄与を述べる。
2. **Model and conventions**：スピンハミルトニアン（J・D の規約を明示）、Liouville マスター方程式、超微細結合の表。
3. **Calibration**：(a) DFT のフラグメント結合 $t(r)$、(b) 超交換 $J$、(c) 4点法の λ ＋ 外圏、(d) photolyase の実測値による裏付け。
4. **One coupling sets both J and τ**：$(J/\hbar)\tau = 1/(\pi\Delta\,{\rm FC})$。断熱極限との補間。
5. **Catalytic competence bounds the lifetime**：詳細釣り合いから Δ ≤ 0.6–0.9 eV、τ ≲ 1 ps。
6. **Magnetic field effects**：MAO-A/B・DAO の軌跡。接触距離の双極子による上限（≤ 2×10⁻⁶ %）。CRY 対照（0.2–0.9 %、方位依存、J の両符号、核スピン 1）。
7. **Discussion**：暗所と光駆動の対比。限界（点双極子、xtb 構造、気相の λ_i、非断熱と断熱の補間、ラジカル三重項機構は範囲外）。
8. **Conclusions**

## 旧版から削除するもの（投稿版の衛生規則）

- 「交換ではなく寿命が決定的」という主張と、その根拠だった J = 750 MHz、τ = 10 ps、T₂ᵉ = 1 ns
- 19.5 mT の交差磁場と、それを使った反証実験の提案
- CRY の単一値 +1.0 %
- 撤回した数値の痕跡（本文・キャプション・コメント・ESI のすべて）と、改訂経緯の記述
- プレプリントとの差分を論じる節（cover letter で開示する）
