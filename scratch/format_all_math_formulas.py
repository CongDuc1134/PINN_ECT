# -*- coding: utf-8 -*-
"""
scratch/format_all_math_formulas.py
Cleans up and formats ALL remaining formulas across the report template,
converting raw LaTeX strings into crystal-clear, beautifully styled .math-card blocks
and clean Unicode mathematical notation.
"""

import os
import re

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "domain_adaptation", "PINN_ECT_Report_template.html")

def clean_remaining_formulas():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Format Chuyên Đề 1 (Kendall Homoscedastic Multi-Task Loss)
    old_chuyen_de_1 = r"""  $$\mathcal{L}_{\text{data}} = \exp(-s_{\text{clf}}) \mathcal{L}_{\text{CE}}(\hat{y}, y) + \frac{1}{2} s_{\text{clf}} + \sum_{i \in \{W, L, D\}} \left( \frac{1}{2} \exp(-s_i) \text{MSE}(\hat{r}_i, r_i) + \frac{1}{2} s_i \right)$$"""
    new_chuyen_de_1 = """  <div class="math-card">
    <div class="math-eq">
      <span>ℒ<sub>data</sub> = exp(-s<sub>clf</sub>) · ℒ<sub>CE</sub>(ŷ, y) + ½ s<sub>clf</sub> + ∑<sub>i∈{W,L,D}</sub> [ ½ exp(-s<sub>i</sub>) · MSE(r̂<sub>i</sub>, r<sub>i</sub>) + ½ s<sub>i</sub> ]</span>
      <span class="eq-no">(Công thức Kendall)</span>
    </div>
    <div class="math-desc">
      <strong>Chú thích cơ chế tự cân bằng bất định (Homoscedastic Task Uncertainty):</strong>
      <ul>
        <li><span class="var">s = log(σ²)</span>: Tham số học được (learnable parameters) đại diện cho logarit của phương sai bất định của từng tác vụ.</li>
        <li><span class="var">exp(-s) = 1/σ²</span>: Đóng vai trò hệ số trọng số thích nghi (adaptive precision weight). Tác vụ nào có phương sai nhiễu lớn sẽ tự động bị giảm trọng số.</li>
        <li><span class="var">½ s</span>: Số hạng phạt điều hòa (regularizer) ngăn chặn nghiệm suy biến khi $s \\to \\infty$.</li>
      </ul>
    </div>
  </div>"""

    if old_chuyen_de_1 in text:
        text = text.replace(old_chuyen_de_1, new_chuyen_de_1)
        print("[OK] Formatted Chuyên Đề 1 equation.")

    # 2. Format Chuyên Đề 2 (PINN Total Loss & Volume Loss)
    old_chuyen_de_2_1 = r"""  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \alpha \cdot \mathcal{L}_{\text{PINN}}$$"""
    new_chuyen_de_2_1 = """  <div class="math-card">
    <div class="math-eq">
      <span>ℒ<sub>total</sub> = ℒ<sub>data</sub> + α · ℒ<sub>PINN</sub>, &nbsp;&nbsp;&nbsp; với &nbsp; ℒ<sub>PINN</sub> = log(1 + SSE / ∑ B<sub>true</sub><sup>2</sup>)</span>
      <span class="eq-no">(Công thức PINN)</span>
    </div>
    <div class="math-desc">
      <strong>Hàm mất mát tích hợp vật lý giải tích từ trường (Dipole Forward Model):</strong>
      <ul>
        <li><span class="var">SSE</span>: Tổng bình phương sai lệch giữa trường từ dự đoán của mạng và trường giải tích lưỡng cực vi sai.</li>
        <li><span class="var">α ∈ {1, 10, 100, 1000}</span>: Trọng số siêu tham số vật lý quy định mức độ chi phối của định luật Maxwell.</li>
      </ul>
    </div>
  </div>"""

    if old_chuyen_de_2_1 in text:
        text = text.replace(old_chuyen_de_2_1, new_chuyen_de_2_1)
        print("[OK] Formatted Chuyên Đề 2 PINN equation.")

    old_chuyen_de_2_2 = r"""    $$\mathcal{L}_{\text{vol}} = \left\| (W \cdot L \cdot D) - \kappa \cdot \max |H| \right\|^2$$"""
    new_chuyen_de_2_2 = """  <div class="math-card">
    <div class="math-eq">
      <span>ℒ<sub>vol</sub> = ||(W · L · D) - κ · max |H|||<sup>2</sup></span>
      <span class="eq-no">(Bảo toàn thể tích)</span>
    </div>
    <div class="math-desc">
      <strong>Định luật bảo toàn thể tích cảm ứng dòng xoáy theo Faraday:</strong>
      <ul>
        <li><span class="var">Vol = W · L · D</span>: Thể tích hình học của khuyết tật kim loại bị chiếm chỗ.</li>
        <li><span class="var">max |H|</span>: Biên độ cực đại của trường phản hồi cảm ứng vi sai đo được trên bề mặt.</li>
        <li><span class="var">κ</span>: Hằng số độ nhạy cảm ứng phụ thuộc độ dẫn điện $\\sigma$ và tần số 5 kHz.</li>
      </ul>
    </div>
  </div>"""

    if old_chuyen_de_2_2 in text:
        text = text.replace(old_chuyen_de_2_2, new_chuyen_de_2_2)
        print("[OK] Formatted Chuyên Đề 2 Volume equation.")

    # 3. Clean up table cell math in Table (around lines 2010-2070)
    table_replacements = [
        (
            r"""$$\mathcal{L} = \mathcal{L}_{\text{CE}}(\hat{y}, y) + \sum_{i \in \{W, L, D\}} \text{MSE}(\hat{r}_i, r_i)$$""",
            """<span class="var">ℒ = ℒ<sub>CE</sub>(ŷ, y) + ∑<sub>i∈{W,L,D}</sub> MSE(r̂<sub>i</sub>, r<sub>i</sub>)</span>"""
        ),
        (
            r"""$$\mathcal{L} = \mathcal{L}_{\text{task}} - \lambda_p \mathcal{L}_{\text{domain}}$$ với $\mathcal{L}_{\text{domain}} = \text{BCE}(D(f(x)), d)$.""",
            """<span class="var">ℒ = ℒ<sub>task</sub> - λ<sub>p</sub> · ℒ<sub>domain</sub></span><br>với <span class="var">ℒ<sub>domain</sub> = BCE(D(f(x)), d)</span>"""
        ),
        (
            r"""$$\mathcal{L} = \mathcal{L}_{\text{task}} + \beta \mathcal{D}_{\text{MMD}}^2(\mathcal{P}_s, \mathcal{P}_t) + \alpha \mathcal{L}_{\text{vol}}$$""",
            """<span class="var">ℒ = ℒ<sub>task</sub> + β · MMD<sup>2</sup>(𝒫<sub>s</sub>, 𝒫<sub>t</sub>) + α · ℒ<sub>vol</sub></span>"""
        ),
        (
            r"""$$\mathcal{L} = \mathcal{L}_{\text{kendall}} + \lambda_{\text{phys}} \mathcal{L}_{\text{vol}} + 0.1 \mathcal{L}_{\text{pair}}$$ với $\mathcal{L}_{\text{vol}} = \|WLD - \kappa \max |H|\|^2$.""",
            """<span class="var">ℒ = ℒ<sub>kendall</sub> + λ<sub>phys</sub> · ℒ<sub>vol</sub> + 0.1 · ℒ<sub>pair</sub></span><br>với <span class="var">ℒ<sub>vol</sub> = ||WLD - κ · max |H|||<sup>2</sup></span>"""
        ),
        (
            r"""$$\mathcal{L} = \mathcal{L}_{\text{kendall}} + \lambda_{\text{phys}} \mathcal{L}_{\text{vol}} + \lambda_{\text{anchor}} \|W_{\text{clf}} - W_{\text{clf}}^{(0)}\|^2$$""",
            """<span class="var">ℒ = ℒ<sub>kendall</sub> + λ<sub>phys</sub> · ℒ<sub>vol</sub> + λ<sub>anchor</sub> ||W<sub>clf</sub> - W<sub>clf</sub><sup>(0)</sup>||<sup>2</sup></span>"""
        ),
        (
            r"""$$\hat{y} = \arg\max_k \frac{f(x) \cdot \mathbf{c}_k}{\|f(x)\| \|\mathbf{c}_k\|}$$ Tầng hồi quy kích thước tối ưu qua MSE.""",
            """<span class="var">ŷ = argmax<sub>k</sub> [ (f(x) · c<sub>k</sub>) / (||f(x)|| ||c<sub>k</sub>||) ]</span><br>Tầng hồi quy tối ưu qua MSE."""
        ),
        (
            r"""$$\mathcal{L}_{\text{train}} = \mathcal{L}_{\text{kendall}} + \lambda_{\text{vol}} \mathcal{L}_{\text{vol}}$$ Suy luận: Cổng phân tách độ cong trường thế $\nabla^2 H < 0.10$.""",
            """<span class="var">ℒ<sub>train</sub> = ℒ<sub>kendall</sub> + λ<sub>vol</sub> · ℒ<sub>vol</sub></span><br>Suy luận: Cổng phân tách độ cong ∇²H < 0.10."""
        ),
    ]

    for old_s, new_s in table_replacements:
        if old_s in text:
            text = text.replace(old_s, new_s)
            print("[OK] Replaced table cell math.")
        else:
            print("[WARN] Table cell math not found:", old_s[:40])

    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        f.write(text)

    print("[SUCCESS] All remaining equations cleanly formatted!")

if __name__ == "__main__":
    clean_remaining_formulas()
