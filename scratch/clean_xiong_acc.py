import os
import re

PROJECT_ROOT = r"c:\Users\Admin\Documents\paper\PINN_ECT"
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "domain_adaptation", "PINN_ECT_Report_template.html")
EXPAND_PATH = os.path.join(PROJECT_ROOT, "scratch", "expand_methods_report.py")

def clean_expand_script():
    with open(EXPAND_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    # Replace any aux or N/A* accuracy in Xiong rows
    # Method 1
    content = content.replace(
        '<td><strong>N/A*</strong> <span style="font-size:7pt;color:#64748b;">(17.5% aux)</span></td>',
        '<td><strong>&mdash;</strong></td>'
    )
    # Method 2 PINN
    content = content.replace(
        '<td><strong>N/A*</strong> <span style="font-size:7pt;color:#64748b;">(40.0% aux)</span></td>',
        '<td><strong>&mdash;</strong></td>'
    )
    # Method 3
    content = content.replace(
        '<td><strong>N/A*</strong> <span style="font-size:7pt;color:#64748b;">(18.3% aux)</span></td>',
        '<td><strong>&mdash;</strong></td>'
    )
    # Method 4
    content = content.replace(
        '<td><strong>N/A*</strong> <span style="font-size:7pt;color:#64748b;">(12.9% aux)</span></td>',
        '<td><strong>&mdash;</strong></td>'
    )
    # Method 13
    content = content.replace(
        '<td><strong>N/A*</strong> <span style="font-size:7pt;color:#64748b;">(30.0% aux)</span></td>\n        <td><strong>N/A*</strong></td>',
        '<td><strong>&mdash;</strong></td>\n        <td><strong>&mdash;</strong></td>'
    )

    with open(EXPAND_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print("[OK] Updated scratch/expand_methods_report.py")

def clean_html_template():
    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 1. Regex replacements for any remaining aux spans
    text = re.sub(r'<strong>N/A\*</strong>\s*<span style="font-size:7(?:.5)?pt;color:#64748b;">\([^)]*aux[^)]*\)</span>', '<strong>&mdash;</strong>', text)
    text = re.sub(r'N/A\*\s*<span style="font-size:7(?:.5)?pt;color:#64748b;">\([^)]*aux[^)]*\)</span>', '&mdash;', text)
    
    # 2. Section 2.4 table (line ~2715)
    text = re.sub(
        r'<td><strong>N/A\*</strong>\s*<span style="font-size:7\.5pt;color:#64748b;">\(30% aux\)</span></td>',
        '<td><strong>&mdash;</strong></td>',
        text
    )

    # 3. In footnote 2643: remove mention of auxiliary head accuracy
    old_note = 'Cột kết quả phân loại được ghi nhận là <strong>N/A*</strong>; trong các thực nghiệm benchmark thích ứng miền, việc bổ sung đầu phân loại tuyến tính phụ (auxiliary head) chỉ đạt mức đoán mò ~20&ndash;30%.'
    new_note = 'Cột kết quả phân loại được ghi nhận là <strong>&mdash;</strong> (Không áp dụng); mô hình thuần hồi quy 3 chiều (<i>W</i>, <i>L</i>, <i>D</i>) nên hoàn toàn không chứa tham số hay tác vụ phân loại hình dạng.'
    text = text.replace(old_note, new_note)

    # 4. In Section 6.2 (Table 24 MLP checkpoints), replace N/A* in Xiong rows with &mdash;
    def clean_table_xiong(match):
        row = match.group(0)
        # replace <td>N/A*</td> with <td>&mdash;</td>
        row = row.replace('<td>N/A*</td>', '<td>&mdash;</td>')
        return row

    text = re.sub(r'<tr>\s*<td>MLP_Xiong.*?</tr>', clean_table_xiong, text, flags=re.DOTALL)

    with open(TEMPLATE_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print("[OK] Updated domain_adaptation/PINN_ECT_Report_template.html")

if __name__ == "__main__":
    clean_expand_script()
    clean_html_template()
