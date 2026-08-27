"""Create the final Chinese research report from verified project artifacts."""

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "final_research_report_zh.docx"


def set_cell_shading(cell, fill: str) -> None:
    props = cell._tc.get_or_add_tcPr()
    shading = props.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        props.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_width(cell, width_dxa: int) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_w = tc_pr.find(qn("w:tcW"))
    if tc_w is None:
        tc_w = OxmlElement("w:tcW")
        tc_pr.append(tc_w)
    tc_w.set(qn("w:w"), str(width_dxa))
    tc_w.set(qn("w:type"), "dxa")


def set_table_geometry(table, widths: list[int]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "120")
    tbl_ind.set(qn("w:type"), "dxa")
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for cell, width in zip(row.cells, widths, strict=True):
            set_cell_width(cell, width)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(2)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    set_table_geometry(table, widths)
    for cell, text in zip(table.rows[0].cells, headers, strict=True):
        cell.text = text
        set_cell_shading(cell, "E8EEF5")
        for run in cell.paragraphs[0].runs:
            run.bold = True
    for values in rows:
        cells = table.add_row().cells
        for cell, text in zip(cells, values, strict=True):
            cell.text = text
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_bullets(doc: Document, values: list[str]) -> None:
    for value in values:
        paragraph = doc.add_paragraph(style="List Bullet")
        paragraph.add_run(value)


def main() -> None:
    two_tower = json.loads(
        (ROOT / "results/video_games_2018/two_tower_v3_final_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    final_test = json.loads(
        (ROOT / "results/video_games_2018/final_test_metrics.json").read_text(encoding="utf-8")
    )
    serving_benchmark = json.loads(
        (ROOT / "results/video_games_2018/serving_benchmark.json").read_text(encoding="utf-8")
    )
    doc = Document()
    section = doc.sections[0]
    section.top_margin = section.bottom_margin = Inches(1)
    section.left_margin = section.right_margin = Inches(1)
    section.header_distance = section.footer_distance = Inches(0.492)
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Calibri")
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.1
    for name, size, color, before, after in [
        ("Heading 1", 16, "2E74B5", 16, 8),
        ("Heading 2", 13, "2E74B5", 12, 6),
        ("Heading 3", 12, "1F4D78", 8, 4),
    ]:
        style = styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style.font.bold = True
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    title.paragraph_format.space_after = Pt(4)
    run = title.add_run("两阶段推荐系统\n最终研究报告")
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:ascii"), "Calibri")
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = RGBColor(11, 37, 69)
    subtitle = doc.add_paragraph("Amazon Video Games · 离线研究原型与可观测 Serving Demo")
    subtitle.paragraph_format.space_after = Pt(16)
    subtitle.runs[0].font.color.rgb = RGBColor(85, 85, 85)
    add_table(
        doc,
        ["项目状态", "数据规模", "评估口径", "报告日期"],
        [["Phase 1–5 完成", "197,597 交互 / 19,621 用户", "严格时间切分 + 全目录", "2026-08-27"]],
        [2100, 2500, 2500, 2260],
    )
    doc.add_heading("执行摘要", level=1)
    doc.add_paragraph(
        "本项目已经完成一个可复现的两阶段推荐系统研究原型，并进一步封装为可部署、"
        "可监控的本地 Serving Demo。系统先用 Popularity、ItemCF 和修复后的 Two-Tower "
        "生成候选，再用 FAISS、RRF 和 LightGBM LambdaRank 完成排序。所有最终数字来自真实运行产物；"
        "项目没有使用线上曝光日志，因此不宣称真实商业收益。"
    )
    doc.add_heading("关键结论", level=2)
    add_bullets(
        doc,
        [
            "Two-Tower 的测试 Recall@100 为 9.51%，超过 Popularity 的 6.41%，"
            "但仍低于 ItemCF 的 15.70%。",
            "多路候选测试 Recall@200 为 21.00%；LambdaRank 将测试 NDCG@10 从 RRF 的 1.78% "
            "提升到 2.99%。",
            "FAISS IndexFlatIP 在 100 位用户上的 Top-100 与 NumPy 精确点积 100% 一致，"
            "单用户 P95 为 2.15 ms。",
            f"Serving 100 次真实请求的 P50/P95/QPS 为 {serving_benchmark['p50_ms']:.2f} ms / "
            f"{serving_benchmark['p95_ms']:.2f} ms / {serving_benchmark['qps']:.2f}；"
            "P95 高于参考目标，"
            "瓶颈是 Python 侧 ItemCF 与特征构造。",
        ],
    )
    doc.add_heading("1. 数据与评估设计", level=1)
    doc.add_paragraph(
        "正式数据来自 UCSD McAuley Lab 的 Amazon Reviews 2018 Video_Games 5-core。"
        "原始评论按用户、商品和日期整理，只保留正向隐式反馈，并为每位用户按不同日期构造 "
        "retrieval_train、rank_train、validation 和 test。主召回指标使用完整 13,923 商品训练目录，"
        "而不是 sampled-negative 评估。"
    )
    add_table(
        doc,
        ["对象", "数量", "用途"],
        [
            ["原始评论", "497,577", "数据下载与审计"],
            ["正式交互", "197,597", "清洗、切分与训练"],
            ["可评估用户", "19,621", "每人一个 validation/test 目标"],
            ["商品总数", "14,391", "清洗后商品实体"],
            ["训练召回目录", "13,923", "双塔、ItemCF、FAISS 共同目录"],
        ],
        [2400, 1800, 5160],
    )
    doc.add_heading("2. 失败诊断与模型修复", level=1)
    doc.add_paragraph(
        "第一版双塔在测试集只有 Recall@100=3.02%，低于 Popularity。代码检查证明 checkpoint、"
        "向量归一化、全目录检索和已见过滤均正常。继续训练到 30 轮后验证集也只有 4.88%，"
        "因此问题不是简单欠训练。"
    )
    doc.add_paragraph(
        "进一步分析发现，批内负样本按出现频率被抽取，热门商品更容易成为负例，"
        "未校正模型推荐商品平均热度只有真实目标的 39%。加入 logQ 采样校正后，双塔验证 Recall@100 "
        "达到 11.48%，冻结测试为 9.51%。这项修复在未参与调参的测试集上仍有效。"
    )
    doc.add_heading("3. 最终离线结果", level=1)
    tt_test = two_tower["splits"]["test"]["metrics"]
    add_table(
        doc,
        ["模型/阶段", "Recall@10", "Recall@100", "NDCG@10", "说明"],
        [
            ["Popularity", "—", "6.41%", "—", "最简单基线"],
            ["ItemCF", "—", "15.70%", "—", "最强单路召回基线"],
            [
                "Two-Tower + logQ",
                f"{tt_test['recall@10']:.2%}",
                f"{tt_test['recall@100']:.2%}",
                f"{tt_test['ndcg@10']:.2%}",
                "完整目录测试",
            ],
            [
                "多路 RRF",
                f"{final_test['retrieval_order_metrics']['recall@10']:.2%}",
                "21.00%候选召回",
                f"{final_test['retrieval_order_metrics']['ndcg@10']:.2%}",
                "200 候选",
            ],
            [
                "多路 + LambdaRank",
                f"{final_test['lambdarank_metrics']['recall@10']:.2%}",
                "21.00%候选召回",
                f"{final_test['lambdarank_metrics']['ndcg@10']:.2%}",
                "冻结测试",
            ],
        ],
        [2600, 1300, 1500, 1300, 2660],
    )
    doc.add_paragraph(
        "LambdaRank 的测试 Recall@10 绝对提升 2.06 个百分点，95% 成对 bootstrap 区间为 "
        "[1.77, 2.34]；NDCG@10 绝对提升 1.21 个百分点，区间为 [1.04, 1.37]。新增命中 623 人、"
        "损失 219 人，McNemar 精确检验 p≈1.3×10^-45。"
    )
    doc.add_heading("4. 工程化 Serving", level=1)
    doc.add_paragraph(
        "ServingPipeline 在启动时一次性加载冻结模型、FAISS、ItemCF、Popularity 和 LambdaRank。"
        "known user 使用完整流程；unknown user 使用稳定的 Popularity fallback。API 只暴露 /health、"
        "/recommend/{user_id} 和 /metrics，k 限制为 1–50。Redis 使用 "
        "rec:v1:{user_id}:{k}、TTL 300 秒，"
        "连接失败在 50 ms 后自动绕过。"
    )
    add_table(
        doc,
        ["验收项", "结果"],
        [
            ["离线 parity", "20/20 固定用户 top-10 完全一致"],
            ["FAISS 正确性", "100 位用户逐行 100% 匹配 NumPy"],
            ["重复请求", "结果确定性一致；无重复、无已见商品"],
            [
                "API benchmark",
                f"P50 {serving_benchmark['p50_ms']:.2f} ms / "
                f"P95 {serving_benchmark['p95_ms']:.2f} ms / "
                f"{serving_benchmark['qps']:.2f} QPS",
            ],
            ["Docker", "Docker CLI 未安装；已提供 Dockerfile 与 Compose，未虚报容器通过"],
        ],
        [2600, 6760],
    )
    doc.add_heading("5. 限制与下一步", level=1)
    add_bullets(
        doc,
        [
            "数据是公开 Amazon Reviews，只有离线反馈，没有曝光日志、线上反馈或 A/B 实验。",
            "721 个测试目标未进入训练目录；长尾目标的 Candidate Recall@200 明显低于头部商品。",
            "正式商品元数据中的 price、avg_rating、rating_number 为空，"
            "没有用常数填充制造伪特征。",
            "下一项高价值扩展是补齐可靠的 title、brand、细粒度 category，"
            "用内容召回改善 cold/long-tail；若字段仍不完整，应停止扩展。",
        ],
    )
    doc.add_heading("6. 项目定位与简历表述", level=1)
    doc.add_paragraph(
        "准确定位：End-to-End Production-Style Two-Stage Recommendation System，"
        "可部署、可监控的离线研究原型。它适合用于研究生申请和推荐算法实习，含金量来自失败诊断、"
        "防泄漏评估、采样校正、ANN 正确性、排序显著性和可观测 Serving，而不是虚构线上业务收益。"
    )
    doc.add_paragraph(
        "推荐简历表述：Built a leakage-aware two-stage recommender on 197K temporally split "
        "interactions; corrected in-batch sampling bias to improve full-catalog Two-Tower "
        "Recall@100 from 3.02% to 9.51%, verified exact FAISS retrieval at 2.15 ms P95, and "
        "improved test NDCG@10 by 67.7% with multi-channel LambdaRank ranking."
    )
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Two-stage Recommendation System | Final Research Report")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
