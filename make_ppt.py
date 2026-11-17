"""Generates Vision_Transformers_Recent_Developments.pptx (6 slides, 16:9)."""
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

BG = RGBColor(0x0F, 0x17, 0x2A)
PANEL = RGBColor(0x1E, 0x29, 0x3B)
WHITE = RGBColor(0xF8, 0xFA, 0xFC)
MUTED = RGBColor(0x94, 0xA3, 0xB8)
CYAN = RGBColor(0x22, 0xD3, 0xEE)
VIOLET = RGBColor(0xA7, 0x8B, 0xFA)
PINK = RGBColor(0xF4, 0x72, 0xB6)
AMBER = RGBColor(0xFB, 0xBF, 0x24)
GREEN = RGBColor(0x34, 0xD3, 0x99)
FONT = "Segoe UI"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def new_slide():
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = BG
    return s


def box(slide, x, y, w, h, fill=None, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08):
    shp = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        shp.adjustments[0] = radius
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(1.5)
    shp.shadow.inherit = False
    return shp


def text(slide, x, y, w, h, runs, size=16, color=WHITE, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=1.1):
    """runs: str, or list of paragraphs; each paragraph is str or list of (text, overrides) tuples."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    paras = [runs] if isinstance(runs, str) else runs
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        for seg in ([(para, {})] if isinstance(para, str) else para):
            t, o = seg if isinstance(seg, tuple) else (seg, {})
            r = p.add_run()
            r.text = t
            r.font.name = FONT
            r.font.size = Pt(o.get("size", size))
            r.font.bold = o.get("bold", bold)
            r.font.color.rgb = o.get("color", color)
    return tb


def header(slide, kicker, title, accent=CYAN, num=None):
    box(slide, 0.6, 0.55, 0.12, 0.85, fill=accent, shape=MSO_SHAPE.RECTANGLE)
    text(slide, 0.9, 0.45, 10, 0.4, kicker.upper(), size=13, color=accent, bold=True)
    text(slide, 0.9, 0.8, 11.5, 0.7, title, size=30, bold=True)
    if num:
        text(slide, 12.0, 6.95, 1.0, 0.35, f"{num} / 6", size=11, color=MUTED, align=PP_ALIGN.RIGHT)


# ---------------- Slide 1: Title ----------------
s = new_slide()
# decorative patch grid (image -> patches motif)
cols = [CYAN, VIOLET, PINK, AMBER]
for r in range(4):
    for c in range(4):
        box(s, 8.6 + c * 1.0, 1.35 + r * 1.0, 0.85, 0.85, fill=PANEL,
            line=cols[(r + c) % 4], radius=0.15)
for i, c in enumerate(cols):
    box(s, 8.75 + i * 1.0, 1.5 + i * 1.0, 0.55, 0.55, fill=c, radius=0.2)
text(s, 0.8, 1.5, 7.6, 0.4, "RECENT DEVELOPMENTS IN DEEP LEARNING", size=14, color=CYAN, bold=True)
text(s, 0.8, 2.0, 7.6, 2.2, [[("Vision Transformers", {})], [("From CNNs to Attention", {"color": VIOLET})]],
     size=44, bold=True, spacing=1.0)
text(s, 0.8, 4.15, 7.4, 1.0,
     "How self-attention is reshaping computer vision: a review of IEEE, ACM and Elsevier publications (2021–2023)",
     size=17, color=MUTED)
box(s, 0.8, 5.5, 6.6, 1.1, fill=PANEL)
text(s, 1.05, 5.62, 6.2, 0.9, [[("Group members:  ", {"color": MUTED}), ("Name 1   •   Name 2", {"bold": True})],
                              [("Subject:  ", {"color": MUTED}), ("Deep Learning, Group Assignment", {})]],
     size=15, spacing=1.3)

# ---------------- Slide 2: Background ----------------
s = new_slide()
header(s, "Background", "Why move beyond CNNs?", CYAN, 2)
# left: CNN vs attention
box(s, 0.6, 1.75, 5.9, 4.95, fill=PANEL)
text(s, 0.9, 1.95, 5.4, 0.4, "The limitation", size=18, color=PINK, bold=True)
text(s, 0.9, 2.4, 5.4, 1.6, [
    "A CNN filter sees only a small local window (e.g. 3×3). Global context is built up slowly by stacking many layers.",
], size=15, color=WHITE)
text(s, 0.9, 3.75, 5.4, 0.4, "The idea: self-attention", size=18, color=GREEN, bold=True)
text(s, 0.9, 4.2, 5.4, 2.4, [
    "Every image region attends to every other region in a single layer, so long-range relationships are learned directly.",
    "",
    [("Attention(Q,K,V) = softmax(QKᵀ / √d) · V", {"color": CYAN, "bold": True, "size": 15})],
], size=15)
# right: ViT pipeline
text(s, 7.0, 1.8, 5.8, 0.4, "Vision Transformer (ViT) pipeline [1]", size=18, color=AMBER, bold=True)
steps = [("Image", "224 × 224 pixels", CYAN),
         ("Split into patches", "16 × 16 each → 196 patches", VIOLET),
         ("Patch embedding", "each patch → token vector + position", PINK),
         ("Transformer encoder", "multi-head self-attention layers", AMBER),
         ("Classification head", "[CLS] token → class label", GREEN)]
for i, (t, sub, c) in enumerate(steps):
    y = 2.35 + i * 0.88
    box(s, 7.0, y, 5.7, 0.7, fill=PANEL, line=c)
    text(s, 7.2, y + 0.05, 5.4, 0.6, [[(t + "   ", {"bold": True, "color": c}), (sub, {"color": MUTED, "size": 13})]],
         size=15, anchor=MSO_ANCHOR.MIDDLE)
    if i < len(steps) - 1:
        a = box(s, 9.7, y + 0.7, 0.3, 0.18, fill=MUTED, shape=MSO_SHAPE.DOWN_ARROW)

# ---------------- Slide 3: Key papers ----------------
s = new_slide()
header(s, "Literature review", "Key recent papers", VIOLET, 3)
papers = [
    ("Swin Transformer", "Liu et al., IEEE/CVF ICCV 2021 [2]", CYAN,
     "Attention inside small shifted windows + hierarchical feature maps. Cost grows linearly with image size instead of quadratically.",
     "87.3% ImageNet-1K top-1 • 58.7 box AP on COCO • ICCV Marr Prize"),
    ("ConvNeXt", "Liu et al., IEEE/CVF CVPR 2022 [3]", PINK,
     "Modernised a ResNet step by step using Transformer design choices (larger kernels, LayerNorm, GELU, fewer activations).",
     "87.8% ImageNet top-1 with a pure CNN; beats Swin on COCO and ADE20K"),
    ("Segment Anything (SAM)", "Kirillov et al., IEEE/CVF ICCV 2023 [4]", AMBER,
     "A ViT-based foundation model that segments any object from a click, box or mask prompt, without retraining.",
     "Trained on SA-1B: 11M images, 1.1B masks • strong zero-shot transfer"),
    ("Surveys", "IEEE TPAMI 2023 [5] • ACM CSUR 2022 [6] • Elsevier MedIA 2023 [7]", GREEN,
     "Classify ViT variants (backbones, detection, segmentation, video, low-level vision) and their use in medical imaging.",
     "Transformers match or beat CNNs, but need more data and compute"),
]
for i, (name, src, c, idea, result) in enumerate(papers):
    x = 0.6 + (i % 2) * 6.15
    y = 1.75 + (i // 2) * 2.6
    box(s, x, y, 5.95, 2.4, fill=PANEL)
    box(s, x, y, 0.1, 2.4, fill=c, shape=MSO_SHAPE.RECTANGLE)
    text(s, x + 0.3, y + 0.12, 5.5, 0.4, name, size=19, color=c, bold=True)
    text(s, x + 0.3, y + 0.52, 5.5, 0.35, src, size=12, color=MUTED)
    text(s, x + 0.3, y + 0.9, 5.5, 0.95, idea, size=13.5)
    text(s, x + 0.3, y + 1.85, 5.5, 0.45, [[("▶ ", {"color": c}), (result, {"bold": True})]], size=13)

# ---------------- Slide 4: Comparison ----------------
s = new_slide()
header(s, "Analysis", "CNN vs Vision Transformers: comparison", PINK, 4)
rows = [
    ["", "CNN (ResNet)", "ViT [1]", "Swin [2]", "ConvNeXt [3]"],
    ["Core operation", "Local convolution", "Global self-attention", "Windowed self-attention", "Modernised convolution"],
    ["Built-in image prior", "Strong (locality)", "Weak", "Medium", "Strong"],
    ["Cost vs image size", "Linear", "Quadratic", "Linear", "Linear"],
    ["Data needed", "Moderate", "Very large", "Large", "Moderate–large"],
    ["Detection / segmentation", "Good", "Needs adaptation", "Excellent", "Excellent"],
]
tbl = s.shapes.add_table(len(rows), 5, Inches(0.6), Inches(1.75), Inches(12.1), Inches(3.3)).table
tbl.columns[0].width = Inches(2.9)
for j in range(1, 5):
    tbl.columns[j].width = Inches(2.3)
hdr_cols = [PANEL, MUTED, CYAN, VIOLET, PINK]
for i, row in enumerate(rows):
    for j, val in enumerate(row):
        cell = tbl.cell(i, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0x2A, 0x37, 0x4B) if i == 0 else (PANEL if i % 2 else RGBColor(0x17, 0x22, 0x36))
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf = cell.text_frame
        tf.paragraphs[0].text = ""
        r = tf.paragraphs[0].add_run()
        r.text = val
        r.font.name = FONT
        r.font.size = Pt(14 if i else 15)
        r.font.bold = (i == 0 or j == 0)
        r.font.color.rgb = hdr_cols[j] if i == 0 else (AMBER if j == 0 else WHITE)
        tf.paragraphs[0].alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
box(s, 0.6, 5.35, 12.1, 1.4, fill=PANEL, line=AMBER)
text(s, 0.9, 5.47, 11.6, 1.2, [
    [("Key insight:  ", {"color": AMBER, "bold": True}),
     ("The field is converging. Transformers borrowed CNN ideas (local windows, hierarchy) and CNNs borrowed "
      "Transformer ideas (large kernels, LayerNorm). The best models in 2022–2023 mix both, and the gap is "
      "now mainly about data, compute and pre-training.", {})],
], size=15, anchor=MSO_ANCHOR.MIDDLE)

# ---------------- Slide 5: Applications, challenges, future ----------------
s = new_slide()
header(s, "Impact & outlook", "Applications, challenges and future directions", AMBER, 5)
cols3 = [
    ("Applications", GREEN, [
        "Medical imaging: tumour segmentation, diagnosis, image reconstruction, report generation [7]",
        "Autonomous driving: object detection and segmentation",
        "Video understanding and action recognition [5]",
        "Image editing and annotation with SAM [4]",
    ]),
    ("Challenges", PINK, [
        "Data hungry: plain ViT needs huge pre-training sets",
        "Global attention cost grows quadratically with image size",
        "High training compute and energy use",
        "Large models are hard to deploy on mobile and edge devices",
        "Attention maps do not fully explain decisions",
    ]),
    ("Future directions", CYAN, [
        "Efficient attention and lightweight hybrid CNN-ViT models",
        "Self-supervised pre-training to reduce labelled data",
        "Vision foundation models that work zero-shot, like SAM",
        "Multimodal models that combine vision and language",
        "Alternatives to attention, e.g. state-space models",
    ]),
]
for i, (title, c, items) in enumerate(cols3):
    x = 0.6 + i * 4.1
    box(s, x, 1.75, 3.9, 5.0, fill=PANEL)
    box(s, x, 1.75, 3.9, 0.08, fill=c, shape=MSO_SHAPE.RECTANGLE)
    text(s, x + 0.25, 1.95, 3.5, 0.45, title, size=20, color=c, bold=True)
    text(s, x + 0.25, 2.5, 3.45, 4.1, [[("■  ", {"color": c, "size": 10}), (it, {})] for it in items],
         size=14, spacing=1.15)
    # paragraph spacing
    for p in s.shapes[-1].text_frame.paragraphs:
        p.space_after = Pt(9)

# ---------------- Slide 6: Conclusion + references ----------------
s = new_slide()
header(s, "Conclusion", "Takeaways and references", GREEN, 6)
box(s, 0.6, 1.75, 4.6, 5.0, fill=PANEL)
take = [
    ("Attention works for images:", " ViT showed an image can be treated as a sequence of patches."),
    ("Efficiency matters:", " Swin's windowed attention made Transformers practical for detection and segmentation."),
    ("CNNs are not dead:", " ConvNeXt matches Transformers with modern training and design."),
    ("Foundation models:", " SAM points to general-purpose vision models that need no retraining."),
]
text(s, 0.85, 1.95, 4.2, 4.7, [[("✔ ", {"color": GREEN}), (a, {"bold": True, "color": GREEN}), (b, {})] for a, b in take],
     size=14, spacing=1.1)
for p in s.shapes[-1].text_frame.paragraphs:
    p.space_after = Pt(12)
refs = [
    '[1] A. Dosovitskiy et al., "An image is worth 16x16 words: Transformers for image recognition at scale," in Proc. ICLR, 2021.',
    '[2] Z. Liu et al., "Swin Transformer: Hierarchical vision transformer using shifted windows," in Proc. IEEE/CVF ICCV, 2021, pp. 10012–10022.',
    '[3] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in Proc. IEEE/CVF CVPR, 2022, pp. 11976–11986.',
    '[4] A. Kirillov et al., "Segment anything," in Proc. IEEE/CVF ICCV, 2023, pp. 4015–4026.',
    '[5] K. Han et al., "A survey on vision transformer," IEEE Trans. Pattern Anal. Mach. Intell., vol. 45, no. 1, pp. 87–110, Jan. 2023.',
    '[6] S. Khan, M. Naseer, M. Hayat, S. W. Zamir, F. S. Khan, and M. Shah, "Transformers in vision: A survey," ACM Comput. Surv., vol. 54, no. 10s, pp. 1–41, 2022.',
    '[7] F. Shamshad et al., "Transformers in medical imaging: A survey," Med. Image Anal. (Elsevier), vol. 88, Art. no. 102802, 2023.',
]
box(s, 5.45, 1.75, 7.25, 5.0, fill=PANEL)
text(s, 5.7, 1.9, 6.8, 0.4, "References (IEEE format)", size=16, color=GREEN, bold=True)
text(s, 5.7, 2.35, 6.85, 4.3, refs, size=12.5, color=MUTED, spacing=1.05)
for p in s.shapes[-1].text_frame.paragraphs:
    p.space_after = Pt(9)

out = "Vision_Transformers_Recent_Developments.pptx"
prs.save(out)
print("Saved", out)
