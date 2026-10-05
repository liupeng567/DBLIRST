"""ITTD 数据解析：VOC XML 标注 / 官方 GT txt / 官方输出 txt。

格式实测（方案 10.9 / ITTD 数据论文表 3）：
- 标注：Annotation/{v}/{f:03d}.xml，Pascal VOC，<name> = 视频内跟踪 ID（1..8，跨帧一致），
        <bndbox> = x1,y1,x2,y2（像素，含端点）。
- 官方 GT：Evaluation/cm_GT/{v}.txt，首行 `targetnum: N`，之后每帧一行：
        `frame:001 n object:1 x1 y1 x2 y2 object:2 x1 y1 x2 y2 ...`（n=0 时无 object 字段）。
- 官方输出：同 GT 结构但 object 后为中心点 `object:1 x y`（检测跟踪任务"对每个目标输出一个坐标点"）。

序列目录名 = 十进制无补零（`1`…`87`）；帧文件名 = 3 位补零（`001`…`250`）。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

FRAME_RE = re.compile(r"^frame:(\d+)\s+(\d+)(.*)$")

N_SEQS = 87
FRAMES_PER_SEQ = 250
TOTAL_FRAMES = N_SEQS * FRAMES_PER_SEQ  # 21,750
TOTAL_INSTANCES = 89_174
TOTAL_TRACKS = 393


@dataclass
class Instance:
    """单个标注实例：track_id 为视频内跟踪 ID，box = [x1, y1, x2, y2]（含端点像素）。

    检测中心点输出以退化框 (x, y, x, y) 表示，坐标允许 .5 等半像素值。
    """

    track_id: int
    box: tuple[float, float, float, float]  # x1, y1, x2, y2

    @property
    def cx(self) -> float:
        return (self.box[0] + self.box[2]) / 2.0

    @property
    def cy(self) -> float:
        return (self.box[1] + self.box[3]) / 2.0


@dataclass
class SeqAnnotation:
    """一段序列的全部帧级标注。frames 缺失的帧号 = 空标注（合法负样本）。"""

    seq_id: int
    n_target: int = 0
    frames: dict[int, list[Instance]] = field(default_factory=dict)

    @property
    def n_instances(self) -> int:
        return sum(len(v) for v in self.frames.values())

    @property
    def track_ids(self) -> set[int]:
        return {ins.track_id for v in self.frames.values() for ins in v}

    @property
    def n_tracks(self) -> int:
        return len(self.track_ids)


def frame_files(seq_dir: Path, ext: str = "bmp") -> list[int]:
    """枚举一段序列的帧号（按数值升序），缺失文件自动跳过（与官方口径一致）。"""
    return sorted(
        int(p.stem) for p in Path(seq_dir).glob(f"*.{ext}") if p.stem.isdigit()
    )


def parse_annotation_xml(xml_path: str | Path) -> tuple[int, list[Instance]]:
    """解析单个 VOC XML，返回 (帧号, 实例列表)。"""
    root = ET.parse(xml_path).getroot()
    stem = Path(xml_path).stem
    frame_no = int(stem)
    instances: list[Instance] = []
    for obj in root.findall("object"):
        track_id = int(obj.findtext("name"))
        bnd = obj.find("bndbox")
        box = (
            int(float(bnd.findtext("xmin"))),
            int(float(bnd.findtext("ymin"))),
            int(float(bnd.findtext("xmax"))),
            int(float(bnd.findtext("ymax"))),
        )
        instances.append(Instance(track_id=track_id, box=box))
    return frame_no, instances


def load_seq_annotation(
    seq_dir: str | Path, frames_per_seq: int = FRAMES_PER_SEQ
) -> SeqAnnotation:
    """载入一段序列的全部 XML 标注。seq_dir = Annotation/{seq_id}（已含序列号层级）。

    帧号按 1..frames_per_seq 补齐（缺 XML = 空帧）。
    """
    seq_dir = Path(seq_dir)
    ann = SeqAnnotation(seq_id=int(seq_dir.name))
    for f in range(1, frames_per_seq + 1):
        xml_path = seq_dir / f"{f:03d}.xml"
        if xml_path.exists():
            frame_no, instances = parse_annotation_xml(xml_path)
            if frame_no != f:
                raise ValueError(
                    f"帧号不连续: {xml_path} 文件名帧 {f} != 内容帧 {frame_no}"
                )
            ann.frames[f] = instances
        else:
            ann.frames[f] = []
    return ann


def parse_official_txt(path: str | Path) -> SeqAnnotation:
    """解析官方 GT txt（box 版本）或预测 txt（中心点版本）。

    预测 txt 的 object 只有 x y 两个数，此时 box = (x, y, x, y)（退化框，中心即该点）。
    """
    seq_ann = SeqAnnotation(seq_id=0)
    with open(path, encoding="utf-8") as fp:
        header = fp.readline().strip()
        m = re.match(r"^targetnum:\s*(\d+)$", header)
        if not m:
            raise ValueError(f"{path} 首行不是 targetnum 头: {header!r}")
        seq_ann.n_target = int(m.group(1))
        for line in fp:
            line = line.strip()
            if not line:
                continue
            m = FRAME_RE.match(line)
            if not m:
                raise ValueError(f"{path} 无法解析行: {line!r}")
            frame_no, n_obj, rest = int(m.group(1)), int(m.group(2)), m.group(3)
            # 浮点感知：中心点输出可能出现 440.5 这类半像素坐标
            nums = [float(t) for t in re.findall(r"-?\d+(?:\.\d+)?", rest)]
            instances: list[Instance] = []
            # 以声明的 n_obj 消歧（15 个数字 = 3 框或 5 点），box 版本优先
            if n_obj > 0 and len(nums) == 5 * n_obj:
                for k in range(0, len(nums), 5):
                    tid, x1, y1, x2, y2 = nums[k : k + 5]
                    instances.append(Instance(track_id=int(tid), box=(x1, y1, x2, y2)))
            elif n_obj > 0 and len(nums) == 3 * n_obj:
                for k in range(0, len(nums), 3):
                    tid, x, y = nums[k : k + 3]
                    instances.append(Instance(track_id=int(tid), box=(x, y, x, y)))
            elif n_obj > 0:
                raise ValueError(
                    f"{path} frame:{frame_no:03d} 声明 {n_obj} 目标但数字数 {len(nums)} 不可解析"
                )
            if len(instances) != n_obj:
                raise ValueError(
                    f"{path} frame:{frame_no:03d} 声明 {n_obj} 个目标，解析得 {len(instances)}"
                )
            seq_ann.frames[frame_no] = instances
    return seq_ann


def write_official_txt(
    path: str | Path,
    seq_id: int,
    detections: dict[int, list[Instance]],
    frames_per_seq: int = FRAMES_PER_SEQ,
) -> None:
    """按官方表 3 格式写出预测 txt（中心点版本）。

    detections: {frame_no: [Instance(track_id, box)]}——box 取中心点输出；
    track_id 为预测航迹 ID（从 1 连续编号）。
    """
    max_track = max(
        (ins.track_id for v in detections.values() for ins in v), default=0
    )
    lines = [f"targetnum: {max_track}"]
    for f in range(1, frames_per_seq + 1):
        dets = detections.get(f, [])
        parts = [f"frame:{f:03d}", str(len(dets))]
        for ins in dets:
            parts.append(f"object:{ins.track_id} {ins.cx:g} {ins.cy:g}")
        lines.append(" ".join(parts))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
