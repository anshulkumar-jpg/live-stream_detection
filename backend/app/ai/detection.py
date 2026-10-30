"""YOLO detection (Ultralytics) and a deterministic mock detector for development/tests."""
from __future__ import annotations

import os
import warnings

import cv2
import numpy as np

from app.ai.base import Detector, ModelUnavailable, RawDetection, register


class MockDetector(Detector):
    """Content-based detector: finds bright blobs (threshold) so results depend on the real pixels.
    Option `fixed_box: [x1,y1,x2,y2]` always emits one detection (used by overlay/e2e tests).
    NOT a real model; reported as backend 'mock' in heartbeats."""

    def load(self) -> None:
        self.class_names = {0: self.spec.options.get("class_name", "blob")}
        self.ready = True

    def detect(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        opts = self.spec.options
        out: list[list[RawDetection]] = []
        for f in frames:
            if "fixed_box" in opts:
                x1, y1, x2, y2 = (float(v) for v in opts["fixed_box"])
                out.append([RawDetection(0, self.class_names[0], 0.9, (x1, y1, x2, y2))])
                continue
            h, w = f.shape[:2]
            small = cv2.cvtColor(cv2.resize(f, (128, 72), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            _, th = cv2.threshold(small, int(opts.get("threshold", 235)), 255, cv2.THRESH_BINARY)
            n, _, stats, _ = cv2.connectedComponentsWithStats(th)
            dets = []
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < int(opts.get("min_area", 6)):
                    continue
                sx, sy = w / 128, h / 72
                dets.append(RawDetection(0, self.class_names[0], min(0.99, 0.5 + area / 400),
                                         (x * sx, y * sy, (x + bw) * sx, (y + bh) * sy)))
            dets.sort(key=lambda d: -d.confidence)
            out.append(dets[: int(opts.get("max_det", 5))])
        return out


class UltralyticsDetector(Detector):
    def load(self) -> None:
        w = self.spec.weights
        if not w:
            raise ModelUnavailable(f"model '{self.name}': 'weights' path is required for backend ultralytics")
        if not os.path.isfile(w):
            raise ModelUnavailable(
                f"model '{self.name}': weights file not found at {w}. Mount your weights into the worker "
                f"container (see README 'Model weights'); nothing is downloaded or substituted automatically.")
        try:
            from ultralytics import YOLO  # heavy import, lazy
        except ImportError as e:
            raise ModelUnavailable("ultralytics is not installed (pip install -r requirements-gpu.txt)") from e
        self.model = YOLO(w)
        self.class_names = {int(k): v for k, v in self.model.names.items()}
        self.ready = True

    def warmup(self) -> None:
        size = int(self.spec.options.get("imgsz", 640))
        self.detect([np.zeros((size, size, 3), np.uint8)])

    def detect(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        o = self.spec.options
        half = bool(o.get("half", False)) and self.device.startswith("cuda")
        res = self.model.predict(frames, conf=float(o.get("conf", 0.25)), iou=float(o.get("iou", 0.45)),
                                 imgsz=int(o.get("imgsz", 640)), half=half, device=self.device,
                                 classes=o.get("classes"), verbose=False)
        out = []
        for r in res:
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy() if len(b) else np.zeros((0, 4))
            cls = b.cls.cpu().numpy().astype(int) if len(b) else []
            conf = b.conf.cpu().numpy() if len(b) else []
            out.append([RawDetection(int(c), self.class_names.get(int(c), str(int(c))), float(s),
                                     tuple(float(v) for v in bb))  # type: ignore[arg-type]
                        for bb, c, s in zip(xyxy, cls, conf)])
        return out


class UltralyticsPoseDetector(UltralyticsDetector):
    """YOLOv8-pose: person boxes with 17 COCO keypoints each."""

    def detect(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        o = self.spec.options
        half = bool(o.get("half", False)) and self.device.startswith("cuda")
        res = self.model.predict(frames, conf=float(o.get("conf", 0.25)), iou=float(o.get("iou", 0.45)),
                                 imgsz=int(o.get("imgsz", 640)), half=half, device=self.device, verbose=False)
        out = []
        for r in res:
            b = r.boxes
            if not len(b):
                out.append([])
                continue
            xyxy, conf = b.xyxy.cpu().numpy(), b.conf.cpu().numpy()
            kp = r.keypoints.data.cpu().numpy() if r.keypoints is not None else None
            out.append([RawDetection(0, self.class_names.get(0, "person"), float(c), tuple(float(v) for v in bb),  # type: ignore[arg-type]
                                     [[float(x), float(y), float(s)] for x, y, s in kp[i]] if kp is not None else None)
                        for i, (bb, c) in enumerate(zip(xyxy, conf))])
        return out


class YoloV5Detector(Detector):
    """Legacy YOLOv5 checkpoints (ultralytics/yolov5 repo), loaded through the `yolov5` pip package."""

    def load(self) -> None:
        w = self.spec.weights
        if not w:
            raise ModelUnavailable(f"model '{self.name}': 'weights' path is required for backend yolov5")
        if not os.path.isfile(w):
            raise ModelUnavailable(f"model '{self.name}': weights file not found at {w}.")
        try:
            import torch
            import yolov5
        except ImportError as e:
            raise ModelUnavailable("yolov5 is not installed (see backend/Dockerfile, WITH_GPU=1)") from e
        o = self.spec.options
        orig = torch.load  # torch>=2.6 defaults weights_only=True, which rejects YOLOv5 pickles (trusted local file)
        torch.load = lambda *a, **k: orig(*a, **{**k, "weights_only": False})
        try:
            self.model = yolov5.load(w, device=self.device)
        finally:
            torch.load = orig
        self.model.conf = float(o.get("conf", 0.25))
        self.model.iou = float(o.get("iou", 0.45))
        self.model.classes = o.get("classes")
        self.model.max_det = int(o.get("max_det", 300))
        if bool(o.get("half", False)) and self.device.startswith("cuda"):
            self.model.half()
        self.class_names = {int(k): v for k, v in self.model.names.items()}
        self.ready = True

    def warmup(self) -> None:
        size = int(self.spec.options.get("imgsz", 640))
        self.detect([np.zeros((size, size, 3), np.uint8)])

    def detect(self, frames: list[np.ndarray]) -> list[list[RawDetection]]:
        size = int(self.spec.options.get("imgsz", 640))
        with warnings.catch_warnings():  # yolov5 emits a torch.cuda.amp FutureWarning on every call
            warnings.simplefilter("ignore", FutureWarning)
            res = self.model([f[:, :, ::-1] for f in frames], size=size)  # AutoShape expects RGB; frames are BGR
        return [[RawDetection(int(c), self.class_names.get(int(c), str(int(c))), float(s),
                              (float(x1), float(y1), float(x2), float(y2)))
                 for x1, y1, x2, y2, s, c in det.cpu().numpy()]
                for det in res.xyxy]


register("detection", "mock", lambda n, s, d: MockDetector(n, s, d))
register("detection", "ultralytics", lambda n, s, d: UltralyticsDetector(n, s, d))
register("detection", "ultralytics_pose", lambda n, s, d: UltralyticsPoseDetector(n, s, d))
register("detection", "yolov5", lambda n, s, d: YoloV5Detector(n, s, d))
