"""Qt desktop interface for local, real-time produce recognition."""
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFontDatabase
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QHBoxLayout, QVBoxLayout, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QDialog)
from live_engine import LiveEngine, Settings, load_catalog, confirm_snapshot
from pipeline_core import PRODUCE, weight_kg, decide
from prototype import ROOT, DEFAULT_MODEL
from ui_theme import apply_theme, DitherBand, VideoViewport, StatusLabel

TEAL = "#34412a"


def annotated_image(packet):
    image = Image.fromarray(packet.frame.image[:, :, ::-1])
    draw = ImageDraw.Draw(image)
    geometry = packet.output["geometry"]
    sx, sy = geometry["scale_xy"]
    px, py = geometry["pad_left_top"]
    for detection in packet.output["detections"]:
        if detection["label"] not in PRODUCE | {"fruit", "vegetable"}:
            continue
        x1, y1, x2, y2 = detection["box_model_px"]
        box = [max(0, (x1-px)/sx), max(0, (y1-py)/sy),
               min(image.width-1, (x2-px)/sx), min(image.height-1, (y2-py)/sy)]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        draw.rectangle(box, outline=TEAL, width=3)
        label = detection["label"].title() + " " + format(detection["score"], ".0%")
        width = draw.textbbox((0,0), label)[2] + 12
        y = max(0, box[1]-20)
        draw.rectangle([box[0], y, min(image.width, box[0]+width), y+20], fill=TEAL)
        draw.text((box[0]+5, y+3), label, fill="white")
    return image


class ProduceWindow(QMainWindow):
    def __init__(self, args, engine=None):
        super().__init__()
        self.args = args
        self.setWindowTitle("Produce Scanner")
        self.resize(1160, 820)
        self.setMinimumSize(1040, 760)
        self.engine = engine or LiveEngine(args.model, args.device, args.log, args.catalog)
        self.held = self.last_packet = self.last_error_id = None
        self.confirmed = False
        self.image_cache = self.image_key = None
        self.product_labels = {}
        for font in (ROOT / "assets/fonts").glob("*.ttf"):
            QFontDatabase.addApplicationFont(str(font))
        apply_theme()
        self.build()
        self.refresh_products()
        self.weight.textChanged.connect(self.update_settings)
        self.unit.currentTextChanged.connect(self.update_settings)
        self.size.currentTextChanged.connect(self.update_settings)
        self.update_settings()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(40)
        if args.image:
            QTimer.singleShot(250, lambda: self.open_image(args.image))
        elif not args.no_camera:
            QTimer.singleShot(250, self.start_camera)

    def build(self):
        page = QWidget(objectName="page")
        self.setCentralWidget(page)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24,20,24,16)
        layout.setSpacing(12)
        header = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(2)
        title.addWidget(QLabel("produce /",objectName="title"))
        title.addWidget(QLabel("LOCAL VISION  /  SIMPLE CHECKOUT",objectName="eyebrow"))
        header.addLayout(title)
        header.addStretch()
        self.badge = QLabel("LOADING MODEL",objectName="badge")
        self.badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.badge.setFixedSize(190,34)
        header.addWidget(self.badge)
        layout.addLayout(header)
        layout.addWidget(DitherBand())
        body = QHBoxLayout()
        body.setSpacing(18)
        layout.addLayout(body,1)
        left = QWidget(objectName="card")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(14,14,14,14)
        ll.setSpacing(10)
        ll.addWidget(QLabel("01 / CAMERA",objectName="eyebrow"))
        self.preview = VideoViewport()
        ll.addWidget(self.preview,1)
        row = QHBoxLayout()
        row.setSpacing(7)
        row.addWidget(QLabel("Camera",objectName="muted"))
        self.camera = QSpinBox()
        self.camera.setRange(0,9)
        self.camera.setValue(self.args.camera)
        self.camera.setFixedWidth(52)
        row.addWidget(self.camera)
        self.connect_button = self.button(row,"Connect",self.start_camera)
        self.connect_button.setFixedWidth(88)
        self.stop_button = self.button(row,"Stop",self.stop)
        self.stop_button.setFixedWidth(62)
        row.addStretch()
        self.pause_button = self.button(row,"Freeze frame",self.toggle_pause)
        self.pause_button.setFixedWidth(132)
        ll.addLayout(row)
        self.open_button = self.button(ll,"Open image",self.open_image)
        note = QLabel("One item at a time. Freeze to review.",objectName="muted")
        note.setFixedHeight(18)
        ll.addWidget(note)
        body.addWidget(left,1)
        right = QWidget(objectName="card")
        right.setFixedWidth(316)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(16,16,16,16)
        rl.setSpacing(6)
        rl.addWidget(QLabel("02 / RESULT",objectName="eyebrow"))
        self.prediction = QLabel("Waiting for a frame",objectName="prediction")
        self.prediction.setFixedHeight(44)
        self.prediction.setToolTip("Produce category from the analyzed frame")
        rl.addWidget(self.prediction)
        self.tree = QTableWidget(0,2)
        self.tree.setHorizontalHeaderLabels(["Category","Confidence"])
        self.tree.horizontalHeader().setSectionResizeMode(0,QHeaderView.ResizeMode.Stretch)
        self.tree.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeMode.ResizeToContents)
        self.tree.verticalHeader().hide()
        self.tree.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.tree.setFixedHeight(82)
        rl.addWidget(self.tree)
        rl.addSpacing(6)
        rl.addWidget(QLabel("WEIGHT",objectName="eyebrow"))
        row = QHBoxLayout()
        row.setSpacing(6)
        self.weight = QLineEdit()
        self.weight.setObjectName("weight")
        self.weight.setPlaceholderText("Measured weight")
        row.addWidget(self.weight,1)
        self.unit = QComboBox()
        self.unit.addItems(["g","kg","lb"])
        self.unit.setFixedWidth(64)
        row.addWidget(self.unit)
        rl.addLayout(row)
        rl.addWidget(QLabel("ESTIMATED TOTAL",objectName="eyebrow"))
        self.price = QLabel("--",objectName="price")
        self.price.setFixedHeight(48)
        rl.addWidget(self.price)
        self.decision = QLabel("Enter weight to estimate price.",objectName="muted")
        self.decision.setWordWrap(True)
        self.decision.setFixedHeight(46)
        rl.addWidget(self.decision)
        self.manual = QCheckBox("Select product manually")
        self.manual.setFixedHeight(24)
        self.manual.toggled.connect(self.manual_changed)
        rl.addWidget(self.manual)
        self.product_box = QComboBox()
        self.product_box.setEnabled(False)
        self.product_box.currentIndexChanged.connect(self.product_changed)
        rl.addWidget(self.product_box)
        self.confirm_button = self.button(rl,"Confirm item",self.confirm)
        self.confirm_button.setObjectName("accent")
        self.confirm_button.setEnabled(False)
        self.confirm_text = QLabel("",objectName="muted")
        self.confirm_text.setWordWrap(True)
        self.confirm_text.setFixedHeight(32)
        rl.addWidget(self.confirm_text)
        rl.addStretch()
        self.settings_button = self.button(rl,"Settings / pipeline checks",self.failure_lab)
        body.addWidget(right)
        self.stats = StatusLabel("Starting local inference...",objectName="muted")
        layout.addWidget(self.stats)
        self.error_text = StatusLabel("",objectName="error")
        layout.addWidget(self.error_text)
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("Settings / pipeline checks")
        self.settings_dialog.setFixedSize(460,320)
        dl = QVBoxLayout(self.settings_dialog)
        dl.setContentsMargins(24,24,24,24)
        dl.setSpacing(12)
        dl.addWidget(QLabel("PREVIEW",objectName="eyebrow"))
        row = QHBoxLayout()
        row.addWidget(QLabel("Inference size"))
        self.size = QComboBox()
        self.size.addItems(["320","416","640"])
        self.size.setCurrentText("416")
        row.addWidget(self.size)
        dl.addLayout(row)
        self.lab_text = QLabel("Resume the preview before injecting a fault. Each check rejects one frame, writes a log, then recovers.")
        self.lab_text.setWordWrap(True)
        self.lab_text.setFixedHeight(70)
        dl.addWidget(self.lab_text)
        self.button(dl,"Inject normalization fault",lambda:self.run_injection("bad-normalization"))
        self.button(dl,"Inject label-map fault",lambda:self.run_injection("wrong-label-map"))
        self.button(dl,"Done",self.settings_dialog.accept)

    def button(self,layout,text,callback):
        button=QPushButton(text)
        button.clicked.connect(lambda checked=False: callback())
        layout.addWidget(button)
        return button

    def refresh_products(self):
        catalog=load_catalog(self.args.catalog)
        labels={i["name"]+"  ["+i["sku"]+"]":i["sku"] for i in catalog["items"] if i["active"]}
        if labels != self.product_labels:
            previous=self.product_box.currentData()
            self.product_labels=labels
            self.product_box.blockSignals(True)
            self.product_box.clear()
            self.product_box.addItem("Select active product","")
            for name,sku in labels.items():
                self.product_box.addItem(name,sku)
            index=self.product_box.findData(previous)
            self.product_box.setCurrentIndex(max(0,index))
            self.product_box.blockSignals(False)
        return catalog

    def update_settings(self,*_):
        self.engine.set_settings(Settings("","g",int(self.size.currentText()),0.60))
        if self.held:
            self.confirmed=False
            self.confirm_text.clear()
            self.update_held_price()
        elif self.last_packet:
            self.update_live_price(self.last_packet)

    def reset_review(self):
        self.held=None
        self.confirmed=False
        self.product_box.setCurrentIndex(0)
        self.manual.setChecked(False)
        self.confirm_text.clear()
        self.pause_button.setText("Freeze frame")
        self.confirm_button.setEnabled(False)
        self.product_box.setEnabled(False)
        self.last_packet=None
        self.image_cache=None
        self.image_key=None
        self.prediction.setText("Waiting for a frame")
        self.price.setText("--")
        self.tree.setRowCount(0)
        self.decision.setText("Enter weight to estimate price.")

    def start_camera(self):
        self.reset_review()
        self.error_text.clear()
        try:
            self.engine.start_camera(self.camera.value())
        except Exception as exc:
            self.error_text.setText(str(exc))

    def stop(self):
        self.engine.stop_camera()
        self.reset_review()
        self.prediction.setText("Camera stopped")
        self.price.setText("--")
        self.tree.setRowCount(0)
        self.preview.clear()
        self.preview.setText("Connect a camera or open a saved image")

    def open_image(self,filename=None):
        import cv2
        if filename is None:
            filename,_=QFileDialog.getOpenFileName(self,"Open produce image",str(ROOT / "samples"),"Images (*.jpg *.jpeg *.png *.bmp)",
                options=QFileDialog.Option.DontUseNativeDialog)
        if not filename:
            return
        try:
            self.engine.start_image(cv2.imread(str(filename),cv2.IMREAD_COLOR))
            self.reset_review()
            self.error_text.clear()
        except Exception as exc:
            self.error_text.setText(str(exc))

    def toggle_pause(self):
        if self.held:
            self.engine.set_paused(False)
            self.reset_review()
            return
        packet=self.engine.hold_result()
        if packet is None:
            self.error_text.setText("Wait for an analyzed frame before freezing.")
            return
        self.held=packet
        self.pause_button.setText("Resume live")
        self.product_box.setEnabled(self.manual.isChecked())
        self.show_packet(packet)
        self.update_held_price()

    def manual_changed(self,*_):
        self.product_box.setEnabled(bool(self.held and self.manual.isChecked()))
        if self.held:
            self.confirmed=False
            self.confirm_text.clear()
            self.update_held_price()

    def product_changed(self,*_):
        if self.held:
            self.confirmed=False
            self.confirm_text.clear()
            self.update_held_price()

    def update_held_price(self,*_):
        if not self.held:
            return
        try:
            catalog=self.refresh_products()
            if not self.manual.isChecked():
                from pipeline_core import decide
                policy=decide(self.held.output["detections"],None,catalog)
                sku=policy["candidates"][0]["sku"] if len(policy["candidates"])==1 else ""
                self.product_box.blockSignals(True)
                self.product_box.setCurrentIndex(max(0,self.product_box.findData(sku)))
                self.product_box.blockSignals(False)
            result=confirm_snapshot(self.held,self.weight.text(),self.unit.currentText(),catalog,
                                    self.product_box.currentData(),self.manual.isChecked())
            self.price.setText("$"+result["selected"]["estimated_total_usd"])
            self.decision.setText("Review the frozen image and selected product, then confirm.")
            self.confirm_button.setEnabled(not self.confirmed)
        except Exception as exc:
            self.price.setText("--")
            self.decision.setText(str(exc))
            self.confirm_button.setEnabled(False)

    def confirm(self):
        if self.held is None or self.confirmed:
            return
        try:
            result=confirm_snapshot(self.held,self.weight.text(),self.unit.currentText(),self.refresh_products(),
                                    self.product_box.currentData(),self.manual.isChecked())
            event={"scan_id":self.held.output["scan_id"],"timestamp_unix_s":time.time(),
                   "status":"confirmed_demo","kind":"customer_confirmation","output":result}
            if not self.engine._log(event):
                raise RuntimeError("Confirmation was not saved because logging failed")
            self.confirmed=True
            self.confirm_text.setText("Confirmed: "+result["selected"]["name"]+" / $"+result["selected"]["estimated_total_usd"]+" (demo)")
            self.confirm_button.setEnabled(False)
        except Exception as exc:
            self.error_text.setText(str(exc))

    def failure_lab(self):
        self.lab_text.setText("Resume the preview before injecting a fault. Each check rejects one frame, writes a log, then recovers.")
        self.settings_dialog.exec()

    def run_injection(self,name):
        try:
            self.engine.inject(name)
            self.error_text.setText("Running check: "+name)
            self.settings_dialog.accept()
        except Exception as exc:
            self.lab_text.setText(str(exc))

    def update_live_price(self,packet):
        try:
            value=self.weight.text().strip()
            mass=weight_kg(value,self.unit.currentText()) if value else None
            output=decide(packet.output["detections"],mass,load_catalog(self.args.catalog))
            candidates=output["candidates"]
            total=candidates[0]["estimated_total_usd"] if len(candidates)==1 else None
            self.price.setText("$"+total if total is not None else "--")
            reasons={"enter_weight_to_estimate_price":"Enter weight to estimate price.",
                     "one_active_match":"Freeze the frame, then review and confirm.",
                     "low_confidence":"Low confidence. Freeze and select manually.",
                     "multiple_produce_detections":"Present one item, or select manually.",
                     "no_supported_produce":"Place produce in view, or select manually.",
                     "not_in_active_catalog":"No active catalog match. Select manually.",
                     "ambiguous_catalog_mapping":"Freeze and select the exact product."}
            self.decision.setText(reasons.get(output["reason"],output["reason"]))
        except Exception as exc:
            self.price.setText("--")
            self.decision.setText(str(exc))

    def show_packet(self,packet):
        self.last_packet=packet
        detections=sorted((d for d in packet.output["detections"] if d["label"] in PRODUCE | {"fruit","vegetable"}),key=lambda d:d["score"],reverse=True)
        self.tree.setRowCount(len(detections))
        for row,detection in enumerate(detections):
            self.tree.setItem(row,0,QTableWidgetItem(detection["label"].title()))
            self.tree.setItem(row,1,QTableWidgetItem(format(detection["score"],".0%")))
        self.prediction.setText(detections[0]["label"].title() if len(detections)==1 else str(len(detections))+" produce detections" if detections else "No produce detected")
        if not self.held:
            self.update_live_price(packet)
        self.image_cache=annotated_image(packet)
        self.image_key=None

    def render(self):
        if self.image_cache is None:
            return
        key=id(self.image_cache)
        if key==self.image_key:
            return
        self.image_key=key
        image=self.image_cache
        pixels=image.tobytes()
        qimage=QImage(pixels,image.width,image.height,image.width*3,QImage.Format.Format_RGB888).copy()
        self.preview.setPixmap(QPixmap.fromImage(qimage))

    def tick(self):
        state=self.engine.snapshot()
        self.badge.setText("FROZEN FOR REVIEW" if self.held else "LOADING MODEL" if not state["ready"] and not state["error"] else "MODEL ERROR" if not state["ready"] else "LIVE WEBCAM" if state["active"] and state["source"].startswith("webcam") else "SAMPLE IMAGE" if state["active"] else "CAMERA STOPPED")
        self.stop_button.setEnabled(state["active"])
        self.pause_button.setEnabled(bool(self.held or (state["active"] and state["result"] is not None)))
        packet=self.held or state["result"]
        if packet is not None and packet is not self.last_packet:
            self.show_packet(packet)
        elif packet is None and state["active"] and not self.held:
            frame=self.engine.frames.latest()
            if frame and frame.number != getattr(self,"raw_number",None):
                self.raw_number=frame.number
                self.image_cache=Image.fromarray(frame.image[:,:,::-1])
                self.image_key=None
            if self.last_packet:
                self.last_packet=None
                self.tree.setRowCount(0)
                self.prediction.setText("Waiting for valid prediction")
                self.price.setText("--")
        self.render()
        error=state["error"]
        if error and error.get("scan_id") != self.last_error_id:
            self.last_error_id=error.get("scan_id")
            self.error_text.setText(("Failure test: "+error["injection"]+" | " if error.get("injection") else "")+error.get("boundary","runtime")+": "+error["message"])
        latencies=state["latencies"]
        timing=""
        if latencies:
            timing=" | latest "+format(latencies[-1],".0f")+" ms | p95 "+format(float(np.percentile(latencies,95)),".0f")+" ms"
        self.stats.setText(state["status"]+timing+" | analyzed "+str(state["completed"])+" | errors "+str(state["errors"])+" | skipped "+str(state["frames_skipped"])+" frames")

    def closeEvent(self,event):
        self.timer.stop()
        self.engine.close()
        event.accept()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera",type=int,default=0)
    parser.add_argument("--model",type=Path,default=DEFAULT_MODEL)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--catalog",type=Path,default=ROOT / "catalog.json")
    parser.add_argument("--log",type=Path,default=ROOT / "logs/live.jsonl")
    parser.add_argument("--image",type=Path)
    parser.add_argument("--no-camera",action="store_true")
    args=parser.parse_args()
    app=QApplication([])
    window=ProduceWindow(args)
    window.show()
    window.showNormal()
    return app.exec()

if __name__=="__main__":
    raise SystemExit(main())
