import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from unittest.mock import patch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from live_app import ProduceWindow
from live_engine import LatestFrame, Frame, Packet, Settings
from prototype import ROOT, DEFAULT_MODEL


def packet(number=1, shape=(200,200,3), score=.9):
    h,w=shape[:2]
    detections=[{"label":"apple","score":score,"box_model_px":[0,0,100,100]}]
    output={"scan_id":"ui-test-"+str(number),"detections":detections,"candidates":[],
        "status":"weight_required","reason":"enter_weight_to_estimate_price",
        "geometry":{"scale_xy":[100/w,100/h],"pad_left_top":[0,0]},"confirmed":False}
    return Packet(Frame(np.full(shape,100,dtype=np.uint8),number,0,1),output,{"threshold":.6},0)


class Engine:
    def __init__(self):
        self.settings=Settings()
        self.frames=LatestFrame()
        self.logs=[]
        self.state={"result":packet(),"error":None,"status":"Live recognition","ready":True,
            "active":True,"paused":False,"source":"sample image","session":1,"revision":0,
            "completed":1,"errors":0,"frames_skipped":0,"latencies":[25]}

    def snapshot(self):
        return dict(self.state)

    def set_settings(self,settings):
        if settings!=self.settings:
            self.settings=settings
            self.state["revision"]+=1

    def hold_result(self):
        if not self.state["active"] or self.state["result"] is None:
            return None
        self.state["paused"]=True
        return self.state["result"]

    def set_paused(self,value):
        self.state["paused"]=value

    def start_image(self,image):
        self.opened_image=image
        self.state.update(active=True,paused=False,result=packet(99,image.shape))

    def stop_camera(self):
        self.state.update(active=False,paused=False,result=None)

    def _log(self,event):
        self.logs.append(event)
        return True

    def close(self):
        self.state["active"]=False

    def inject(self,name):
        self.injection=name


class UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def setUp(self):
        self.engine=Engine()
        self.args=SimpleNamespace(model=DEFAULT_MODEL,device="cpu",log=ROOT/"logs/ui-tests.jsonl",
            catalog=ROOT/"catalog.json",camera=0,image=None,no_camera=True)
        self.window=ProduceWindow(self.args,self.engine)
        self.window.show()
        self.app.processEvents()
        self.window.tick()
        self.app.processEvents()

    def tearDown(self):
        self.window.close()
        self.app.processEvents()

    def test_frame_changes_do_not_move_buttons_during_mouse_clicks(self):
        self.window.weight.setText("500")
        for number in range(1,13):
            button=self.window.pause_button
            initial=button.geometry()
            QTest.mousePress(button,Qt.MouseButton.LeftButton)
            self.engine.state["result"]=packet(number,shape=(480,1280,3) if number%2 else (640,320,3))
            self.window.tick()
            self.app.processEvents()
            self.assertEqual(initial,button.geometry())
            QTest.mouseRelease(button,Qt.MouseButton.LeftButton)
            self.assertIsNotNone(self.window.held)
            QTest.mouseClick(button,Qt.MouseButton.LeftButton)
            self.assertIsNone(self.window.held)
            self.window.tick()

    def test_weight_edits_do_not_interrupt_live_recognition(self):
        revision=self.engine.state["revision"]
        self.window.weight.setText("0")
        self.assertIsNotNone(self.engine.state["result"])
        self.assertEqual(self.engine.state["revision"],revision)
        self.assertEqual(self.window.price.text(),"--")
        self.window.weight.setText("500")
        QTest.mouseClick(self.window.pause_button,Qt.MouseButton.LeftButton)
        self.assertIsNotNone(self.window.held)
        self.assertEqual(self.window.price.text(),"$1.50")
        self.assertTrue(self.window.confirm_button.isEnabled())

    def test_confirm_manual_selection_change_and_stop_are_clickable(self):
        self.window.weight.setText("500")
        QTest.mouseClick(self.window.pause_button,Qt.MouseButton.LeftButton)
        QTest.mouseClick(self.window.confirm_button,Qt.MouseButton.LeftButton)
        self.assertTrue(self.window.confirmed)
        self.assertEqual(len(self.engine.logs),1)
        QTest.mouseClick(self.window.manual,Qt.MouseButton.LeftButton)
        self.window.product_box.setCurrentIndex(self.window.product_box.findData("DEMO_BANANA"))
        self.assertFalse(self.window.confirmed)
        self.assertTrue(self.window.confirm_button.isEnabled())
        self.assertEqual(self.window.price.text(),"$0.75")
        QTest.mouseClick(self.window.confirm_button,Qt.MouseButton.LeftButton)
        self.assertTrue(self.window.confirmed)
        QTest.mouseClick(self.window.stop_button,Qt.MouseButton.LeftButton)
        self.assertFalse(self.engine.state["active"])
        self.assertIsNone(self.window.held)

    def test_theme_overrides_dark_system_palette_for_popups_and_dialogs(self):
        dark=QPalette()
        dark.setColor(QPalette.ColorRole.Base,QColor("#303030"))
        dark.setColor(QPalette.ColorRole.Text,QColor("#182c28"))
        self.app.setPalette(dark)
        from ui_theme import apply_theme
        apply_theme(self.app)
        for widget in [self.window.unit.view(),self.window.settings_dialog,self.window.product_box.view()]:
            palette=widget.palette()
            bg=palette.color(QPalette.ColorRole.Base)
            fg=palette.color(QPalette.ColorRole.Text)
            def luminance(color):
                channels=[color.redF(),color.greenF(),color.blueF()]
                linear=[v/12.92 if v<=.04045 else ((v+.055)/1.055)**2.4 for v in channels]
                return .2126*linear[0]+.7152*linear[1]+.0722*linear[2]
            hi,lo=sorted([luminance(bg),luminance(fg)],reverse=True)
            self.assertGreaterEqual((hi+.05)/(lo+.05),7)

    def test_long_error_text_does_not_shift_controls(self):
        original=self.window.pause_button.geometry()
        self.window.error_text.setText("A pipeline boundary error "*60)
        self.app.processEvents()
        self.assertEqual(original,self.window.pause_button.geometry())
        self.assertTrue(self.window.error_text.toolTip().startswith("A pipeline"))

    def test_open_image_button_dispatches_and_uses_a_readable_qt_dialog(self):
        image=ROOT/"samples/Granny-Smith-crop.jpg"
        with patch("live_app.QFileDialog.getOpenFileName",return_value=(str(image),"")) as dialog:
            QTest.mouseClick(self.window.open_button,Qt.MouseButton.LeftButton)
            self.assertEqual(self.engine.opened_image.shape,(200,200,3))
            from PySide6.QtWidgets import QFileDialog
            self.assertEqual(dialog.call_args.kwargs["options"],QFileDialog.Option.DontUseNativeDialog)

    def test_settings_button_opens_dialog_on_first_click(self):
        seen=[]
        def dismiss():
            seen.append(self.window.settings_dialog.isVisible())
            self.window.settings_dialog.accept()
        QTimer.singleShot(10,dismiss)
        QTest.mouseClick(self.window.settings_button,Qt.MouseButton.LeftButton)
        self.assertEqual(seen,[True])

if __name__=="__main__":
    unittest.main()
