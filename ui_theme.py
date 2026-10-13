"""Minimal paper/ink theme and stable, code-drawn dither accents."""
from pathlib import Path
from PySide6.QtCore import Qt, QSize, QRectF
from PySide6.QtGui import QColor, QPalette, QPainter, QFont, QPixmap
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QSizePolicy

PAPER = "#f5f4ef"
SURFACE = "#fcfcf8"
INK = "#20221d"
MUTED = "#66695f"
ACCENT = "#dbe6b4"
BAYER = ((0,8,2,10),(12,4,14,6),(3,11,1,9),(15,7,13,5))

STYLE = """
QWidget { font-family: 'DejaVu Sans'; font-size: 13px; color: #20221d; }
QMainWindow, QWidget#page, QDialog, QFileDialog { background: #f5f4ef; }
QWidget#card { background: #fcfcf8; border: 1px solid #dbddd2; border-radius: 3px; }
QLabel { background: transparent; border: none; }
QLabel#muted { color: #66695f; font-size: 11px; }
QLabel#eyebrow { color: #66695f; font-family: 'DejaVu Sans Mono'; font-size: 10px; letter-spacing: 1px; }
QLabel#title { font-size: 34px; font-weight: 600; }
QLabel#prediction { font-size: 24px; font-weight: 600; }
QLabel#price { font-size: 36px; font-weight: 600; }
QLabel#badge { color: #30352a; background: #dbe6b4; padding: 8px; border-radius: 2px; font-family: 'DejaVu Sans Mono'; font-size: 10px; }
QLabel#error { color: #9c3930; font-size: 11px; }
QPushButton { background: #fcfcf8; color: #20221d; border: 1px solid #c6c9ba; border-radius: 3px; padding: 10px 12px; min-height: 18px; }
QPushButton:hover { background: #edf0df; border-color: #757d60; }
QPushButton:pressed { background: #dbe6b4; border-color: #20221d; }
QPushButton:focus { border: 1px solid #20221d; }
QPushButton:disabled { color: #85897b; background: #f0f0e9; border-color: #dedfd5; }
QPushButton#accent { background: #20221d; color: #fcfcf8; border-color: #20221d; }
QPushButton#accent:hover { background: #3b4230; }
QPushButton#accent:pressed { background: #59644a; }
QPushButton#accent:disabled { background: #e3e5d9; color: #777d6b; border-color: #d7d9cd; }
QLineEdit, QComboBox, QSpinBox { background: #fcfcf8; color: #20221d; border: 1px solid #c6c9ba; border-radius: 3px; padding: 8px; min-height: 18px; selection-background-color: #20221d; selection-color: #fcfcf8; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border-color: #20221d; }
QLineEdit[invalid="true"] { border-color: #a5473a; }
QComboBox:disabled, QSpinBox:disabled { background: #eeefe6; color: #707767; border-color: #dadccf; }
QComboBox::drop-down { width: 24px; border: none; }
QComboBox::down-arrow { image: url(@CHEVRON@); width: 12px; height: 8px; }
QComboBox QAbstractItemView { background: #fcfcf8; color: #20221d; border: 1px solid #aab09a; outline: none; selection-background-color: #20221d; selection-color: #fcfcf8; padding: 4px; }
QComboBox QAbstractItemView::item { min-height: 28px; color: #20221d; background: #fcfcf8; padding: 4px 8px; }
QComboBox QAbstractItemView::item:selected { color: #fcfcf8; background: #20221d; }
QMenu { background: #fcfcf8; color: #20221d; border: 1px solid #aab09a; padding: 4px; }
QMenu::item { padding: 8px 18px; color: #20221d; }
QMenu::item:selected { background: #20221d; color: #fcfcf8; }
QToolTip { background: #20221d; color: #fcfcf8; border: none; padding: 5px; }
QCheckBox { spacing: 8px; background: transparent; color: #20221d; }
QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #aab09a; background: #fcfcf8; border-radius: 2px; }
QCheckBox::indicator:checked { background: #20221d; border: 3px solid #dbe6b4; }
QCheckBox:disabled { color: #85897b; }
QTableWidget, QListView, QTreeView { background: #fcfcf8; color: #20221d; alternate-background-color: #efefe6; gridline-color: #e5e6dc; border: none; selection-background-color: #20221d; selection-color: #fcfcf8; outline: none; }
QHeaderView::section { background: #efefe6; color: #626659; border: none; padding: 7px; font-size: 11px; font-weight: 400; }
QDialog QLabel { color: #20221d; background: transparent; }
QScrollBar:vertical { width: 10px; background: #efefe6; border: none; }
QScrollBar::handle:vertical { background: #aab09a; min-height: 24px; }
"""


def apply_theme(app=None):
    app = app or QApplication.instance()
    app.setStyle("Fusion")
    palette = QPalette()
    colors = {"Window":PAPER,"WindowText":INK,"Base":SURFACE,"AlternateBase":"#efefe6",
              "Text":INK,"Button":SURFACE,"ButtonText":INK,"ToolTipBase":INK,
              "ToolTipText":SURFACE,"Highlight":INK,"HighlightedText":SURFACE,
              "PlaceholderText":MUTED,"Light":"#ffffff","Mid":"#c6c9ba","Dark":"#66695f"}
    for name,color in colors.items():
        palette.setColor(getattr(QPalette.ColorRole,name),QColor(color))
    for role in (QPalette.ColorRole.WindowText,QPalette.ColorRole.Text,QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled,role,QColor("#85897b"))
    app.setPalette(palette)
    app.setFont(QFont("DejaVu Sans",10))
    icon=(Path(__file__).resolve().parent / "assets/icons/chevron-down.svg").as_posix()
    app.setStyleSheet(STYLE.replace("@CHEVRON@",icon))


class DitherBand(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setFixedHeight(12)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.fillRect(self.rect(),QColor(PAPER))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#777e65"))
        for y in range(0,self.height(),3):
            for x in range(0,self.width(),3):
                density=max(0,11-int(14*x/max(1,self.width())))
                if BAYER[(y//3)%4][(x//3)%4] < density:
                    painter.drawRect(x,y,1,1)


class VideoViewport(QWidget):
    """Painting a frame never changes layout geometry or steals mouse input."""
    def __init__(self,parent=None):
        super().__init__(parent)
        self.pixmap=QPixmap()
        self.message="CONNECT A CAMERA"
        self.setMinimumSize(500,300)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)

    def sizeHint(self):
        return QSize(640,480)

    def setPixmap(self,pixmap):
        self.pixmap=pixmap
        self.update()

    def clear(self):
        self.pixmap=QPixmap()
        self.update()

    def setText(self,text):
        self.message=text
        self.update()

    def paintEvent(self,event):
        painter=QPainter(self)
        painter.fillRect(self.rect(),QColor("#20251d"))
        if self.pixmap.isNull():
            painter.setPen(QColor("#394332"))
            for y in range(0,self.height(),8):
                for x in range(0,self.width(),8):
                    if BAYER[(y//8)%4][(x//8)%4] < 4:
                        painter.drawPoint(x,y)
            painter.setPen(QColor("#dbe6b4"))
            painter.setFont(QFont("DejaVu Sans Mono",10))
            painter.drawText(self.rect(),Qt.AlignmentFlag.AlignCenter,self.message)
        else:
            fitted=self.pixmap.size().scaled(self.size(),Qt.AspectRatioMode.KeepAspectRatio)
            target=QRectF((self.width()-fitted.width())/2,(self.height()-fitted.height())/2,
                          fitted.width(),fitted.height())
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            painter.drawPixmap(target,self.pixmap,QRectF(self.pixmap.rect()))


class StatusLabel(QLabel):
    """Long runtime messages never resize the controls during a click."""
    def __init__(self,text="",parent=None,**kwargs):
        super().__init__(parent,**kwargs)
        self.full_text=text
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Fixed)
        self.setText(text)

    def setText(self,text):
        self.full_text=text
        self.setToolTip(text)
        QLabel.setText(self,self.fontMetrics().elidedText(text,Qt.TextElideMode.ElideRight,max(1,self.width())))

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.setText(self.full_text)
