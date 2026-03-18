"""Real-time camera viewer with Fourier spectrum display for off-axis digital holographic microscopy (DHM).

Uses Harvesters (GenICam) for vendor-neutral camera acquisition via GenTL producers.
Compatible with any GenICam-compliant camera (Basler, FLIR, Allied Vision, IDS, etc.).
"""

import os
import sys
import time
import numpy as np
from harvesters.core import Harvester
from genicam.gentl import TimeoutException
from PySide6 import QtWidgets, QtGui, QtCore
from PySide6.QtCore import QSettings
import scipy.fft
from scipy.ndimage import zoom as _ndimage_zoom


def extract_val(key, filepath, dtype=float):
    """Parse a 'KEY value' or 'KEY=value' pair from a config file."""
    if not os.path.exists(filepath):
        print(f"Config file not found: {filepath}")
        return None

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.strip().startswith('#') or len(line.strip()) == 0:
                    continue
                parts = line.replace('=', ' ').split()
                if len(parts) >= 2 and parts[0] == key:
                    return dtype(parts[1])
    except Exception as e:
        print(f"Error reading {key} from {filepath}: {e}")
    return None


class ClickableLabel(QtWidgets.QLabel):
    """QLabel subclass that emits a clicked(QPoint) signal."""
    clicked = QtCore.Signal(QtCore.QPoint)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.clicked.emit(event.position().toPoint())
        super().mousePressEvent(event)


class FourierWindow(QtWidgets.QWidget):
    """Detached window for the Fourier magnitude display."""
    def __init__(self, parent_viewer):
        super().__init__()
        self.parent_viewer = parent_viewer
        self.setWindowTitle("Fourier Plane")

        layout = QtWidgets.QVBoxLayout(self)
        self.label = ClickableLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(1, 1)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.label.clicked.connect(self.on_label_clicked)
        layout.addWidget(self.label)

        self.current_pixmap = None
        self.fourier_image_size = (0, 0)
        self.resize(512, 512)

    def update_fft(self, pixmap, image_size=None):
        self.current_pixmap = pixmap
        if image_size:
            self.fourier_image_size = image_size
        if self.current_pixmap:
            self.label.setPixmap(self.current_pixmap.scaled(
                self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def on_label_clicked(self, pos):
        """Map label click to Fourier-plane coordinates and update CX/CY."""
        if not self.parent_viewer.pick_mode:
            return
        if not self.current_pixmap or self.fourier_image_size[0] == 0:
            return

        label_size = self.label.size()
        pixmap = self.label.pixmap()
        if not pixmap:
            return

        pm_size = pixmap.size()
        offset_x = (label_size.width() - pm_size.width()) // 2
        offset_y = (label_size.height() - pm_size.height()) // 2

        click_x = pos.x() - offset_x
        click_y = pos.y() - offset_y

        if click_x < 0 or click_y < 0 or click_x >= pm_size.width() or click_y >= pm_size.height():
            return

        img_w, img_h = self.fourier_image_size
        
        fft_x = click_x * img_w / pm_size.width()
        fft_y = click_y * img_h / pm_size.height()
        
        if self.parent_viewer:
            # Use per-axis scaling so click mapping stays correct on rectangular FFTs.
            scale_x, scale_y = self.parent_viewer._get_fft_scale_factors(img_w, img_h)

            new_cx = int(fft_x / scale_x)
            new_cy = int(fft_y / scale_y)

            self.parent_viewer.set_circle_position(new_cx, new_cy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_pixmap:
            self.label.setPixmap(self.current_pixmap.scaled(
                self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def closeEvent(self, event):
        if self.parent_viewer:
            self.parent_viewer.detach_check.setChecked(False)
            self.parent_viewer.fourier_window = None
        super().closeEvent(event)


class CameraViewer(QtWidgets.QWidget):
    SETTINGS_ORG = "irimas"
    SETTINGS_APP = "dhm-fourier-viewer"

    def __init__(self, ia, harvester, model_name, parent=None):
        super().__init__(parent)
        self.ia = ia
        self.harvester = harvester
        self.model_name = model_name
        self.node_map = ia.remote_device.node_map
        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setMinimumSize(1, 1)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)

        self._last_camera_image = None   # uint8 numpy array
        self._last_fourier_image = None  # uint8 numpy array

        self.dim_roi = 1024
        self.crop_x = 0
        self.crop_y = 0
        self.nxmax = 0
        self.cx = 512 + 150
        self.cy = 512
        self.config_loaded = False

        # Optical parameters for NXMAX computation (see _compute_nxmax)
        self.na = None
        self.n0 = None
        self.lambda0 = None
        self.f_tube = None
        self.f_obj = None
        self.tpcam = None
        self.rf = None

        self.load_configuration()

        self._default_cx = self.cx
        self._default_cy = self.cy
        self._default_nxmax = self.nxmax

        # Snapshot all defaults right after config load (before UI overrides)
        self._defaults = {
            'fourier_enabled': False,
            'detach': False,
            'orders_visible': True,
            'crop_enabled': True,
            'dim_roi': self.dim_roi,
            'crop_x': 0,
            'crop_y': 0,
            'resample_enabled': False,
            'resample_target': 512,
            'gamma': 1.0,
            'clip_pct': 100.0,
            'nxmax': self.nxmax,
            'cx': self.cx,
            'cy': self.cy,
            'auto_calc': True,
        }

        self.pick_mode = False
        self.fourier_enabled = False
        self.crop_enabled = True
        self.resample_enabled = False
        self.fourier_image_size = (0, 0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        layout.addWidget(self.label, 1)

        # Fourier and exposure controls
        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(5)

        self.fourier_check = QtWidgets.QCheckBox('Fourier', self)
        self.fourier_check.setToolTip('Fourier magnitude display (F)')
        self.fourier_check.toggled.connect(self.toggle_fourier)
        controls.addWidget(self.fourier_check)

        self.detach_check = QtWidgets.QCheckBox('Detach')
        self.detach_check.setChecked(False)
        self.detach_check.setEnabled(False)
        self.detach_check.setToolTip('Detach Fourier view into its own window')
        self.detach_check.toggled.connect(self.toggle_detach)
        controls.addWidget(self.detach_check)

        self.orders_check = QtWidgets.QCheckBox('Orders')
        self.orders_check.setChecked(True)
        self.orders_check.setEnabled(False)
        self.orders_check.setToolTip('Overlay diffraction orders (DC, +1, -1) on Fourier plane')
        controls.addWidget(self.orders_check)

        self.auto_exp_check = QtWidgets.QCheckBox('Auto Exp', self)
        self.auto_exp_check.setToolTip('Continuous auto-exposure')
        self.auto_exp_check.toggled.connect(self.toggle_exposure)
        controls.addWidget(self.auto_exp_check)

        controls.addWidget(QtWidgets.QLabel('Exp:'))
        self.exp_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.exp_slider.setRange(100, 50000)
        self.exp_slider.setValue(5000)
        self.exp_slider.setFixedWidth(80)
        self.exp_slider.setToolTip('Exposure time (µs)')
        self.exp_slider.valueChanged.connect(self.on_exp_slider_changed)
        controls.addWidget(self.exp_slider)

        self.exp_spin = QtWidgets.QSpinBox()
        self.exp_spin.setRange(100, 100000)
        self.exp_spin.setValue(5000)
        self.exp_spin.setSingleStep(100)
        self.exp_spin.setFixedWidth(75)
        self.exp_spin.setSuffix(" µs")
        self.exp_spin.valueChanged.connect(self.on_exp_spin_changed)
        controls.addWidget(self.exp_spin)
        controls.addStretch()

        layout.addLayout(controls)

        # ROI crop, resample, and Fourier display controls
        crop_controls = QtWidgets.QHBoxLayout()
        crop_controls.setSpacing(5)

        self.roi_crop_check = QtWidgets.QCheckBox('ROI Crop')
        self.roi_crop_check.setChecked(True)
        self.roi_crop_check.setToolTip('Crop to square ROI at (X, Y) offset')
        self.roi_crop_check.toggled.connect(self.toggle_crop)
        crop_controls.addWidget(self.roi_crop_check)

        self.crop_size_spin = QtWidgets.QSpinBox()
        self.crop_size_spin.setRange(64, 4096)
        self.crop_size_spin.setValue(self.dim_roi)
        self.crop_size_spin.setSingleStep(64)
        self.crop_size_spin.setFixedWidth(60)
        self.crop_size_spin.setToolTip('ROI size in pixels (DIM_ROI)')
        self.crop_size_spin.valueChanged.connect(self.on_crop_size_changed)
        crop_controls.addWidget(self.crop_size_spin)

        crop_controls.addWidget(QtWidgets.QLabel('X:'))
        self.crop_x_spin = QtWidgets.QSpinBox()
        self.crop_x_spin.setRange(0, 4096)
        self.crop_x_spin.setValue(self.crop_x)
        self.crop_x_spin.setSingleStep(64)
        self.crop_x_spin.setFixedWidth(60)
        self.crop_x_spin.setToolTip('ROI crop X offset (px)')
        self.crop_x_spin.valueChanged.connect(self.on_crop_offset_changed)
        crop_controls.addWidget(self.crop_x_spin)

        crop_controls.addWidget(QtWidgets.QLabel('Y:'))
        self.crop_y_spin = QtWidgets.QSpinBox()
        self.crop_y_spin.setRange(0, 4096)
        self.crop_y_spin.setValue(self.crop_y)
        self.crop_y_spin.setSingleStep(64)
        self.crop_y_spin.setFixedWidth(60)
        self.crop_y_spin.setToolTip('ROI crop Y offset (px)')
        self.crop_y_spin.valueChanged.connect(self.on_crop_offset_changed)
        crop_controls.addWidget(self.crop_y_spin)

        self.resample_check = QtWidgets.QCheckBox('Resample')
        self.resample_check.setChecked(False)
        self.resample_check.setToolTip('Resample to target size (bilinear)')
        self.resample_check.toggled.connect(self.toggle_resample)
        crop_controls.addWidget(self.resample_check)

        self.resample_spin = QtWidgets.QSpinBox()
        self.resample_spin.setRange(64, 4096)
        self.resample_spin.setValue(512)
        self.resample_spin.setSingleStep(64)
        self.resample_spin.setFixedWidth(60)
        self.resample_spin.setToolTip('Target size (px)')
        self.resample_spin.setEnabled(False)
        crop_controls.addWidget(self.resample_spin)

        crop_controls.addWidget(QtWidgets.QLabel('Gamma:'))
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(10, 1000)
        self.gamma_slider.setValue(100)
        self.gamma_slider.setFixedWidth(60)
        self.gamma_slider.setToolTip('Fourier display gamma (>1 suppresses weak spatial frequencies, <1 boosts them)')
        self.gamma_slider.valueChanged.connect(self.on_gamma_slider_changed)
        crop_controls.addWidget(self.gamma_slider)

        self.gamma_spin = QtWidgets.QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 10.0)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setDecimals(1)
        self.gamma_spin.setFixedWidth(50)
        self.gamma_spin.valueChanged.connect(self.on_gamma_spin_changed)
        crop_controls.addWidget(self.gamma_spin)

        crop_controls.addWidget(QtWidgets.QLabel('Clip %:'))
        self.clip_spin = QtWidgets.QDoubleSpinBox()
        self.clip_spin.setRange(90.0, 100.0)
        self.clip_spin.setValue(100.0)
        self.clip_spin.setSingleStep(0.01)
        self.clip_spin.setDecimals(2)
        self.clip_spin.setFixedWidth(60)
        self.clip_spin.setToolTip('Upper percentile clip for magnitude display (e.g. 99.5)')
        crop_controls.addWidget(self.clip_spin)

        self.restore_defaults_btn = QtWidgets.QPushButton('Defaults')
        self.restore_defaults_btn.setToolTip('Restore all settings to initial defaults')
        self.restore_defaults_btn.setFixedWidth(55)
        self.restore_defaults_btn.clicked.connect(self.restore_defaults)
        crop_controls.addWidget(self.restore_defaults_btn)

        self.save_btn = QtWidgets.QPushButton('Save')
        self.save_btn.setToolTip('Save camera and Fourier images (S)')
        self.save_btn.setFixedWidth(45)
        self.save_btn.clicked.connect(self.save_images)
        crop_controls.addWidget(self.save_btn)

        crop_controls.addStretch()
        layout.addLayout(crop_controls)

        # Labeled separator
        sep_layout = QtWidgets.QHBoxLayout()
        sep_layout.setSpacing(5)
        sep_label = QtWidgets.QLabel('Diffraction Order Parameters')
        sep_label.setStyleSheet('color: grey; font-size: 10px; font-weight: bold;')
        sep_layout.addWidget(sep_label)
        separator = QtWidgets.QFrame()
        separator.setFrameShape(QtWidgets.QFrame.HLine)
        separator.setFrameShadow(QtWidgets.QFrame.Sunken)
        sep_layout.addWidget(separator, 1)
        layout.addLayout(sep_layout)
        fft_params = QtWidgets.QHBoxLayout()
        fft_params.setSpacing(5)

        fft_params.addWidget(QtWidgets.QLabel('NXMAX:'))
        self.nxmax_spin = QtWidgets.QSpinBox()
        self.nxmax_spin.setRange(0, 500)
        self.nxmax_spin.setValue(self.nxmax)
        self.nxmax_spin.setFixedWidth(50)
        self.nxmax_spin.setToolTip('Filtering aperture radius (Fourier pixels)')
        self.nxmax_spin.valueChanged.connect(self.on_nxmax_changed)
        fft_params.addWidget(self.nxmax_spin)

        fft_params.addWidget(QtWidgets.QLabel('CX:'))
        self.cx_spin = QtWidgets.QSpinBox()
        self.cx_spin.setRange(0, 4096)
        self.cx_spin.setValue(self.cx)
        self.cx_spin.setFixedWidth(55)
        self.cx_spin.setToolTip('+1 order horizontal position (px)')
        self.cx_spin.valueChanged.connect(self.on_cx_changed)
        fft_params.addWidget(self.cx_spin)

        fft_params.addWidget(QtWidgets.QLabel('CY:'))
        self.cy_spin = QtWidgets.QSpinBox()
        self.cy_spin.setRange(0, 4096)
        self.cy_spin.setValue(self.cy)
        self.cy_spin.setFixedWidth(55)
        self.cy_spin.setToolTip('+1 order vertical position (px)')
        self.cy_spin.valueChanged.connect(self.on_cy_changed)
        fft_params.addWidget(self.cy_spin)

        self.pick_mode_btn = QtWidgets.QCheckBox('Pick')
        self.pick_mode_btn.setToolTip('Pick +1 order position from Fourier display')
        self.pick_mode_btn.toggled.connect(self.on_pick_toggled)
        fft_params.addWidget(self.pick_mode_btn)

        self.auto_calc_check = QtWidgets.QCheckBox('Auto Theo.')
        self.auto_calc_check.setToolTip('Derive NXMAX and +1 order positions from optical parameters')
        fft_params.addWidget(self.auto_calc_check)

        self.theo_cxy_combo = QtWidgets.QComboBox()
        self.theo_cxy_combo.setToolTip('Theoretical +1 order candidates (carrier offset sign permutations)')
        self.theo_cxy_combo.addItem("+1 Order")
        self.theo_cxy_combo.currentIndexChanged.connect(self.on_theo_cxy_selected)
        fft_params.addWidget(self.theo_cxy_combo)

        # Connect toggled after combo box is instantiated
        self.auto_calc_check.setChecked(True)
        self.auto_calc_check.toggled.connect(self.on_auto_calc_toggled)

        self.theoretical_cxys = []

        self._update_nxmax_ui()

        self.reset_cxy_btn = QtWidgets.QPushButton('Reset')
        self.reset_cxy_btn.setToolTip('Reset NXMAX, CX, CY to defaults')
        self.reset_cxy_btn.setFixedWidth(45)
        self.reset_cxy_btn.clicked.connect(self.reset_params)
        fft_params.addWidget(self.reset_cxy_btn)

        fft_params.addStretch()
        layout.addLayout(fft_params)


        self.status_label = QtWidgets.QLabel('')
        self.status_label.setStyleSheet('color: grey; font-size: 10px;')
        layout.addWidget(self.status_label)

        self.setWindowTitle("DHM Fourier Viewer")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.current_pixmap = None

        self._frame_count = 0
        self._fps_time = time.monotonic()
        self._current_fps = 0.0
        self._status_msg_until = 0.0

        self.fourier_window = None

        self.label.mousePressEvent = self.on_main_label_clicked

        QtGui.QShortcut(QtGui.QKeySequence('F'), self, lambda: self.fourier_check.toggle())
        QtGui.QShortcut(QtGui.QKeySequence('S'), self, self.save_images)

        try:
            self.ia.start()
        except Exception as e:
            print(f"Error starting camera: {e}")
            self.status_label.setText(f"Camera error: {e}")
            return

        self.timer.start(30)
        self._init_exposure_controls()
        self._set_initial_size()
        self._update_status()

        # Apply default auto-calculation
        self.on_auto_calc_toggled(self.auto_calc_check.isChecked())

        # Restore persisted settings (overrides config-file values with last session)
        self.load_settings()

    def closeEvent(self, event):
        self.save_settings()
        self.timer.stop()
        if self.fourier_window:
            self.fourier_window.close()
            self.fourier_window = None
        try:
            if self.ia.is_acquiring():
                self.ia.stop()
            self.ia.destroy()
            self.harvester.reset()
        except Exception:
            pass
        super().closeEvent(event)

    def save_settings(self):
        """Persist current UI state to QSettings."""
        s = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        s.setValue('fourier_enabled', self.fourier_check.isChecked())
        s.setValue('detach', self.detach_check.isChecked())
        s.setValue('orders_visible', self.orders_check.isChecked())
        s.setValue('crop_enabled', self.roi_crop_check.isChecked())
        s.setValue('dim_roi', self.crop_size_spin.value())
        s.setValue('crop_x', self.crop_x_spin.value())
        s.setValue('crop_y', self.crop_y_spin.value())
        s.setValue('resample_enabled', self.resample_check.isChecked())
        s.setValue('resample_target', self.resample_spin.value())
        s.setValue('gamma', self.gamma_spin.value())
        s.setValue('clip_pct', self.clip_spin.value())
        s.setValue('nxmax', self.nxmax_spin.value())
        s.setValue('cx', self.cx_spin.value())
        s.setValue('cy', self.cy_spin.value())
        s.setValue('auto_calc', self.auto_calc_check.isChecked())
        s.setValue('geometry', self.saveGeometry())
        if self.fourier_window:
            s.setValue('fourier_window_geometry', self.fourier_window.saveGeometry())
        else:
            s.remove('fourier_window_geometry')

    def load_settings(self):
        """Restore UI state from QSettings, falling back to current widget values."""
        s = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        if not s.contains('fourier_enabled'):
            return  # No saved settings yet

        # Block signals during bulk restore to avoid triggering side effects
        widgets = [
            self.fourier_check,
            self.detach_check, self.orders_check,
            self.roi_crop_check, self.crop_size_spin,
            self.crop_x_spin, self.crop_y_spin,
            self.resample_check,
            self.resample_spin, self.gamma_slider, self.gamma_spin,
            self.clip_spin, self.nxmax_spin, self.cx_spin, self.cy_spin,
            self.auto_calc_check,
        ]
        for w in widgets:
            w.blockSignals(True)

        fourier = s.value('fourier_enabled', 'false') == 'true'
        self.fourier_check.setChecked(fourier)
        self.fourier_enabled = fourier

        orders = s.value('orders_visible', 'true') != 'false'
        self.orders_check.setChecked(orders)

        crop = s.value('crop_enabled', 'true') != 'false'
        self.roi_crop_check.setChecked(crop)
        self.crop_enabled = crop

        dim_roi = int(s.value('dim_roi', self._defaults['dim_roi']))
        self.crop_size_spin.setValue(dim_roi)
        self.dim_roi = dim_roi

        crop_x = int(s.value('crop_x', self._defaults['crop_x']))
        self.crop_x_spin.setValue(crop_x)
        self.crop_x = crop_x

        crop_y = int(s.value('crop_y', self._defaults['crop_y']))
        self.crop_y_spin.setValue(crop_y)
        self.crop_y = crop_y

        resample = s.value('resample_enabled', 'false') == 'true'
        self.resample_check.setChecked(resample)
        self.resample_enabled = resample
        self.resample_spin.setEnabled(resample)

        self.resample_spin.setValue(int(s.value('resample_target', self._defaults['resample_target'])))

        gamma = float(s.value('gamma', self._defaults['gamma']))
        self.gamma_spin.setValue(gamma)
        self.gamma_slider.setValue(min(int(gamma * 100), self.gamma_slider.maximum()))

        self.clip_spin.setValue(float(s.value('clip_pct', self._defaults['clip_pct'])))

        nxmax = int(s.value('nxmax', self._defaults['nxmax']))
        self.nxmax_spin.setValue(nxmax)
        self.nxmax = nxmax

        cx = int(s.value('cx', self._defaults['cx']))
        cy = int(s.value('cy', self._defaults['cy']))
        self.cx_spin.setValue(cx)
        self.cy_spin.setValue(cy)
        self.cx = cx
        self.cy = cy

        auto_calc = s.value('auto_calc', 'true') != 'false'
        self.auto_calc_check.setChecked(auto_calc)

        # Sync combo box selection with restored CX/CY
        self._sync_theo_combo_to_cxy()

        for w in widgets:
            w.blockSignals(False)

        # Enable/disable dependent widgets
        self.detach_check.setEnabled(fourier)
        self.orders_check.setEnabled(fourier and self.nxmax > 0)
        self._update_nxmax_ui()

        # Restore window geometry
        geom = s.value('geometry')
        if geom:
            self.restoreGeometry(geom)

        # Restore detached Fourier window
        detach = s.value('detach', 'false') == 'true'
        if detach and fourier:
            self.detach_check.setChecked(True)
            self.toggle_detach(True)
            fw_geom = s.value('fourier_window_geometry')
            if fw_geom and self.fourier_window:
                self.fourier_window.restoreGeometry(fw_geom)

    def restore_defaults(self):
        """Clear QSettings and reset all controls to their initial defaults."""
        s = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        s.clear()

        d = self._defaults

        # Block signals during bulk reset
        widgets = [
            self.fourier_check,
            self.detach_check, self.orders_check,
            self.roi_crop_check, self.crop_size_spin,
            self.crop_x_spin, self.crop_y_spin,
            self.resample_check,
            self.resample_spin, self.gamma_slider, self.gamma_spin,
            self.clip_spin, self.nxmax_spin, self.cx_spin, self.cy_spin,
            self.auto_calc_check,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.fourier_check.setChecked(d['fourier_enabled'])
        self.detach_check.setChecked(d['detach'])
        self.orders_check.setChecked(d['orders_visible'])
        self.roi_crop_check.setChecked(d['crop_enabled'])
        self.crop_size_spin.setValue(d['dim_roi'])
        self.crop_x_spin.setValue(d['crop_x'])
        self.crop_y_spin.setValue(d['crop_y'])
        self.resample_check.setChecked(d['resample_enabled'])
        self.resample_spin.setValue(d['resample_target'])
        self.gamma_spin.setValue(d['gamma'])
        self.gamma_slider.setValue(int(d['gamma'] * 100))
        self.clip_spin.setValue(d['clip_pct'])
        self.nxmax_spin.setValue(d['nxmax'])
        self.cx_spin.setValue(d['cx'])
        self.cy_spin.setValue(d['cy'])
        self.auto_calc_check.setChecked(d['auto_calc'])

        for w in widgets:
            w.blockSignals(False)

        # Apply state from defaults
        self.fourier_enabled = d['fourier_enabled']
        self.crop_enabled = d['crop_enabled']
        self.dim_roi = d['dim_roi']
        self.crop_x = d['crop_x']
        self.crop_y = d['crop_y']
        self.resample_enabled = d['resample_enabled']
        self.nxmax = d['nxmax']
        self.cx = d['cx']
        self.cy = d['cy']

        # Update dependent widget states
        self.detach_check.setEnabled(d['fourier_enabled'])
        self.orders_check.setEnabled(d['fourier_enabled'] and d['nxmax'] > 0)
        self.resample_spin.setEnabled(d['resample_enabled'])
        self._update_nxmax_ui()

        # Close detached Fourier window if open
        if self.fourier_window:
            self.fourier_window.close()
            self.fourier_window = None

        # Re-run auto calculation if enabled
        if d['auto_calc']:
            self.on_auto_calc_toggled(True)

        self._show_status_message("All settings restored to defaults.", 3)

    def save_images(self):
        """Save current camera and Fourier images to disk."""
        if self._last_camera_image is None:
            self._show_status_message("No image to save.", 3)
            return

        path, filt = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save images", "", "PNG (*.png);;TIFF (*.tif)")
        if not path:
            return

        base, ext = os.path.splitext(path)
        if not ext:
            ext = '.tif' if 'tif' in filt.lower() else '.png'

        cam_path = f"{base}_camera{ext}"
        h, w = self._last_camera_image.shape
        cam_qimg = QtGui.QImage(self._last_camera_image.data, w, h,
                                self._last_camera_image.strides[0],
                                QtGui.QImage.Format_Grayscale8)
        cam_qimg.save(cam_path)

        if self._last_fourier_image is not None:
            fft_path = f"{base}_fourier{ext}"
            fh, fw = self._last_fourier_image.shape
            fft_qimg = QtGui.QImage(self._last_fourier_image.data, fw, fh,
                                    self._last_fourier_image.strides[0],
                                    QtGui.QImage.Format_Grayscale8)
            fft_qimg.save(fft_path)
            self._show_status_message(f"Saved to {base}_*{ext}", 3)
        else:
            self._show_status_message(f"Saved to {cam_path}", 3)

    @property
    def fourier_detached(self):
        return self.fourier_window is not None

    def load_configuration(self):
        """Load acquisition and optical parameters from gui_tomo.conf / config_manip.txt."""
        try:
            home = os.environ.get('HOME', '')
            gui_conf_path = os.path.join(home, ".config/gui_tomo.conf")

            if not os.path.exists(gui_conf_path):
                print(f"Warning: {gui_conf_path} not found. Using defaults.")
                return

            config_dir = extract_val("CHEMIN_CONFIG_PC_ACQUIS", gui_conf_path, dtype=str)

            if config_dir:
                manip_conf_path = os.path.join(config_dir, "config_manip.txt")
                print(f"Loading manip config from: {manip_conf_path}")

                val_roi = extract_val("DIM_ROI", manip_conf_path, int)
                val_nxmax = extract_val("NXMAX", manip_conf_path, int)
                val_cx = extract_val("CIRCLE_CX", manip_conf_path, int)
                val_cy = extract_val("CIRCLE_CY", manip_conf_path, int)

                if val_roi is not None: self.dim_roi = val_roi
                if val_nxmax is not None: self.nxmax = val_nxmax
                if val_cx is not None: self.cx = val_cx
                if val_cy is not None: self.cy = val_cy

                # Optical parameters
                val_na = extract_val("NA", manip_conf_path, float)
                val_n0 = extract_val("N0", manip_conf_path, float)
                val_lambda = extract_val("LAMBDA", manip_conf_path, float)
                val_f_tube = extract_val("F_TUBE", manip_conf_path, float)
                val_f_obj = extract_val("F_OBJ", manip_conf_path, float)
                val_tpcam = extract_val("TPCAM", manip_conf_path, float)
                val_rf = extract_val("RF", manip_conf_path, float)

                if val_na is not None: self.na = val_na
                if val_n0 is not None: self.n0 = val_n0
                if val_lambda is not None: self.lambda0 = val_lambda
                if val_f_tube is not None: self.f_tube = val_f_tube
                if val_f_obj is not None: self.f_obj = val_f_obj
                if val_tpcam is not None: self.tpcam = val_tpcam
                if val_rf is not None: self.rf = val_rf

                if val_nxmax is None:
                    if not self._compute_nxmax():
                        print("Warning: NXMAX not set in config and cannot be computed. "
                              "Order overlays will be hidden until NXMAX is set.")

                self.config_loaded = True
                print(f"Config Loaded: ROI={self.dim_roi}, NXMAX={self.nxmax}, CX={self.cx}, CY={self.cy}")
                print(f"Optical params: NA={self.na}, n_0={self.n0}, lambda_0={self.lambda0}, t_{{p,cam}}={self.tpcam}, R_f={self.rf}")
            else:
                print("Could not find CHEMIN_CONFIG_PC_ACQUIS in gui_tomo.conf")

        except Exception as e:
            print(f"Configuration load failed: {e}")

    def update_frame(self):
        if not self.ia.is_acquiring():
            self.timer.stop()
            self.status_label.setText("Camera stopped grabbing")
            return

        try:
            with self.ia.fetch(timeout=0.5) as buffer:
                component = buffer.payload.components[0]
                img = component.data.reshape(
                    component.height, component.width).copy()

            # Sync exposure slider/spinbox with camera value during auto-exposure
            if self.auto_exp_check.isChecked():
                try:
                    curv = int(self.node_map.ExposureTime.value)
                    if curv != self.exp_spin.value():
                        self.exp_slider.blockSignals(True)
                        self.exp_spin.blockSignals(True)
                        self.exp_slider.setValue(min(curv, self.exp_slider.maximum()))
                        self.exp_spin.setValue(curv)
                        self.exp_slider.blockSignals(False)
                        self.exp_spin.blockSignals(False)
                except Exception as e:
                    print(f"Exposure sync error: {e}")

            if self.crop_enabled:
                img = self._apply_crop(img)

            if self.resample_enabled:
                img = self._apply_resize(img)

            if not self.fourier_enabled:
                disp = self._normalize_image(img)
                self._last_camera_image = disp
                self._last_fourier_image = None
                qimg = QtGui.QImage(disp.data, disp.shape[1], disp.shape[0],
                                    disp.strides[0], QtGui.QImage.Format_Grayscale8)
                self.current_pixmap = QtGui.QPixmap.fromImage(qimg)
            else:
                self._render_fourier(img)

            self._update_label_pixmap()

            self._frame_count += 1
            now = time.monotonic()
            elapsed = now - self._fps_time
            if elapsed >= 1.0:
                self._current_fps = self._frame_count / elapsed
                self._frame_count = 0
                self._fps_time = now
                self._update_status()

        except TimeoutException:
            pass
        except Exception as e:
            print(f"Frame error: {e}")

    def _render_fourier(self, img):
        """Compute 2D Fourier magnitude and render with gamma/clipping adjustments."""
        h, w = img.shape

        disp_orig = self._normalize_image(img)

        f = scipy.fft.fft2(img.astype(np.float32), workers=-1)
        fshift = scipy.fft.fftshift(f)
        magnitude = np.log1p(np.abs(fshift))

        # Percentile clipping and gamma correction
        clip_pct = self.clip_spin.value()
        if clip_pct < 100.0:
            clip_max = np.percentile(magnitude, clip_pct)
            magnitude = np.clip(magnitude, 0, clip_max)

        mx = magnitude.max()
        if mx > 0:
            normalized = magnitude / mx
            gamma = self.gamma_spin.value()
            disp_float = np.power(normalized, gamma) * 255.0
        else:
            disp_float = magnitude

        disp_fft = disp_float.astype(np.uint8)

        self._last_camera_image = disp_orig
        self._last_fourier_image = disp_fft

        if self.fourier_detached and self.fourier_window:
            qimg_orig = QtGui.QImage(disp_orig.data, w, h, disp_orig.strides[0],
                                     QtGui.QImage.Format_Grayscale8)
            self.current_pixmap = QtGui.QPixmap.fromImage(qimg_orig)

            qimg_fft = QtGui.QImage(disp_fft.data, w, h, disp_fft.strides[0],
                                    QtGui.QImage.Format_Grayscale8)
            fft_pixmap = QtGui.QPixmap.fromImage(qimg_fft)

            if self.orders_check.isChecked():
                self._draw_overlays_on_pixmap(fft_pixmap, w, h)

            self.fourier_window.update_fft(fft_pixmap, (w, h))
        else:
            self.fourier_image_size = (w, h)
            combined = np.hstack([disp_orig, disp_fft])
            h_comb, w_comb = combined.shape

            qimg = QtGui.QImage(combined.data, w_comb, h_comb, combined.strides[0],
                                QtGui.QImage.Format_Grayscale8)
            self.current_pixmap = QtGui.QPixmap.fromImage(qimg)

            if self.orders_check.isChecked():
                self._draw_overlays_side_by_side(w, h)

    def _draw_order_overlays(self, painter, img_w, img_h, offset_x=0):
        """Draw 0th, +1, and -1 order circles and crosshairs on the Fourier plane."""
        pen = QtGui.QPen()
        pen.setWidth(2)
        painter.setBrush(QtCore.Qt.NoBrush)

        scale_x, scale_y = self._get_fft_scale_factors(img_w, img_h)

        eff_cx = self.cx * scale_x
        eff_cy = self.cy * scale_y
        # Keep radii data-scaled per axis so overlays match rectangular FFT geometry.
        eff_nxmax_x = self.nxmax * scale_x
        eff_nxmax_y = self.nxmax * scale_y

        center_x = img_w // 2
        center_y = img_h // 2

        # 0th order (DC) — yellow
        pen.setColor(QtGui.QColor("#FFFF00"))
        painter.setPen(pen)
        painter.drawEllipse(QtCore.QPoint(offset_x + center_x, center_y),
                            int(eff_nxmax_x * 2), int(eff_nxmax_y * 2))

        # +1 order (real image) — cyan
        pen.setColor(QtGui.QColor("#00FFFF"))
        painter.setPen(pen)
        painter.drawEllipse(QtCore.QPoint(offset_x + int(eff_cx), int(eff_cy)),
                            int(eff_nxmax_x), int(eff_nxmax_y))

        # -1 order (virtual image) — magenta
        x2 = img_w - int(eff_cx)
        y2 = img_h - int(eff_cy)
        pen.setColor(QtGui.QColor("#FF00FF"))
        painter.setPen(pen)
        painter.drawEllipse(QtCore.QPoint(offset_x + x2, y2),
                            int(eff_nxmax_x), int(eff_nxmax_y))

        # +1 crosshairs (cyan)
        rx = int(eff_nxmax_x * 1.15)
        ry = int(eff_nxmax_y * 1.15)
        pen.setColor(QtGui.QColor("#00FFFF"))
        painter.setPen(pen)
        painter.drawLine(offset_x + int(eff_cx) - rx, int(eff_cy),
                         offset_x + int(eff_cx) + rx, int(eff_cy))
        painter.drawLine(offset_x + int(eff_cx), int(eff_cy) - ry,
                         offset_x + int(eff_cx), int(eff_cy) + ry)

        # -1 crosshairs (magenta)
        pen.setColor(QtGui.QColor("#FF00FF"))
        painter.setPen(pen)
        painter.drawLine(offset_x + x2 - rx, y2,
                         offset_x + x2 + rx, y2)
        painter.drawLine(offset_x + x2, y2 - ry,
                         offset_x + x2, y2 + ry)

    def _draw_overlays_on_pixmap(self, pixmap, img_w, img_h):
        """Draw order overlays on a standalone Fourier pixmap."""
        if not self.nxmax or self.nxmax <= 0:
            return
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self._draw_order_overlays(painter, img_w, img_h)
        painter.end()

    def _draw_overlays_side_by_side(self, img_w, img_h):
        """Draw order overlays on the right half of the combined image."""
        if not self.nxmax or self.nxmax <= 0:
            return
        painter = QtGui.QPainter(self.current_pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        self._draw_order_overlays(painter, img_w, img_h, offset_x=img_w)
        painter.end()

    def _normalize_image(self, img):
        if img.dtype == np.uint8:
            return img
        if np.issubdtype(img.dtype, np.integer):
            info = np.iinfo(img.dtype)
            if info.max > 255:
                return (img / info.max * 255).astype(np.uint8)
        return img.astype(np.uint8)

    def _get_fft_scale_factors(self, img_w, img_h):
        """Return CX/CY-to-FFT scaling factors for x and y axes."""
        if self.dim_roi <= 0:
            return 1.0, 1.0
        # CX/CY are defined in DIM_ROI coordinates; use per-axis scaling for non-square FFTs.
        return img_w / self.dim_roi, img_h / self.dim_roi

    def _apply_crop(self, img):
        crop_size = self.dim_roi
        h, w = img.shape[:2]

        if h <= crop_size and w <= crop_size:
            return img

        start_x = min(self.crop_x, max(0, w - crop_size))
        start_y = min(self.crop_y, max(0, h - crop_size))
        end_y = start_y + min(crop_size, h)
        end_x = start_x + min(crop_size, w)

        return img[start_y:end_y, start_x:end_x].copy()

    def _apply_resize(self, img):
        target_size = self.resample_spin.value()
        h, w = img.shape[:2]

        if h == target_size and w == target_size:
            return img

        zoom_factor = target_size / max(h, w)
        return _ndimage_zoom(img, zoom_factor, order=1).astype(img.dtype)

    def _update_label_pixmap(self):
        if self.current_pixmap:
            self.label.setPixmap(self.current_pixmap.scaled(
                self.label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def _init_exposure_controls(self):
        try:
            min_exp = int(self.node_map.ExposureTime.min)
            max_exp = int(self.node_map.ExposureTime.max)
            cur_exp = int(self.node_map.ExposureTime.value)
            self.exp_slider.blockSignals(True)
            self.exp_spin.blockSignals(True)
            self.exp_slider.setRange(min_exp, min(max_exp, 50000))
            self.exp_spin.setRange(min_exp, max_exp)
            self.exp_slider.setValue(cur_exp)
            self.exp_spin.setValue(cur_exp)
            self.exp_slider.blockSignals(False)
            self.exp_spin.blockSignals(False)
            is_auto = self.node_map.ExposureAuto.value != 'Off'
            self.auto_exp_check.setChecked(is_auto)
            self.exp_slider.setEnabled(not is_auto)
            self.exp_spin.setEnabled(not is_auto)
        except Exception as e:
            print(f"Exposure init error: {e}")

    def on_exp_slider_changed(self, val):
        self.exp_spin.blockSignals(True)
        self.exp_spin.setValue(val)
        self.exp_spin.blockSignals(False)
        try:
            self.node_map.ExposureTime.value = int(val)
        except Exception as e:
            print(f"Exposure set error: {e}")

    def on_exp_spin_changed(self, val):
        self.exp_slider.blockSignals(True)
        self.exp_slider.setValue(min(val, self.exp_slider.maximum()))
        self.exp_slider.blockSignals(False)
        try:
            self.node_map.ExposureTime.value = int(val)
        except Exception as e:
            print(f"Exposure set error: {e}")

    def toggle_exposure(self, checked):
        try:
            if checked:
                self.node_map.ExposureAuto.value = 'Continuous'
                self.exp_slider.setEnabled(False)
                self.exp_spin.setEnabled(False)
            else:
                self.node_map.ExposureAuto.value = 'Off'
                self.exp_slider.setEnabled(True)
                self.exp_spin.setEnabled(True)
        except Exception as e:
            print(f"Auto exposure toggle error: {e}")

    def toggle_fourier(self, checked):
        self.fourier_enabled = checked
        self.orders_check.setEnabled(checked and self.nxmax > 0)
        self.detach_check.setEnabled(checked)
        if not checked and self.fourier_window:
            self.fourier_window.close()
            self.fourier_window = None

    def toggle_detach(self, checked):
        if checked:
            if not self.fourier_window:
                self.fourier_window = FourierWindow(self)
                self.fourier_window.show()
        else:
            if self.fourier_window:
                self.fourier_window.close()
                self.fourier_window = None

    def on_pick_toggled(self, checked):
        self.pick_mode = checked

    def toggle_crop(self, checked):
        self.crop_enabled = checked

    def toggle_resample(self, checked):
        self.resample_enabled = checked
        self.resample_spin.setEnabled(checked)

    def on_crop_size_changed(self, value):
        self.dim_roi = value
        if self.auto_calc_check.isChecked():
            self.calculate_all_theo()

    def on_crop_offset_changed(self):
        self.crop_x = self.crop_x_spin.value()
        self.crop_y = self.crop_y_spin.value()

    def on_nxmax_changed(self, value):
        self.nxmax = value
        self._update_nxmax_ui()

    def on_cx_changed(self, value):
        self.cx = value

    def on_cy_changed(self, value):
        self.cy = value

    def set_circle_position(self, cx, cy):
        self.cx = cx
        self.cy = cy
        self.cx_spin.blockSignals(True)
        self.cy_spin.blockSignals(True)
        self.cx_spin.setValue(cx)
        self.cy_spin.setValue(cy)
        self.cx_spin.blockSignals(False)
        self.cy_spin.blockSignals(False)
        self._sync_theo_combo_to_cxy()

    def on_main_label_clicked(self, event):
        """Map click on the right half of the side-by-side view to CX/CY."""
        if not self.pick_mode:
            return
        if not self.fourier_enabled or self.fourier_detached:
            return
        if event.button() != QtCore.Qt.LeftButton:
            return
        if self.fourier_image_size[0] == 0:
            return

        label_size = self.label.size()
        pixmap = self.label.pixmap()
        if not pixmap:
            return

        pm_size = pixmap.size()
        offset_x = (label_size.width() - pm_size.width()) // 2
        offset_y = (label_size.height() - pm_size.height()) // 2

        click_x = event.position().toPoint().x() - offset_x
        click_y = event.position().toPoint().y() - offset_y

        if click_x < 0 or click_y < 0 or click_x >= pm_size.width() or click_y >= pm_size.height():
            return

        img_w, img_h = self.fourier_image_size
        combined_w = img_w * 2

        scale_x = combined_w / pm_size.width()
        scale_y = img_h / pm_size.height()

        orig_x = click_x * scale_x
        orig_y = click_y * scale_y

        if orig_x >= img_w:
            fft_x = orig_x - img_w
            fft_y = orig_y

            scale_x, scale_y = self._get_fft_scale_factors(img_w, img_h)

            new_cx = int(fft_x / scale_x)
            new_cy = int(fft_y / scale_y)
            
            self.set_circle_position(new_cx, new_cy)

    def on_gamma_slider_changed(self, val):
        gamma = val / 100.0
        self.gamma_spin.blockSignals(True)
        self.gamma_spin.setValue(gamma)
        self.gamma_spin.blockSignals(False)

    def on_gamma_spin_changed(self, val):
        self.gamma_slider.blockSignals(True)
        slider_val = min(int(val * 100), self.gamma_slider.maximum())
        self.gamma_slider.setValue(slider_val)
        self.gamma_slider.blockSignals(False)

    def reset_params(self):
        if self.auto_calc_check.isChecked():
            self.calculate_all_theo()
        else:
            self.nxmax_spin.setValue(self._default_nxmax)
            self.set_circle_position(self._default_cx, self._default_cy)

    def on_auto_calc_toggled(self, checked):
        if checked:
            self.calculate_all_theo()
        self.theo_cxy_combo.setEnabled(checked)


    def _compute_nxmax(self):
        """
        Compute NXMAX from the optical setup parameters.
        Returns True if successful, False if parameters are missing.

        G = f_tube / f_obj,  G_t = G / R_f
        pixel_holo = t_{p,cam} / G_t
        R_Ewald = DIM_ROI * pixel_holo * n_0 / lambda_0
        NXMAX = R_Ewald * NA / n_0
        """
        params = {'NA': self.na, 'N0': self.n0, 'LAMBDA': self.lambda0,
                  'F_TUBE': self.f_tube, 'F_OBJ': self.f_obj,
                  'TPCAM': self.tpcam, 'RF': self.rf}
        missing = [k for k, v in params.items() if v is None]
        if missing:
            print(f"Warning: Cannot compute NXMAX, missing optical parameters: {', '.join(missing)}")
            return False
        g = self.f_tube / self.f_obj
        gt = g / self.rf
        pixel_holo_nm = self.tpcam / gt * 1e9
        r_ewald = self.dim_roi * pixel_holo_nm * self.n0 / (self.lambda0 * 1e9)
        nxmax = r_ewald * self.na / self.n0
        self.nxmax = int(round(nxmax))
        print(f"NXMAX computed: {nxmax:.1f} -> {self.nxmax}  "
              f"(G={g:.1f}, pixel_holo={pixel_holo_nm:.2f} nm, R_Ewald={r_ewald:.1f})")
        return True

    def calculate_nxmax_theo(self):
        try:
            if not self._compute_nxmax():
                self._show_status_message("Cannot compute NXMAX: missing optical parameters", 5)
                return
            self.nxmax_spin.setValue(self.nxmax)
        except Exception as e:
            print(f"Error calculating NXMAX: {e}")

    def calculate_all_theo(self):
        """Calculate both NXMAX and theoretical CX/CY."""
        self.calculate_nxmax_theo()
        self.calculate_theoretical_cxy()

    def calculate_theoretical_cxy(self):
        """Calculate four theoretical +1 order positions from sign permutations of the carrier offset."""
        if not self.nxmax or self.nxmax <= 0:
            if not self._compute_nxmax():
                self._show_status_message("Cannot compute theoretical CX/CY: NXMAX not set", 5)
                return

        try:
            # 3×NXMAX distance from DC, projected onto each axis (45° carrier assumption)
            carrier_coord = 3.0 * self.nxmax / np.sqrt(2.0)
            f1 = int(round(self.dim_roi / 2.0 - carrier_coord))
            f2 = int(round(self.dim_roi / 2.0 + carrier_coord))

            print(f"Theoretical CX/CY computed: carrier_coord={carrier_coord:.2f} -> f1={f1}, f2={f2} "
                  f"(NXMAX={self.nxmax}, DIM_ROI={self.dim_roi})")

            self.theoretical_cxys = [
                (f1, f1),
                (f1, f2),
                (f2, f1),
                (f2, f2)
            ]

            self.theo_cxy_combo.blockSignals(True)
            self.theo_cxy_combo.clear()
            self.theo_cxy_combo.addItem("+1 Order")
            for idx, (cx, cy) in enumerate(self.theoretical_cxys):
                self.theo_cxy_combo.addItem(f"({cx}, {cy})")
            self.theo_cxy_combo.blockSignals(False)

            # Auto-select the first candidate to update the spinboxes
            self.theo_cxy_combo.setCurrentIndex(1)
            self._show_status_message("Theoretical +1 positions computed.", 3)

        except Exception as e:
            print(f"Error calculating theoretical CX/CY: {e}")

    def _sync_theo_combo_to_cxy(self):
        """Set combo box selection to match the current CX/CY, or index 0 if no match."""
        self.theo_cxy_combo.blockSignals(True)
        matched = 0
        for i, (cx, cy) in enumerate(self.theoretical_cxys):
            if cx == self.cx and cy == self.cy:
                matched = i + 1
                break
        self.theo_cxy_combo.setCurrentIndex(matched)
        self.theo_cxy_combo.blockSignals(False)

    def on_theo_cxy_selected(self, index):
        """Handle selection of a theoretical CX/CY profile."""
        if index > 0 and index <= len(self.theoretical_cxys):
            cx, cy = self.theoretical_cxys[index - 1]
            self.set_circle_position(cx, cy)

    def _update_nxmax_ui(self):
        has_nxmax = self.nxmax > 0
        if has_nxmax:
            self.orders_check.setEnabled(self.fourier_enabled)
            self.orders_check.setToolTip('Overlay diffraction orders (DC, +1, -1) on Fourier plane')
        else:
            self.orders_check.setEnabled(False)
            self.orders_check.setToolTip('Set NXMAX > 0 to enable')

    def _show_status_message(self, msg, seconds=5):
        self.status_label.setText(msg)
        self._status_msg_until = time.monotonic() + seconds

    def _update_status(self):
        if time.monotonic() < self._status_msg_until:
            return
        model = self.model_name or "Unknown"

        config_str = "config loaded" if self.config_loaded else "defaults"
        fft_str = " | Fourier on" if self.fourier_enabled else ""
        self.status_label.setText(
            f"{model} | {self._current_fps:.1f} fps | {config_str}{fft_str}"
        )

    def _set_initial_size(self):
        try:
            cam_width = self.node_map.Width.value
            cam_height = self.node_map.Height.value
        except Exception:
            cam_width = 1024
            cam_height = 1024

        aspect_ratio = cam_width / cam_height

        screen = QtWidgets.QApplication.primaryScreen().availableGeometry()
        max_width = int(screen.width() * 0.7)
        max_height = int(screen.height() * 0.7)

        controls_estimate = 80  # px, approximate height of UI controls

        if aspect_ratio > max_width / (max_height - controls_estimate):
            win_width = max_width
            win_height = int(max_width / aspect_ratio) + controls_estimate
        else:
            win_height = max_height
            win_width = int((max_height - controls_estimate) * aspect_ratio)

        self.resize(win_width, win_height)


def _find_cti_files():
    """Discover GenTL producer (.cti) files from environment and CLI arguments."""
    import argparse
    import glob

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--cti', action='append', default=[],
                        help='Path to a GenTL producer .cti file')
    args, _ = parser.parse_known_args()

    cti_files = list(args.cti)

    for env_var in ('GENICAM_GENTL64_PATH', 'GENICAM_GENTL32_PATH'):
        path = os.environ.get(env_var, '')
        if path:
            for directory in path.split(os.pathsep):
                cti_files.extend(glob.glob(os.path.join(directory, '*.cti')))

    return cti_files


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    try:
        h = Harvester()
        cti_paths = _find_cti_files()
        if not cti_paths:
            raise RuntimeError(
                "No GenTL producer (.cti) files found. Install a GenTL producer "
                "(e.g. Basler pylon, FLIR Spinnaker, Allied Vision Vimba) and set "
                "GENICAM_GENTL64_PATH, or pass --cti <path>."
            )
        for p in cti_paths:
            h.add_file(p)
        h.update()

        if not h.device_info_list:
            raise RuntimeError(
                "No camera detected! Check the connection and ensure a GenTL "
                "producer is installed and GENICAM_GENTL64_PATH is set."
            )

        model = h.device_info_list[0].model
        print(f"Found camera: {model}")
        ia = h.create(0)

        viewer = CameraViewer(ia, h, model)
        viewer.show()

        sys.exit(app.exec())

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setText("Application Failed to Start")
        msg.setInformativeText(str(e))
        msg.setWindowTitle("Error")
        msg.exec()
        sys.exit(1)