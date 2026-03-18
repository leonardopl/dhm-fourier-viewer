# DHM Fourier Viewer

Real-time camera viewer with live Fourier-plane display for off-axis digital
holographic microscopy (DHM).

Displays the camera feed alongside its 2D Fourier magnitude spectrum.
Overlay markers indicate the 0th-order (DC), +1 (real image), and −1
(virtual image) diffraction orders.

Built on [Harvesters](https://github.com/genicam/harvesters) (the GenICam
committee's Python acquisition library) and PySide6. Works with any
GenICam-compliant camera that has a GenTL producer (`.cti` file), including
Basler, FLIR, Allied Vision, IDS, Ximea, etc.

## Features

- Side-by-side or detached Fourier magnitude view (gamma correction and
  percentile-based intensity clipping)
- Overlay circles: 0th order/DC in yellow (2×NXMAX), +1 order (real image) in
  cyan, and -1 order (virtual image) in magenta, with crosshairs
- Click-to-set +1 order position (CX, CY) on the Fourier panel
- Compute the theoretical ±1 diffraction order positions from the cutoff radius
- Automatic NXMAX computation from NA, system magnification, pixel pitch, and
  relay factor
- Crop to configurable ROI (size and position), optional spatial resampling
- Manual and continuous auto-exposure
- Save camera and Fourier images to PNG/TIFF

## Frequency cutoff (NXMAX)

When NXMAX is not provided in the config file the program computes it from
the optical chain:

$$
G_t = \frac{f_{\text{tube}}}{f_{\text{obj}} \cdot R_f}, \qquad
R_{\text{Ewald}} = \frac{\text{DIM\\_ROI} \cdot t_{p,\text{cam}}}{G_t}
  \cdot \frac{n_0}{\lambda_0}, \qquad
\text{NXMAX} = R_{\text{Ewald}} \cdot \frac{\text{NA}}{n_0}
$$

where $t_{p,\text{cam}}$ is the camera pixel pitch, $R_f$ the relay factor,
$n_0$ the immersion medium index, and $\lambda_0$ the vacuum wavelength.
${\text{DIM\\_ROI}}$ is the width of the cropped region used for the Fourier
transform. $R_{\text{Ewald}}$ is the Ewald sphere radius $n_0/\lambda_0$
expressed in Fourier pixels of the cropped hologram.

The theoretical carrier offset used to populate the +1 order combo box is
a distance of $3 \times \text{NXMAX}$ from DC, projected onto
each Fourier axis assuming a 45° carrier angle:

$$d_{\text{carrier}} = \frac{3 \cdot \text{NXMAX}}{\sqrt{2}}$$

The four sign permutations of this offset produce the candidate positions
shown in the combo box.

See Cuche et al. (1999) and Colomb et al. (2006) for background on the
off-axis DHM reconstruction geometry.

## Requirements

- Python >= 3.10
- A GenTL producer (`.cti` file) for your camera. Common sources:
  - [Basler pylon SDK](https://www.baslerweb.com/en/downloads/software/)
  - [FLIR Spinnaker SDK](https://www.flir.com/products/spinnaker-sdk/)
  - [Allied Vision Vimba X](https://www.alliedvision.com/en/products/software/vimba-x-sdk/)
- Set `GENICAM_GENTL64_PATH` to the directory containing the `.cti` file,
  or pass `--cti <path>` on the command line.
- **Linux USB3:** udev rules are typically needed for camera access
  (e.g. install the Basler pylon SDK or copy the vendor-provided rules).
- Only standard SFNC features (`ExposureTime`, `ExposureAuto`, `Width`,
  `Height`) are used.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python3 dhm_fourier_viewer.py
```

Or use the wrapper script:

```bash
./run.sh
```

### Keyboard shortcuts

| Key | Action              |
|-----|---------------------|
| F   | Toggle Fourier view |
| S   | Save images         |

## Configuration

The program reads `~/.config/gui_tomo.conf`, which must contain a
`CHEMIN_CONFIG_PC_ACQUIS` entry pointing to a directory with
`config_manip.txt`.

Recognised keys in `config_manip.txt`:

| Key         | Type  | Description                            |
|-------------|-------|----------------------------------------|
| DIM_ROI     | int   | ROI crop size (px)                     |
| NXMAX       | int   | Frequency cutoff radius (px)           |
| CIRCLE_CX   | int   | +1 order X position (px)               |
| CIRCLE_CY   | int   | +1 order Y position (px)               |
| NA          | float | Numerical aperture                     |
| N0          | float | Medium refractive index                |
| LAMBDA      | float | Vacuum wavelength (m)                  |
| F_TUBE      | float | Tube lens focal length (m)             |
| F_OBJ       | float | Objective focal length (m)             |
| TPCAM       | float | Camera pixel pitch (m)                 |
| RF          | float | Relay factor                           |

If NXMAX is absent it is computed from the optical parameters above. When
neither NXMAX nor the full optical parameter set is available, overlay
markers stay disabled until NXMAX is set in the GUI.

UI settings (ROI size, overlay positions, etc.) persist between
sessions via QSettings and can be reset with the Defaults button.
Exposure time and auto-exposure mode are stored by the camera and are
not affected by the Defaults button.

## References

1. E. Cuche, P. Marquet, and C. Depeursinge, "Simultaneous amplitude-contrast
   and quantitative phase-contrast microscopy by numerical reconstruction of
   Fresnel off-axis holograms," Appl. Opt. **38**(34), 6994–7001 (1999).
2. T. Colomb, E. Cuche, F. Charrière, J. Kühn, N. Aspert, F. Montfort,
   P. Marquet, and C. Depeursinge, "Automatic procedure for aberration
   compensation in digital holographic microscopy and applications to
   specimen shape compensation," Appl. Opt. **45**(4), 851–863 (2006).