# SkullMatrix 💀

A real-time 3D ASCII skull renderer for the terminal.

SkullMatrix loads a 3D skull model and renders it as animated ANSI/ASCII art with lighting, colors, glitches, CRT effects, trails, sparks, background effects, skull rain, and more.

## Requirements

- Linux
- Python 3
- ANSI-compatible terminal
- Kitty recommended

## Installation

Clone the repository:

    git clone https://github.com/AnungUnRama133/Skullmatrix.git
    cd Skullmatrix

Run the installer:

    chmod +x install.sh
    ./install.sh

Then launch:

    skull

The installer creates the Python virtual environment, installs the required dependencies, and creates the `skull` command in `~/.local/bin`.

## Controls

### Main Skull

| Key | Effect |
|---|---|
| J | Toggle main 3D skull on/off |
| R | Reset rotation |
| Space | Pause/resume |
| + / - | Rotation speed |
| Q | Quit |

### Skull Effects

| Key | Effect |
|---|---|
| 0 | Toggle all visual effects |
| 1 | Red |
| 2 | Purple |
| 3 | Green |
| 4 | Cyan |
| 5 | Multi-color |
| 6 | Full-skull rainbow |
| 7 | Continuous 360° rotation |
| E | Electric |
| C | CRT scanlines |
| H | Hologram |
| T | Ghost trail |
| S | Sparks |
| B | Breathing |
| V | Reverse spin |
| F | Spin burst |
| D | Disintegration |
| X | Screen shake |
| G | Glitch |
| P | Pulse |
| L | Moving light |
| [ / ] | Light speed |

### Background

| Key | Effect |
|---|---|
| 8 | Toggle background effect |
| 9 | Cycle background colors |

### Skull Rain

| Key | Effect |
|---|---|
| K | Toggle real mini 3D skull rain |
| M | Toggle simple `☠` skull rain |
| J + K | Main skull + 3D skull rain |
| J + M | Main skull + simple skull rain |
| K + M | Both skull rain effects |
| J + K + M | Main skull + both rain effects |

The skull rain effects are independent, so they can be combined in different ways.

## Terminal Resizing

SkullMatrix automatically detects terminal size changes while running.

The renderer dynamically adjusts the skull projection so the skull remains properly scaled when the terminal window is resized.

## Dependencies

- NumPy
- Numba
- Pillow
- llvmlite

Exact versions are listed in `requirements.txt`.

The installer automatically creates an isolated Python virtual environment and installs the required versions.

## Project Features

- Real-time 3D skull rendering
- Software-based 3D lighting
- ANSI/ASCII shading
- Multiple skull color modes
- Full-skull rainbow mode
- Moving light
- Glitch effects
- Electric effects
- CRT scanlines
- Hologram flicker
- Ghost trails
- Sparks
- Breathing effect
- Reverse spin
- Spin bursts
- Disintegration
- Screen shake
- Animated background
- Background color cycling
- Real mini 3D skull rain
- Simple `☠` skull rain
- Live terminal resizing
- Portable OBJ model discovery
- Portable installation script

## Notes

The repository includes the skull model, texture, and material files required by the renderer.

Kitty is recommended for the intended visual experience.

The renderer is designed to run locally and does not require an internet connection after installation and dependency setup.

## License

No license has currently been specified.
