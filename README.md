# SkullMatrix 💀

A real-time 3D ASCII skull renderer for the terminal.

SkullMatrix loads a 3D skull model and renders it as animated ANSI/ASCII art with lighting, colors, glitches, CRT effects, trails, sparks, background effects, and more.

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

The installer creates the Python virtual environment, installs the required dependencies, and creates the skull command.

## Controls

| Key | Effect |
|---|---|
| 0 | Toggle all effects |
| 1 | Red |
| 2 | Purple |
| 3 | Green |
| 4 | Cyan |
| 5 | Multi-color |
| 6 | Full-skull rainbow |
| 7 | 360° rotation |
| 8 | Background on/off |
| 9 | Background color |
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
| Space | Pause |
| + / - | Rotation speed |
| R | Reset rotation |
| Q | Quit |

## Dependencies

- NumPy
- Numba
- Pillow
- llvmlite

Exact versions are listed in requirements.txt.

## Notes

The repository includes the skull model and texture required by the renderer.

Kitty is recommended for the intended visual experience.

## License

No license has currently been specified.
