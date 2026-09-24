import os
import glob
import time
import shutil
import random
import sys
import select
import termios
import tty
import numpy as np
import numba

# Use the actual terminal dimensions.
_terminal = shutil.get_terminal_size(fallback=(110, 42))

WIDTH = max(40, _terminal.columns)
HEIGHT = max(20, _terminal.lines)

CHARS = " .,:;irsXA253hMHGS#9B&@"

# Scale the skull with the terminal while preserving the
# original character-cell aspect compensation.
#
# Reference renderer size: 110x42
# Original projection: SX=48, SY=23.5
#
# Fit the skull to whichever terminal dimension is limiting.
scale = min(
    WIDTH / 110.0,
    HEIGHT / 42.0
)

SX = 48.0 * scale
SY = 23.5 * scale

# ------------------------------------------------------------
# Find OBJ
# ------------------------------------------------------------

obj_files = glob.glob(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "**", "*.obj"),
    recursive=True
)

if not obj_files:
    print("No OBJ file found.")
    raise SystemExit

OBJ = obj_files[0]

# ------------------------------------------------------------
# Load OBJ
# ------------------------------------------------------------

vertices = []
faces = []

with open(OBJ, "r", errors="ignore") as f:

    for line in f:

        if line.startswith("v "):

            p = line.split()

            vertices.append([
                float(p[1]),
                float(p[2]),
                float(p[3])
            ])

        elif line.startswith("f "):

            parts = line.split()[1:]
            face = []

            for p in parts:
                face.append(
                    int(p.split("/")[0]) - 1
                )

            if len(face) >= 3:

                for i in range(1, len(face) - 1):

                    faces.append([
                        face[0],
                        face[i],
                        face[i + 1]
                    ])

vertices = np.array(vertices, dtype=np.float32)
faces = np.array(faces, dtype=np.int32)

# ------------------------------------------------------------
# Center and normalize
# ------------------------------------------------------------

mins = vertices.min(axis=0)
maxs = vertices.max(axis=0)

center = (mins + maxs) / 2.0

vertices -= center

model_size = np.max(maxs - mins)

vertices /= model_size

# ------------------------------------------------------------
# Smooth vertex normals
# ------------------------------------------------------------

vertex_normals = np.zeros_like(vertices)

for face in faces:

    a, b, c = face

    v0 = vertices[a]
    v1 = vertices[b]
    v2 = vertices[c]

    normal = np.cross(
        v1 - v0,
        v2 - v0
    )

    length = np.sqrt(
        normal[0] ** 2
        + normal[1] ** 2
        + normal[2] ** 2
    )

    if length > 0:
        normal /= length

    vertex_normals[a] += normal
    vertex_normals[b] += normal
    vertex_normals[c] += normal

lengths = np.sqrt(
    vertex_normals[:, 0] ** 2
    + vertex_normals[:, 1] ** 2
    + vertex_normals[:, 2] ** 2
)

lengths[lengths == 0] = 1

vertex_normals /= lengths[:, None]

# ------------------------------------------------------------
# Light
# ------------------------------------------------------------

light = np.array(
    [-0.35, -1.0, 0.55],
    dtype=np.float32
)

light /= np.sqrt(
    light[0] ** 2
    + light[1] ** 2
    + light[2] ** 2
)


# ============================================================
# NUMBA RENDERER
# ============================================================

@numba.njit(cache=True, fastmath=True)
def render_arrays(
    angle,
    vertices,
    faces,
    vertex_normals,
    light,
    width,
    height,
    sx,
    sy
):

    rotated = np.empty_like(vertices)

    c = np.cos(angle)
    s = np.sin(angle)

    for i in range(vertices.shape[0]):

        x = vertices[i, 0]
        y = vertices[i, 1]

        rotated[i, 0] = x * c - y * s
        rotated[i, 1] = x * s + y * c
        rotated[i, 2] = vertices[i, 2]

    shade = np.zeros(
        (height, width),
        dtype=np.float32
    )

    depth = np.full(
        (height, width),
        -9999.0,
        dtype=np.float32
    )

    for f in range(faces.shape[0]):

        ia = faces[f, 0]
        ib = faces[f, 1]
        ic = faces[f, 2]

        v0x = rotated[ia, 0]
        v0y = rotated[ia, 2]
        v0z = -rotated[ia, 1]

        v1x = rotated[ib, 0]
        v1y = rotated[ib, 2]
        v1z = -rotated[ib, 1]

        v2x = rotated[ic, 0]
        v2y = rotated[ic, 2]
        v2z = -rotated[ic, 1]

        sx0 = width / 2.0 + v0x * sx
        sy0 = height / 2.0 - v0y * sy

        sx1 = width / 2.0 + v1x * sx
        sy1 = height / 2.0 - v1y * sy

        sx2 = width / 2.0 + v2x * sx
        sy2 = height / 2.0 - v2y * sy

        min_x = max(
            0,
            int(min(sx0, sx1, sx2))
        )

        max_x = min(
            width - 1,
            int(max(sx0, sx1, sx2))
        )

        min_y = max(
            0,
            int(min(sy0, sy1, sy2))
        )

        max_y = min(
            height - 1,
            int(max(sy0, sy1, sy2))
        )

        if min_x > max_x or min_y > max_y:
            continue

        area = (
            (sx1 - sx0) * (sy2 - sy0)
            - (sy1 - sy0) * (sx2 - sx0)
        )

        if abs(area) < 1e-6:
            continue

        n0x = vertex_normals[ia, 0]
        n0y = vertex_normals[ia, 1]
        n0z = vertex_normals[ia, 2]

        n1x = vertex_normals[ib, 0]
        n1y = vertex_normals[ib, 1]
        n1z = vertex_normals[ib, 2]

        n2x = vertex_normals[ic, 0]
        n2y = vertex_normals[ic, 1]
        n2z = vertex_normals[ic, 2]

        for y in range(min_y, max_y + 1):

            py = y + 0.5

            for x in range(min_x, max_x + 1):

                px = x + 0.5

                w0 = (
                    (sx1 - px) * (sy2 - py)
                    - (sy1 - py) * (sx2 - px)
                ) / area

                w1 = (
                    (sx2 - px) * (sy0 - py)
                    - (sy2 - py) * (sx0 - px)
                ) / area

                w2 = 1.0 - w0 - w1

                if w0 < 0.0 or w1 < 0.0 or w2 < 0.0:
                    continue

                z = (
                    w0 * v0z
                    + w1 * v1z
                    + w2 * v2z
                )

                if z <= depth[y, x]:
                    continue

                nx = (
                    n0x * w0
                    + n1x * w1
                    + n2x * w2
                )

                ny = (
                    n0y * w0
                    + n1y * w1
                    + n2y * w2
                )

                nz = (
                    n0z * w0
                    + n1z * w1
                    + n2z * w2
                )

                normal_length = np.sqrt(
                    nx * nx
                    + ny * ny
                    + nz * nz
                )

                if normal_length > 0.0:

                    nx /= normal_length
                    ny /= normal_length
                    nz /= normal_length

                brightness = (
                    nx * light[0]
                    + ny * light[1]
                    + nz * light[2]
                )

                brightness = (
                    brightness + 0.15
                ) / 1.15

                if brightness < 0.0:
                    brightness = 0.0

                elif brightness > 1.0:
                    brightness = 1.0

                brightness = brightness ** 1.05

                shade[y, x] = brightness
                depth[y, x] = z

    return shade, depth


# ============================================================
# CAVITY ENHANCEMENT
# ============================================================

def enhance_cavities(shade, depth):

    height, width = depth.shape

    for y in range(1, height - 1):

        for x in range(1, width - 1):

            if depth[y, x] <= -9990:
                continue

            neighbours = (
                depth[y - 1, x],
                depth[y + 1, x],
                depth[y, x - 1],
                depth[y, x + 1]
            )

            total = 0.0
            count = 0

            for n in neighbours:

                if n > -9990:

                    total += n
                    count += 1

            if count == 0:
                continue

            local_average = total / count

            cavity = max(
                0.0,
                local_average - depth[y, x]
            )

            factor = min(
                0.48,
                cavity * 5.5
            )

            shade[y, x] *= (
                1.0 - factor
            )

            if cavity > 0.055:

                shade[y, x] *= 0.72


# ============================================================
# ASCII + CYBER GLITCH
# ============================================================

previous_screen = None

def make_frame(shade):

    global previous_screen

    frame_height, frame_width = shade.shape

    # --------------------------------------------------------
    # Animated full-screen background
    # --------------------------------------------------------

    background_active = background_enabled

    background_mask = np.zeros(
        (frame_height, frame_width),
        dtype=np.bool_
    )

    if background_active:

        screen = np.full(
            (frame_height, frame_width),
            " ",
            dtype="<U1"
        )

        bg_chars = ".:*+·'`"
        bg_time = time.monotonic()

        for y in range(frame_height):

            for x in range(frame_width):

                # Several overlapping waves create a moving
                # digital atmosphere across the whole terminal.
                wave = (
                    np.sin(
                        bg_time * 1.8
                        + x * 0.19
                        + y * 0.11
                    )
                    + np.sin(
                        bg_time * 0.9
                        - x * 0.07
                        + y * 0.23
                    )
                ) * 0.5

                # Mostly sparse, with occasional brighter areas.
                threshold = 0.48 + random.uniform(-0.12, 0.12)

                if wave > threshold:

                    index = int(
                        (
                            wave + 1.0
                        )
                        / 2.0
                        * (len(bg_chars) - 1)
                    )

                    index = max(
                        0,
                        min(
                            len(bg_chars) - 1,
                            index
                        )
                    )

                    screen[y, x] = bg_chars[index]
                    background_mask[y, x] = True

                # Random digital particles.
                elif random.random() < 0.018:

                    screen[y, x] = random.choice(
                        bg_chars
                    )
                    background_mask[y, x] = True

        # Moving scanline.
        scanline = int(
            (bg_time * 8.0) % HEIGHT
        )

        for x in range(frame_width):

            if random.random() < 0.75:
                screen[scanline, x] = random.choice(
                    ".:=+"
                )

        # Occasional background signal burst.
        if random.random() < 0.08:

            burst_y = random.randrange(frame_height)

            for x in range(frame_width):

                if random.random() < 0.20:
                    screen[burst_y, x] = random.choice(
                        "-=+*#"
                    )

    else:

        screen = np.full(
            (frame_height, frame_width),
            " ",
            dtype="<U1"
        )

    # --------------------------------------------------------
    # Simple skull rain
    # --------------------------------------------------------

    if simple_skull_rain_enabled:

        if not hasattr(make_frame, "_simple_rain"):
            make_frame._simple_rain = []

        simple_rain = make_frame._simple_rain

        target_count = max(
            5,
            min(
                24,
                frame_width // 7
            )
        )

        while len(simple_rain) < target_count:

            simple_rain.append({
                "x": random.uniform(
                    0,
                    max(1, frame_width - 1)
                ),
                "y": random.uniform(
                    -frame_height,
                    -1.0
                ),
                "speed": random.uniform(
                    0.10,
                    0.35
                ),
                "drift": random.uniform(
                    -0.03,
                    0.03
                )
            })

        if len(simple_rain) > target_count:
            del simple_rain[target_count:]

        for drop in simple_rain:

            drop["y"] += drop["speed"]
            drop["x"] += drop["drift"]

            if (
                drop["y"] > frame_height
                or drop["x"] < -1
                or drop["x"] >= frame_width
            ):

                drop["x"] = random.uniform(
                    0,
                    max(1, frame_width - 1)
                )

                drop["y"] = random.uniform(
                    -12.0,
                    -1.0
                )

                drop["speed"] = random.uniform(
                    0.10,
                    0.35
                )

                drop["drift"] = random.uniform(
                    -0.03,
                    0.03
                )

            draw_x = int(drop["x"])
            draw_y = int(drop["y"])

            if (
                0 <= draw_x < frame_width
                and 0 <= draw_y < frame_height
            ):

                screen[
                    draw_y,
                    draw_x
                ] = "☠"

                background_mask[
                    draw_y,
                    draw_x
                ] = False

    # --------------------------------------------------------
    # Mini 3D skull rain
    # --------------------------------------------------------

    if skull_rain_enabled:

        if not hasattr(make_frame, "_rain"):
            make_frame._rain = []

        rain = make_frame._rain

        # Find the visible part of the actual 3D skull.
        visible = shade > 0.025

        if np.any(visible):

            ys, xs = np.where(visible)

            skull_min_x = int(xs.min())
            skull_max_x = int(xs.max())
            skull_min_y = int(ys.min())
            skull_max_y = int(ys.max())

            skull_crop = shade[
                skull_min_y:skull_max_y + 1,
                skull_min_x:skull_max_x + 1
            ]

            crop_h, crop_w = skull_crop.shape

            # Keep the falling skulls small enough to look
            # like individual objects rather than copies
            # of the main skull.
            rain_h = 9
            rain_w = max(
                5,
                min(
                    15,
                    int(
                        crop_w
                        * (rain_h / max(1, crop_h))
                    )
                )
            )

            target_count = max(
                3,
                min(
                    12,
                    frame_width // 18
                )
            )

            while len(rain) < target_count:

                rain.append({
                    "x": random.uniform(
                        -rain_w,
                        max(0, frame_width - 1)
                    ),
                    "y": random.uniform(
                        -frame_height,
                        -2.0
                    ),
                    "speed": random.uniform(
                        0.10,
                        0.24
                    ),
                    "drift": random.uniform(
                        -0.018,
                        0.018
                    )
                })

            if len(rain) > target_count:
                del rain[target_count:]

            # Resize the real skull crop into a tiny sprite.
            sprite = np.zeros(
                (rain_h, rain_w),
                dtype=np.float32
            )

            for sy in range(rain_h):

                source_y = int(
                    sy
                    * crop_h
                    / rain_h
                )

                source_y = min(
                    crop_h - 1,
                    max(0, source_y)
                )

                for sx in range(rain_w):

                    source_x = int(
                        sx
                        * crop_w
                        / rain_w
                    )

                    source_x = min(
                        crop_w - 1,
                        max(0, source_x)
                    )

                    sprite[sy, sx] = (
                        skull_crop[
                            source_y,
                            source_x
                        ]
                    )

            for drop in rain:

                drop["y"] += drop["speed"]
                drop["x"] += drop["drift"]

                if (
                    drop["y"] > frame_height
                    or drop["x"] < -rain_w
                    or drop["x"] >= frame_width
                ):

                    drop["x"] = random.uniform(
                        0,
                        max(0, frame_width - rain_w)
                    )

                    drop["y"] = random.uniform(
                        -15.0,
                        -2.0
                    )

                    drop["speed"] = random.uniform(
                        0.10,
                        0.24
                    )

                    drop["drift"] = random.uniform(
                        -0.018,
                        0.018
                    )

                base_x = int(drop["x"])
                base_y = int(drop["y"])

                for sy in range(rain_h):

                    for sx in range(rain_w):

                        value = sprite[sy, sx]

                        if value <= 0.12:
                            continue

                        draw_x = base_x + sx
                        draw_y = base_y + sy

                        if (
                            0 <= draw_x < frame_width
                            and 0 <= draw_y < frame_height
                        ):

                            index = int(
                                value
                                * (len(CHARS) - 1)
                            )

                            index = max(
                                0,
                                min(
                                    len(CHARS) - 1,
                                    index
                                )
                            )

                            screen[
                                draw_y,
                                draw_x
                            ] = CHARS[index]

                            background_mask[
                                draw_y,
                                draw_x
                            ] = False

    levels = len(CHARS) - 1

    # Very subtle brightness flicker
    flicker = random.uniform(
        0.97,
        1.02
    )

    # Slow cyberpunk energy pulse
    if pulse_enabled or all_effects:

        pulse = (
            np.sin(time.monotonic() * 2.2)
            + 1.0
        ) * 0.5

        pulse = 0.94 + pulse * 0.06

    else:

        pulse = 0.94

    for y in range(frame_height):

        for x in range(frame_width):

            value = shade[y, x] * flicker

            value = max(
                0.0,
                min(1.0, value)
            )

            index = int(
                value * levels
            )

            # Only replace the background where the skull
            # actually has visible geometry.
            if (
                main_skull_enabled
                and value > 0.025
            ):
                screen[y, x] = CHARS[index]
                background_mask[y, x] = False

    # --------------------------------------------------------
    # Rare scanline glitch
    # --------------------------------------------------------

    if glitches_enabled and random.random() < 0.035:

        y = random.randrange(
            4,
            HEIGHT - 4
        )

        shift = random.choice(
            [-3, -2, 2, 3]
        )

        original = screen[y].copy()

        for x in range(frame_width):

            source = x - shift

            if 0 <= source < WIDTH:

                screen[y, x] = original[source]

    # --------------------------------------------------------
    # Very rare tiny character corruption
    # --------------------------------------------------------

    if glitches_enabled and random.random() < 0.015:

        y = random.randrange(
            5,
            HEIGHT - 5
        )

        x = random.randrange(
            10,
            WIDTH - 10
        )

        for i in range(
            random.randint(2, 5)
        ):

            xx = x + i

            if 0 <= xx < WIDTH:

                screen[y, xx] = random.choice(
                    "01#@$%"
                )

    # --------------------------------------------------------
    # Rotation-reactive digital tear
    # --------------------------------------------------------

    # Glitch becomes slightly stronger near rotation limits
    rotation_factor = abs(angle) / 0.75

    tear_chance = (
        0.006
        + rotation_factor * 0.018
    )

    if glitches_enabled and random.random() < tear_chance:

        if HEIGHT > 22:
            start_y = random.randrange(
                10,
                HEIGHT - 12
            )
        else:
            start_y = max(
                0,
                (HEIGHT - 1) // 2
            )

        tear_height = random.randint(
            1,
            3
        )

        shift_amount = int(
            5 + rotation_factor * 5
        )

        shift = random.choice(
            [-shift_amount, shift_amount]
        )

        for y in range(
            start_y,
            min(
                HEIGHT,
                start_y + tear_height
            )
        ):

            original = screen[y].copy()

            for x in range(frame_width):

                source = x - shift

                if 0 <= source < WIDTH:

                    screen[y, x] = original[source]

    # --------------------------------------------------------
    # Holographic ghost-frame afterimage
    # --------------------------------------------------------

    if (
        'shift_amount' in locals()
        and previous_screen is not None
        and glitches_enabled
        and random.random() < 0.75
    ):

        ghost_rows = random.randint(
            1,
            3
        )

        for _ in range(ghost_rows):

            gy = random.randrange(
                8,
                HEIGHT - 8
            )

            shift = random.choice(
                [-4, -3, 3, 4]
            )

            for x in range(frame_width):

                source = x - shift

                if (
                    0 <= source < WIDTH
                    and previous_screen[gy, source] != " "
                    and random.random() < 0.28
                ):

                    # Only leave fragments, not a full duplicate skull.
                    screen[gy, x] = previous_screen[gy, source]

    # --------------------------------------------------------
    # Cyan signal burst
    # --------------------------------------------------------

    # Only trigger during a digital tear.
    # Creates a few tiny fragments around the skull silhouette.
    if glitches_enabled and 'shift_amount' in locals() and random.random() < 0.65:

        fragments = random.randint(3, 7)

        for _ in range(fragments):

            y = random.randrange(
                7,
                HEIGHT - 7
            )

            x = random.randrange(
                5,
                WIDTH - 5
            )

            # Look for the edge of the skull.
            if screen[y, x] != " ":
                continue

            nearby = False

            for yy in range(
                max(0, y - 1),
                min(HEIGHT, y + 2)
            ):

                for xx in range(
                    max(0, x - 1),
                    min(WIDTH, x + 2)
                ):

                    if screen[yy, xx] != " ":
                        nearby = True

            if nearby:

                length = random.randint(
                    1,
                    4
                )

                direction = random.choice(
                    [-1, 1]
                )

                for i in range(length):

                    xx = x + i * direction

                    if 0 <= xx < WIDTH:
                        screen[y, xx] = random.choice(
                            ".:-=+"
                        )

    # ========================================================
    # EXTRA EFFECT PACK
    # ========================================================

    now = time.monotonic()

    # --------------------------------------------------------
    # Breathing / holographic size shimmer
    # --------------------------------------------------------

    if breathing_enabled or all_effects:

        breath = (
            np.sin(now * 2.0)
            + 1.0
        ) * 0.5

        # Stronger breathing shimmer.
        # The skull periodically becomes visibly brighter/denser.
        if breath > 0.65:
            chance = 0.035 + (breath - 0.65) * 0.20

            for y in range(frame_height):
                for x in range(frame_width):
                    if screen[y, x] != " ":
                        if random.random() < chance:
                            screen[y, x] = random.choice(
                                "@#&$%MH"
                            )

    # --------------------------------------------------------
    # Electric arcs
    # --------------------------------------------------------

    if (electric_enabled or all_effects) and random.random() < 0.28:

        for _ in range(random.randint(1, 3)):

            y = random.randrange(5, HEIGHT - 5)
            x = random.randrange(5, WIDTH - 5)

            for _ in range(random.randint(3, 9)):

                if not (
                    0 <= x < WIDTH
                    and 0 <= y < HEIGHT
                ):
                    break

                if screen[y, x] == " ":
                    screen[y, x] = random.choice(
                        "/\\|+-"
                    )

                x += random.choice([-1, 0, 1])
                y += random.choice([-1, 0, 1])

    # --------------------------------------------------------
    # Spark particles
    # --------------------------------------------------------

    if sparks_enabled or all_effects:

        for _ in range(random.randint(4, 9)):

            y = random.randrange(4, HEIGHT - 4)
            x = random.randrange(4, WIDTH - 4)

            nearby = False

            for yy in range(
                max(0, y - 2),
                min(HEIGHT, y + 3)
            ):

                for xx in range(
                    max(0, x - 2),
                    min(WIDTH, x + 3)
                ):

                    if screen[yy, xx] != " ":
                        nearby = True
                        break

                if nearby:
                    break

            if nearby:
                screen[y, x] = random.choice(
                    ".*+x"
                )

    # --------------------------------------------------------
    # Extra ghost trail
    # --------------------------------------------------------

    if (
        trail_enabled
        and previous_screen is not None
    ):

        for _ in range(2):

            gy = random.randrange(
                5,
                HEIGHT - 5
            )

            shift = random.choice(
                [-6, -5, 5, 6]
            )

            for x in range(frame_width):

                source = x - shift

                if (
                    0 <= source < WIDTH
                    and previous_screen[gy, source] != " "
                    and random.random() < 0.18
                ):

                    screen[gy, x] = previous_screen[
                        gy,
                        source
                    ]

    # --------------------------------------------------------
    # CRT scanlines
    # --------------------------------------------------------

    if crt_enabled or all_effects:

        for y in range(0, HEIGHT, 2):

            for x in range(frame_width):

                if (
                    screen[y, x] != " "
                    and random.random() < 0.38
                ):
                    screen[y, x] = "."

    # --------------------------------------------------------
    # Hologram flicker
    # --------------------------------------------------------

    if hologram_enabled or all_effects:

        if random.random() < 0.12:

            for y in range(frame_height):

                for x in range(frame_width):

                    if (
                        screen[y, x] != " "
                        and random.random() < 0.12
                    ):
                        screen[y, x] = " "

    # --------------------------------------------------------
    # Digital disintegration
    # --------------------------------------------------------

    if disintegrate_enabled or all_effects:

        for y in range(frame_height):

            for x in range(frame_width):

                if (
                    screen[y, x] != " "
                    and random.random() < 0.018
                ):

                    screen[y, x] = random.choice(
                        " .:;01"
                    )

        if random.random() < 0.20:

            for _ in range(random.randint(2, 5)):

                y = random.randrange(
                    5,
                    HEIGHT - 5
                )

                x = random.randrange(
                    5,
                    WIDTH - 5
                )

                screen[y, x] = random.choice(
                    "01#@$%"
                )

    # --------------------------------------------------------
    # Screen shake
    # --------------------------------------------------------

    if (shake_enabled or all_effects) and random.random() < 0.055:

        shift = random.choice(
            [-2, -1, 1, 2]
        )

        shaken = np.full(
            (frame_height, frame_width),
            " ",
            dtype="<U1"
        )

        for y in range(frame_height):

            for x in range(frame_width):

                source = x - shift

                if 0 <= source < WIDTH:
                    shaken[y, x] = screen[y, source]

        screen = shaken

    background_colors = [
        "\033[2;34m",
        "\033[2;35m",
        "\033[2;31m",
        "\033[2;32m",
        "\033[2;33m",
        "\033[2;36m",
        "\033[0m",
    ]

    background_color = background_colors[background_color_mode]

    # --------------------------------------------------------
    # Full-skull rainbow color mode
    # --------------------------------------------------------
    # The ENTIRE skull uses one color at a time.
    # The color smoothly cycles through the spectrum.
    # No spatial stripes.

    if color_mode == "rainbow":

        previous_screen = screen.copy()

        rainbow = [
            "\033[91m",  # red
            "\033[93m",  # yellow
            "\033[92m",  # green
            "\033[96m",  # cyan
            "\033[94m",  # blue
            "\033[95m",  # purple
        ]

        # Slow continuous color movement.
        position = (now * 0.8) % len(rainbow)

        index = int(position) % len(rainbow)

        # Use the current color for the entire skull.
        color = rainbow[index]

        output = []

        for y, row in enumerate(screen):

            rendered = ""

            for x, char in enumerate(row):

                if (
                    background_active
                    and background_mask[y, x]
                    and char != " "
                ):
                    rendered += (
                        background_color
                        + char
                    )
                else:
                    rendered += (
                        color
                        + char
                    )

            output.append(
                rendered
                + "\033[0m"
            )

        previous_screen = screen.copy()

        return "\n".join(output)

    # --------------------------------------------------------
    # Switch subtly between normal and bright selected color
    if color_mode == "multi":
        multi_colors = [
            ("\033[91m", "\033[31m"),  # red
            ("\033[95m", "\033[35m"),  # purple
            ("\033[94m", "\033[34m"),  # blue
            ("\033[96m", "\033[36m"),  # cyan
            ("\033[92m", "\033[32m"),  # green
            ("\033[93m", "\033[33m"),  # yellow
        ]

        previous_screen = screen.copy()

        output = []

        for y, row in enumerate(screen):

            rendered = ""

            for x, char in enumerate(row):

                if (
                    background_active
                    and background_mask[y, x]
                    and char != " "
                ):
                    rendered += (
                        background_color
                        + char
                    )

                else:
                    bright, normal = multi_colors[
                        (x + y) % len(multi_colors)
                    ]

                    rendered += (
                        bright if pulse > 0.975 else normal
                    ) + char

            output.append(
                rendered
                + "\033[0m"
            )

        previous_screen = screen.copy()

        return "\n".join(output)

    if color_mode == "red":
        color = "\033[91m" if pulse > 0.975 else "\033[31m"
    elif color_mode == "purple":
        color = "\033[95m" if pulse > 0.975 else "\033[35m"
    elif color_mode == "green":
        color = "\033[92m" if pulse > 0.975 else "\033[32m"
    else:
        color = "\033[96m" if pulse > 0.975 else "\033[36m"

    previous_screen = screen.copy()

    output = []

    for y, row in enumerate(screen):

        rendered = ""

        for x, char in enumerate(row):

            if (
                background_active
                and background_mask[y, x]
                and char != " "
            ):
                rendered += (
                    background_color
                    + char
                )
            else:
                rendered += (
                    color
                    + char
                )

        output.append(
            rendered
            + "\033[0m"
        )

    return "\n".join(output)


# ============================================================
# INTERACTIVE CONTROLS
# ============================================================

rotation_speed = 0.035
rotation_paused = False
glitches_enabled = True
pulse_enabled = True
moving_light = True
light_speed = 1.8
quit_requested = False
color_mode = "cyan"
all_effects = False
background_enabled = False
background_color_mode = 0

main_skull_enabled = True
skull_rain_enabled = False
simple_skull_rain_enabled = False

# Extra visual effects
rainbow_enabled = False
electric_enabled = False
crt_enabled = False
hologram_enabled = False
trail_enabled = False
sparks_enabled = False
breathing_enabled = False
reverse_spin = False
spin_burst = 0.0
full_rotate = False
disintegrate_enabled = False
shake_enabled = False

old_terminal_settings = termios.tcgetattr(sys.stdin)
tty.setcbreak(sys.stdin.fileno())


def read_keys():
    global rotation_speed
    global rotation_paused
    global glitches_enabled
    global pulse_enabled
    global moving_light
    global light_speed
    global angle
    global direction
    global quit_requested
    global color_mode
    global all_effects
    global background_enabled
    global background_color_mode
    global main_skull_enabled
    global skull_rain_enabled
    global simple_skull_rain_enabled
    global rainbow_enabled
    global electric_enabled
    global crt_enabled
    global hologram_enabled
    global trail_enabled
    global sparks_enabled
    global breathing_enabled
    global reverse_spin
    global spin_burst
    global full_rotate
    global disintegrate_enabled
    global shake_enabled

    while select.select([sys.stdin], [], [], 0)[0]:

        key = sys.stdin.read(1)

        if key in ("q", "Q"):
            quit_requested = True

        elif key == " ":
            rotation_paused = not rotation_paused

        elif key in ("+", "="):
            rotation_speed = min(
                0.12,
                rotation_speed + 0.005
            )

        elif key in ("-", "_"):
            rotation_speed = max(
                0.005,
                rotation_speed - 0.005
            )

        elif key in ("g", "G"):
            glitches_enabled = not glitches_enabled

        elif key in ("p", "P"):
            pulse_enabled = not pulse_enabled

        elif key in ("l", "L"):
            moving_light = not moving_light

        elif key in ("j", "J"):
            main_skull_enabled = not main_skull_enabled

        elif key in ("k", "K"):
            skull_rain_enabled = not skull_rain_enabled

        elif key in ("m", "M"):
            simple_skull_rain_enabled = not simple_skull_rain_enabled

        elif key == "[":
            light_speed = max(
                0.2,
                light_speed - 0.2
            )

        elif key == "]":
            light_speed = min(
                6.0,
                light_speed + 0.2
            )

        elif key == "1":
            color_mode = "red"

        elif key == "2":
            color_mode = "purple"

        elif key == "3":
            color_mode = "green"

        elif key == "4":
            color_mode = "cyan"

        elif key == "5":
            color_mode = "multi"
        elif key == "0":
            all_effects = not all_effects

        elif key == "6":
            color_mode = "rainbow"

        elif key in ("e", "E"):
            electric_enabled = not electric_enabled

        elif key in ("c", "C"):
            crt_enabled = not crt_enabled

        elif key in ("h", "H"):
            hologram_enabled = not hologram_enabled

        elif key in ("t", "T"):
            trail_enabled = not trail_enabled

        elif key in ("s", "S"):
            sparks_enabled = not sparks_enabled

        elif key in ("b", "B"):
            breathing_enabled = not breathing_enabled

        elif key in ("v", "V"):
            reverse_spin = not reverse_spin

        elif key in ("f", "F"):
            spin_burst = 1.0

        elif key == "7":
            full_rotate = not full_rotate

        elif key == "8":
            background_enabled = not background_enabled

        elif key == "9":
            background_color_mode = (
                background_color_mode + 1
            ) % 7

        elif key in ("d", "D"):
            disintegrate_enabled = not disintegrate_enabled

        elif key in ("x", "X"):
            shake_enabled = not shake_enabled

        elif key in ("r", "R"):
            angle = 0.0
            direction = 1.0


# ============================================================
# ANIMATION
# ============================================================


angle = 0.0
direction = 1.0

print(
    "\033[?1049h"
    "\033[2J"
    "\033[H"
    "\033[?25l",
    end="",
    flush=True
)

try:

    while True:

        # ------------------------------------------------
        # Live terminal resize detection
        # ------------------------------------------------

        terminal = shutil.get_terminal_size(
            fallback=(110, 42)
        )

        new_width = max(40, terminal.columns)
        new_height = max(20, terminal.lines)

        if (
            new_width != WIDTH
            or new_height != HEIGHT
        ):
            WIDTH = new_width
            HEIGHT = new_height

            scale = min(
                WIDTH / 110.0,
                HEIGHT / 42.0
            )

            SX = 48.0 * scale
            SY = 23.5 * scale

        read_keys()

        if quit_requested:
            break

        if moving_light or all_effects:
            light_angle = time.monotonic() * light_speed

            light = np.array(
                [
                    np.sin(light_angle) * 0.75,
                    -1.0,
                    0.35 + np.cos(light_angle) * 0.45
                ],
                dtype=np.float32
            )

            light /= np.sqrt(
                light[0] ** 2
                + light[1] ** 2
                + light[2] ** 2
            )

        shade, depth = render_arrays(
            angle,
            vertices,
            faces,
            vertex_normals,
            light,
            WIDTH,
            HEIGHT,
            SX,
            SY
        )

        enhance_cavities(
            shade,
            depth
        )

        frame = make_frame(
            shade
        )

        print(
            "\033[H" + frame,
            end="",
            flush=True
        )

        if not rotation_paused:

            # ------------------------------------------------
            # Motion system
            # ------------------------------------------------

            effective_direction = -1.0 if reverse_spin else direction

            # Spin burst: rapidly accelerate, then smoothly decay.
            if spin_burst > 0.0:
                burst_multiplier = 1.0 + (spin_burst * 5.0)
                spin_burst = max(
                    0.0,
                    spin_burst - 0.025
                )
            else:
                burst_multiplier = 1.0

            step = (
                rotation_speed
                * effective_direction
                * burst_multiplier
            )

            # Full rotation mode ignores the old
            # back-and-forth limits and continuously
            # rotates through 360 degrees.
            if full_rotate:
                angle += step

                if angle > np.pi * 2:
                    angle -= np.pi * 2

                elif angle < -np.pi * 2:
                    angle += np.pi * 2

            else:
                angle += step

                if angle >= 0.75:
                    direction = -1.0

                elif angle <= -0.75:
                    direction = 1.0

        time.sleep(0.015)

except KeyboardInterrupt:

    pass

finally:

    termios.tcsetattr(
        sys.stdin,
        termios.TCSADRAIN,
        old_terminal_settings
    )

    print(
        "\033[?25h"
        "\033[?1049l",
        end="",
        flush=True
    )
