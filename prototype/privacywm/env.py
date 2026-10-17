"""Dependency-light first-person grid world, with a software ray caster.

Appearance NEVER enters physics, reward, oracle actions, or scene sampling.
This is an engineering fallback, not MiniWorld or a full 3D human simulator.
"""
from collections import deque
import colorsys
import numpy as np
from PIL import Image, ImageDraw

N_IDENTITIES = 30
ACTIONS = ("forward", "left", "right", "interact", "wait")
DIRECTIONS = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])


def identity_sequence(mode, length, episode, seed=0, k=8):
    # Separate stream: identity draws cannot perturb geometry or actions.
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode, 713]))
    start = (episode + seed) % N_IDENTITIES
    if mode == "S0":
        return np.full(length, start, dtype=np.int64)
    if mode == "S1":
        # Random block phase makes the change clock stationary across episodes.
        phase = int(rng.integers(k))
        return rng.integers(N_IDENTITIES, size=(length + phase) // k + 1)[
            (np.arange(length) + phase) // k]
    if mode == "S2":
        return rng.integers(N_IDENTITIES, size=length)
    if mode == "S3":
        return (np.arange(length) + start) % N_IDENTITIES
    raise ValueError(f"Unknown identity schedule: {mode}")


class World:
    size = 64

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.walls = np.zeros((9, 9), dtype=bool)
        self.walls[[0, -1], :] = True
        self.walls[:, [0, -1]] = True
        self.walls[2:7, 4] = True
        self.walls[4, 4] = False
        cells = [(x, y) for y in range(1, 8) for x in range(1, 8)
                 if not self.walls[y, x]]
        indices = self.rng.choice(len(cells), 3, replace=False)
        self.agent, self.person, self.goal = [tuple(cells[i]) for i in indices]
        self.heading = int(self.rng.integers(4))
        self.person_heading = int(self.rng.integers(4))
        self.identity = 0
        self.carrying = False
        self.success = False
        self.collisions = 0
        self.steps = 0

    def blocked(self, cell):
        x, y = cell
        return (x < 0 or y < 0 or x >= 9 or y >= 9 or
                self.walls[y, x] or tuple(cell) == self.person)

    def step(self, action):
        reward = -0.01
        if action == 0:
            dest = tuple(np.array(self.agent) + DIRECTIONS[self.heading])
            if self.blocked(dest):
                self.collisions += 1
                reward -= 0.1
            else:
                self.agent = dest
        elif action == 1:
            self.heading = (self.heading - 1) % 4
        elif action == 2:
            self.heading = (self.heading + 1) % 4
        elif action == 3:
            front = tuple(np.array(self.agent) + DIRECTIONS[self.heading])
            if front == self.person and not self.carrying:
                self.carrying = True
                reward += 1
        elif action != 4:
            raise ValueError(action)
        if self.carrying and self.agent == self.goal and not self.success:
            self.success = True
            reward += 3
        self.steps += 1
        return reward, self.success

    def oracle(self):
        """Shortest path in (x,y,heading); knows physical state, never identity."""
        start = (*self.agent, self.heading)
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            (x, y, h), path = queue.popleft()
            front = tuple(np.array((x, y)) + DIRECTIONS[h])
            if self.carrying and (x, y) == self.goal:
                return path[0] if path else 4
            if not self.carrying and front == self.person:
                return path[0] if path else 3
            candidates = [((x, y, (h - 1) % 4), 1),
                          ((x, y, (h + 1) % 4), 2)]
            if not self.blocked(front):
                candidates.append(((*front, h), 0))
            for state, action in candidates:
                if state not in seen:
                    seen.add(state)
                    queue.append((state, path + [action]))
        return 4

    def metadata(self):
        # Four by four bins covering the seven by seven walkable interior.
        px, py = self.person
        pos = min(3, (py - 1) * 4 // 7) * 4 + min(3, (px - 1) * 4 // 7)
        return {"identity": int(self.identity), "position": int(pos),
                "agent": list(self.agent), "person": list(self.person),
                "heading": self.heading, "person_heading": self.person_heading,
                "carrying": self.carrying, "success": self.success}

    def render(self, identity=None, mask_person=False, person_override=None):
        """64x64 RGB. Ray depth occludes billboards. No metadata in pixels."""
        ident = self.identity if identity is None else identity
        s = self.size
        pixels = np.zeros((s, s, 3), dtype=np.uint8)
        pixels[:32] = [43, 52, 66]
        for y in range(32, s):
            pixels[y] = [64 + (y - 32), 66 + (y - 32), 70 + (y - 32)]
        forward = DIRECTIONS[self.heading].astype(float)
        right = np.array([-forward[1], forward[0]])
        origin = np.array(self.agent, dtype=float) + 0.5
        depth = np.full(s, 20.0)
        rays = forward + ((np.arange(s) + 0.5) / s - 0.5)[:, None] * 1.4 * right
        distances = np.arange(0.03, 15, 0.045)
        points = np.floor(origin + distances[:, None, None] * rays[None]).astype(int)
        gx, gy = points[:, :, 0], points[:, :, 1]
        hit = self.walls[np.clip(gy, 0, 8), np.clip(gx, 0, 8)]
        depth = distances[hit.argmax(axis=0)]
        for col, dist in enumerate(depth):
            height = min(64, int(42 / dist))
            top, bottom = max(0, 32 - height // 2), min(s, 32 + height // 2)
            shade = 0.55 + 0.45 / (1 + dist * 0.15)
            tint = np.array([147, 158, 169]) * shade
            pixels[top:bottom, col] = tint.astype(np.uint8)
            if top < bottom:
                pixels[top:top + 1, col] = tint * 0.7
        canvas = Image.fromarray(pixels)
        visible = False
        objects = [(self.goal, "goal")]
        if not mask_person:
            objects.append((self.person if person_override is None else person_override, "person"))
        objects.sort(key=lambda item: -np.linalg.norm(np.array(item[0]) + 0.5 - origin))
        for loc, kind in objects:
            delta = np.array(loc) + 0.5 - origin
            distance = float(delta @ forward)
            lateral = float(delta @ right)
            if distance < 0.25:
                continue
            center = int(32 + lateral / distance * s / 1.4)
            height = min(160, int(39 / distance))
            width = max(2, int(height * (0.38 + 0.025 * (ident // 6))))
            top, bottom = 32 - height // 2, 32 + height // 2
            if kind == "person":
                # The selection mask uses the physical center, never identity-dependent
                # silhouette width. Otherwise visible-only probes have different samples.
                visible = bool(0 <= center < s and distance < depth[min(s - 1, max(0, center))]
                               and height >= 3)
            layer = Image.new("RGBA", (s, s))
            draw = ImageDraw.Draw(layer)
            if kind == "goal":
                draw.rectangle((center - 3, bottom - 4, center + 3, bottom), fill=(74, 221, 166, 255))
            else:
                color = tuple(int(v * 255) for v in colorsys.hsv_to_rgb((ident % 6) / 6, 0.7, 0.87))
                draw.ellipse((center - width // 3, top, center + width // 3, top + height // 4), fill=(211, 182, 153, 255))
                draw.rectangle((center - width // 2, top + height // 4, center + width // 2,
                                top + height * 3 // 4), fill=(*color, 255))
                spacing = 2 + ident // 6
                for yy in range(top + height // 3, top + height * 3 // 4, spacing):
                    draw.line((center - width // 2, yy, center + width // 2, yy), fill=(239, 231, 220, 255))
                draw.rectangle((center - width // 3, top + height * 3 // 4,
                                center + width // 3, bottom), fill=(35, 42, 56, 255))
                # Orientation cue varies independently of identity.
                side = 1 if self.person_heading % 2 else -1
                draw.line((center, top + height // 3, center + side * max(2, width // 2),
                           top + height // 3), fill=(15, 20, 30, 255), width=2)
                if not self.carrying:
                    # Fixed physical anchor, unaffected by identity body width.
                    box_w = max(2, height // 5)
                    draw.rectangle((center, top + height // 2, center + box_w,
                                    top + height * 2 // 3), fill=(250, 208, 65, 255))
            arr = np.array(layer)
            valid = (distance < depth)[None, :]
            alpha = (arr[:, :, 3] > 0) & valid
            base = np.array(canvas)
            base[alpha] = arr[:, :, :3][alpha]
            canvas = Image.fromarray(base)
        return np.array(canvas), visible
