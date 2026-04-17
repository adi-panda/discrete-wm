"""
Interactive play mode for the discrete diffusion world model.

Lets you control an agent in the world model's imagination, seeing
predicted frames in real time.

Usage:
    # With pygame window (needs display / X11 forwarding):
    python play_interactive.py --checkpoint /vast/adi/discrete_wm/checkpoints/model_best.pt --mode pygame

    # Terminal mode (no display needed, saves frames as GIF):
    python play_interactive.py --checkpoint /vast/adi/discrete_wm/checkpoints/model_best.pt --mode terminal

    # Jupyter mode (use from notebook):
    #   from play_interactive import InteractiveWorldModel
    #   wm = InteractiveWorldModel(checkpoint_path, tokenizer_path, data_path)
    #   wm.run_notebook()

Controls (pygame mode):
    Arrow keys: LEFT / RIGHT
    Space:      FIRE
    No input:   NOOP
    R:          Reset to a random dataset frame
    S:          Save current rollout as GIF
    Q / ESC:    Quit

Controls (terminal mode):
    a / left:   LEFT
    d / right:  RIGHT
    space / f:  FIRE
    enter:      NOOP
    r:          Reset
    s:          Save GIF
    q:          Quit
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import imageio

from models.discrete_diffusion import DiscreteWorldModel, PatchVQVAE
from utils import AtariTokenizedDataset


# Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
ACTION_NAMES = {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"}


class InteractiveWorldModel:
    """Wraps the discrete diffusion world model for interactive use."""

    def __init__(self, checkpoint_path, tokenizer_path, data_path,
                 gen_steps=8, temperature=0.9, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gen_steps = gen_steps
        self.temperature = temperature

        # Load tokenizer
        tok_ckpt = torch.load(tokenizer_path, map_location=self.device)
        tok_args = tok_ckpt['args']
        self.tokenizer = PatchVQVAE(
            patch_size=tok_args['patch_size'],
            num_channels=3,
            vocab_size=tok_args['vocab_size'],
            embed_dim=tok_args['embed_dim'],
        ).to(self.device)
        self.tokenizer.load_state_dict(tok_ckpt['model'])
        self.tokenizer.eval()

        # Load dataset (for initial frames)
        self.dataset = AtariTokenizedDataset(data_path)

        # Load model
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        model_args = ckpt.get('args', {})
        grid_size = int(math.sqrt(self.dataset.prev_tokens.shape[1]))

        self.model = DiscreteWorldModel(
            vocab_size=tok_args['vocab_size'],
            grid_h=grid_size,
            grid_w=grid_size,
            d_model=model_args.get('d_model', 512),
            n_layers=model_args.get('n_layers', 8),
            n_heads=model_args.get('n_heads', 8),
            n_actions=self.dataset.num_actions,
            cond_dim=model_args.get('d_model', 512),
        ).to(self.device)
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self.n_actions = self.dataset.num_actions
        self.current_tokens = None
        self.frame_history = []
        self.step_count = 0

        print(f"Model loaded: step {ckpt.get('step', '?')}, "
              f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M params")
        print(f"Gen steps: {gen_steps}, temperature: {temperature}")
        print(f"Actions: {ACTION_NAMES}")

    def reset(self, idx=None):
        """Reset to a frame from the dataset."""
        if idx is None:
            idx = np.random.randint(len(self.dataset))
        prev_tokens, _, _ = self.dataset[idx]
        self.current_tokens = prev_tokens.unsqueeze(0).to(self.device).long()
        self.frame_history = [self._decode_frame()]
        self.step_count = 0
        return self.frame_history[-1]

    def step(self, action):
        """Take one step: predict next frame given action."""
        action_tensor = torch.tensor([action], device=self.device)
        with torch.no_grad():
            self.current_tokens = self.model.generate(
                self.current_tokens, action_tensor,
                num_steps=self.gen_steps,
                temperature=self.temperature,
                device=self.device,
            )
        frame = self._decode_frame()
        self.frame_history.append(frame)
        self.step_count += 1
        return frame

    def _decode_frame(self):
        """Decode current tokens to a uint8 numpy image [H, W, 3]."""
        with torch.no_grad():
            frames = self.tokenizer.decode_tokens(self.current_tokens)
            frame = ((frames[0].clamp(-1, 1) + 1) / 2 * 255).byte().cpu().numpy()
        return frame

    def save_gif(self, path="rollout.gif", fps=10):
        """Save frame history as GIF."""
        if not self.frame_history:
            print("No frames to save.")
            return
        imageio.mimsave(path, self.frame_history, fps=fps)
        print(f"Saved {len(self.frame_history)} frames to {path}")

    # --- Pygame mode ---

    def run_pygame(self, window_size=512, fps=10):
        """Interactive play with a pygame window."""
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((window_size, window_size + 40))
        pygame.display.set_caption("Discrete Diffusion World Model")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("mono", 16)

        frame = self.reset()
        running = True

        while running:
            action = 0  # NOOP by default

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        frame = self.reset()
                        continue
                    elif event.key == pygame.K_s:
                        self.save_gif(f"rollout_interactive_{int(time.time())}.gif")
                        continue

            # Read held keys for movement
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                action = 3  # LEFT
            elif keys[pygame.K_RIGHT]:
                action = 2  # RIGHT
            elif keys[pygame.K_SPACE]:
                action = 1  # FIRE

            # Step the world model
            frame = self.step(action)

            # Render frame
            img = np.array(
                __import__('PIL.Image', fromlist=['Image']).Image.fromarray(frame)
                .resize((window_size, window_size), __import__('PIL.Image', fromlist=['Image']).Image.NEAREST)
            )
            surface = pygame.surfarray.make_surface(img.transpose(1, 0, 2))
            screen.fill((0, 0, 0))
            screen.blit(surface, (0, 40))

            # HUD
            hud = f"Step: {self.step_count:4d}  Action: {ACTION_NAMES.get(action, '?'):>5s}  [R]eset [S]ave [Q]uit"
            screen.blit(font.render(hud, True, (255, 255, 255)), (5, 10))

            pygame.display.flip()
            clock.tick(fps)

        pygame.quit()
        print(f"Session ended after {self.step_count} steps.")

    # --- Terminal mode ---

    def run_terminal(self, max_steps=200):
        """Interactive play in the terminal (no display needed). Saves GIF at the end."""
        frame = self.reset()
        print(f"\nTerminal mode — type action each step, rollout saved as GIF on quit.")
        print(f"  a=LEFT  d=RIGHT  f/space=FIRE  enter=NOOP  r=reset  s=save  q=quit\n")

        while self.step_count < max_steps:
            try:
                user = input(f"[step {self.step_count:3d}] action> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if user in ('q', 'quit'):
                break
            elif user in ('r', 'reset'):
                frame = self.reset()
                print("  → Reset to new frame")
                continue
            elif user in ('s', 'save'):
                self.save_gif(f"rollout_terminal_{int(time.time())}.gif")
                continue
            elif user in ('a', 'left'):
                action = 3
            elif user in ('d', 'right'):
                action = 2
            elif user in ('f', 'space', ' '):
                action = 1
            else:
                action = 0

            t0 = time.time()
            frame = self.step(action)
            dt = (time.time() - t0) * 1000
            print(f"  → {ACTION_NAMES[action]} ({dt:.0f}ms)")

        out_path = f"rollout_terminal_{int(time.time())}.gif"
        self.save_gif(out_path)

    # --- Jupyter notebook mode ---

    def run_live(self, fps=10, scale=4, start_idx=None):
        """Auto-play the world model as a continuous video.

        Buttons set the *current* action; it stays in effect until another
        action button is clicked. Video auto-advances at ``fps`` while Play
        is toggled on. FIRE is one-shot (reverts to NOOP after one frame).
        """
        return LivePlayer(self, fps=fps, scale=scale, start_idx=start_idx).show()

    def run_notebook(self, start_idx=None):
        """
        Interactive widget for Jupyter notebooks.

        Usage in a notebook cell:
            from play_interactive import InteractiveWorldModel
            wm = InteractiveWorldModel(
                '/vast/adi/discrete_wm/checkpoints/model_best.pt',
                '/vast/adi/discrete_wm/checkpoints/tokenizer_final.pt',
                '/vast/adi/discrete_wm/data/tokenized_data.npz',
            )
            wm.run_notebook()
        """
        from IPython.display import display, clear_output
        import ipywidgets as widgets
        import matplotlib.pyplot as plt

        frame = self.reset(start_idx)

        out = widgets.Output()
        label = widgets.Label(f"Step: 0 | Action: NOOP")

        def show_frame(frame, action_name="NOOP"):
            with out:
                clear_output(wait=True)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(frame)
                ax.set_title(f"Step {self.step_count} | {action_name}")
                ax.axis('off')
                plt.tight_layout()
                plt.show()

        def on_action(action, name):
            nonlocal frame
            frame = self.step(action)
            label.value = f"Step: {self.step_count} | Action: {name}"
            show_frame(frame, name)

        btn_left = widgets.Button(description="← LEFT")
        btn_noop = widgets.Button(description="NOOP")
        btn_right = widgets.Button(description="RIGHT →")
        btn_fire = widgets.Button(description="FIRE ●")
        btn_reset = widgets.Button(description="Reset")
        btn_save = widgets.Button(description="Save GIF")

        btn_left.on_click(lambda _: on_action(3, "LEFT"))
        btn_right.on_click(lambda _: on_action(2, "RIGHT"))
        btn_noop.on_click(lambda _: on_action(0, "NOOP"))
        btn_fire.on_click(lambda _: on_action(1, "FIRE"))
        btn_reset.on_click(lambda _: (self.reset(), show_frame(self.frame_history[-1], "RESET")))
        btn_save.on_click(lambda _: self.save_gif(f"rollout_notebook_{int(time.time())}.gif"))

        controls = widgets.HBox([btn_left, btn_noop, btn_fire, btn_right, btn_reset, btn_save])
        show_frame(frame)
        display(widgets.VBox([label, out, controls]))


class LivePlayer:
    """Auto-stepping notebook player that renders the world model as a video.

    A Play toggle starts an asyncio loop that calls ``wm.step(current_action)``
    at the target FPS. Action buttons change ``current_action`` on the fly.
    FIRE is one-shot and reverts to the previously-held action after one frame.
    """

    def __init__(self, wm, fps=10, scale=4, start_idx=None):
        import asyncio
        import io
        import ipywidgets as widgets
        from PIL import Image as PILImage

        self.wm = wm
        self.fps = fps
        self.scale = scale
        self._asyncio = asyncio
        self._io = io
        self._PILImage = PILImage

        self.current_action = 0  # NOOP — the persistent held action
        self._fire_oneshot = False
        self._task = None

        # Initialise the world model to a starting frame.
        wm.reset(start_idx)

        self.image = widgets.Image(format='png')
        self.status = widgets.Label(value="Ready. Press ▶ Play.")
        self.fps_slider = widgets.IntSlider(
            value=fps, min=1, max=30, step=1, description='FPS',
            continuous_update=False,
            layout=widgets.Layout(width='220px'),
        )
        self.gen_steps_slider = widgets.IntSlider(
            value=wm.gen_steps, min=2, max=32, step=2, description='gen steps',
            continuous_update=False,
            layout=widgets.Layout(width='220px'),
        )
        self.fps_slider.observe(self._on_fps, names='value')
        self.gen_steps_slider.observe(self._on_gen_steps, names='value')

        btn_layout = widgets.Layout(width='90px')
        self.btn_play = widgets.ToggleButton(value=False, description='▶ Play',
                                             button_style='success', layout=btn_layout)
        self.btn_left = widgets.Button(description='← LEFT', layout=btn_layout)
        self.btn_noop = widgets.Button(description='NOOP', layout=btn_layout)
        self.btn_right = widgets.Button(description='RIGHT →', layout=btn_layout)
        self.btn_fire = widgets.Button(description='● FIRE', layout=btn_layout,
                                        button_style='warning')
        self.btn_reset = widgets.Button(description='Reset', layout=btn_layout)
        self.btn_save = widgets.Button(description='Save GIF', layout=btn_layout)

        self.btn_play.observe(self._on_play, names='value')
        self.btn_left.on_click(lambda _: self._set_action(3))
        self.btn_noop.on_click(lambda _: self._set_action(0))
        self.btn_right.on_click(lambda _: self._set_action(2))
        self.btn_fire.on_click(lambda _: self._fire())
        self.btn_reset.on_click(self._on_reset)
        self.btn_save.on_click(self._on_save)

        row1 = widgets.HBox([self.btn_play, self.btn_reset, self.btn_save])
        row2 = widgets.HBox([self.btn_left, self.btn_noop, self.btn_fire, self.btn_right])
        row3 = widgets.HBox([self.fps_slider, self.gen_steps_slider])
        self.widget = widgets.VBox([self.image, self.status, row1, row2, row3])

        self._render_frame(self.wm.frame_history[-1])

    def _set_action(self, action):
        self.current_action = action
        self._update_status()

    def _fire(self):
        self._fire_oneshot = True
        self._update_status()

    def _on_fps(self, change):
        self.fps = int(change['new'])
        self._update_status()

    def _on_gen_steps(self, change):
        self.wm.gen_steps = int(change['new'])
        self._update_status()

    def _on_reset(self, _):
        self.wm.reset()
        self._render_frame(self.wm.frame_history[-1])

    def _on_save(self, _):
        path = f"rollout_notebook_{int(time.time())}.gif"
        self.wm.save_gif(path, fps=self.fps)

    def _on_play(self, change):
        if change['new']:
            self.btn_play.description = '⏸ Pause'
            if self._task is None or self._task.done():
                self._task = self._asyncio.ensure_future(self._loop())
        else:
            self.btn_play.description = '▶ Play'

    async def _loop(self):
        while self.btn_play.value:
            t0 = time.time()
            if self._fire_oneshot:
                action = 1
                self._fire_oneshot = False
            else:
                action = self.current_action
            try:
                frame = self.wm.step(action)
                self._render_frame(frame)
            except Exception as e:
                self.status.value = f"Error: {e}"
                self.btn_play.value = False
                break
            elapsed = time.time() - t0
            await self._asyncio.sleep(max(0.0, 1.0 / self.fps - elapsed))

    def _render_frame(self, frame_np):
        h, w = frame_np.shape[:2]
        img = self._PILImage.fromarray(frame_np).resize(
            (w * self.scale, h * self.scale), self._PILImage.NEAREST
        )
        buf = self._io.BytesIO()
        img.save(buf, format='PNG')
        self.image.value = buf.getvalue()
        self._update_status()

    def _update_status(self):
        held = ACTION_NAMES.get(self.current_action, '?')
        pending = ' +FIRE' if self._fire_oneshot else ''
        self.status.value = (
            f"Step {self.wm.step_count}  |  held: {held}{pending}  |  "
            f"fps={self.fps}  gen_steps={self.wm.gen_steps}"
        )

    def show(self):
        from IPython.display import display
        display(self.widget)
        return self


def main():
    parser = argparse.ArgumentParser(description="Interactive world model play")
    parser.add_argument("--checkpoint", default="/vast/adi/discrete_wm/checkpoints/model_best.pt")
    parser.add_argument("--tokenizer-ckpt", default="/vast/adi/discrete_wm/checkpoints/tokenizer_final.pt")
    parser.add_argument("--data", default="/vast/adi/discrete_wm/data/tokenized_data.npz")
    parser.add_argument("--mode", choices=["pygame", "terminal"], default="terminal",
                        help="pygame = graphical window, terminal = text input + GIF output")
    parser.add_argument("--gen-steps", type=int, default=8, help="Denoising steps per frame")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--fps", type=int, default=10, help="Target FPS (pygame mode)")
    parser.add_argument("--window-size", type=int, default=512, help="Window size (pygame mode)")
    args = parser.parse_args()

    wm = InteractiveWorldModel(
        args.checkpoint, args.tokenizer_ckpt, args.data,
        gen_steps=args.gen_steps, temperature=args.temperature,
    )

    if args.mode == "pygame":
        wm.run_pygame(window_size=args.window_size, fps=args.fps)
    else:
        wm.run_terminal()


if __name__ == "__main__":
    main()
