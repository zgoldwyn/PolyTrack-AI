# env.py
import os
import time
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from mss import mss
from pynput.keyboard import Controller, Key

class PolytrackEnv(gym.Env):
    metadata = {"render_modes": []}

    # Threshold for template matching (0-1, lower = stricter match)
    STUCK_MATCH_THRESHOLD = 0.8
    CHECKPOINT_MATCH_THRESHOLD = 0.8

    # Reward for reaching each checkpoint (index = checkpoint number)
    CHECKPOINT_REWARDS = [0.0, 5.0, 5.0, 10.0]  # 0/3=nothing, 1/3=+5, 2/3=+5, 3/3=+10

    # Out-of-bounds green detection (HSV range)
    OOB_HSV_LOW = np.array([55, 40, 80])
    OOB_HSV_HIGH = np.array([90, 150, 180])
    OOB_THRESHOLD = 0.75  # 75% of bottom half is green = out of bounds

    # On-track grey detection (HSV range)
    TRACK_HSV_LOW = np.array([92, 33, 90])
    TRACK_HSV_HIGH = np.array([122, 93, 170])
    TRACK_CHECK_REGION = (0.57, 0.59, 0.45, 0.54)  # (top%, bottom%, left%, right%)
    TRACK_THRESHOLD = 0.3  # 30% of check region is grey = on track

    def __init__(self, monitor_region):
        super().__init__()
        self.monitor_region = monitor_region
        self.sct = mss()
        self.kb = Controller()

        # 6 discrete actions
        # 0 = forward only (no steering)
        # 1 = forward + left
        # 2 = forward + right
        # 3 = coast (no gas, no steering)
        # 4 = coast + left
        # 5 = coast + right
        self.action_space = spaces.Discrete(6)

        # grayscale stacked frames: 4 x 84 x 84
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(4, 84, 84), dtype=np.uint8
        )

        # Load the stuck popup template for detection
        base_dir = os.path.dirname(__file__)
        template_path = os.path.join(base_dir, "stuck_template.png")
        self.stuck_template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if self.stuck_template is None:
            print(f"WARNING: Could not load {template_path} — stuck detection disabled")

        # Load checkpoint templates (0/3 through 3/3)
        self.checkpoint_templates = []
        for i in range(4):
            path = os.path.join(base_dir, f"checkpoint_{i}.png")
            tmpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tmpl is None:
                print(f"WARNING: Could not load {path}")
            self.checkpoint_templates.append(tmpl)

        # Load finish screen template
        finish_path = os.path.join(base_dir, "finish_template.png")
        self.finish_template = cv2.imread(finish_path, cv2.IMREAD_GRAYSCALE)
        if self.finish_template is None:
            print(f"WARNING: Could not load {finish_path} — finish detection disabled")

        self.frame_stack = []
        self.prev_frame = None
        self.last_full_frame = None
        self.last_full_color = None
        self.cumulative_progress = 0.0
        self.last_progress = 0.0
        self.last_time = time.time()
        self.stuck_steps = 0
        self.checkpoints_reached = 0
        self.step_count = 0
        self.stuck_popup_streak = 0
        self.episode_reward = 0.0
        self.episode_num = 0

    def _grab_frame(self):
        raw = np.array(self.sct.grab(self.monitor_region))[:, :, :3]
        self.last_full_color = raw  # keep color for OOB detection
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        self.last_full_frame = gray  # keep full-res for template matching
        small = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return small

    def _is_out_of_bounds(self):
        """Check if the car is on the green out-of-bounds area."""
        if self.last_full_color is None:
            return False
        h = self.last_full_color.shape[0]
        # Check bottom half of the frame (where the ground is)
        bottom = self.last_full_color[h // 2:, :]
        hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.OOB_HSV_LOW, self.OOB_HSV_HIGH)
        green_ratio = np.count_nonzero(mask) / mask.size
        return green_ratio >= self.OOB_THRESHOLD

    def _is_on_track(self):
        """Check if grey track surface is visible in front of the car."""
        if self.last_full_color is None:
            return False
        h, w = self.last_full_color.shape[:2]
        t, b, l, r = self.TRACK_CHECK_REGION
        region = self.last_full_color[int(t*h):int(b*h), int(l*w):int(r*w)]
        if region.size == 0:
            return False
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.TRACK_HSV_LOW, self.TRACK_HSV_HIGH)
        grey_ratio = np.count_nonzero(mask) / mask.size
        return grey_ratio >= self.TRACK_THRESHOLD

    def _is_stuck_popup(self):
        """Check if the stuck/respawn popup is visible using template matching."""
        if self.stuck_template is None or self.last_full_frame is None:
            return False
        result = cv2.matchTemplate(
            self.last_full_frame, self.stuck_template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= self.STUCK_MATCH_THRESHOLD

    def _is_finish_screen(self):
        """Check if the finish/completion screen is visible."""
        if self.finish_template is None or self.last_full_frame is None:
            return False
        result = cv2.matchTemplate(
            self.last_full_frame, self.finish_template, cv2.TM_CCOEFF_NORMED
        )
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val >= self.STUCK_MATCH_THRESHOLD

    def _detect_checkpoints(self):
        """Return the highest checkpoint state that matches the current frame."""
        if self.last_full_frame is None:
            return 0
        best = 0
        best_score = -1.0
        for i, tmpl in enumerate(self.checkpoint_templates):
            if tmpl is None:
                continue
            result = cv2.matchTemplate(
                self.last_full_frame, tmpl, cv2.TM_CCOEFF_NORMED
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if max_val >= self.CHECKPOINT_MATCH_THRESHOLD and max_val > best_score:
                best = i
                best_score = max_val
        return best

    def _get_obs(self):
        frame = self._grab_frame()
        if len(self.frame_stack) == 0:
            self.frame_stack = [frame] * 4
        else:
            self.frame_stack.pop(0)
            self.frame_stack.append(frame)
        return np.stack(self.frame_stack, axis=0).astype(np.uint8)

    def _release_all(self):
        for k in ['w', 'a', 'd']:
            try:
                self.kb.release(k)
            except Exception:
                pass

    def _apply_action(self, action):
        self._release_all()
        if action <= 2:
            self.kb.press('w')          # gas held for actions 0-2
        if action in (1, 4):
            self.kb.press('a')          # left
        elif action in (2, 5):
            self.kb.press('d')          # right

    def _estimate_progress(self, obs):
        # Optical flow direction: measures whether the scene is expanding
        # outward (forward movement) or contracting (backward/stopped).
        # Returns positive for forward, negative for backward, ~0 for still.
        current_frame = self.frame_stack[-1]
        if self.prev_frame is None:
            self.prev_frame = current_frame
            return self.cumulative_progress

        # Compute dense optical flow between consecutive frames
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_frame, current_frame,
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self.prev_frame = current_frame

        # flow[:,:,1] is vertical movement of pixels
        # When driving forward in a 3D game, the bottom half of the screen
        # has pixels moving downward (positive y-flow) as the ground rushes past.
        # When reversing, pixels move upward (negative y-flow).
        h = flow.shape[0]
        bottom_half_vy = flow[h // 2:, :, 1]  # vertical flow in bottom half
        avg_flow = np.mean(bottom_half_vy)

        # Positive avg_flow = pixels moving down = driving forward
        # Negative avg_flow = pixels moving up = driving backward
        speed_estimate = avg_flow / 10.0  # normalize
        self.cumulative_progress += speed_estimate
        return self.cumulative_progress

    def step(self, action):
        self._apply_action(action)
        time.sleep(0.05)  # 20 FPS control loop

        obs = self._get_obs()
        progress = self._estimate_progress(obs)
        self.step_count += 1

        reward = max(0.001, 0.01 - (self.step_count * 0.00001))  # alive bonus decays over time (~900 steps to bottom out)

        # Small reward for frame-diff movement
        if progress > self.last_progress:
            reward += (progress - self.last_progress) * 1.0
            self.stuck_steps = 0
        else:
            self.stuck_steps += 1
            reward -= 0.01

        self.last_progress = progress

        # Check templates every 10 steps to save CPU (except stuck — check always)
        terminated = False
        truncated = False
        term_reason = ""

        if self._is_stuck_popup():
            self.stuck_popup_streak += 1
        else:
            self.stuck_popup_streak = 0

        if self.stuck_popup_streak >= 3:
            reward -= 1.0
            terminated = True
            term_reason = "stuck_popup"

        # Penalize driving on green out-of-bounds ground — end episode immediately
        if not terminated and self._is_out_of_bounds():
            reward -= 2.0
            terminated = True
            term_reason = "out_of_bounds"

        # Reward for staying on the grey track
        if not terminated and self._is_on_track():
            reward += 0.05

        if not terminated and self.step_count % 10 == 0:
            # Detect finish screen — big reward!
            if self._is_finish_screen():
                reward += 20.0
                terminated = True
                term_reason = "FINISHED"

            # Checkpoint reward — scaled by speed (fewer steps = bigger reward)
            if not terminated:
                current_cp = self._detect_checkpoints()
                if current_cp > self.checkpoints_reached:
                    base_reward = self.CHECKPOINT_REWARDS[current_cp]
                    # At step 1 multiplier is 2.0, at step 600+ it's 1.0
                    speed_mult = max(1.0, 2.0 - (self.step_count / 600.0))
                    reward += base_reward * speed_mult
                    self.checkpoints_reached = current_cp
                    self.stuck_steps = 0

        if self.stuck_steps > 600:
            reward -= 1.0
            terminated = True
            term_reason = "no_progress"

        self.episode_reward += reward

        if terminated:
            elapsed = self.step_count / 20.0  # approximate seconds
            print(f"EP {self.episode_num} | {elapsed:.1f}s | steps={self.step_count} | cp={self.checkpoints_reached}/3 | reward={self.episode_reward:.2f} | end={term_reason}")

        info = {"progress": progress, "checkpoints": self.checkpoints_reached}
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._release_all()
        self.kb.press('t')
        self.kb.release('t')
        time.sleep(1.0)

        self.frame_stack = []
        self.prev_frame = None
        self.last_full_frame = None
        self.last_full_color = None
        self.cumulative_progress = 0.0
        self.last_progress = 0.0
        self.stuck_steps = 0
        self.checkpoints_reached = 0
        self.step_count = 0
        self.stuck_popup_streak = 0
        self.episode_reward = 0.0
        self.episode_num += 1

        obs = self._get_obs()
        return obs, {}

    def close(self):
        self._release_all()
