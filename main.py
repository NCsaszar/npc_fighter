# npc_fighter/main.py
import os
import time
import math
import random
import ctypes
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import cv2
import mss
import pyautogui
import keyboard
import pygetwindow as gw

from config import CONFIG

# -------------------------- Windows DPI awareness --------------------------
try:
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# -------------------------- Windows timer resolution (critical for speed fidelity) --------------------------
_TIME_PERIOD_SET = False

def _enable_high_res_timer() -> None:
    global _TIME_PERIOD_SET
    try:
        winmm = ctypes.WinDLL("winmm")
        res = winmm.timeBeginPeriod(1)
        if res == 0:
            _TIME_PERIOD_SET = True
    except Exception:
        pass

def _disable_high_res_timer() -> None:
    global _TIME_PERIOD_SET
    if not _TIME_PERIOD_SET:
        return
    try:
        winmm = ctypes.WinDLL("winmm")
        winmm.timeEndPeriod(1)
    except Exception:
        pass
    _TIME_PERIOD_SET = False

# -------------------------- Globals / State --------------------------
STOP = False
PAUSED = False

pyautogui.FAILSAFE = True
pyautogui.PAUSE = float(CONFIG["MOUSE"].get("autoDelaySec", 0.0))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(SCRIPT_DIR, "images")
DEBUG_DIR = os.path.join(SCRIPT_DIR, "debug")

_sct = mss.mss()

# -------------------------- Logging --------------------------
def ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int((time.time()%1)*1000):03d}Z"

class LogColor:
    RESET = "\033[0m"
    INFO = "\033[92m"      # Green
    WARN = "\033[93m"      # Yellow
    DEBUG = "\033[90m"     # Gray
    ERROR = "\033[91m"     # Red

LEVEL_COLOR = {
    "INFO": LogColor.INFO,
    "WARN": LogColor.WARN,
    "DEBUG": LogColor.DEBUG,
    "ERROR": LogColor.ERROR,
}

def log(level: str, msg: str) -> None:
    color = LEVEL_COLOR.get(level, "")
    reset = LogColor.RESET if color else ""
    print(f"{color}[{ts()}] [{level}] {msg}{reset}", flush=True)

_LAST_LOG_MS: Dict[str, float] = {}

def log_throttled(key: str, every_ms: int, level: str, msg: str) -> None:
    if every_ms <= 0:
        log(level, msg)
        return
    now_ms = time.time() * 1000.0
    last = _LAST_LOG_MS.get(key, 0.0)
    if (now_ms - last) >= every_ms:
        _LAST_LOG_MS[key] = now_ms
        log(level, msg)

# -------------------------- Data helpers --------------------------
@dataclass
class Rect:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

def rand_int(a: int, b: int) -> int:
    return random.randint(a, b)

def ensure_debug_dir() -> None:
    os.makedirs(DEBUG_DIR, exist_ok=True)

def hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    h = hex_str.strip().lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)

def window_region(win: gw.Win32Window) -> Rect:
    return Rect(int(win.left), int(win.top), int(win.width), int(win.height))

def region_from_window_offsets(win: gw.Win32Window, offsets: dict) -> Rect:
    b = window_region(win)

    left_off = int(offsets["left"])
    top_off = int(offsets["top"])
    width_off = int(offsets["width"])
    height_off = int(offsets["height"])

    if width_off <= 0:
        width_off = int(b.width - left_off)
    if height_off <= 0:
        height_off = int(b.height - top_off)

    return Rect(
        left=b.left + left_off,
        top=b.top + top_off,
        width=width_off,
        height=height_off,
    )

def point_from_window_offsets(win: gw.Win32Window, x: int, y: int) -> Tuple[int, int]:
    b = window_region(win)
    return (b.left + int(x), b.top + int(y))

# -------------------------- Window focus / gating --------------------------
def find_target_window() -> gw.Win32Window:
    needle = CONFIG["WINDOW"]["titleContains"].lower().strip()
    if not needle:
        raise RuntimeError("CONFIG.WINDOW.titleContains is empty")

    for w in gw.getAllWindows():
        try:
            title = (w.title or "").lower()
        except Exception:
            continue
        if needle in title and w.width > 0 and w.height > 0:
            return w

    raise RuntimeError(f'Target window not found (titleContains="{CONFIG["WINDOW"]["titleContains"]}")')

def is_window_active(win: gw.Win32Window) -> bool:
    try:
        active = gw.getActiveWindow()
        return bool(active and (active._hWnd == win._hWnd))
    except Exception:
        return False

def focus_window(win: gw.Win32Window) -> None:
    try:
        if getattr(win, "isMinimized", False):
            win.restore()
        win.activate()
        time.sleep(0.02)
    except Exception as e:
        log("WARN", f"Failed to focus window: {e}")

def ensure_window_focused(win: gw.Win32Window) -> bool:
    if not is_window_active(win):
        focus_window(win)
    return True

# -------------------------- Screen capture --------------------------
def grab_region(rect: Rect) -> np.ndarray:
    monitor = {"left": rect.left, "top": rect.top, "width": rect.width, "height": rect.height}
    img = np.array(_sct.grab(monitor))  # BGRA
    return img[:, :, :3]  # BGR

def capture_region_to_file(rect: Rect, filepath: str) -> None:
    ensure_debug_dir()
    bgr = grab_region(rect)
    cv2.imwrite(filepath, bgr)
    log("INFO", f"Captured region -> {filepath}")

# -------------------------- Mouse helpers --------------------------
def _smoothstep(t: float) -> float:
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)

def human_move_to(x: int, y: int, speed_px_s: float) -> None:
    cur_x, cur_y = pyautogui.position()
    dx = x - cur_x
    dy = y - cur_y
    dist = math.hypot(dx, dy)

    if dist <= 0.5:
        return

    if speed_px_s <= 0:
        pyautogui.moveTo(x, y, duration=0)
        return

    total_t = dist / float(speed_px_s)

    if total_t <= 0.002:
        pyautogui.moveTo(x, y, duration=0)
        return

    start = time.perf_counter()
    end = start + total_t
    target_dt = 1.0 / 240.0  # ~240 Hz updates

    while True:
        now = time.perf_counter()
        if now >= end:
            break

        t = (now - start) / total_t
        ease = _smoothstep(t)

        nx = int(round(cur_x + dx * ease))
        ny = int(round(cur_y + dy * ease))
        pyautogui.moveTo(nx, ny, duration=0)

        remaining = end - now
        sleep_for = min(target_dt, max(0.0, remaining - 0.0005))
        if sleep_for > 0:
            time.sleep(sleep_for)

    pyautogui.moveTo(x, y, duration=0)

def click_point(x: int, y: int, win: gw.Win32Window) -> None:
    if not ensure_window_focused(win):
        return

    speed = float(CONFIG["MOUSE"].get("speed", 20000))
    human_move_to(x, y, speed)

    after_move_ms = int(CONFIG.get("MOUSE", {}).get("afterMoveDelayMs", 0))
    if after_move_ms > 0:
        time.sleep(after_move_ms / 1000.0)

    move_delay_ms = int(CONFIG.get("LOOP", {}).get("moveDelayMs", 0))
    if move_delay_ms > 0:
        time.sleep(move_delay_ms / 1000.0)

    pyautogui.click(button="left")

    after_click_ms = int(CONFIG.get("MOUSE", {}).get("afterClickDelayMs", 0))
    if after_click_ms > 0:
        time.sleep(after_click_ms / 1000.0)

def close_pos_interface(win: gw.Win32Window) -> None:
    pos_cfg = CONFIG.get("POS", {})
    if not pos_cfg.get("enabled", True):
        return

    pt = pos_cfg.get("closePoint", {"x": 459, "y": 64})
    rel_x = int(pt.get("x", 459))
    rel_y = int(pt.get("y", 64))

    delay_ms = int(pos_cfg.get("delayAfterUnpauseMs", 0))
    if delay_ms > 0:
        time.sleep(delay_ms / 1000.0)

    sx, sy = point_from_window_offsets(win, rel_x, rel_y)
    log("INFO", f"Closing POS: window-relative ({rel_x},{rel_y}) -> screen ({sx},{sy})")
    click_point(sx, sy, win)

# -------------------------- Color search --------------------------
def find_any_color_in_region(rect: Rect, hex_colors: List[str], tolerance: int) -> Optional[Tuple[int, int, str]]:
    if not hex_colors:
        return None

    img = grab_region(rect)  # BGR
    tol = int(max(0, min(255, tolerance)))

    for hx in hex_colors:
        b, g, r = hex_to_bgr(hx)

        lower = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)], dtype=np.uint8)
        upper = np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)], dtype=np.uint8)

        mask = cv2.inRange(img, lower, upper)
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue

        idx = random.randrange(len(xs))
        rx = int(xs[idx])
        ry = int(ys[idx])

        pb, pg, pr = img[ry, rx]
        if (abs(int(pb) - b) <= tol) and (abs(int(pg) - g) <= tol) and (abs(int(pr) - r) <= tol):
            sx = rect.left + rx
            sy = rect.top + ry
            return (sx, sy, hx)

    return None

# NEW: lightweight boolean check for a single color (used by hover failsafe)
def any_color_present_in_region(rect: Rect, hex_color: str, tolerance: int) -> bool:
    img = grab_region(rect)  # BGR
    tol = int(max(0, min(255, tolerance)))
    b, g, r = hex_to_bgr(hex_color)

    lower = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)], dtype=np.uint8)
    upper = np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)], dtype=np.uint8)

    mask = cv2.inRange(img, lower, upper)
    return bool(np.any(mask))

# NEW: hover-text failsafe confirmation
def hover_failsafe_confirm(win: gw.Win32Window) -> bool:
    cfg = CONFIG.get("HOVER_FAILSAFE", {})
    if not cfg.get("enabled", True):
        return True

    rect = region_from_window_offsets(win, cfg["hoverTextRegion"])
    color_hex = str(cfg.get("hoverTextColorHex", "ffff00"))
    tol = int(cfg.get("tolerance", 6))
    settle_ms = int(cfg.get("settleDelayMs", 30))
    timeout_ms = int(cfg.get("confirmTimeoutMs", 150))
    poll_ms = int(cfg.get("pollEveryMs", 15))

    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)

    start_ms = time.time() * 1000.0
    while (time.time() * 1000.0 - start_ms) <= timeout_ms:
        if any_color_present_in_region(rect, color_hex, tol):
            return True
        if poll_ms > 0:
            time.sleep(poll_ms / 1000.0)
        else:
            break

    return False

def is_target_box_visible(win: gw.Win32Window) -> bool:
    rect = region_from_window_offsets(win, CONFIG["TARGET_BOX_REGION_OFFSETS"])
    tol = int(CONFIG["LOOP"].get("colorTolerance", 4))
    found = find_any_color_in_region(rect, CONFIG["TARGET_BOX_COLORS_HEX"], tol)
    return found is not None

def wait_for_target_box_state(win: gw.Win32Window, want_visible: bool, timeout_ms: int) -> bool:
    if timeout_ms <= 0:
        return False

    start = time.time()
    poll_ms = int(CONFIG["LOOP"].get("boxPollEveryMs", 80))
    timeout_s = timeout_ms / 1000.0

    while (time.time() - start) < timeout_s:
        visible = is_target_box_visible(win)
        if visible == want_visible:
            return True
        time.sleep(poll_ms / 1000.0)
    return False

# -------------------------- Template matching --------------------------
def load_template(template_name: str) -> np.ndarray:
    p = os.path.join(IMAGES_DIR, template_name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Template image not found: {p}")
    tpl = cv2.imread(p, cv2.IMREAD_COLOR)
    if tpl is None:
        raise RuntimeError(f"Failed to load template image: {p}")
    return tpl

def template_match_in_region(rect: Rect, template_bgr: np.ndarray, threshold: float) -> Optional[Tuple[int, int, int, int, float]]:
    img = grab_region(rect)
    h, w = template_bgr.shape[:2]
    if img.shape[0] < h or img.shape[1] < w:
        return None

    res = cv2.matchTemplate(img, template_bgr, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val >= threshold:
        ml, mt = max_loc
        return (rect.left + ml, rect.top + mt, w, h, float(max_val))
    return None

# -------------------------- Healing --------------------------
HEAL_TEMPLATE = None
FULL_HP_TEMPLATE = None

def is_full_hp_visible(win: gw.Win32Window, threshold: float) -> bool:
    global FULL_HP_TEMPLATE
    heal_cfg = CONFIG.get("HEAL", {})
    if FULL_HP_TEMPLATE is None:
        FULL_HP_TEMPLATE = load_template(str(heal_cfg.get("fullHpImageName", "full_hp.png")))

    hp_rect = region_from_window_offsets(win, heal_cfg["fullHpRegion"])
    found = template_match_in_region(hp_rect, FULL_HP_TEMPLATE, threshold)
    return found is not None

def try_heal_if_needed(win: gw.Win32Window, now_ms: float, last_heal_click_ms: Optional[float]) -> Optional[float]:
    """
    Clicks heal.png in inventory ONLY if full_hp.png is NOT visible in the HP region.
    Enforces cooldown (default 10s).
    Returns updated last_heal_click_ms (or the original if nothing was done).
    """
    global HEAL_TEMPLATE

    heal_cfg = CONFIG.get("HEAL", {})
    if not heal_cfg.get("enabled", True):
        return last_heal_click_ms

    cooldown_ms = int(heal_cfg.get("cooldownMs", 10_000))
    if last_heal_click_ms is not None and (now_ms - last_heal_click_ms) < cooldown_ms:
        return last_heal_click_ms

    threshold = float(heal_cfg.get("confidence", 0.80))

    # If full HP indicator is visible, do nothing.
    try:
        if is_full_hp_visible(win, threshold=threshold):
            return last_heal_click_ms
    except Exception as e:
        log("WARN", f"Full-HP check failed: {e}")
        return last_heal_click_ms

    # Not full HP -> find heal potion in inventory and click it.
    if HEAL_TEMPLATE is None:
        HEAL_TEMPLATE = load_template(str(heal_cfg.get("healImageName", "heal.png")))

    inv_rect = region_from_window_offsets(win, CONFIG["INVENTORY_SCREEN_REGION_OFFSETS"])
    found = template_match_in_region(inv_rect, HEAL_TEMPLATE, threshold)
    if not found:
        return last_heal_click_ms

    left, top, w, h, score = found
    cx = left + (w // 2)
    cy = top + (h // 2)

    log("INFO", f"Heal needed (full_hp not visible). Clicking heal at ({cx},{cy}) score={score:.3f}")
    click_point(cx, cy, win)

    post_ms = int(heal_cfg.get("postClickDelayMs", 0))
    if post_ms > 0:
        time.sleep(post_ms / 1000.0)

    return now_ms

# -------------------------- Compass --------------------------
def next_compass_at_ms() -> float:
    mn = int(CONFIG["COMPASS"].get("intervalMinMs", 60_000))
    mx = int(CONFIG["COMPASS"].get("intervalMaxMs", 120_000))
    return time.time() * 1000.0 + rand_int(mn, mx)

def click_compass(win: gw.Win32Window) -> None:
    cx = int(CONFIG["COMPASS"]["point"]["x"])
    cy = int(CONFIG["COMPASS"]["point"]["y"])
    sx, sy = point_from_window_offsets(win, cx, cy)

    log("INFO", f"Clicking compass at window-relative ({cx}, {cy}) -> screen ({sx}, {sy})")
    click_point(sx, sy, win)

    after = int(CONFIG["COMPASS"].get("postClickDelayMs", 150))
    if after > 0:
        time.sleep(after / 1000.0)

# -------------------------- Login template matching --------------------------
LOGIN_TEMPLATE = None

def handle_login_if_visible(win: gw.Win32Window) -> bool:
    global LOGIN_TEMPLATE

    login_cfg = CONFIG.get("LOGIN", {})
    check_every_ms = int(login_cfg.get("checkEveryMs", 10_000))
    post_click_ms = int(login_cfg.get("postClickDelayMs", 250))
    max_block_ms = int(login_cfg.get("maxBlockMs", 0))
    threshold = float(login_cfg.get("confidence", 0.75))

    if LOGIN_TEMPLATE is None:
        LOGIN_TEMPLATE = load_template(login_cfg["imageName"])

    search_rect = region_from_window_offsets(win, CONFIG["FULL_APP_SCREEN_REGION_OFFSETS"])
    first = template_match_in_region(search_rect, LOGIN_TEMPLATE, threshold)
    if not first:
        return False

    log("WARN", f"Login button detected (score={first[4]:.3f}). Clicking and blocking until it disappears.")
    start_ms = time.time() * 1000.0

    while True:
        found = template_match_in_region(search_rect, LOGIN_TEMPLATE, threshold)
        if not found:
            log("INFO", "Login button no longer visible. Resuming normal operation.")
            return True

        left, top, w, h, score = found
        cx = left + (w // 2)
        cy = top + (h // 2)

        log("INFO", f"Clicking login button center at ({cx}, {cy}) score={score:.3f}")
        click_point(cx, cy, win)

        if post_click_ms > 0:
            time.sleep(post_click_ms / 1000.0)

        if max_block_ms > 0 and (time.time() * 1000.0 - start_ms) > max_block_ms:
            log("WARN", "Login handling maxBlockMs reached; giving up and resuming.")
            return True

        time.sleep(check_every_ms / 1000.0)

# -------------------------- Prayer template matching --------------------------
PRAY_ACTIVATED_TEMPLATE = None
PRAY_NOT_ACTIVATED_TEMPLATE = None

def handle_prayer_if_needed(win: gw.Win32Window) -> bool:
    global PRAY_ACTIVATED_TEMPLATE, PRAY_NOT_ACTIVATED_TEMPLATE

    prayer_cfg = CONFIG.get("PRAYER", {})
    threshold = float(prayer_cfg.get("confidence", 0.75))
    post_click_ms = int(prayer_cfg.get("postClickDelayMs", 150))

    if PRAY_ACTIVATED_TEMPLATE is None:
        PRAY_ACTIVATED_TEMPLATE = load_template(prayer_cfg["activatedImageName"])
    if PRAY_NOT_ACTIVATED_TEMPLATE is None:
        PRAY_NOT_ACTIVATED_TEMPLATE = load_template(prayer_cfg["notActivatedImageName"])

    search_rect = region_from_window_offsets(win, CONFIG["FULL_APP_SCREEN_REGION_OFFSETS"])

    activated = template_match_in_region(search_rect, PRAY_ACTIVATED_TEMPLATE, threshold)
    if activated:
        return False

    not_activated = template_match_in_region(search_rect, PRAY_NOT_ACTIVATED_TEMPLATE, threshold)
    if not_activated:
        left, top, w, h, score = not_activated
        cx = left + (w // 2)
        cy = top + (h // 2)
        log("INFO", f"Prayer not activated detected (score={score:.3f}). Clicking center at ({cx}, {cy}) to activate.")
        click_point(cx, cy, win)

        if post_click_ms > 0:
            time.sleep(post_click_ms / 1000.0)

        return True

    return False

# -------------------------- Inventory template matching --------------------------
OVERLOAD_TEMPLATE = None

def handle_overload_click(win: gw.Win32Window) -> bool:
    global OVERLOAD_TEMPLATE

    ov_cfg = CONFIG.get("OVERLOAD", {})
    if not ov_cfg.get("enabled", True):
        return False

    threshold = float(ov_cfg.get("confidence", CONFIG.get("LOGIN", {}).get("confidence", 0.75)))
    post_click_ms = int(ov_cfg.get("postClickDelayMs", 150))
    image_name = str(ov_cfg.get("imageName", "overload.png"))

    if OVERLOAD_TEMPLATE is None:
        OVERLOAD_TEMPLATE = load_template(image_name)

    inv_rect = region_from_window_offsets(win, CONFIG["INVENTORY_SCREEN_REGION_OFFSETS"])
    found = template_match_in_region(inv_rect, OVERLOAD_TEMPLATE, threshold)
    if not found:
        return False

    left, top, w, h, score = found
    cx = left + (w // 2)
    cy = top + (h // 2)

    log("INFO", f"Overload detected in inventory (score={score:.3f}). Clicking center at ({cx}, {cy}).")
    click_point(cx, cy, win)

    if post_click_ms > 0:
        time.sleep(post_click_ms / 1000.0)

    return True

# -------------------------- Hotkeys --------------------------
def on_pause_hotkey():
    global PAUSED
    PAUSED = not PAUSED
    log("INFO", "Paused" if PAUSED else "Resumed")

    if not PAUSED:
        try:
            win = find_target_window()
            focus_window(win)
            close_pos_interface(win)
        except Exception as e:
            log("WARN", f"Unable to focus/close POS on unpause: {e}")

def on_stop_hotkey():
    global STOP
    STOP = True
    log("INFO", "Stop requested")

def on_capture_hotkey():
    try:
        win = find_target_window()
        rect = region_from_window_offsets(win, CONFIG["MAIN_SCREEN_REGION_OFFSETS"])
        ensure_debug_dir()
        out = os.path.join(
            DEBUG_DIR,
            f"window_region_{int(time.time()*1000)}_win({win.left},{win.top},{win.width}x{win.height})"
            f"_region({rect.left},{rect.top},{rect.width}x{rect.height}).png"
        )
        capture_region_to_file(rect, out)
    except Exception as e:
        log("WARN", f"Capture failed: {e}")

def register_hotkeys():
    hk = CONFIG.get("HOTKEYS", {})
    keyboard.add_hotkey(hk.get("pause", "ctrl+shift+p"), on_pause_hotkey)
    keyboard.add_hotkey(hk.get("stop", "ctrl+shift+q"), on_stop_hotkey)

    if CONFIG.get("LOGIN", {}).get("debug", {}).get("enableCaptureHotkey", True):
        keyboard.add_hotkey(hk.get("capture", "ctrl+shift+c"), on_capture_hotkey)

# -------------------------- Main loop --------------------------
def main():
    _enable_high_res_timer()
    register_hotkeys()

    log("INFO", "Starting npc_fighter (window-following)...")
    log("INFO", f"Pause: {CONFIG['HOTKEYS']['pause']} | Stop: {CONFIG['HOTKEYS']['stop']}")
    if CONFIG.get("LOGIN", {}).get("debug", {}).get("enableCaptureHotkey", True):
        log("INFO", f"Debug capture: {CONFIG['HOTKEYS']['capture']} (captures window search region)")

    clicks = 0
    last_no_match_log = 0.0
    last_window_err_log = 0.0

    compass_next_at_ms = next_compass_at_ms()
    did_start_compass_click = False

    next_login_check_at_ms = time.time() * 1000.0
    next_prayer_check_at_ms = time.time() * 1000.0

    # healing timers/state
    last_heal_click_ms: Optional[float] = None
    next_heal_check_at_ms: float = time.time() * 1000.0

    target_box_visible_since_ms: Optional[float] = None

    log_cfg = CONFIG.get("LOGGING", {})
    combat_wait_every_ms = int(log_cfg.get("combatWaitEveryMs", 2000))
    combat_timeout_every_ms = int(log_cfg.get("combatTimeoutEveryMs", 2000))

    ov_cfg = CONFIG.get("OVERLOAD", {})
    overload_interval_ms = int(ov_cfg.get("intervalMs", 30 * 60 * 1000))
    next_overload_at_ms: Optional[float] = (time.time() * 1000.0)

    # hover failsafe logging throttle (NEW)
    hover_cfg = CONFIG.get("HOVER_FAILSAFE", {})
    hover_reject_every_ms = int(hover_cfg.get("logRejectionsEveryMs", 1000))

    try:
        win0 = find_target_window()
        focus_window(win0)
    except Exception:
        pass

    try:
        while not STOP:
            if PAUSED:
                time.sleep(0.1)
                continue

            try:
                win = find_target_window()
            except Exception as e:
                now = time.time()
                if (now - last_window_err_log) > 2.0:
                    log("WARN", str(e))
                    last_window_err_log = now
                time.sleep(0.5)
                continue

            ensure_window_focused(win)
            now_ms = time.time() * 1000.0

            # --- healing (runs on its own timer; respects cooldown internally) ---
            heal_cfg = CONFIG.get("HEAL", {})
            if heal_cfg.get("enabled", True) and now_ms >= next_heal_check_at_ms:
                next_heal_check_at_ms = now_ms + int(heal_cfg.get("checkEveryMs", 250))
                try:
                    last_heal_click_ms = try_heal_if_needed(win, now_ms, last_heal_click_ms)
                except Exception as e:
                    log("WARN", f"Heal check failed: {e}")

            # --- login check ---
            if now_ms >= next_login_check_at_ms:
                next_login_check_at_ms = now_ms + int(CONFIG["LOGIN"].get("checkEveryMs", 10_000))
                log("INFO", "Checking for login button...")
                try:
                    handle_login_if_visible(win)
                except Exception as e:
                    log("WARN", f"Login check failed: {e}")

            # --- prayer check ---
            if now_ms >= next_prayer_check_at_ms:
                next_prayer_check_at_ms = now_ms + int(CONFIG.get("PRAYER", {}).get("checkEveryMs", 5_000))
                log("INFO", "Checking prayer state...")
                try:
                    handle_prayer_if_needed(win)
                except Exception as e:
                    log("WARN", f"Prayer check failed: {e}")

            # --- compass ---
            if CONFIG["COMPASS"].get("clickOnStart", True) and not did_start_compass_click:
                try:
                    click_compass(win)
                except Exception as e:
                    log("WARN", f"Compass click on start failed: {e}")
                finally:
                    did_start_compass_click = True
                    compass_next_at_ms = next_compass_at_ms()

            if now_ms >= compass_next_at_ms:
                try:
                    in_combat = is_target_box_visible(win) if CONFIG["TARGET_BOX_COLORS_HEX"] else False
                    if in_combat:
                        compass_next_at_ms = now_ms + 2000
                    else:
                        click_compass(win)
                        compass_next_at_ms = next_compass_at_ms()
                except Exception as e:
                    log("WARN", f"Compass click failed: {e}")
                    compass_next_at_ms = next_compass_at_ms()

            # --- overload periodic ---
            if ov_cfg.get("enabled", True) and next_overload_at_ms is not None and now_ms >= next_overload_at_ms:
                try:
                    avoid_combat = bool(ov_cfg.get("avoidDuringCombat", True))
                    in_combat = is_target_box_visible(win) if (avoid_combat and CONFIG["TARGET_BOX_COLORS_HEX"]) else False

                    if in_combat:
                        next_overload_at_ms = now_ms + 5000
                    else:
                        clicked = handle_overload_click(win)
                        if clicked:
                            log("INFO", f"Overload click completed. Next scheduled in {overload_interval_ms}ms.")
                            next_overload_at_ms = now_ms + overload_interval_ms
                        else:
                            next_overload_at_ms = now_ms + 10_000
                except Exception as e:
                    log("WARN", f"Overload periodic click failed: {e}")
                    next_overload_at_ms = now_ms + 30_000

            # --- combat gating: if box is visible, wait for it to disappear ---
            if CONFIG["TARGET_BOX_COLORS_HEX"]:
                try:
                    visible = is_target_box_visible(win)

                    if visible:
                        if target_box_visible_since_ms is None:
                            target_box_visible_since_ms = now_ms

                        max_visible_ms = int(CONFIG["LOOP"].get("maxTargetBoxVisibleMs", 30_000))
                        visible_for_ms = int(now_ms - target_box_visible_since_ms)

                        if visible_for_ms >= max_visible_ms:
                            log("WARN", f"Target box stuck visible for {visible_for_ms}ms (>= {max_visible_ms}ms). Ignoring gating and resuming.")
                            target_box_visible_since_ms = None
                        else:
                            log_throttled(
                                key="combat_wait",
                                every_ms=int(CONFIG.get("LOGGING", {}).get("combatWaitEveryMs", 2000)),
                                level="DEBUG",
                                msg="Target box visible; waiting for fight to end...",
                            )

                            ended = wait_for_target_box_state(
                                win,
                                want_visible=False,
                                timeout_ms=int(CONFIG["LOOP"].get("waitForBoxDisappearTimeoutMs", 5000)),
                            )

                            if ended:
                                target_box_visible_since_ms = None
                            else:
                                log_throttled(
                                    key="combat_timeout",
                                    every_ms=int(CONFIG.get("LOGGING", {}).get("combatTimeoutEveryMs", 2000)),
                                    level="WARN",
                                    msg="Timed out waiting for target box to disappear; resuming.",
                                )

                            continue
                    else:
                        target_box_visible_since_ms = None

                except Exception as e:
                    log("WARN", f"Combat gating error: {e}")

            # --- find and click monster (UPDATED: hover failsafe before click) ---
            search_rect = region_from_window_offsets(win, CONFIG["MAIN_SCREEN_REGION_OFFSETS"])
            tol = int(CONFIG["LOOP"].get("colorTolerance", 4))
            found = find_any_color_in_region(search_rect, CONFIG["MONSTER_COLORS_HEX"], tol)

            if found:
                x, y, matched_hex = found

                # Move mouse to candidate first
                speed = float(CONFIG["MOUSE"].get("speed", 20000))
                human_move_to(x, y, speed)

                # Confirm hover text color (ffff00) appears before clicking
                if not hover_failsafe_confirm(win):
                    log_throttled(
                        key="hover_reject",
                        every_ms=hover_reject_every_ms,
                        level="DEBUG",
                        msg="Hover failsafe rejected click (ffff00 not found in hover text region).",
                    )
                    time.sleep(int(CONFIG["LOOP"].get("idleDelayMs", 60)) / 1000.0)
                    continue

                # Perform click using existing delay logic (without re-moving)
                after_move_ms = int(CONFIG.get("MOUSE", {}).get("afterMoveDelayMs", 0))
                if after_move_ms > 0:
                    time.sleep(after_move_ms / 1000.0)

                move_delay_ms = int(CONFIG.get("LOOP", {}).get("moveDelayMs", 0))
                if move_delay_ms > 0:
                    time.sleep(move_delay_ms / 1000.0)

                pyautogui.click(button="left")

                after_click_ms = int(CONFIG.get("MOUSE", {}).get("afterClickDelayMs", 0))
                if after_click_ms > 0:
                    time.sleep(after_click_ms / 1000.0)

                clicks += 1
                log("INFO", f"Click #{clicks} at ({x}, {y}) monsterHex={matched_hex}")

                # After clicking a monster, wait up to this long for the target box to appear.
                appear_timeout_ms = int(CONFIG["LOOP"].get("waitForBoxAppearTimeoutMs", 0))
                if appear_timeout_ms > 0 and CONFIG["TARGET_BOX_COLORS_HEX"]:
                    appeared = wait_for_target_box_state(win, want_visible=True, timeout_ms=appear_timeout_ms)
                    if appeared:
                        log("DEBUG", f"Target box appeared within {appear_timeout_ms}ms after click.")
                    else:
                        log("DEBUG", f"Target box did NOT appear within {appear_timeout_ms}ms after click (continuing).")

                time.sleep(int(CONFIG["LOOP"].get("postClickDelayMs", 150)) / 1000.0)

            else:
                now = time.time()
                if (now - last_no_match_log) * 1000.0 > int(CONFIG["LOOP"].get("logNoMatchEveryMs", 2000)):
                    log("DEBUG", "No monster match in region")
                    last_no_match_log = now

                time.sleep(int(CONFIG["LOOP"].get("idleDelayMs", 60)) / 1000.0)

        log("INFO", "Stopped cleanly")
    finally:
        _disable_high_res_timer()

if __name__ == "__main__":
    main()
