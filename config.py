# npc_fighter/config.py
CONFIG = {
    "WINDOW": {"titleContains": "Lunite - Ceezur"},
    "MAIN_SCREEN_REGION_OFFSETS": {"left": 0, "top": 0, "width": 517, "height": 362},
    "FULL_APP_SCREEN_REGION_OFFSETS": {"left": 0, "top": 0, "width": -1, "height": -1},
    "INVENTORY_SCREEN_REGION_OFFSETS": {"left": 551, "top": 238, "width": 186, "height": 253},

    "MONSTER_COLORS_HEX": ["22001f"],
    #"MONSTER_COLORS_HEX": ["7e4d4a","9d9973"], bork

    "TARGET_BOX_REGION_OFFSETS": {"left": 9, "top": 52, "width": 145, "height": 95},
    "TARGET_BOX_COLORS_HEX": ["007c00"],

    # Hover-text failsafe (must see ffff00 after hovering before clicking)
    "HOVER_FAILSAFE": {
        "enabled": True,
        "hoverTextRegion": {"left": 0, "top": 0, "width": 248, "height": 51},  # window-relative
        "hoverTextColorHex": "ffff00",
        "tolerance": 6,              # allow slight variation; tune if needed
        "settleDelayMs": 500,         # after moving mouse, wait briefly for hover text to render
        "confirmTimeoutMs": 150,     # how long to wait for hover color to appear
        "pollEveryMs": 15,           # polling cadence within confirmTimeout
        "logRejectionsEveryMs": 1000 # throttle logs when hover fails
    },

    "MOUSE": {
        "speed": 1250,
        "autoDelaySec": 0.0,
        "afterMoveDelayMs": 250,
        "afterClickDelayMs": 200
    },

    "POS": {
        "enabled": True,
        "closePoint": {"x": 459, "y": 64},
        "delayAfterUnpauseMs": 3000,
    },

    "HEAL": {
        "enabled": True,
        "healImageName": "heal.png",
        "fullHpImageName": "full_hp.png",
        "fullHpRegion": {"left": 691, "top": 42, "width": 66, "height": 36},
        "confidence": 0.60,
        "cooldownMs": 5_000,
        "checkEveryMs": 250,
        "postClickDelayMs": 150,
    },

    "COMPASS": {
        "point": {"x": 548, "y": 53},
        "clickOnStart": True,
        "intervalMinMs": 30_000,
        "intervalMaxMs": 60_000,
        "postClickDelayMs": 150,
    },

    "LOGIN": {
        "imageName": "login_button.png",
        "confidence": 0.75,
        "checkEveryMs": 10_000,
        "postClickDelayMs": 250,
        "maxBlockMs": 0,
        "debug": {"enableCaptureHotkey": True},
    },

    "PRAYER": {
        "activatedImageName": "pray_activated.png",
        "notActivatedImageName": "pray_not_activated.png",
        "confidence": 0.75,
        "checkEveryMs": 5_000,
        "postClickDelayMs": 150,
        "debug": {"enableCaptureHotkey": True},
    },

    "OVERLOAD": {
        "enabled": True,
        "imageName": "overload.png",
        "confidence": 0.6,
        "intervalMs": 30 * 60 * 1000,
        "postClickDelayMs": 150,
        "avoidDuringCombat": False,
    },

    "LOOP": {
        "postClickDelayMs": 150,
        "idleDelayMs": 60,
        "moveDelayMs": 0,
        "logNoMatchEveryMs": 2000,
        "waitForBoxAppearTimeoutMs": 0,
        "waitForBoxDisappearTimeoutMs": 0,
        "boxPollEveryMs": 80,
        "maxTargetBoxVisibleMs": 30_000,
        "colorTolerance": 4,
    },

    "LOGGING": {"combatWaitEveryMs": 2000, "combatTimeoutEveryMs": 2000},
    "HOTKEYS": {"pause": "ctrl+shift+p", "stop": "ctrl+shift+q", "capture": "ctrl+shift+c"},
}
