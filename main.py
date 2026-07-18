import importlib
import logging
import os
import sys
import subprocess
import threading
import time

logging.getLogger("discord").setLevel(logging.WARNING)

import utils
from brawl_industry_logger import get_logger
from utils import (
    load_toml_as_dict, save_dict_as_toml,
    current_wall_model_is_latest, update_wall_model_classes,
    get_latest_wall_model_file, start_config_watcher,
)
from bot_shared_state import BotState
import heartbeat

log = get_logger("brawl_industry.main")

REQUIRED_PACKAGES = {
    "discord":     "discord.py",
    "cv2":         "opencv-python",
    "numpy":       "numpy",
    "onnxruntime": "onnxruntime",
    "av":          "av",
    "adbutils":    "adbutils",
    "toml":        "toml",
    "requests":    "requests",
}

WATCHDOG_TIMEOUT          = 120
RECONNECT_WINDOW_SECS     = 60
MAX_RECONNECTS_IN_WINDOW  = 3
NO_DETECTIONS_THRESHOLD_S = 60 * 8


def _hard_exit(code: int = 1):
    for h in logging.root.handlers:
        try:
            h.flush()
        except Exception:
            pass
    os._exit(code)


def _heartbeat_watchdog():
    while True:
        time.sleep(30)
        elapsed = time.time() - heartbeat.last_beat()
        if elapsed > WATCHDOG_TIMEOUT:
            log.error(f"Watchdog: no heartbeat for {elapsed:.0f}s, forcing exit")
            _hard_exit(1)


def check_dependencies():
    missing = [pip for mod, pip in REQUIRED_PACKAGES.items()
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    log.warning(f"Missing packages: {', '.join(missing)}")
    if input("Install them? [Y/n]: ").strip().lower() in ("n", "no"):
        log.warning("Skipped, bot may not work")
        return
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)


def ensure_discord_token() -> str:
    config = load_toml_as_dict("cfg/general_config.toml")
    token = config.get("discord_token", "").strip()
    if token:
        return token
    token = input("Paste your Discord bot token: ").strip()
    if not token:
        log.error("No token provided, exiting")
        sys.exit(1)
    config["discord_token"] = token
    save_dict_as_toml(config, "cfg/general_config.toml")
    return token


def _rewire_controller(wc, bot_state, play, lobby_automator, stage_manager):
    play.window_controller             = wc
    play.brawler_ranges                = None
    lobby_automator.window_controller  = wc
    stage_manager.window_controller    = wc
    bot_state.get_screenshot           = wc.get_latest_frame
    bot_state.restart_func             = wc.restart_brawl_stars


def game_loop(bot_state: BotState):
    from bluestacks_control import recover_emulator
    from play import Play
    from time_management import TimeManagement
    from lobby_automation import LobbyAutomation
    from stage_manager import StageManager
    from state_finder import get_state
    from _paths import MODEL_MAIN, MODEL_TILES

    try:
        log.info("Waiting for emulator connection")
        window_controller = recover_emulator(bump_heartbeat=heartbeat.bump)
        if window_controller is None:
            log.error("Could not connect to emulator at startup, forcing exit")
            time.sleep(5)
            _hard_exit(1)

        play            = Play(MODEL_MAIN, MODEL_TILES, window_controller)
        time_mgmt       = TimeManagement()
        lobby_automator = LobbyAutomation(window_controller)
        stage_manager   = StageManager(lobby_automator, window_controller)
        trophy_observer = stage_manager.Trophy_observer

        from utils import load_brawlers_info
        brawlers_info   = load_brawlers_info()
        default_brawler = next(iter(brawlers_info), "shelly") if brawlers_info else "shelly"
        play.current_brawler = default_brawler
        bot_state.set_current_brawler(default_brawler)
        log.info(f"Default brawler: {default_brawler}")

        bot_state.session_stats_func = trophy_observer.get_session_stats
        bot_state.reset_stats_func   = trophy_observer.reset_history
        bot_state.get_screenshot     = window_controller.get_latest_frame
        bot_state.restart_func       = window_controller.restart_brawl_stars

        try:
            max_ips = int(load_toml_as_dict("cfg/general_config.toml").get("max_ips", 8))
        except (ValueError, TypeError):
            max_ips = 8

        log.info("Ready, waiting for Start command from Discord")

        s_time             = time.time()
        frame_count        = 0
        last_notif_check   = 0.0
        last_processed_fid = -1

        reconnect_count = 0
        reconnect_start = 0.0

        def _reconnect(force: bool):
            nonlocal window_controller
            try:
                window_controller.keys_up()
            except Exception:
                pass
            try:
                window_controller.close()
            except Exception:
                pass
            new_wc = recover_emulator(bump_heartbeat=heartbeat.bump, force_relaunch=force)
            if new_wc is None:
                log.error("All recovery attempts failed, forcing exit for .bat restart")
                time.sleep(5)
                _hard_exit(1)
            window_controller = new_wc
            _rewire_controller(new_wc, bot_state, play, lobby_automator, stage_manager)

        def _handle_disconnect(err: Exception):
            nonlocal reconnect_count, reconnect_start
            log.error(f"Emulator connection lost: {err}")
            now = time.time()
            if reconnect_count == 0 or now - reconnect_start > RECONNECT_WINDOW_SECS:
                reconnect_start = now
                reconnect_count = 0
            reconnect_count += 1
            force = False
            if reconnect_count >= MAX_RECONNECTS_IN_WINDOW:
                log.warning(f"Rapid reconnects ({reconnect_count} in {now - reconnect_start:.0f}s), forcing kill+relaunch")
                force = True
                reconnect_count = 0
            _reconnect(force)

        while True:
            heartbeat.bump()
            frame_start = time.perf_counter()

            if not bot_state.is_running():
                try:
                    window_controller.keys_up()
                except Exception:
                    pass
                time.sleep(0.5)
                s_time      = time.time()
                frame_count = 0
                bot_state.set_session_start_time(None)
                continue

            if bot_state.take_pending_reload():
                play.reload_config()
                log.info("Config reloaded")

            if bot_state.get_session_start_time() is None:
                bot_state.set_session_start_time(time.time())
                stats = trophy_observer.get_session_stats()
                bot_state.set_last_notif_wins(stats["wins"])
                bot_state.set_last_notif_games(stats["games"])

            elapsed = time.time() - s_time
            if elapsed > 1:
                bot_state.update_ips(frame_count / elapsed)
                trophy_observer.add_time_played(elapsed)
                s_time      = time.time()
                frame_count = 0

            if time.time() - last_notif_check >= 60:
                last_notif_check = time.time()
                stats = trophy_observer.get_session_stats()
                last_wins = bot_state.get_last_notif_wins()
                if stats["wins"] > 0 and stats["wins"] % 10 == 0 and stats["wins"] != last_wins:
                    bot_state.set_last_notif_wins(stats["wins"])
                    bot_state.push_notification(
                        f"{stats['wins']} wins, winrate {stats['winrate']:.1f}%"
                    )
                last_games = bot_state.get_last_notif_games()
                if stats["games"] > 0 and stats["games"] % 50 == 0 and stats["games"] != last_games:
                    bot_state.set_last_notif_games(stats["games"])
                    bot_state.push_notification(
                        f"{stats['games']} games, W:{stats['wins']} L:{stats['losses']} D:{stats['draws']}"
                    )

            try:
                if window_controller.frame_id == last_processed_fid:
                    if time.time() - window_controller.last_frame_time < 5.0:
                        time.sleep(0.005)
                        continue
                last_processed_fid = window_controller.frame_id
                frame = window_controller.screenshot()
            except (ConnectionError, TimeoutError) as e:
                _handle_disconnect(e)
                continue

            if frame is None:
                continue

            try:
                if time_mgmt.state_check():
                    state = get_state(frame)
                    if state != "match":
                        play.time_since_last_proceeding = time.time()
                    stage_manager.do_state(state)

                if time_mgmt.no_detections_check():
                    now = time.time()
                    for last_seen in play.time_since_detections.values():
                        if now - last_seen > NO_DETECTIONS_THRESHOLD_S:
                            window_controller.restart_brawl_stars()
                            play.time_since_detections["player"] = now
                            play.time_since_detections["enemy"]  = now
                            break

                if time_mgmt.idle_check():
                    lobby_automator.check_for_idle(frame)

                play.main(frame, bot_state.get_current_brawler() or default_brawler)

            except (ConnectionError, OSError, TimeoutError) as e:
                _handle_disconnect(e)
                continue

            frame_count += 1

            work   = time.perf_counter() - frame_start
            target = 1.0 / max_ips
            if work < target:
                time.sleep(target - work)

    except Exception as e:
        log.error(f"Fatal error in game loop: {e}")
        import traceback
        traceback.print_exc()
        bot_state.set_running(False)
        bot_state.set_stuck_alert(True)
        log.error("Game loop is dead, forcing process exit in 30s for .bat restart")

        def _delayed_exit():
            time.sleep(30)
            _hard_exit(1)
        threading.Thread(target=_delayed_exit, daemon=True).start()


def setup():
    print("Brawl Industry setup\n")
    config_path = "cfg/general_config.toml"
    config      = load_toml_as_dict(config_path)

    current_token = config.get("discord_token", "").strip()
    if current_token:
        masked = current_token[:15] + "..." if len(current_token) > 20 else current_token
        print(f"  token: {masked}")
        if input("  keep it? [Y/n]: ").strip().lower() in ("n", "no"):
            current_token = ""
    if not current_token:
        current_token = input("  paste Discord token: ").strip()

    current_port = config.get("emulator_port", 5037)
    port_input   = input(f"  emulator port (current: {current_port}): ").strip()
    if port_input:
        try:
            current_port = int(port_input)
        except ValueError:
            print(f"  not a number, keeping {current_port}")

    config["discord_token"] = current_token
    config["emulator_port"] = current_port
    save_dict_as_toml(config, config_path)

    with open("start.bat", "w") as f:
        f.write(
            "@echo off\n"
            "cd /d \"%~dp0\"\n"
            ":restart_loop\n"
            "echo [%date% %time%] Starting Brawl Industry...\n"
            "python main.py\n"
            "set EXITCODE=%errorlevel%\n"
            "echo [%date% %time%] Process exited (code %EXITCODE%). Restarting in 10s...\n"
            "timeout /t 10 /nobreak >nul\n"
            "goto restart_loop\n"
        )

    print(f"\nSetup done (port {current_port}). Double-click start.bat to launch the bot.")


def main():
    if "--setup" in sys.argv:
        check_dependencies()
        setup()
        return

    check_dependencies()
    start_config_watcher()

    if utils.api_base_url != "localhost":
        update_wall_model_classes()
        if not current_wall_model_is_latest():
            log.info("New wall detection model found, downloading")
            get_latest_wall_model_file()

    token     = ensure_discord_token()
    bot_state = BotState()

    threading.Thread(target=_heartbeat_watchdog, daemon=True).start()
    threading.Thread(target=game_loop, args=(bot_state,), daemon=True).start()

    from discord_bot import create_discord_bot
    discord_bot = create_discord_bot(bot_state)
    while True:
        try:
            discord_bot.run(token, log_handler=None)
        except Exception as e:
            log.error(f"Discord bot crashed: {e}, restarting in 15s")
            time.sleep(15)
            try:
                discord_bot = create_discord_bot(bot_state)
            except Exception as e2:
                log.error(f"Failed to recreate Discord bot: {e2}")


if __name__ == "__main__":
    main()
