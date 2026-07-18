import subprocess
import threading
import time

from brawl_industry_logger import get_logger
from utils import load_toml_as_dict
from window_controller import (
    WindowController,
    _resolve_bluestacks_instance,
    BLUESTACKS_EXE,
    restart_adb_server,
)

log = get_logger("brawl_industry.recovery")

WC_CONNECT_TIMEOUT    = 30
FRAMES_VERIFY_TIMEOUT = 12
QUICK_RETRY_ATTEMPTS  = 3
RELAUNCH_CYCLES       = ((60, 3), (90, 3))


def sleep_hb(seconds: float, bump_heartbeat=None):
    end = time.time() + seconds
    while time.time() < end:
        if bump_heartbeat is not None:
            bump_heartbeat()
        time.sleep(min(1.0, max(0.0, end - time.time())))


def _run_ps(cmd: str, timeout: int = 15):
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, timeout=timeout,
        )
    except Exception as e:
        log.warning(f"PowerShell call failed: {e}")


def cleanup_adb(port: int):
    log.info(f"Cleaning up ADB for port {port}")
    try:
        subprocess.run(
            ["adb", "disconnect", f"127.0.0.1:{port}"],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def kill_bluestacks(instance_name: str, port: int):
    log.info(f"Killing BlueStacks instance '{instance_name}'")
    _run_ps(
        "Get-CimInstance Win32_Process -Filter \"Name='HD-Player.exe'\" | "
        f"Where-Object {{ $_.CommandLine -match '{instance_name}' }} | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }"
    )
    try:
        subprocess.run(
            ["taskkill", "/F", "/FI", f"WINDOWTITLE eq {instance_name}*"],
            capture_output=True, timeout=15,
        )
    except Exception:
        pass
    time.sleep(3)
    cleanup_adb(port)


def _create_wc_with_timeout(timeout: int, bump_heartbeat) -> WindowController:
    result, error = [None], [None]

    def _worker():
        try:
            result[0] = WindowController()
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    deadline = time.time() + timeout
    while t.is_alive() and time.time() < deadline:
        if bump_heartbeat is not None:
            bump_heartbeat()
        time.sleep(1)

    if t.is_alive():
        raise TimeoutError(f"WindowController creation hung for {timeout}s")

    if error[0]:
        raise error[0]
    return result[0]


def _verify_frames_flowing(wc: WindowController, timeout: int, bump_heartbeat) -> bool:
    initial_id = wc.frame_id
    deadline   = time.time() + timeout

    got_first = False
    while time.time() < deadline:
        if bump_heartbeat is not None:
            bump_heartbeat()
        time.sleep(0.5)
        if wc.frame_id > initial_id:
            got_first = True
            break
    if not got_first:
        return False

    drain_end = min(time.time() + 4.0, deadline)
    while time.time() < drain_end:
        if bump_heartbeat is not None:
            bump_heartbeat()
        time.sleep(0.5)
    if time.time() >= deadline:
        return False

    checkpoint = wc.frame_id
    while time.time() < deadline:
        if bump_heartbeat is not None:
            bump_heartbeat()
        time.sleep(0.5)
        if wc.frame_id > checkpoint:
            return True
    return False


def try_connect(bump_heartbeat=None) -> WindowController | None:
    try:
        wc = _create_wc_with_timeout(WC_CONNECT_TIMEOUT, bump_heartbeat)
    except TimeoutError as e:
        log.error(str(e))
        return None
    except Exception as e:
        log.error(f"WindowController init failed: {e}")
        return None

    if _verify_frames_flowing(wc, FRAMES_VERIFY_TIMEOUT, bump_heartbeat):
        return wc

    log.warning("Connected but frames not flowing")
    try:
        wc.close()
    except Exception:
        pass
    return None


def full_kill_and_relaunch(instance: str, port: int, boot_wait: int,
                           post_retries: int, bump_heartbeat=None) -> WindowController | None:
    kill_bluestacks(instance, port)

    log.info(f"Relaunching BlueStacks instance {instance}, waiting {boot_wait}s")
    subprocess.Popen([BLUESTACKS_EXE, "--instance", instance])
    sleep_hb(boot_wait, bump_heartbeat)

    for retry in range(1, post_retries + 1):
        log.info(f"Post-relaunch connection attempt {retry}/{post_retries}")
        wc = try_connect(bump_heartbeat)
        if wc:
            return wc
        if retry < post_retries:
            sleep_hb(10, bump_heartbeat)
            cleanup_adb(port)
    return None


def recover_emulator(bump_heartbeat=None, force_relaunch: bool = False) -> WindowController | None:
    if not force_relaunch:
        for attempt in range(1, QUICK_RETRY_ATTEMPTS + 1):
            wait = 5 * attempt
            log.info(f"Reconnection attempt {attempt}/{QUICK_RETRY_ATTEMPTS} in {wait}s")
            sleep_hb(wait, bump_heartbeat)
            wc = try_connect(bump_heartbeat)
            if wc:
                log.info("Reconnected")
                return wc

        log.warning("Quick reconnect exhausted, bouncing adb daemon")
        restart_adb_server()
        wc = try_connect(bump_heartbeat)
        if wc:
            log.info("Reconnected after adb server restart")
            return wc

    log.warning("Entering kill+relaunch cycles")
    try:
        cfg = load_toml_as_dict("cfg/general_config.toml")
        config_port = int(cfg.get("emulator_port", 5037))
        instance, live_port = _resolve_bluestacks_instance(config_port)
    except Exception as e:
        log.error(f"Could not resolve BlueStacks instance: {e}")
        return None

    if not instance:
        log.error("BlueStacks instance name unknown, cannot relaunch")
        return None

    for i, (boot_wait, retries) in enumerate(RELAUNCH_CYCLES, start=1):
        log.info(f"Kill+Relaunch cycle {i}/{len(RELAUNCH_CYCLES)}")
        wc = full_kill_and_relaunch(instance, live_port, boot_wait, retries, bump_heartbeat)
        if wc:
            log.info(f"Reconnected after relaunch cycle {i}")
            return wc

    return None
