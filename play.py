import math
import random
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from state_finder import get_state
from detect import Detect
from pathfinding import NavGrid, pixel_dir_to_wasd
from utils import load_toml_as_dict, count_hsv_pixels, load_brawlers_info
from brawl_industry_logger import get_logger

log = get_logger("brawl_industry.play")

_PARALLEL_EXEC = ThreadPoolExecutor(max_workers=2)

_lobby_cfg            = load_toml_as_dict("./cfg/lobby_config.toml")
_pca                  = _lobby_cfg.get("pixel_counter_crop_area", {})
super_crop_area       = _pca.get("super",       [1460, 830, 1560,  930])
gadget_crop_area      = _pca.get("gadget",      [1580, 930, 1700, 1050])
hypercharge_crop_area = _pca.get("hypercharge", [1350, 940, 1450, 1050])


class Movement:

    def __init__(self, window_controller, bot_config=None, time_config=None):
        if bot_config is None:
            bot_config  = load_toml_as_dict("cfg/bot_config.toml")
        if time_config is None:
            time_config = load_toml_as_dict("cfg/time_thresholds.toml")
        self.fix_movement_keys = {
            "delay_to_trigger": bot_config.get("unstuck_movement_delay",     3.0),
            "duration":         bot_config.get("unstuck_movement_hold_time", 1.5),
            "toggled":          False,
            "started_at":       time.time(),
            "fixed":            "",
        }
        gadget_value           = bot_config.get("bot_uses_gadgets", "yes")
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.super_threshold        = time_config.get("super",          0.1)
        self.gadget_threshold       = time_config.get("gadget",         0.5)
        self.hypercharge_threshold  = time_config.get("hypercharge",    1.0)
        self.walls_threshold        = time_config.get("wall_detection", 0.2)
        self.keep_walls_in_memory   = self.walls_threshold <= 1
        self.last_walls_data        = []
        self.keys_hold              = []
        self.time_since_different_movement  = time.time()
        self.time_since_gadget_checked      = time.time()
        self.is_gadget_ready                = False
        self.time_since_hypercharge_checked = time.time()
        self.is_hypercharge_ready           = False
        self.window_controller              = window_controller
        self.TILE_SIZE = bot_config.get("tile_size_pixels", 60)

    @staticmethod
    def get_enemy_pos(enemy):
        return (enemy[0] + enemy[2]) / 2, (enemy[1] + enemy[3]) / 2

    @staticmethod
    def get_player_pos(player_data):
        return (player_data[0] + player_data[2]) / 2, (player_data[1] + player_data[3]) / 2

    @staticmethod
    def get_distance(enemy_coords, player_coords):
        return math.hypot(enemy_coords[0] - player_coords[0], enemy_coords[1] - player_coords[1])

    @staticmethod
    def is_there_enemy(enemy_data):
        return bool(enemy_data)

    @staticmethod
    def get_horizontal_move_key(direction_x, opposite=False):
        if opposite:
            return "A" if direction_x > 0 else "D"
        return "D" if direction_x > 0 else "A"

    @staticmethod
    def get_vertical_move_key(direction_y, opposite=False):
        if opposite:
            return "W" if direction_y > 0 else "S"
        return "S" if direction_y > 0 else "W"

    def attack(self, touch_up=True, touch_down=True):
        self.window_controller.press_key("M", touch_up=touch_up, touch_down=touch_down)

    def use_hypercharge(self):
        log.info("Using hypercharge")
        self.window_controller.press_key("H")

    def use_gadget(self):
        log.info("Using gadget")
        self.window_controller.press_key("G")

    def use_super(self):
        log.info("Using super")
        self.window_controller.press_key("E")

    @staticmethod
    def reverse_movement(movement):
        return movement.lower().translate(str.maketrans("wasd", "sdwa"))

    @staticmethod
    def _closest_teammate_pos(player_pos, teammate_data):
        if not teammate_data:
            return None
        best_pos  = None
        best_dist = float("inf")
        for t in teammate_data:
            pos  = ((t[0] + t[2]) / 2, (t[1] + t[3]) / 2)
            dist = math.hypot(pos[0] - player_pos[0], pos[1] - player_pos[1])
            if dist < best_dist:
                best_dist = dist
                best_pos  = pos
        return best_pos

    def get_strafe_movement(self, player_pos, enemy_coords, walls, teammate_data=None):
        dx = enemy_coords[0] - player_pos[0]
        dy = enemy_coords[1] - player_pos[1]

        candidates = ["W", "S"] if abs(dx) >= abs(dy) else ["A", "D"]

        t_pos = self._closest_teammate_pos(player_pos, teammate_data)
        if t_pos:
            t_dx = t_pos[0] - player_pos[0]
            t_dy = t_pos[1] - player_pos[1]

            def _toward_teammate(move):
                if move == "W": return -t_dy
                if move == "S": return  t_dy
                if move == "A": return -t_dx
                if move == "D": return  t_dx
                return 0

            candidates.sort(key=_toward_teammate, reverse=True)

        for move in candidates:
            if not self.is_path_blocked(player_pos, move, walls):
                return move
        return candidates[0]

    def get_retreat_movement(self, player_pos, enemy_coords, walls, teammate_data=None):
        dx = enemy_coords[0] - player_pos[0]
        dy = enemy_coords[1] - player_pos[1]
        h_away = self.get_horizontal_move_key(dx, opposite=True)
        v_away = self.get_vertical_move_key(dy, opposite=True)

        t_pos = self._closest_teammate_pos(player_pos, teammate_data)
        if t_pos:
            t_dx = t_pos[0] - player_pos[0]
            t_dy = t_pos[1] - player_pos[1]
            h_toward = self.get_horizontal_move_key(t_dx)
            v_toward = self.get_vertical_move_key(t_dy)
            candidates = [
                h_toward + v_away,
                h_away   + v_toward,
                h_toward + v_toward,
                h_away   + v_away,
                h_away,
                v_away,
            ]
        else:
            candidates = [h_away + v_away, h_away, v_away]

        for move in candidates:
            if not self.is_path_blocked(player_pos, move, walls):
                return move

        if self._nav_grid is not None:
            dist = max(1.0, math.hypot(dx, dy))
            cell = self.TILE_SIZE * self.window_controller.scale_factor
            retreat_target = (
                player_pos[0] - (dx / dist) * cell * 3,
                player_pos[1] - (dy / dist) * cell * 3,
            )
            if t_pos:
                retreat_target = (
                    (retreat_target[0] + t_pos[0]) / 2,
                    (retreat_target[1] + t_pos[1]) / 2,
                )
            nxt = self._nav_grid.find_next_step(player_pos, retreat_target)
            if nxt is not None:
                return pixel_dir_to_wasd(player_pos, nxt)

        return h_away + v_away

    def unstuck_movement_if_needed(self, movement, current_time=None):
        if current_time is None:
            current_time = time.time()
        movement = movement.lower()

        if self.fix_movement_keys["toggled"]:
            if current_time - self.fix_movement_keys["started_at"] > self.fix_movement_keys["duration"]:
                self.fix_movement_keys["toggled"] = False
            return self.fix_movement_keys["fixed"]

        if "".join(self.keys_hold) != movement and movement[::-1] != "".join(self.keys_hold):
            self.time_since_different_movement = current_time

        if current_time - self.time_since_different_movement > self.fix_movement_keys["delay_to_trigger"]:
            reversed_movement = self.reverse_movement(movement)
            if reversed_movement == "s":
                reversed_movement = random.choice(["aw", "dw"])
            elif reversed_movement == "w":
                reversed_movement = random.choice(["as", "ds"])
            self.fix_movement_keys["fixed"]      = reversed_movement
            self.fix_movement_keys["toggled"]    = True
            self.fix_movement_keys["started_at"] = current_time
            return reversed_movement

        return movement


class Play(Movement):

    def __init__(self, main_info_model, tile_detector_model, window_controller):
        bot_config  = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_thresholds.toml")
        super().__init__(window_controller, bot_config, time_config)

        self.Detect_main_info = Detect(main_info_model, classes=["enemy", "teammate", "player"])
        self.Detect_tile_detector = Detect(
            tile_detector_model,
            classes=bot_config.get("wall_model_classes", ["wall", "bush", "close_bush"]),
        )

        self._nav_grid = None

        self.time_since_movement            = time.time()
        self.time_since_gadget_checked      = time.time()
        self.time_since_hypercharge_checked = time.time()
        self.time_since_super_checked       = time.time()
        self.time_since_walls_checked       = time.time()
        self.time_since_movement_change     = time.time()
        self.time_since_player_last_found   = time.time()
        self.current_brawler       = None
        self.is_hypercharge_ready  = False
        self.is_gadget_ready       = False
        self.is_super_ready        = False
        self.brawlers_info         = load_brawlers_info()
        self.brawler_ranges        = None
        self.time_since_detections = {
            "player": time.time(),
            "enemy":  time.time(),
        }
        self.time_since_last_proceeding = time.time()

        self.last_movement       = ""
        self.last_movement_time  = time.time()
        self.wall_history_length = 3
        self.wall_history        = deque(maxlen=self.wall_history_length)

        self.minimum_movement_delay                    = bot_config.get("minimum_movement_delay",                    0.1)
        self.no_detection_proceed_delay                = time_config.get("no_detection_proceed",                     6.5)
        self.gadget_pixels_minimum                     = bot_config.get("gadget_pixels_minimum",                  1300.0)
        self.hypercharge_pixels_minimum                = bot_config.get("hypercharge_pixels_minimum",             2000.0)
        self.super_pixels_minimum                      = bot_config.get("super_pixels_minimum",                   2400.0)
        self.wall_detection_confidence                 = bot_config.get("wall_detection_confidence",                 0.9)
        self.entity_detection_confidence               = bot_config.get("entity_detection_confidence",               0.6)
        self.time_since_holding_attack                 = None
        self.seconds_to_hold_attack_after_reaching_max = bot_config.get("seconds_to_hold_attack_after_reaching_max", 1.5)

    def reload_config(self):
        bot_config  = load_toml_as_dict("cfg/bot_config.toml")
        time_config = load_toml_as_dict("cfg/time_thresholds.toml")

        self.brawler_ranges = None
        self._nav_grid      = None

        gadget_value           = bot_config.get("bot_uses_gadgets", "yes")
        self.should_use_gadget = str(gadget_value).lower() in ("yes", "true", "1")
        self.super_threshold        = time_config.get("super",          0.1)
        self.gadget_threshold       = time_config.get("gadget",         0.5)
        self.hypercharge_threshold  = time_config.get("hypercharge",    1.0)
        self.walls_threshold        = time_config.get("wall_detection", 0.2)
        self.keep_walls_in_memory   = self.walls_threshold <= 1

        self.minimum_movement_delay                    = bot_config.get("minimum_movement_delay",                    0.1)
        self.no_detection_proceed_delay                = time_config.get("no_detection_proceed",                     6.5)
        self.gadget_pixels_minimum                     = bot_config.get("gadget_pixels_minimum",                  1300.0)
        self.hypercharge_pixels_minimum                = bot_config.get("hypercharge_pixels_minimum",             2000.0)
        self.super_pixels_minimum                      = bot_config.get("super_pixels_minimum",                   2400.0)
        self.wall_detection_confidence                 = bot_config.get("wall_detection_confidence",                 0.9)
        self.entity_detection_confidence               = bot_config.get("entity_detection_confidence",               0.6)
        self.seconds_to_hold_attack_after_reaching_max = bot_config.get("seconds_to_hold_attack_after_reaching_max", 1.5)

    def load_brawler_ranges(self, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        scale = self.window_controller.scale_factor
        ranges = {}
        for brawler, info in brawlers_info.items():
            s  = info.get("safe_range",   300)
            a  = info.get("attack_range", 400)
            su = info.get("super_range",  500)
            ranges[brawler] = [int(s * scale), int(a * scale), int(su * scale)]
        return ranges

    @staticmethod
    def can_attack_through_walls(brawler, skill_type, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        info = brawlers_info.get(brawler)
        if info is None:
            return False
        if skill_type == "attack":
            return info.get("ignore_walls_for_attacks", False)
        if skill_type == "super":
            return info.get("ignore_walls_for_supers", False)
        raise ValueError("skill_type must be either 'attack' or 'super'")

    @staticmethod
    def must_brawler_hold_attack(brawler, brawlers_info=None):
        if not brawlers_info:
            brawlers_info = load_brawlers_info()
        info = brawlers_info.get(brawler)
        if info is None:
            return False
        return info.get("hold_attack", 0) > 0

    @staticmethod
    def walls_block_line_of_sight(p1, p2, walls):
        if not walls:
            return False
        p1_t = (int(p1[0]), int(p1[1]))
        p2_t = (int(p2[0]), int(p2[1]))
        min_x = min(p1_t[0], p2_t[0]); max_x = max(p1_t[0], p2_t[0])
        min_y = min(p1_t[1], p2_t[1]); max_y = max(p1_t[1], p2_t[1])
        w = np.array(walls, dtype=np.float32)

        mask = ~(
            (max_x < w[:, 0]) | (min_x > w[:, 2]) |
            (max_y < w[:, 1]) | (min_y > w[:, 3])
        )
        for wall in w[mask]:
            rect = (int(wall[0]), int(wall[1]), int(wall[2] - wall[0]), int(wall[3] - wall[1]))
            if cv2.clipLine(rect, p1_t, p2_t)[0]:
                return True
        return False

    def no_enemy_movement(self, player_data, walls):
        player_position = self.get_player_pos(player_data)

        for move in ["W", "WA", "WD"]:
            if not self.is_path_blocked(player_position, move, walls):
                return move

        if self._nav_grid is not None:
            goal_x = self._nav_grid.cols * self._nav_grid.cell_size / 2
            goal_y = self._nav_grid.cell_size * 2
            nxt = self._nav_grid.find_next_step(player_position, (goal_x, goal_y))
            if nxt is not None:
                return pixel_dir_to_wasd(player_position, nxt)

        for move in ["A", "D", "S"]:
            if not self.is_path_blocked(player_position, move, walls):
                return move
        return "W"

    def is_enemy_hittable(self, player_pos, enemy_pos, walls, skill_type):
        if self.can_attack_through_walls(self.current_brawler, skill_type, self.brawlers_info):
            return True
        return not self.walls_block_line_of_sight(player_pos, enemy_pos, walls)

    def find_closest_enemy(self, enemy_data, player_coords, walls, skill_type):
        closest_hittable_dist   = float("inf")
        closest_unhittable_dist = float("inf")
        closest_hittable   = None
        closest_unhittable = None
        for enemy in enemy_data:
            enemy_pos = self.get_enemy_pos(enemy)
            distance  = self.get_distance(enemy_pos, player_coords)
            if self.is_enemy_hittable(player_coords, enemy_pos, walls, skill_type):
                if distance < closest_hittable_dist:
                    closest_hittable_dist = distance
                    closest_hittable = (enemy_pos, distance)
            else:
                if distance < closest_unhittable_dist:
                    closest_unhittable_dist = distance
                    closest_unhittable = (enemy_pos, distance)
        if closest_hittable:
            return closest_hittable
        if closest_unhittable:
            return closest_unhittable
        return (None, None)

    def get_main_data(self, frame):
        return self.Detect_main_info.detect_objects(frame, conf_thresh=self.entity_detection_confidence)

    def is_path_blocked(self, player_pos, move_direction, walls, distance=None):
        if distance is None:
            distance = self.TILE_SIZE * self.window_controller.scale_factor
        dx, dy = 0, 0
        md = move_direction.lower()
        if "w" in md: dy -= distance
        if "s" in md: dy += distance
        if "a" in md: dx -= distance
        if "d" in md: dx += distance
        new_pos = (player_pos[0] + dx, player_pos[1] + dy)
        return self.walls_block_line_of_sight(player_pos, new_pos, walls)

    @staticmethod
    def validate_game_data(data):
        if "player" not in data or not data["player"]:
            return None
        data.setdefault("enemy",    None)
        data.setdefault("teammate", None)
        if "wall" not in data or not data["wall"]:
            data["wall"] = []
        return data

    def track_no_detections(self, data):
        if not data:
            data = {"enemy": None, "player": None}
        now = time.time()
        for key in self.time_since_detections:
            if key in data and data[key]:
                self.time_since_detections[key] = now

    def do_movement(self, movement):
        movement = movement.lower()
        keys_to_keyDown = [k for k in "wasd" if k in movement]
        if keys_to_keyDown:
            self.window_controller.keys_down(keys_to_keyDown)
        else:
            self.window_controller.keys_up()
        self.keys_hold = keys_to_keyDown

    def get_brawler_range(self, brawler):
        if self.brawler_ranges is None:
            self.brawler_ranges = self.load_brawler_ranges(self.brawlers_info)
        ranges = self.brawler_ranges.get(brawler)
        if ranges is None:
            log.warning(f"Brawler '{brawler}' not found in brawlers_info, using default ranges")
            scale = self.window_controller.scale_factor
            return [int(300 * scale), int(400 * scale), int(500 * scale)]
        return ranges

    def loop(self, brawler, data, current_time):
        movement = self.get_movement(
            player_data=data["player"][0],
            enemy_data=data["enemy"],
            walls=data["wall"],
            brawler=brawler,
            current_time=current_time,
            teammate_data=data.get("teammate"),
        )
        if current_time - self.time_since_movement > self.minimum_movement_delay:
            movement = self.unstuck_movement_if_needed(movement, current_time)
            self.do_movement(movement)
            self.time_since_movement = current_time
        return movement

    def _crop(self, frame, crop_area):
        wr, hr = self.window_controller.width_ratio, self.window_controller.height_ratio
        x1, y1 = int(crop_area[0] * wr), int(crop_area[1] * hr)
        x2, y2 = int(crop_area[2] * wr), int(crop_area[3] * hr)
        return frame[y1:y2, x1:x2]

    def check_if_hypercharge_ready(self, frame):
        purple_pixels = count_hsv_pixels(self._crop(frame, hypercharge_crop_area), (137, 158, 159), (179, 255, 255))
        return purple_pixels > self.hypercharge_pixels_minimum

    def check_if_gadget_ready(self, frame):
        green_pixels = count_hsv_pixels(self._crop(frame, gadget_crop_area), (57, 219, 165), (62, 255, 255))
        return green_pixels > self.gadget_pixels_minimum

    def check_if_super_ready(self, frame):
        yellow_pixels = count_hsv_pixels(self._crop(frame, super_crop_area), (17, 170, 200), (27, 255, 255))
        return yellow_pixels > self.super_pixels_minimum

    def get_tile_data(self, frame):
        return self.Detect_tile_detector.detect_objects(frame, conf_thresh=self.wall_detection_confidence)

    def process_tile_data(self, tile_data):
        walls = []
        for class_name, boxes in tile_data.items():
            if class_name != "bush":
                walls.extend(boxes)
        self.wall_history.append(walls)
        return self.combine_walls_from_history()

    def combine_walls_from_history(self):
        return [list(w) for w in {tuple(w) for frame_walls in self.wall_history for w in frame_walls}]

    def get_movement(self, player_data, enemy_data, walls, brawler, current_time=None, teammate_data=None):
        if current_time is None:
            current_time = time.time()

        brawler_info = self.brawlers_info.get(brawler)
        if not brawler_info:
            log.warning(f"Brawler '{brawler}' not found in brawlers_info, skipping movement AI this frame")
            return "W"

        must_hold = self.must_brawler_hold_attack(brawler, self.brawlers_info)

        if (must_hold
                and self.time_since_holding_attack is not None
                and current_time - self.time_since_holding_attack
                    >= brawler_info.get("hold_attack", 0) + self.seconds_to_hold_attack_after_reaching_max):
            self.attack(touch_up=True, touch_down=False)
            self.time_since_holding_attack = None

        safe_range, attack_range, super_range = self.get_brawler_range(brawler)
        player_pos = self.get_player_pos(player_data)

        if not self.is_there_enemy(enemy_data):
            return self.no_enemy_movement(player_data, walls)

        enemy_coords, enemy_distance = self.find_closest_enemy(enemy_data, player_pos, walls, "attack")
        if enemy_coords is None:
            return self.no_enemy_movement(player_data, walls)

        direction_x = enemy_coords[0] - player_pos[0]
        direction_y = enemy_coords[1] - player_pos[1]

        if enemy_distance > attack_range:
            move_horizontal = self.get_horizontal_move_key(direction_x)
            move_vertical   = self.get_vertical_move_key(direction_y)
            movement_options = [move_horizontal + move_vertical, move_vertical, move_horizontal]

            movement = None
            for move in movement_options:
                if not self.is_path_blocked(player_pos, move, walls):
                    movement = move
                    break

            if movement is None and self._nav_grid is not None:
                nxt = self._nav_grid.find_next_step(player_pos, enemy_coords)
                if nxt is not None:
                    movement = pixel_dir_to_wasd(player_pos, nxt)

            if movement is None:
                movement = move_horizontal + move_vertical

        elif enemy_distance > safe_range:
            movement = self.get_strafe_movement(player_pos, enemy_coords, walls, teammate_data)
        else:
            movement = self.get_retreat_movement(player_pos, enemy_coords, walls, teammate_data)

        if movement != self.last_movement:
            if current_time - self.last_movement_time >= self.minimum_movement_delay:
                self.last_movement      = movement
                self.last_movement_time = current_time
            else:
                movement = self.last_movement
        else:
            self.last_movement_time = current_time

        if self.is_super_ready and self.time_since_holding_attack is None:
            super_type     = brawler_info.get("super_type", "other")
            enemy_hittable = self.is_enemy_hittable(player_pos, enemy_coords, walls, "super")
            if enemy_hittable and (
                    enemy_distance <= super_range
                    or super_type in ["spawnable", "other"]
                    or (brawler in ["stu", "surge"]
                        and super_type == "charge"
                        and enemy_distance <= super_range + attack_range)
            ):
                if self.is_hypercharge_ready:
                    self.use_hypercharge()
                    self.time_since_hypercharge_checked = current_time
                    self.is_hypercharge_ready = False
                self.use_super()
                self.time_since_super_checked = current_time
                self.is_super_ready = False

        if enemy_distance <= attack_range:
            if self.is_enemy_hittable(player_pos, enemy_coords, walls, "attack"):
                if self.should_use_gadget and self.is_gadget_ready and self.time_since_holding_attack is None:
                    self.use_gadget()
                    self.time_since_gadget_checked = current_time
                    self.is_gadget_ready = False

                if not must_hold:
                    self.attack()
                else:
                    if self.time_since_holding_attack is None:
                        self.time_since_holding_attack = current_time
                        self.attack(touch_up=False, touch_down=True)
                    elif current_time - self.time_since_holding_attack >= brawler_info.get("hold_attack", 0):
                        self.attack(touch_up=True, touch_down=False)
                        self.time_since_holding_attack = None

        return movement

    def main(self, frame, brawler):
        current_time = time.time()

        if current_time - self.time_since_walls_checked > self.walls_threshold:
            main_future = _PARALLEL_EXEC.submit(self.get_main_data, frame)
            tile_future = _PARALLEL_EXEC.submit(self.get_tile_data, frame)
            data = main_future.result()
            tile_data = tile_future.result()
            walls     = self.process_tile_data(tile_data)
            self.time_since_walls_checked = current_time
            self.last_walls_data          = walls
            data["wall"] = walls

            h, w = frame.shape[:2]
            cell = self.TILE_SIZE * self.window_controller.scale_factor
            self._nav_grid = NavGrid(walls, w, h, cell)
        else:
            data = self.get_main_data(frame)
            if self.keep_walls_in_memory:
                data["wall"] = self.last_walls_data

        data = self.validate_game_data(data)
        self.track_no_detections(data)

        if data is not None:
            self.time_since_player_last_found = current_time

        if data is None:
            if current_time - self.time_since_player_last_found > 1.0:
                self.window_controller.keys_up()
            self.time_since_different_movement = current_time
            if current_time - self.time_since_last_proceeding > self.no_detection_proceed_delay:
                if get_state(frame) == "match":
                    self.window_controller.press_key("Q")
                self.time_since_last_proceeding = current_time
            return

        self.time_since_last_proceeding = current_time

        if current_time - self.time_since_hypercharge_checked > self.hypercharge_threshold:
            self.is_hypercharge_ready           = self.check_if_hypercharge_ready(frame)
            self.time_since_hypercharge_checked = current_time

        if current_time - self.time_since_gadget_checked > self.gadget_threshold:
            self.is_gadget_ready           = self.check_if_gadget_ready(frame)
            self.time_since_gadget_checked = current_time

        if current_time - self.time_since_super_checked > self.super_threshold:
            self.is_super_ready           = self.check_if_super_ready(frame)
            self.time_since_super_checked = current_time

        self.loop(brawler, data, current_time)
