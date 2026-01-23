import cv2
import csv
import os
import sys
import numpy as np


class RegionSelector:
    def __init__(self, frame, num_rewards, modes_to_use=None, initial_regions=None):
        self.frame = frame.copy()
        self.clone = self.frame.copy()
        self.window_name = "Teruletek megjelolese"
        self.drawing = False
        self.start_point = None
        self.current_rect = None

        if modes_to_use is None:
            modes_to_use = ["ablak", "folyoso", "jutalom"]
        full_modes = {
            "ablak": {"max": 8, "color": (0, 255, 0), "regions": []},
            "jutalom": {"max": num_rewards, "color": (0, 0, 255), "regions": []},
            "folyoso": {"max": 4, "color": (255, 0, 0), "regions": []}
        }
        self.modes = {k: full_modes[k] for k in modes_to_use}

        if initial_regions:
            for key in self.modes.keys():
                if key in initial_regions:
                    self.modes[key]["regions"] = initial_regions[key]

        self.mode_keys = list(self.modes.keys())
        self.current_mode_idx = 0
        self.temp_corridor_points = []

    def get_current_mode(self):
        return self.mode_keys[self.current_mode_idx]

    def mouse_callback(self, event, x, y, flags, param):
        mode = self.get_current_mode()

        if mode == "folyoso":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_corridor_points.append((x, y))
                if len(self.temp_corridor_points) == 4:
                    self.modes["folyoso"]["regions"].append(self.temp_corridor_points.copy())
                    self.temp_corridor_points.clear()
        else:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.modes[mode]["regions"]) < self.modes[mode]["max"]:
                    self.start_point = (x, y)
                    self.drawing = True
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.current_rect = (self.start_point, (x, y))
            elif event == cv2.EVENT_LBUTTONUP and self.drawing:
                self.drawing = False
                end_point = (x, y)
                self.modes[mode]["regions"].append((self.start_point, end_point))
                self.current_rect = None

    def run(self, allow_navigation=False):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        user_action = None

        while True:
            display_frame = self.clone.copy()

            for m, config in self.modes.items():
                if m == "folyoso":
                    for idx, poly in enumerate(config["regions"]):
                        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [pts], isClosed=True, color=config["color"], thickness=1)
                        cv2.putText(display_frame, f"Folyoso {idx + 1}",
                                    (poly[0][0], poly[0][1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config["color"], 1)
                else:
                    for idx, (start, end) in enumerate(config["regions"]):
                        cv2.rectangle(display_frame, start, end, config["color"], 1)
                        cv2.putText(display_frame, f"{m.capitalize()} {idx + 1}",
                                    (start[0], start[1] - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, config["color"], 1)

            if self.get_current_mode() == "folyoso":
                for pt in self.temp_corridor_points:
                    cv2.circle(display_frame, pt, 5, (255, 0, 0), -1)
                if len(self.temp_corridor_points) > 1:
                    for i in range(len(self.temp_corridor_points) - 1):
                        cv2.line(display_frame, self.temp_corridor_points[i],
                                 self.temp_corridor_points[i + 1], (255, 0, 0), 1)

            if self.drawing and self.current_rect:
                cv2.rectangle(display_frame, self.current_rect[0], self.current_rect[1],
                              self.modes[self.get_current_mode()]["color"], 1)

            instructions1 = f"Mod: {self.get_current_mode().upper()} rajzolas ({len(self.modes[self.get_current_mode()]['regions'])}/{self.modes[self.get_current_mode()]['max']})"
            if allow_navigation:
                instructions2 = "z = torles | a = kijeloles | c = inditas"
            else:
                instructions2 = "n=kovetkezo mod | z=torles | c=inditas"

            cv2.putText(display_frame, instructions1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, instructions2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):
                if self.current_mode_idx < len(self.mode_keys) - 1:
                    self.current_mode_idx += 1
                else:
                    print("Minden mód kész. Nyomj 'c'-t az induláshoz.")
            elif key == ord('z'):
                mode = self.get_current_mode()
                if self.modes[mode]["regions"]:
                    self.modes[mode]["regions"].pop()
                    print(f"Utolsó {mode} törölve.")
            elif key == ord('a') and allow_navigation:
                user_action = 'a'
                print("Ablak/folyosó kijelölő megnyitása.")
                break
            elif key == ord('c'):
                user_action = 'c'
                print("Területek kiválasztva. Folytatás...")
                break
            elif key == ord('q') or key == 27:
                print("Kilépés a programból.")
                cv2.destroyWindow(self.window_name)
                sys.exit(0)

        cv2.destroyWindow(self.window_name)

        regions = {}
        for key, val in self.modes.items():
            if key == "folyoso":
                regions[key] = val["regions"]
            else:
                rects = []
                for start, end in val["regions"]:
                    x1, y1 = start
                    x2, y2 = end
                    rects.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                regions[key] = rects
        return regions, user_action


def save_regions_to_single_csv(filepath, corridors, windows):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["type", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"])
        for region in corridors:
            row = ["folyoso"]
            for pt in region:
                row.extend(pt)
            writer.writerow(row)
        for rect in windows:
            x1, y1, x2, y2 = rect
            writer.writerow(["ablak", x1, y1, x2, y2, "", "", "", ""])


def load_regions_from_single_csv(filepath):
    corridors = []
    windows = []
    if not os.path.exists(filepath):
        return corridors, windows
    with open(filepath, mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            tp = row[0]
            if tp == "folyoso":
                pts = []
                for i in range(1, 9, 2):
                    pts.append((int(row[i]), int(row[i + 1])))
                corridors.append(pts)
            elif tp == "ablak":
                x1, y1, x2, y2 = map(int, row[1:5])
                windows.append((x1, y1, x2, y2))
    return corridors, windows


def contour_overlaps_rect(contour, rect):
    x1, y1, x2, y2 = rect
    rect_contour = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.int32)
    intersection = cv2.intersectConvexConvex(cv2.convexHull(contour), cv2.convexHull(rect_contour))
    return intersection[0] > 0 and intersection[1] is not None and cv2.contourArea(intersection[1]) > 10


def contour_overlaps_polygon(contour, polygon):
    hull = cv2.convexHull(contour)
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    intersection = cv2.intersectConvexConvex(hull, cv2.convexHull(poly))
    return intersection[0] > 0


def point_in_rect(px, py, rect):
    x1, y1, x2, y2 = rect
    return x1 <= px <= x2 and y1 <= py <= y2


def point_in_polygon(px, py, polygon):
    poly = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    result = cv2.pointPolygonTest(poly, (float(px), float(py)), False)
    return result >= 0


def calculate_polygon_center(polygon):
    pts = np.array(polygon, dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts.reshape(-1, 2)
    center_x = np.mean(pts[:, 0])
    center_y = np.mean(pts[:, 1])
    return (int(center_x), int(center_y))


def save_speed_recordings(filepath, speed_recordings):
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["corridor", "frame", "time_sec", "rat_x", "rat_y", "midpoint_x", "midpoint_y"])
        for event in speed_recordings:
            writer.writerow([
                event['corridor'],
                event['frame'],
                event['time_sec'],
                event['rat_x'],
                event['rat_y'],
                event['midpoint_x'],
                event['midpoint_y']
            ])


def region_selection_menu(first_frame, num_rewards, region_csv):
    corridors, windows = [], []
    if os.path.exists(region_csv):
        corridors, windows = load_regions_from_single_csv(region_csv)
        print(f"Folyosók és ablakok betöltve: {region_csv}")
    else:
        print("Kérlek, jelöld ki a folyosókat és ablakokat.")
        selector = RegionSelector(first_frame, num_rewards, modes_to_use=["ablak", "folyoso"])
        rois, _ = selector.run(allow_navigation=False)
        corridors = rois.get("folyoso", [])
        windows = rois.get("ablak", [])
        save_regions_to_single_csv(region_csv, corridors, windows)
        print(f"Folyosók és ablakok elmentve: {region_csv}")

    rewards = []
    while True:
        print("Jutalom kijelölő. Nyomj 'a'-t új ablak/folyosó kijelölőhöz, vagy 'c'-t az elemzés indításához.")
        selector = RegionSelector(first_frame, num_rewards=num_rewards, modes_to_use=["jutalom"])
        rois, action = selector.run(allow_navigation=True)
        rewards = rois.get("jutalom", [])

        if action == 'a':
            selector = RegionSelector(first_frame, num_rewards, modes_to_use=["ablak", "folyoso"])
            rois, _ = selector.run(allow_navigation=False)
            corridors = rois.get("folyoso", [])
            windows = rois.get("ablak", [])
            save_regions_to_single_csv(region_csv, corridors, windows)
            print(f"Folyosók és ablakok frissítve: {region_csv}")
            continue
        elif action == 'c':
            print("Kijelölés kész, indul az elemzés.")
            break

    return corridors, windows, rewards


def analyze_video(video_path, num_rewards=1, debug_mode=False):
    VAR_THRESHOLD = 60
    MIN_AREA = 450
    MORPH_CLOSE = 5
    MORPH_OPEN = 2
    MAX_LOST_FRAMES = 1000
    WARMUP_FRAMES = 100
    MIDPOINT_THRESHOLD = 30

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = f"{video_name}.csv"
    region_csv = "regions.csv"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Hiba: nem sikerült megnyitni a videót")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Hiba: nem sikerült beolvasni az első képkockát")
        return

    corridors, windows, rewards = region_selection_menu(first_frame, num_rewards, region_csv)

    corridor_midpoints = []
    for corridor in corridors:
        midpoint = calculate_polygon_center(corridor)
        corridor_midpoints.append(midpoint)
        print(f"Folyosó {len(corridor_midpoints)} középpontja: {midpoint}")

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=VAR_THRESHOLD, detectShadows=False)
    
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    print(f"Háttér modell bemelegítése ({WARMUP_FRAMES} képkocka)...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        bg_sub.apply(frame, learningRate=-1)
    print("Bemelegítés kész. Elemzés indul...\n")

    reward_collected = [False] * num_rewards
    last_position = None
    frames_lost = 0

    speed_recordings = []
    midpoint_crossed = [False] * len(corridors)

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    continuous_pos_csv = f"{video_name}_continuous_positions.csv"
    
    with open(output_csv, mode="w", newline="") as f, \
         open(continuous_pos_csv, mode="w", newline="") as f_pos:
        
        writer = csv.writer(f)
        header = ["frame"] + [f"corridor_{i+1}" for i in range(4)] + [f"window_{i+1}" for i in range(8)] + [f"reward_{i+1}" for i in range(num_rewards)]
        writer.writerow(header)

        pos_writer = csv.writer(f_pos)
        pos_writer.writerow(["frame", "x", "y", "estimated"])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display = frame.copy()
            
            fg_mask = bg_sub.apply(frame, learningRate=-1)

            roi_mask = np.zeros(fg_mask.shape, dtype=np.uint8)
            for poly in corridors:
                pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(roi_mask, [pts], 255)
            
            for rect in windows:
                x1, y1, x2, y2 = rect
                cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)

            for rect in rewards:
                x1, y1, x2, y2 = rect
                cv2.rectangle(roi_mask, (x1, y1), (x2, y2), 255, -1)
            
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=roi_mask)

            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close, iterations=MORPH_CLOSE)
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open, iterations=MORPH_OPEN)
            fg_mask = cv2.medianBlur(fg_mask, 5)

            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            corridor_vals = [0] * 4
            window_vals = [0] * 8
            reward_vals = [1 if not reward_collected[i] else 0 for i in range(num_rewards)]

            rat_contour = None
            current_position = None
            is_estimated = False

            if contours:
                valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_AREA]
                
                if valid_contours:
                    if last_position is None:
                        rat_contour = max(valid_contours, key=cv2.contourArea)
                    else:
                        rat_contour = min(valid_contours, key=lambda c: np.linalg.norm(np.array(calculate_polygon_center(c)) - np.array(last_position)))
                    
                    M = cv2.moments(rat_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        current_position = (cx, cy)
                        last_position = current_position
                        frames_lost = 0
                    if debug_mode:
                        cv2.drawContours(display, [rat_contour], -1, (0, 255, 255), 1)

            if rat_contour is not None:
                # Measured position
                pos_writer.writerow([frame_idx, current_position[0], current_position[1], False])

                for i, poly in enumerate(corridors):
                    if contour_overlaps_polygon(rat_contour, poly):
                        corridor_vals[i] = 1
                        
                        midpoint = corridor_midpoints[i]
                        distance = np.sqrt((current_position[0] - midpoint[0])**2 + 
                                          (current_position[1] - midpoint[1])**2)
                        
                        if distance <= MIDPOINT_THRESHOLD and not midpoint_crossed[i]:
                            midpoint_crossed[i] = True
                            
                            last_recorded_corridor = speed_recordings[-1]['corridor'] if speed_recordings else -1
                            current_corridor_id = i + 1

                            if current_corridor_id != last_recorded_corridor:
                                time_sec = frame_idx / fps
                                speed_recordings.append({
                                    'corridor': current_corridor_id,
                                    'frame': frame_idx,
                                    'time_sec': time_sec,
                                    'rat_x': current_position[0],
                                    'rat_y': current_position[1],
                                    'midpoint_x': midpoint[0],
                                    'midpoint_y': midpoint[1]
                                })
                                print(f"Folyosó {current_corridor_id} középpontja elérve a {frame_idx}. képkockán ({time_sec:.2f} sec)")
                        
                        elif distance > MIDPOINT_THRESHOLD * 2:
                            midpoint_crossed[i] = False

                for i, rect in enumerate(windows):
                    if contour_overlaps_rect(rat_contour, rect):
                        window_vals[i] = 1

                for i, rect in enumerate(rewards):
                    if not reward_collected[i] and contour_overlaps_rect(rat_contour, rect):
                        reward_collected[i] = True
                        reward_vals[i] = 0
                        print(f"Jutalom {i+1} begyűjtve a(z) {frame_idx}. képkockán")

            elif last_position is not None and frames_lost < MAX_LOST_FRAMES:
                frames_lost += 1
                cx, cy = last_position
                
                # Estimated position
                pos_writer.writerow([frame_idx, cx, cy, True])
                
                is_in_corridor = False
                for poly in corridors:
                    if point_in_polygon(cx, cy, poly):
                        is_in_corridor = True
                        break
                
                if not is_in_corridor:
                    frames_lost = MAX_LOST_FRAMES 
                else:

                    if debug_mode:
                        cv2.circle(display, (cx, cy), 15, (255, 165, 0), 2)
                        cv2.putText(display, "BECSLES", (cx - 30, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

                    for i, poly in enumerate(corridors):
                        if point_in_polygon(cx, cy, poly):
                            corridor_vals[i] = 1

                    for i, rect in enumerate(windows):
                        if point_in_rect(cx, cy, rect):
                            window_vals[i] = 1

                    for i, rect in enumerate(rewards):
                        if not reward_collected[i] and point_in_rect(cx, cy, rect):
                            reward_collected[i] = True
                            reward_vals[i] = 0
                            print(f"Jutalom {i+1} begyűjtve a(z) {frame_idx}. képkockán (becslés)")
            else:
                # No position known
                pos_writer.writerow([frame_idx, "", "", ""])

            row = [frame_idx] + corridor_vals + window_vals + reward_vals
            writer.writerow(row)

            if debug_mode:
                for i, midpoint in enumerate(corridor_midpoints):
                    cv2.circle(display, midpoint, 8, (255, 255, 0), -1)
                    cv2.putText(display, f"M{i+1}", (midpoint[0]+10, midpoint[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                for i, poly in enumerate(corridors):
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    color = (255, 0, 0) if corridor_vals[i] else (100, 100, 100)
                    thickness = 5 if corridor_vals[i] else 2
                    cv2.polylines(display, [pts], isClosed=True, color=color, thickness=thickness)
                    cv2.putText(display, f"F{i+1}", (poly[0][0] + 5, poly[0][1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                for i, rect in enumerate(windows):
                    x1, y1, x2, y2 = rect
                    color = (0, 255, 0) if window_vals[i] else (100, 100, 100)
                    thickness = 5 if window_vals[i] else 2
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                    cv2.putText(display, f"A{i+1}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                for i, rect in enumerate(rewards):
                    x1, y1, x2, y2 = rect
                    if reward_collected[i]:
                        color = (128, 128, 128)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.line(display, (x1, y1), (x2, y2), color, 4)
                        cv2.line(display, (x2, y1), (x1, y2), color, 4)
                        cv2.putText(display, f"J{i+1} X", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    else:
                        color = (0, 0, 255)
                        thickness = 2
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                        cv2.putText(display, f"J{i+1}", (x1 + 5, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                status_text = "KOVETES" if rat_contour is not None else f"BECSLES ({frames_lost}/{MAX_LOST_FRAMES})"
                status_color = (0, 255, 0) if rat_contour is not None else (255, 165, 0)

                cv2.putText(display, f"Kepkocka: {frame_idx}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Statusz: {status_text}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(display, f"Hatra levo jutalmak: {sum([not c for c in reward_collected])}/{num_rewards}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.imshow("Patkany kovetes", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1

    cap.release()
    if debug_mode:
        cv2.destroyAllWindows()
    
    print(f"\nCSV mentve: {output_csv}")
    print(f"Pozíció adatok mentve: {continuous_pos_csv}")
    print(f"Összes képkocka: {frame_idx}")
    print(f"Végleges jutalom állapot: {['Begyűjtve' if c else 'Nincs gyűjtve' for c in reward_collected]}")

    speed_csv = f"{video_name}_speed_recording.csv"
    save_speed_recordings(speed_csv, speed_recordings)
    print(f"Sebesség mérési események mentve: {speed_csv}")
    print(f"Összes mérési esemény: {len(speed_recordings)}")
    
    return speed_recordings


if __name__ == "__main__":
    video_path = "LE_17_1_64.mpg"
    
    watch_mode = input("Szeretnéd látni a program működését? (y/n): ").strip().lower()
    debug_mode = (watch_mode == 'y')
    
    if debug_mode:
        print("Debug mód: BEKAPCSOLVA - A program működése megjelenik az ablakban.")
    else:
        print("Debug mód: KIKAPCSOLVA - A program a háttérben fut, csak a konzolba ír.")
    
    try:
        num_rewards = int(input("Hány jutalmat szeretnél megjelölni? "))
    except ValueError:
        num_rewards = 1
    
    analyze_video(video_path, num_rewards=num_rewards, debug_mode=debug_mode)
