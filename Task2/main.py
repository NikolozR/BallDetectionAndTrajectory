import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.interpolate import interp1d


def optimize_initial_velocity(x0, y0, observed_positions, observed_times, k_m, g=9.81, h=0.01):

    def trajectory_error(vx0, vy0):
        initial_state = np.array([x0, y0, vx0, vy0])
        t_values, trajectory = solve_ode(
            lambda t, state: ball_trajectory(t, state, k_m, g),
            (0, observed_times[-1]),
            initial_state,
            h
        )

        estimated_trajectory = [(t, state[0], state[1]) for t, state in zip(t_values, trajectory) if state[1] >= 0]
        estimated_x = [point[1] for point in estimated_trajectory[:len(observed_positions)]]
        estimated_y = [point[2] for point in estimated_trajectory[:len(observed_positions)]]

        observed_x = [point[0] for point in observed_positions]
        observed_y = [point[1] for point in observed_positions]

        return np.sum((np.array(observed_x) - np.array(estimated_x)) ** 2 +
                      (np.array(observed_y) - np.array(estimated_y)) ** 2)

    vx0_guess, vy0_guess = 0, 0
    max_iter = 1000
    learning_rate = 0.01
    epsilon = 1e-5

    for i in range(max_iter):
        error = trajectory_error(vx0_guess, vy0_guess)

        grad_vx = (trajectory_error(vx0_guess + epsilon, vy0_guess) - error) / epsilon
        grad_vy = (trajectory_error(vx0_guess, vy0_guess + epsilon) - error) / epsilon

        vx0_guess -= learning_rate * grad_vx
        vy0_guess -= learning_rate * grad_vy

        if abs(grad_vx) < 1e-6 and abs(grad_vy) < 1e-6:
            break

    return vx0_guess, vy0_guess


def detect_and_track_ball(video_path):
    scale = 100
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1 / fps

    min_radius = 5
    max_radius = 120
    prev_frame = None
    prev_ball_pos = None
    search_radius = 100

    trajectory = []
    times = []
    frame_count = 0

    def find_ball_in_region(mask, prev_pos=None, search_rad=None):
        if prev_pos and search_rad:
            region_mask = np.zeros_like(mask)
            x, y = prev_pos
            cv2.circle(region_mask, (x, y), search_rad, 255, -1)
            mask = cv2.bitwise_and(mask, mask, mask=region_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_ball = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:
                continue

            (x, y), radius = cv2.minEnclosingCircle(contour)
            if min_radius <= radius <= max_radius:
                circularity = area / (np.pi * radius ** 2)
                perimeter = cv2.arcLength(contour, True)
                compactness = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                score = (circularity + compactness) / 2
                if prev_pos:
                    dist = np.sqrt((x - prev_pos[0]) ** 2 + (y - prev_pos[1]) ** 2)
                    position_score = max(0, 1 - dist / search_rad)
                    score = score * 0.7 + position_score * 0.3

                if score > best_score and score > 0.7:
                    best_ball = (int(x), int(y), int(radius))
                    best_score = score

        return best_ball

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is None:
            prev_frame = gray
            continue

        frame_delta = cv2.absdiff(prev_frame, gray)
        thresh = cv2.threshold(frame_delta, 3, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=4)

        if prev_ball_pos:
            best_ball = find_ball_in_region(thresh, prev_ball_pos, search_radius)
            if not best_ball:
                best_ball = find_ball_in_region(thresh)
        else:
            best_ball = find_ball_in_region(thresh)

        if best_ball:
            x, y, radius = best_ball
            prev_ball_pos = (x, y)

            current_time = frame_count * frame_time
            trajectory.append((x / scale, y / scale))
            times.append(current_time)

            if len(trajectory) >= 2:
                last_x, last_y = trajectory[-2]
                dx = x / scale - last_x
                dy = y / scale - last_y
                movement = np.sqrt(dx * dx + dy * dy)
                search_radius = int(max(50, min(200, movement * scale * 2)))

        prev_frame = gray

    cap.release()
    cv2.destroyAllWindows()

    if len(trajectory) < 2:
        raise ValueError("Ball not detected or insufficient trajectory points.")

    positions = np.array(trajectory)

    initial_velocity = optimize_initial_velocity(
        positions[0][0], positions[0][1], positions, np.array(times), 0.01
    )
    print(initial_velocity, "ASDASFD")
    res = {
        'initial_position': tuple(positions[0]),
        'final_position': tuple(positions[-1]),
        'initial_velocity': tuple(initial_velocity),
        'total_time': times[-1],
        'trajectory': positions,
        'timestamps': np.array(times),
        'radius': radius
    }

    return res


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ode(f, t_span, y0, h):
    t_values = [t_span[0]]
    y_values = [y0]
    t = t_span[0]
    y = y0
    while t < t_span[1]:
        y = rk4_step(f, t, y, h)
        t += h
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)


def ball_trajectory(t, state, k_m, g):
    x, y, vx, vy = state
    speed = np.sqrt(vx ** 2 + vy ** 2)
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -k_m * vx * speed
    dvy_dt = g - k_m * vy * speed
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


def shooting_error(k_m, x0, y0, vx0, vy0, T, observed_positions, observed_times, g, h=0.01, alpha=1.0):
    initial_state = np.array([x0, y0, vx0, vy0])
    t_values, trajectory = solve_ode(
        lambda t, state: ball_trajectory(t, state, k_m, g),
        (0, observed_times[-1]),
        initial_state,
        h
    )

    estimated_trajectory = [(t, state[0], state[1]) for t, state in zip(t_values, trajectory) if state[1] >= 0]
    last_observed = observed_positions[-1]
    last_observed_x = last_observed[1]
    closest_point = min(estimated_trajectory, key=lambda point: abs(point[1] - last_observed_x))
    error = last_observed[1] - closest_point[1]
    return error


def optimize_km_ratio(x0, y0, vx0, vy0, T, observed_positions, observed_times, g=9.81, h=0.01):
    def trajectory_error(k_m):
        return shooting_error(k_m, x0, y0, vx0, vy0, T, observed_positions, observed_times, g, h)
    k_m = 0.1
    error = 0
    for i in range(400):
        if len(observed_positions) > 1:
            error = trajectory_error(k_m)

            if abs(error) < 0.001:
                break
            k_m -= 0.05 * error

    t_values, trajectory = solve_ode(
        lambda t, state: ball_trajectory(t, state, k_m, g),
        (0, observed_times[-1]),
        np.array([x0, y0, vx0, vy0]),
        h
    )

    return {
        'k_m_ratio': k_m,
        'final_error': error,
        't_values': t_values,
        'trajectory': trajectory
    }


def shooting_method_hit(x0, y0, xt, yt, T, k_m, g, v_guess, h, tol=1e-5, max_iter=100):
    def target_error(v0):
        state0 = [x0, y0, v0[0], v0[1]]
        _, trajectory = solve_ode(lambda t, state: ball_trajectory(t, state, k_m, g), (0, T), state0, h)
        final_x, final_y = trajectory[-1, :2]
        return np.array([final_x - xt, final_y - yt])

    v = np.array(v_guess, dtype=float)
    for _ in range(max_iter):
        error = target_error(v)
        if np.linalg.norm(error) < tol:
            return v
        jacobian = np.array([
            (target_error(v + [h, 0]) - error) / h,
            (target_error(v + [0, h]) - error) / h,
        ]).T
        delta_v = np.linalg.solve(jacobian, -error)
        v += delta_v
    raise RuntimeError("Shooting method did not converge.")


def generate_extended_trajectory_video(tracking_data, optimization_result, video_path,
                                       output_path="extended_predicted_trajectory.mp4", time_multiplier=2, small_ball_trajectory=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    scale = 100
    ball_radius = int(tracking_data['radius'])

    original_t_values = optimization_result['t_values']
    original_trajectory = optimization_result['trajectory']
    extended_t_end = original_t_values[-1] * time_multiplier

    extended_t_values, extended_trajectory = solve_ode(
        lambda t, state: ball_trajectory(t, state, optimization_result['k_m_ratio'], 9.81),
        (0, extended_t_end),
        np.array([tracking_data['initial_position'][0], tracking_data['initial_position'][1],
                  tracking_data['initial_velocity'][0], tracking_data['initial_velocity'][1]]),
        0.01
    )

    extended_trajectory_pixels = extended_trajectory[:, :2] * scale

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    predicted_index = 0
    small_ball_index = 0
    total_frames = int(extended_t_end * fps)

    initial_fixed_frames = int(2 * fps)
    small_ball_initial_pos = small_ball_trajectory[0]

    for frame_idx in range(total_frames + initial_fixed_frames):
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        if frame_idx < initial_fixed_frames:
            small_x, small_y = int(small_ball_initial_pos[0]), int(small_ball_initial_pos[1])
            if 0 <= small_x < frame_width and 0 <= small_y < frame_height:
                cv2.circle(frame, (small_x, small_y), 5, (255, 0, 0), -1)

        else:

            if predicted_index < len(extended_trajectory_pixels):
                pred_pos = extended_trajectory_pixels[predicted_index]
                pred_x, pred_y = int(pred_pos[0]), int(pred_pos[1])
                if 0 <= pred_x < frame_width and 0 <= pred_y < frame_height:
                    cv2.circle(frame, (pred_x, pred_y), ball_radius, (0, 0, 255), -1)

                time_interval = extended_t_values[1] - extended_t_values[0]
                if (frame_idx - initial_fixed_frames) * (1 / fps) >= predicted_index * time_interval:
                    predicted_index += 1

            if small_ball_index < len(small_ball_trajectory):
                small_pos = small_ball_trajectory[small_ball_index]
                small_x, small_y = int(small_pos[0]), int(small_pos[1])
                if 0 <= small_x < frame_width and 0 <= small_y < frame_height:
                    cv2.circle(frame, (small_x, small_y), 5, (255, 0, 0), -1)  # Blue ball (small)

                time_interval = extended_t_values[1] - extended_t_values[0]
                if (frame_idx - initial_fixed_frames) * (1 / fps) >= small_ball_index * time_interval:
                    small_ball_index += 1

        out.write(frame)
        frame_count += 1

    out.release()
    cv2.destroyAllWindows()


def calculate_intercept_velocity(tracking_data, optimization_result, T, k_m, g=9.81, h=0.01):
    t_values, trajectory = solve_ode(
        lambda t, state: ball_trajectory(t, state, k_m, g),
        (0, T),
        np.array([tracking_data['initial_position'][0],
                  tracking_data['initial_position'][1],
                  tracking_data['initial_velocity'][0],
                  tracking_data['initial_velocity'][1]]),
        h
    )

    interpolate_x = interp1d(t_values, trajectory[:, 0], kind='linear', fill_value='extrapolate')
    interpolate_y = interp1d(t_values, trajectory[:, 1], kind='linear', fill_value='extrapolate')
    xt = interpolate_x(T)
    yt = interpolate_y(T)

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    scale = 100
    width_m = frame_width / scale
    height_m = frame_height / scale

    margin_x = width_m * 0.2
    margin_y = height_m * 0.8
    x0 = np.random.uniform(margin_x, width_m - margin_x)
    y0 = np.random.uniform(margin_y, height_m - margin_y)

    dx = xt - x0
    dy = yt - y0
    vx_guess = dx / T
    vy_guess = (dy - 0.5 * g * T * T) / T
    v_guess = [vx_guess, vy_guess]
    print(xt, yt)
    print(f"Small ball starting position: ({x0:.2f}m, {y0:.2f}m)")
    return shooting_method_hit(x0, y0, xt, yt, T, k_m, g, v_guess, h), (x0, y0)


def analyze_ball_video(video_path):
    scale = 100
    tracking_data = detect_and_track_ball(video_path)

    x0, y0 = tracking_data['initial_position']

    vx0, vy0 = tracking_data['initial_velocity']

    T = tracking_data['total_time']

    result = optimize_km_ratio(
        x0, y0, vx0, vy0, T,
        tracking_data['trajectory'],
        tracking_data['timestamps'],
    )

    captured = cv2.VideoCapture(video_path)
    frame_width = int(captured.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(captured.get(cv2.CAP_PROP_FRAME_HEIGHT))
    captured.release()

    plt.figure(figsize=(frame_width / 100, frame_height / 100))

    observed_positions = tracking_data['trajectory'] * scale

    plt.plot(observed_positions[:, 0],
             frame_height - observed_positions[:, 1],
             'g.-', label='Observed Trajectory', alpha=0.5, markersize=2)

    predicted_trajectory = result['trajectory'] * scale
    plt.plot(predicted_trajectory[:, 0],
             frame_height - predicted_trajectory[:, 1],
             'r-', label='Predicted Trajectory', linewidth=2)

    plt.grid(True)
    plt.legend()
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.title(f'Ball Trajectory Analysis\nk/m ratio = {result["k_m_ratio"]:.6f}')

    plt.axis('equal')

    plt.xlim(0, frame_width)
    plt.ylim(0, frame_height)

    plt.show()

    return {
        'tracking_data': tracking_data,
        'optimization_result': result
    }


# RUN FROM HERE

if __name__ == "__main__":
    video_path = "test_10.mp4" # Relative path to input video
    output_path = "output_with_trajectory.mp4"
    results = analyze_ball_video(video_path)

    T = 1
    k_m = results['optimization_result']['k_m_ratio']

    small_ball_velocity, start_pos = calculate_intercept_velocity(results['tracking_data'],
                                                                results['optimization_result'],
                                                                T, k_m)

    t_values, trajectory = solve_ode(
        lambda t, state: ball_trajectory(t, state, k_m, 9.81),
        (0, T),
        np.array([start_pos[0], start_pos[1], small_ball_velocity[0], small_ball_velocity[1]]),
        0.01
    )

    generate_extended_trajectory_video(
        results['tracking_data'],
        results['optimization_result'],
        video_path,
        output_path,
        4,
        trajectory * 100
    )