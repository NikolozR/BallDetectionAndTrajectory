import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma=1):
    k = size // 2
    x, y = np.mgrid[-k:k + 1, -k:k + 1]
    kernel = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return kernel / kernel.sum()


def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    shape = (h - kh + 1, w - kw + 1, kh, kw)
    strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return np.einsum('ijkl,kl->ij', windows, kernel)


def detect_edges(image_path, edge_threshold=0.5):
    image = cv2.imread(image_path)
    gray_image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
    gray_image = gray_image / 255.0
    kernel = gaussian_kernel(7, sigma=1)
    smoothed_image = convolve(gray_image, kernel)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gradient_x = convolve(smoothed_image, sobel_x)
    gradient_y = convolve(smoothed_image, sobel_y)
    gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    binary_edges = (gradient_magnitude > edge_threshold).astype(np.uint8)
    return binary_edges


class SpatialGrid:
    def __init__(self, points, cell_size=None):
        self.points = np.array(points)
        self.n_points, self.n_dims = self.points.shape

        if cell_size is None:
            distances = np.sqrt(np.sum((self.points[1:] - self.points[:-1]) ** 2, axis=1))
            self.cell_size = np.mean(distances) / 2
        else:
            self.cell_size = cell_size

        self.min_bounds = np.min(points, axis=0)
        self.max_bounds = np.max(points, axis=0)
        self.grid_dims = np.ceil((self.max_bounds - self.min_bounds) / self.cell_size).astype(int)
        self.grid = {}
        self._build_grid()

    def _point_to_cell_coords(self, point):
        return tuple(((point - self.min_bounds) / self.cell_size).astype(int))

    def _build_grid(self):
        for i, point in enumerate(self.points):
            cell_coords = self._point_to_cell_coords(point)
            if cell_coords not in self.grid:
                self.grid[cell_coords] = []
            self.grid[cell_coords].append(i)

    def _get_neighbor_cells(self, cell_coords, radius):
        radius_cells = int(np.ceil(radius / self.cell_size))
        neighbor_cells = []

        ranges = [range(max(0, coord - radius_cells),
                        min(dim, coord + radius_cells + 1))
                  for coord, dim in zip(cell_coords, self.grid_dims)]

        for neighbor_coord in np.ndindex(*[len(r) for r in ranges]):
            neighbor_cell = tuple(r[i] for i, r in zip(neighbor_coord, ranges))
            neighbor_cells.append(neighbor_cell)

        return neighbor_cells

    def get_neighbors(self, point_idx, eps):
        point = self.points[point_idx]
        cell_coords = self._point_to_cell_coords(point)
        neighbor_cells = self._get_neighbor_cells(cell_coords, eps)

        neighbors = []
        for cell in neighbor_cells:
            if cell in self.grid:
                for neighbor_idx in self.grid[cell]:
                    if neighbor_idx != point_idx:
                        distance = np.sqrt(np.sum((point - self.points[neighbor_idx]) ** 2))
                        if distance <= eps:
                            neighbors.append(neighbor_idx)

        return np.array(neighbors)


def dbscan_optimized(points, eps, min_samples):
    print(len(points))
    n_points = len(points)
    labels = np.full(n_points, -1)

    grid = SpatialGrid(points)

    cluster_id = 0

    def expand_cluster(point_idx, neighbors, cluster_id):
        neighbors = set(neighbors)
        labels[point_idx] = cluster_id
        queue = list(neighbors)
        while queue:
            neighbor = queue.pop()
            if labels[neighbor] == -1:
                labels[neighbor] = cluster_id
                new_neighbors = grid.get_neighbors(neighbor, eps)
                if len(new_neighbors) >= min_samples:
                    new_neighbors = set(new_neighbors)
                    new_neighbors.difference_update(neighbors)
                    neighbors.update(new_neighbors)
                    queue.extend(new_neighbors)
    for i in range(n_points):
        if labels[i] != -1:
            continue
        neighbors = grid.get_neighbors(i, eps)
        if len(neighbors) < min_samples:
            labels[i] = 0
            continue
        cluster_id += 1
        expand_cluster(i, neighbors, cluster_id)
    return labels


def detect_balls(image_path, edge_threshold=0.5, eps=20, min_samples=10):
    binary_edges = detect_edges(image_path, edge_threshold)

    edge_points = np.array(np.where(binary_edges == 1)).T

    labels = dbscan_optimized(edge_points, eps, min_samples)

    unique_labels = np.unique(labels)
    centroids = []

    for label in unique_labels:
        if label > 0:
            cluster_points = edge_points[labels == label]

            centroid_y = int(np.mean(cluster_points[:, 0]))
            centroid_x = int(np.mean(cluster_points[:, 1]))

            distances = np.sqrt(np.sum((cluster_points - [centroid_y, centroid_x]) ** 2, axis=1))
            radius = int(np.mean(distances))

            centroids.append((centroid_x, centroid_y, radius))

    def is_inside(centroid1, centroid2):
        x_1, y_1, r_1 = centroid1
        x_2, y_2, r_2 = centroid2
        d = np.sqrt((x_2 - x_1) ** 2 + (y_2 - y_1) ** 2)
        return d + r_1 <= r_2

    result = []

    for i, centroid1 in enumerate(centroids):
        inside = False
        for j, centroid2 in enumerate(centroids):
            if i != j and is_inside(centroid1, centroid2):
                inside = True
                break
        if not inside:
            result.append(centroid1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Edge Detection")
    plt.imshow(binary_edges, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Detected Balls")
    plt.imshow(binary_edges, cmap='gray')
    for x, y, r in result:
        circle = plt.Circle((x, y), r, fill=False, color='red')
        plt.gca().add_patch(circle)
        plt.plot(x, y, 'r+')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return result


def rk2_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h, y + h * k1)
    return y + (h / 2) * (k1 + k2)


def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h / 2, y + h * k1 / 2)
    k3 = f(t + h / 2, y + h * k2 / 2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve_ode(f, t_span, y0, h, step_method):
    t_values = [t_span[0]]
    y_values = [y0]
    t = t_span[0]
    y = y0
    while t < t_span[1]:
        y = step_method(f, t, y, h)
        t += h
        t_values.append(t)
        y_values.append(y)
    return np.array(t_values), np.array(y_values)


def ball_trajectory(t, state, k, m, g):
    x, y, vx, vy = state
    speed = np.sqrt(vx**2 + vy**2)
    dx_dt = vx
    dy_dt = vy
    dvx_dt = -(k / m) * vx * speed
    dvy_dt = g - (k / m) * vy * speed
    return np.array([dx_dt, dy_dt, dvx_dt, dvy_dt])


def compare_rk_methods(x0, y0, xt, yt, T, k, m, g, v_guess, h, tol=1e-5):
    results = {}
    methods = {'RK4': rk4_step, 'RK2': rk2_step}
    for method_name, step_method in methods.items():
        try:
            v0 = shooting_method(x0, y0, xt, yt, T, k, m, g, v_guess, h, tol, step_method=step_method)
            initial_state = [x0, y0, v0[0], v0[1]]
            t_values, trajectory = solve_ode(
                lambda t, state: ball_trajectory(t, state, k, m, g),
                (0, T),
                initial_state,
                h,
                step_method
            )
            results[method_name] = {
                "trajectory": trajectory,
                "velocity": v0
            }
        except RuntimeError as e:
            results[method_name] = {"error": str(e)}
    return results


def plot_comparison(results, target_position):
    plt.figure(figsize=(10, 6))
    for method_name, result in results.items():
        if "trajectory" in result:
            trajectory = result["trajectory"]
            plt.plot(trajectory[:, 0], trajectory[:, 1], label=f"{method_name} Trajectory")
        else:
            print(f"{method_name} failed: {result['error']}")

    plt.scatter(target_position[0], target_position[1], color='red', label='Target Position')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Comparison of RK4 and RK2")
    plt.legend()
    plt.grid()
    plt.show()


def shooting_method(x0, y0, xt, yt, T, k, m, g, v_guess, desired_h, tol=1e-5, max_iter=100, step_method=rk4_step):
    h_max = (2.785 * m) / k  # Maximum stable step size
    h = min(h_max, desired_h)  # Ensure step size respects stability limit

    def target_error(v0):
        state0 = [x0, y0, v0[0], v0[1]]
        _, trajectory = solve_ode(lambda t, state: ball_trajectory(t, state, k, m, g), (0, T), state0, h, step_method)
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


def draw_balls_on_canvas(centroids, original_shape, small_ball=None):
    print(centroids)
    canvas = np.full((original_shape[0], original_shape[1], 3), (99, 140, 109), dtype=np.uint8)
    for x, y, r in centroids:
        cv2.circle(canvas, (int(x), int(y)), int(r), (200, 76, 5), -1)

    if small_ball:
        cv2.circle(canvas, (small_ball[0], small_ball[1]), small_ball[2], (80, 0, 115), -1)
    return canvas


def draw_trajectory(canvas, trajectory, color=(0, 255, 255), thickness=2):
    for i in range(len(trajectory) - 1):
        pt1 = (int(trajectory[i][0]), int(trajectory[i][1]))
        pt2 = (int(trajectory[i + 1][0]), int(trajectory[i + 1][1]))
        cv2.line(canvas, pt1, pt2, color, thickness)


def animate_ball_trajectories(canvas, trajectories, small_ball, output_path, fps=30):
    h, w, _ = canvas.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    base_frame = canvas.copy()
    cv2.circle(base_frame, (small_ball[0], small_ball[1]), small_ball[2], (80, 0, 115), -1)

    for _ in range(fps * 2):
        writer.write(base_frame.copy())

    for trajectory in trajectories:
        current_frame = base_frame.copy()
        for i in range(len(trajectory)):
            draw_trajectory(current_frame, trajectory[:i + 1])
            writer.write(current_frame.copy())

    for _ in range(fps * 2):
        writer.write(current_frame)

    writer.release()


def is_ball_too_close(h, w, small_ball_position, centroids, min_distance=100):
    small_x, small_y, r = small_ball_position
    if small_x > w or small_y > h:
        return True
    for centroid_x, centroid_y, rad in centroids:
        distance = np.sqrt((small_x - centroid_x) ** 2 + (small_y - centroid_y) ** 2)
        if distance <= min_distance + rad:
            return True
    return False


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def check_trajectory_collision(trajectory, balls, ball_radius, safety_margin=10):
    total_radius = ball_radius + safety_margin

    for point in trajectory:
        for ball_x, ball_y, ball_r in balls:
            distance = np.sqrt((point[0] - ball_x) ** 2 + (point[1] - ball_y) ** 2)
            if distance <= (ball_r + total_radius):
                return True
    return False


def find_valid_trajectory(small_ball, target_ball, other_balls, total_frames, k_adjusted, m, g_pxframe2, h, step_method=rk4_step):
    dx = target_ball[0] - small_ball[0]
    dy = target_ball[1] - small_ball[1]

    base_speed = np.sqrt(dx * dx + dy * dy) / total_frames
    angles = np.linspace(-np.pi*2, 2*np.pi, 20)

    for angle in angles:
        base_angle = np.arctan2(dy, dx)
        total_angle = base_angle + angle

        vx_guess = base_speed * np.cos(total_angle)
        vy_guess = base_speed * np.sin(total_angle)

        try:
            v0 = shooting_method(small_ball[0], small_ball[1],
                                 target_ball[0], target_ball[1],
                                 total_frames, k_adjusted, m, g_pxframe2,
                                 [vx_guess, vy_guess], h)

            initial_state = [small_ball[0], small_ball[1], v0[0], v0[1]]
            _, trajectory = solve_ode(
                lambda t, state: ball_trajectory(t, state, k_adjusted, m, g_pxframe2),
                (0, total_frames), initial_state, h, step_method=step_method
            )

            if not check_trajectory_collision(trajectory[:, :2], other_balls, small_ball[2]):
                return trajectory[:, :2]

        except RuntimeError:
            continue

    return None


def animate_ball_trajectories_dynamic(canvas, trajectories, hits, small_ball, output_path, fps=30):
    h, w, _ = canvas.shape
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    current_frame = canvas.copy()
    cv2.circle(current_frame, (small_ball[0], small_ball[1]), small_ball[2], (80, 0, 115), -1)

    for _ in range(fps):
        writer.write(current_frame.copy())

    for traj_idx, (trajectory, hit_ball) in enumerate(zip(trajectories, hits)):
        current_frame = canvas.copy()
        for prev_hit in hits[:traj_idx]:
            cv2.circle(current_frame, (int(prev_hit[0]), int(prev_hit[1])),
                       int(prev_hit[2]), (99, 140, 109), -1)

        cv2.circle(current_frame, (small_ball[0], small_ball[1]),
                   small_ball[2], (80, 0, 115), -1)

        for i in range(len(trajectory)):
            frame = current_frame.copy()
            draw_trajectory(frame, trajectory[:i + 1])
            writer.write(frame)

        cv2.circle(current_frame, (int(hit_ball[0]), int(hit_ball[1])),
                   int(hit_ball[2]), (99, 140, 109), -1)

        for _ in range(fps):
            writer.write(current_frame.copy())

    for _ in range(fps):
        writer.write(current_frame)

    writer.release()


def create_video_with_balls(par_image_path, par_output_video_path, duration_seconds, k, m, g, fps=30,
                            edge_threshold=0.65, small_ball_radius=10, min_distance=200, eps=25, min_samples=40):
    centroids = detect_balls(par_image_path, edge_threshold, eps, min_samples)
    print(len(centroids))
    image = cv2.imread(par_image_path)
    height, width, _ = image.shape

    small_ball = (300, 300, small_ball_radius)
    while is_ball_too_close(height, width, small_ball, centroids):
        import random
        x_min, x_max = small_ball_radius + 10, width - small_ball_radius - 10
        y_min, y_max = small_ball_radius + 10, height - small_ball_radius - 10
        random_x = random.randint(x_min, x_max)
        random_y = random.randint(y_min, y_max)

        small_ball = (random_x, random_y, small_ball_radius)

    canvas = draw_balls_on_canvas(centroids, (height, width), small_ball)

    total_frames = int(duration_seconds * fps)
    h = 1

    PIXELS_PER_METER = 100
    g_pxframe2 = g * PIXELS_PER_METER / (fps * fps)
    k_adjusted = k

    trajectories = []
    hits = []
    skipped_balls = []
    remaining_balls = centroids.copy()

    while remaining_balls or skipped_balls:
        if not remaining_balls and skipped_balls:
            remaining_balls = skipped_balls.copy()
            skipped_balls.clear()

        closest_ball = min(remaining_balls, key=lambda x: calculate_distance(small_ball, x))
        other_balls = [ball for ball in remaining_balls if ball != closest_ball] + skipped_balls

        trajectory = find_valid_trajectory(
            small_ball, closest_ball, other_balls,
            total_frames, k_adjusted, m, g_pxframe2, h
        )

        if trajectory is not None:
            trajectories.append(trajectory)
            hits.append(closest_ball)
            remaining_balls.remove(closest_ball)
        else:
            skipped_balls.append(closest_ball)
            remaining_balls.remove(closest_ball)

    animate_ball_trajectories_dynamic(canvas, trajectories, hits, small_ball, par_output_video_path, fps)


image_path = "img_1.png"
output_path = "output_video.mp4"
to_hit_duration_seconds = 1
k = 0.001
m = 1
g = 9.81
fps = 30
create_video_with_balls(image_path, output_path, to_hit_duration_seconds, k, m, g, fps)
print("Video generation completed!")