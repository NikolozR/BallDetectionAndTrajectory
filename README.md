# Project Documentation: Ball Detection, Trajectory Planning, and Video Analysis

## Introduction

This project consists of two main tasks, each focusing on the analysis and simulation of ball motion using computer vision and numerical methods. The first task is centered on detecting balls in images and planning collision-free trajectories, while the second task involves tracking a ball in a video and estimating its physical parameters from observed motion. Both tasks leverage mathematical modeling, numerical integration, and optimization techniques.

---

## Task 1: Ball Detection and Trajectory Planning

### Overview
Task 1 aims to detect multiple balls in a static image, identify their positions and radii, and plan a trajectory for a small ball to reach a target ball without colliding with others. The solution involves image processing, clustering, and solving ordinary differential equations (ODEs) to simulate realistic ball motion under drag and gravity.

### Mathematical Principles
- **Image Processing & Edge Detection:** Gaussian smoothing and Sobel filters are used to detect edges in grayscale images, highlighting ball boundaries.
- **Clustering (DBSCAN):** The DBSCAN algorithm groups edge points into clusters, each representing a ball. A spatial grid accelerates neighbor searches.
- **Centroid and Radius Calculation:** For each cluster, the centroid and average distance to the edge points are computed to estimate the ball's center and radius.
- **Numerical ODE Solvers:** The ball's motion is modeled by a second-order ODE incorporating gravity and quadratic air resistance. Both RK2 and RK4 Runge-Kutta methods are implemented for numerical integration.
- **Shooting Method:** An iterative root-finding technique (Newton's method) is used to find the initial velocity that allows the small ball to reach the target ball in a given time.
- **Collision Detection:** The planned trajectory is checked for intersections with other balls, ensuring a collision-free path.

### Algorithms Used
- **Gaussian Kernel & Convolution:** For image smoothing.
- **Sobel Edge Detection:** For extracting edges.
- **DBSCAN with Spatial Grid:** For efficient clustering of edge points.
- **Runge-Kutta (RK2, RK4):** For solving the ODE of ball motion.
- **Shooting Method:** For trajectory planning.
- **Collision Checking:** For validating the planned path.

### Implementation Details
- **Edge Detection:** The image is converted to grayscale, smoothed, and edges are detected using Sobel filters. Edge points are thresholded to form a binary edge map.
- **Ball Detection:** DBSCAN clusters edge points. For each cluster, the centroid and radius are calculated. Nested balls are filtered out.
- **Trajectory Planning:** The user selects a small ball and a target ball. The shooting method, combined with RK4 or RK2, finds the required initial velocity. The trajectory is simulated and checked for collisions.
- **Visualization:** Detected balls and planned trajectories are visualized using Matplotlib and OpenCV. Animations are generated to show the ball's motion.

### Visualizations and Outputs
- **Detected Balls:** Balls are outlined on the image.
- **Trajectory Animation:** The planned path is animated, showing the small ball moving toward the target.
- **Comparison Plots:** RK2 and RK4 trajectories are compared.

---

## Task 2: Ball Tracking and Parameter Estimation from Video

### Overview
Task 2 focuses on analyzing a video of a moving ball. The goal is to track the ball's position frame-by-frame, estimate its initial velocity and drag coefficient, and predict its future trajectory. The task combines computer vision, parameter optimization, and ODE integration.

### Mathematical Principles
- **Object Tracking:** The ball is detected in each frame using background subtraction, contour analysis, and region-of-interest search based on previous positions.
- **Parameter Estimation:** The initial velocity and drag coefficient (k/m) are estimated by minimizing the error between observed and simulated trajectories using gradient descent.
- **Numerical ODE Solvers:** The ball's motion is modeled with gravity and quadratic drag, solved using RK4 integration.
- **Shooting Method:** Used to compute the velocity required for a second ball to intercept the tracked ball at a future time.

### Algorithms Used
- **Background Subtraction & Contour Detection:** For robust ball tracking.
- **Gradient Descent:** For optimizing initial velocity and drag coefficient.
- **Runge-Kutta (RK4):** For simulating ball motion.
- **Shooting Method:** For interception planning.
- **Interpolation:** For estimating positions at arbitrary times.

### Implementation Details
- **Ball Tracking:** Each frame is processed to detect the moving ball. The search region is dynamically adjusted based on previous detections.
- **Parameter Optimization:** The initial velocity is optimized by minimizing the squared error between observed and simulated positions. The drag coefficient is further refined using a similar approach.
- **Trajectory Prediction:** With estimated parameters, the future trajectory is simulated and visualized.
- **Interception Planning:** The code can compute the velocity needed for a second ball to intercept the tracked ball, simulating both trajectories.
- **Visualization:** The observed and predicted trajectories are overlaid on the video. Extended trajectory videos are generated.

### Visualizations and Outputs
- **Tracked Trajectory:** The detected path of the ball is plotted.
- **Parameter Estimation:** The fit between observed and simulated trajectories is visualized.
- **Predicted Trajectory:** The future path is shown, including possible interception scenarios.
- **Output Videos:** Annotated videos with trajectories and predictions.

---

## Conclusion

This project demonstrates the integration of computer vision, clustering, numerical ODE solving, and optimization for analyzing and simulating ball motion. Task 1 focuses on static image analysis and trajectory planning, while Task 2 extends these ideas to dynamic video analysis and parameter estimation. Both tasks highlight the power of mathematical modeling and algorithmic problem-solving in real-world scenarios. 