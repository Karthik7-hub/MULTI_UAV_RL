# Algorithmic Formulations and Complete Pseudocode

**Document Version:** 1.0  
**Target Repository:** `research_v2`  
**Date:** November 2026  

---

## Algorithm 1: 3D A\* Global Path Planning with Pure Pursuit Tracking

```python
# Implementation: models/astar.py
Algorithm 1: 3D A* Waypoint Path Planning and Pure Pursuit Following
Input: Initial position p_0, Goal station g, Obstacle set C = {(c_x, c_y, r_k, h_k)},
       Grid resolution delta = 2.0 m, Reach tolerance d_reach = 5.0 m, Lookahead d_look = 6.0 m
Output: Commanded 3D displacement delta_p_t at each step t

Procedure PlanPath(p_0, g, C):
    Initialize open_set = PriorityQueue(), closed_set = Set()
    Initialize g_score[p_0] = 0, f_score[p_0] = ||p_0 - g||_2
    open_set.insert(p_0, priority=f_score[p_0])
    
    while open_set is not empty:
        current = open_set.pop_min()
        if ||current - g||_2 <= delta:
            return ReconstructPath(current)
            
        closed_set.add(current)
        for neighbor in Get26ConnectedNeighbors(current, delta):
            if neighbor in closed_set:
                continue
            if CheckStaticObstacleCollision(current, neighbor, C, margin=2.0):
                continue
                
            tentative_g = g_score[current] + ||neighbor - current||_2
            if tentative_g < g_score.get(neighbor, infinity):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + ||neighbor - g||_2
                open_set.insert(neighbor, priority=f_score[neighbor])
                
    return StraightLineFallback(p_0, g)

Procedure PurePursuitController(p_t, waypoints, d_look, v_max):
    # Find active target waypoint along path at lookahead distance
    target_wp = FindWaypointAtDistance(waypoints, p_t, d_look)
    disp = target_wp - p_t
    dist = ||disp||_2
    
    if dist <= 0.1:
        return [0.0, 0.0, 0.0]
        
    speed = min(v_max, dist)
    commanded_velocity = (disp / dist) * speed
    return commanded_velocity * delta_t
```

---

## Algorithm 2: Continuous Quadratic Line-Cylinder Intersection Safety Filter

```python
# Implementation: env/safety_filter.py
Algorithm 2: Continuous Analytical Safety Filter
Input: Current position p_t = (x_0, y_0, z_0), Proposed displacement delta_p = (dx, dy, dz),
       Obstacle set C = {(c_x, c_y, r_k, h_k)}, Safety margin d_margin = 2.0 m
Output: Safe executed displacement delta_p_safe

Procedure EnforceSafetyFilter(p_t, delta_p, C):
    p_prop = p_t + delta_p
    t_min_impact = 1.0
    collision_detected = False
    
    for each obstacle (c_x, c_y, r_k, h_k) in C:
        r_eff = r_k + d_margin
        
        # Formulate quadratic intersection in XY plane: a*t^2 + b*t + c = 0
        a = dx^2 + dy^2
        if a < 1e-6:
            continue  # Pure vertical motion
            
        b = 2.0 * (dx * (x_0 - c_x) + dy * (y_0 - c_y))
        c = (x_0 - c_x)^2 + (y_0 - c_y)^2 - r_eff^2
        discriminant = b^2 - 4.0 * a * c
        
        if discriminant >= 0.0:
            sqrt_disc = sqrt(discriminant)
            t_xy1 = (-b - sqrt_disc) / (2.0 * a)
            t_xy2 = (-b + sqrt_disc) / (2.0 * a)
            
            # Formulate vertical span interval [t_z1, t_z2]
            if abs(dz) > 1e-6:
                t_z1 = (0.0 - z_0) / dz
                t_z2 = ((h_k + d_margin) - z_0) / dz
                if t_z1 > t_z2: swap(t_z1, t_z2)
            else:
                if 0.0 <= z_0 <= (h_k + d_margin):
                    t_z1, t_z2 = -infinity, infinity
                else:
                    t_z1, t_z2 = infinity, -infinity
                    
            # 3D intersection interval
            t_enter = max(0.0, t_xy1, t_z1)
            t_exit = min(1.0, t_xy2, t_z2)
            
            if t_enter <= t_exit and t_enter < t_min_impact:
                t_min_impact = t_enter
                collision_detected = True
                
    if collision_detected:
        # Scale back displacement to stop safely 10 cm before collision boundary
        safe_fraction = max(0.0, t_min_impact - 0.05)
        return delta_p * safe_fraction
    else:
        return delta_p
```

---

## Algorithm 3: Joint Dual-Head MAPPO Training with Centralized Critic

```python
# Implementation: models/mappo.py, train.py
Algorithm 3: Dual-Head MAPPO with Centralized Critic and Causal Offload Masking
Input: Number of UAVs N = 4, Episode length T = 100, PPO epochs K = 5,
       Clipping epsilon = 0.2, GAE lambda = 0.95, Discount factor gamma = 0.99
Output: Optimized policy parameters theta, Critic parameters phi

Initialize Actor pi_theta(a_mov, a_off | o), Centralized Critic V_phi(s)
Initialize RolloutBuffer B

For episode = 1 to NumEpisodes do:
    Reset environment: obs_0, state_0 = env.reset()
    For step t = 0 to T - 1 do:
        For each UAV j = 1 to N do:
            # Action mask: if no task arrived, force offload action to 0 (no-op)
            mask_j = env.get_action_mask(j)
            a_mov_j, a_off_j, logp_j = pi_theta.sample(obs_j, mask_j)
            actions[j] = {move: a_mov_j, offload: a_off_j}
            
        Execute actions in simulation:
        obs_{t+1}, state_{t+1}, reward_t, done_t, info_t = env.step(actions)
        
        # Store fleet transition in buffer
        B.insert(state_t, obs_t, actions, logp, reward_t, done_t)
        state_t, obs_t = state_{t+1}, obs_{t+1}
        
    # Generalized Advantage Estimation (GAE)
    Compute values V_phi(state_t) for all t
    For t = T - 1 down to 0 do:
        delta_t = reward_t + gamma * V(state_{t+1}) * (1 - done_t) - V(state_t)
        A_hat_t = delta_t + gamma * lambda * (1 - done_t) * A_hat_{t+1}
        Returns_t = A_hat_t + V(state_t)
        
    # Mini-Batch PPO Update
    For epoch = 1 to K do:
        For each mini-batch b in B do:
            Compute probability ratios: r(theta) = exp(logpi_theta - logpi_old)
            Surr1 = r(theta) * A_hat
            Surr2 = clip(r(theta), 1 - epsilon, 1 + epsilon) * A_hat
            Loss_actor = -E[min(Surr1, Surr2)] - c_2 * Entropy(pi_theta)
            Loss_critic = 0.5 * E[(V_phi(state) - Returns)^2]
            
            Update theta with Adam(nabla_theta Loss_actor)
            Update phi with Adam(nabla_phi Loss_critic)
            
    B.clear()
```

---

## Algorithm 4: Multi-Tenant OFDMA MEC Task Offloading and Queue Stepping

```python
# Implementation: env/comms_compute.py, env/multi_uav_env.py
Algorithm 4: Multi-Access OFDMA Communication and MEC Execution Dynamics
Input: Arrived tasks {task_j}, Commanded offload targets {target_j}, Fleet positions {p_j},
       Edge Base Station position p_edge = (100, 100, 0), RF Bandwidth B = 20 MHz
Output: Delay T_delay, Consumed energy E_step, Completed count, Missed count

Procedure ProcessTasksAndQueues():
    edge_offloaders = [j for j in Fleet if target_j == 4]
    peer_offloaders = [j for j in Fleet if 1 <= target_j <= 3]
    local_executors = [j for j in Fleet if target_j == 0]
    
    K_edge = len(edge_offloaders)
    K_tx = max(1, len(edge_offloaders) + len(peer_offloaders))
    
    # Process tasks for each UAV
    for j = 1 to N do:
        task = current_tasks[j]
        if task is None: continue
        
        if target_j == 0:
            # 1. Local CPU Execution
            f_eff = f_local
            t_wait = Q_local[j] / f_eff
            t_comp = task.cycles / f_eff
            delay = t_wait + t_comp
            energy = kappa * (f_eff^2) * task.cycles
            Q_local[j] += task.cycles
            
        elif target_j == 4:
            # 2. Ground Edge Base Station Offloading
            dist = ||p_j - p_edge||_2
            SNR = (P_tx * (beta_0 / dist^2)) / sigma^2
            R_edge = (B / K_tx) * log2(1.0 + SNR)
            t_tx = task.data_size / max(1e3, R_edge)
            e_tx = P_tx * t_tx
            
            f_eff_edge = f_edge / (K_edge + edge_background_load)
            t_wait = Q_edge / f_eff_edge
            t_comp = task.cycles / f_eff_edge
            delay = t_tx + t_wait + t_comp
            energy = e_tx  # Edge CPU grid power is external
            Q_edge += task.cycles
            
        else:
            # 3. Peer UAV Cooperative Offloading
            peer_id = (j + target_j) % N
            dist = ||p_j - p_peer_id||_2
            SNR = (P_tx * (beta_0 / dist^2)) / sigma^2
            R_peer = (B / K_tx) * log2(1.0 + SNR)
            t_tx = task.data_size / max(1e3, R_peer)
            e_tx = P_tx * t_tx
            
            f_eff_peer = f_peer / (2.0 if peer_id in local_executors else 1.0)
            t_wait = Q_peer[peer_id] / f_eff_peer
            t_comp = task.cycles / f_eff_peer
            delay = t_tx + t_wait + t_comp
            energy = e_tx + kappa * (f_eff_peer^2) * task.cycles
            Q_peer[peer_id] += task.cycles
            
        # Deadline Accounting
        if delay <= task.deadline:
            tasks_completed += 1
        else:
            deadline_misses += 1
            
    # Advance time and drain persistent queues:
    for j = 1 to N do:
        Q_local[j] = max(0.0, Q_local[j] - f_local * delta_t)
        Q_peer[j] = max(0.0, Q_peer[j] - f_peer * delta_t)
    Q_edge = max(0.0, Q_edge - f_edge * delta_t)
```

