import pybullet as p
import time
import numpy as np
from random import random
import envs
import pybullet_data

from pybullet_planning.pybullet_tools.pr2_utils import TOP_HOLDING_LEFT_ARM, PR2_URDF, DRAKE_PR2_URDF, \
    SIDE_HOLDING_LEFT_ARM, PR2_GROUPS, open_arm, get_disabled_collisions, REST_LEFT_ARM, rightarm_from_leftarm
from pybullet_planning.pybullet_tools.utils import set_base_values, joint_from_name, quat_from_euler, set_joint_position, \
    set_joint_positions, add_data_path, set_camera_pose, connect, plan_base_motion, plan_joint_motion, enable_gravity, \
    joint_controller, dump_body, load_model, joints_from_names, wait_if_gui, disconnect, get_joint_positions, \
    get_link_pose, link_from_name, HideOutput, get_pose, wait_if_gui, load_pybullet, set_quat, Euler, PI, RED, BLACK, \
    add_line, \
    wait_for_duration, LockRenderer, base_aligned_z, Point, set_point, get_aabb, stable_z_on_aabb, AABB, set_pose, Pose, \
    stable_z, \
    add_segments

from pybullet_planning.pybullet_tools.utils import interval_generator, get_distance_fn, get_extend_fn, all_between, get_self_link_pairs, \
    get_moving_links, can_collide, CollisionPair, parse_body, cached_fn, get_buffered_aabb, aabb_overlap, \
    pairwise_link_collision, \
    product, pairwise_collision, check_initial_end

from pybullet_planning.motion.motion_planners.meta import solve, check_direct
from pybullet_planning.motion.motion_planners.rrt_star import OptimalNode

from pybullet_planning.motion.motion_planners.smoothing import smooth_path
from pybullet_planning.motion.motion_planners.utils import RRT_RESTARTS, RRT_SMOOTHING, INF, irange, elapsed_time, compute_path_cost, \
    default_selector, argmin, \
    BLUE, RED, elapsed_time  



EPSILON = 1e-6
PRINT_FREQUENCY = 100

SLEEP = None  # None | 0.05
MAX_DISTANCE = 0.  # 0. | 1e-3


class Action:
    def __init__(self, body, joints, end_conf, lower_limits, upper_limits, obstacles=[], attachments=[],
                self_collisions=True, disabled_collisions=set(),
                weights=None, resolutions=None, max_distance=MAX_DISTANCE,
                use_aabb=False, cache=True, custom_limits={}, algorithm="rrt_star", **kwargs):

        if (weights is None) and (resolutions is not None):
            weights = np.reciprocal(resolutions)
        self.start_conf = get_joint_positions(body, joints)
        self.end_conf = end_conf
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.custom_limits = custom_limits
        self.algorithm = algorithm
        self.sample_fn = self.get_sample_fn(self.lower_limits, self.upper_limits)
        self.distance_fn = get_distance_fn(body, joints, weights=weights)
        self.extend_fn = get_extend_fn(body, joints, resolutions=resolutions) 
        self.collision_fn = self.get_collision_fn(body, joints, obstacles, attachments,
                            self_collisions, disabled_collisions, 
                            custom_limits = custom_limits, lower_limits = self.lower_limits,
                            upper_limits = self.upper_limits, max_distance = max_distance, 
                            use_aabb = use_aabb, cache = cache)


    def get_sample_fn(self, lower_limits, upper_limits, **kwargs):
        generator = interval_generator(lower_limits, upper_limits, **kwargs)

        def fn():
            return tuple(next(generator))

        return fn

    def get_limits_fn(self, lower_limits, upper_limits, verbose=False):

        def limits_fn(q):
            if not all_between(lower_limits, q, upper_limits):
                return True
            return False

        return limits_fn

    def get_collision_fn(self, body, joints, obstacles=[], attachments=[], self_collisions=True, disabled_collisions=set(),
                     custom_limits={}, lower_limits={}, upper_limits={}, use_aabb=False, cache=False,
                     max_distance=MAX_DISTANCE, **kwargs):
        
        check_link_pairs = get_self_link_pairs(body, joints, disabled_collisions) if self_collisions else []
        moving_links = frozenset(link for link in get_moving_links(body, joints)
                                if can_collide(body, link))  # TODO: propagate elsewhere
        attached_bodies = [attachment.child for attachment in attachments]
        moving_bodies = [CollisionPair(body, moving_links)] + list(map(parse_body, attached_bodies))
        get_obstacle_aabb = cached_fn(get_buffered_aabb, cache=cache, max_distance=max_distance / 2., **kwargs)
        limits_fn = self.get_limits_fn(lower_limits=lower_limits, upper_limits=upper_limits)

        def collision_fn(q, verbose=False):
            if limits_fn(q):
                return True
            set_joint_positions(body, joints, q)
            for attachment in attachments:
                attachment.assign()
            get_moving_aabb = cached_fn(get_buffered_aabb, cache=True, max_distance=max_distance / 2., **kwargs)

            for link1, link2 in check_link_pairs:
                # Self-collisions should not have the max_distance parameter
                # TODO: self-collisions between body and attached_bodies (except for the link adjacent to the robot)
                if (not use_aabb or aabb_overlap(get_moving_aabb(body), get_moving_aabb(body))) and \
                        pairwise_link_collision(body, link1, body, link2):  # , **kwargs):
                    # print(get_body_name(body), get_link_name(body, link1), get_link_name(body, link2))
                    if verbose: print(body, link1, body, link2)
                    return True

            for body1, body2 in product(moving_bodies, obstacles):
                if (not use_aabb or aabb_overlap(get_moving_aabb(body1), get_obstacle_aabb(body2))) \
                        and pairwise_collision(body1, body2, **kwargs):
                    # print(get_body_name(body1), get_body_name(body2))
                    if verbose: print(body1, body2)
                    return True
            return False

        return collision_fn
    
    def safe_path(self, sequence, collision):
        path = []
        for q in sequence:
            if collision(q):
                break
            path.append(q)
        return path


    def setup_action(self):

        # if not check_initial_end(self.start_conf, self.end_conf, self.collision_fn)
        #     return None 
        
        # path = check_direct(self.start_conf, self.end_conf, self.extend_fn, self.collision_fn)
        # if path is not None or algorithm == 'direct':
        #     return path
        self.time_spent = 0
        self.max_iterations = INF
        self.goal_probability = 0.2
        self.max_time = INF
        self.start = self.start_conf 
        self.goal = self.end_conf
        self.nodes = [OptimalNode(self.start)]
        self.goal_n = None
        self.iteration = 0
        self.radius = 1
        self.time_step = 5
        self.start_time_total = time.time()
        self.informed = True
        self.itr = 0 
        self.path = None
        self.check_direct = False

    def refine_action(self):

        if not self.check_direct :
            path = check_direct(self.start_conf, self.end_conf, self.extend_fn, self.collision_fn)
            self.check_direct = True
            if path is not None or self.algorithm == 'direct':
                return 1, 'E', path
        
        path = None
        start_time = time.time()
        while elapsed_time(start_time) < self.time_step and elapsed_time(self.start_time_total) < self.max_time:
            do_goal = self.goal_n is None and (self.iteration == 0 or random() < self.goal_probability)
            s = self.goal if do_goal else self.sample_fn()

            if self.informed and (self.goal_n is not None) and \
                    (self.distance_fn(self.start, s) + self.distance_fn(s, self.goal) >= self.goal_n.cost):
                continue

            if self.iteration%PRINT_FREQUENCY == 0:
                self.success = self.goal_n is not None
                self.cost = self.goal_n.cost if self.success else INF 
                print('Iteration: {} | Time: {:.3f} | Success: {} | {} | Cost: {:.3f}'.format(
                    self.iteration, elapsed_time(start_time), self.success, do_goal, self.cost))
            self.iteration += 1

            nearest = argmin(lambda n_test : self.distance_fn(n_test.config, s), self.nodes)
            path_ = self.safe_path(self.extend_fn(nearest.config, s), self.collision_fn)

            if len(path_) == 0:
                continue

            new = OptimalNode(path_[-1], parent=nearest,
                                d=self.distance_fn(nearest.config, path_[-1]), path=path_[:-1],
                                iteration=self.iteration)
            
            if do_goal and self.distance_fn(new.config, self.goal) < EPSILON:
                self.goal_n = new 
                self.goal_n.set_solution(True)
                self.time_spent = self.time_spent + elapsed_time(start_time)
                print("Path found Plan Cost ", self.goal_n.cost, "Time", self.time_spent)
                print("\n")
                path = self.goal_n.retrace()
                break

            neighbors = filter(lambda n_test: self.distance_fn(n_test.config, new.config) < self.radius, self.nodes)
            self.nodes.append(new)
            # TODO: smooth solution once found to improve the cost bound
            for n in neighbors:
                d = self.distance_fn(n.config, new.config)
                if (n.cost + d) < new.cost:
                    path_ = self.safe_path(self.extend_fn(n.config, new.config), self.collision_fn)
                    if (len(path_) != 0) and (self.distance_fn(new.config, path_[-1]) < EPSILON):
                        new.rewire(n, d, path_[:-1], iteration=self.iteration)
            for n in neighbors:  # TODO - avoid repeating work
                d = self.distance_fn(new.config, n.config)
                if (new.cost + d) < n.cost:
                    path_ = self.safe_path(self.extend_fn(new.config, n.config), self.collision_fn)
                    if (len(path_) != 0) and (self.distance_fn(n.config, path_[-1]) < EPSILON):
                        n.rewire(new, d, path_[:-1], iteration=self.iteration)
            self.time_spent = self.time_spent + elapsed_time(start_time)
            if self.goal_n is None :
                continue
            else :
                break
        if self.goal_n is None:
            return 0, 'P', None
        else :
            path = smooth_path(path, self.extend_fn, self.collision_fn, # sample_fn=sample_fn,
                           distance_fn=self.distance_fn, cost_fn=None,
                           max_iterations=None, max_time=INF, verbose=False)
            return 1, 'E', path


class PR2:
    def __init__(self, num_of_plans, num_of_actions):
        self.pr2 = self.create_pr2()
        self.num_of_plans = num_of_plans 
        self.num_of_actions = num_of_actions
        self.is_done = np.zeros((self.num_of_plans, self.num_of_plans), dtype = int)
        
        self.lower_limits = [[[-3, 0]], [[-3, -4]]]
        self.upper_limits = [[[14, 6]], [[24, 1.5]]]
        self.time_spent = np.zeros((self.num_of_plans, self.num_of_plans), dtype = int)
        self.plans = []
        self.setup_pr2(self.base_start, self.base_goal)
        self.create_actions()

    def reset(self):
        self.time_spent = np.zeros((self.num_of_plans, self.num_of_plans), dtype = int)
        self.plans = []
        self.setup_pr2(self.base_start, self.base_goal)
        self.create_actions()

    
    def setup_pr2(self, base_start, base_goal):
        self.disabled_collisions = get_disabled_collisions(self.pr2)
        base_joints = [joint_from_name(self.pr2, name) for name in PR2_GROUPS['base']]
        set_joint_positions(self.pr2, base_joints, base_start)
        self.base_joints = base_joints[:2]
        self.base_goal = base_goal[:len(self.base_joints)]

    def create_actions(self):
        # Creating actions 
        action1 = Action(self.pr2, self.base_joints, self.base_goal, obstacles = self.obstacles,
                        lower_limits = self.lower_limits[0][0], upper_limits = self.upper_limits[0][0],
                        disables_collisions = self.disabled_collisions)
        action1.setup_action()
        actions = []
        actions.append(action1)
        self.plans.append(actions)
        action2 = Action(self.pr2, self.base_joints, self.base_goal, obstacles = self.obstacles,
                        lower_limits = self.lower_limits[1][0], upper_limits = self.upper_limits[1][0],
                        disables_collisions = self.disabled_collisions)
        action2.setup_action()
        actions = []
        actions.append(action2)
        self.plans.append(actions)

    
    def step(self, action, last_refined_action):

        action_ = self.plans[action][last_refined_action]
        res, mode, base_path = action_.refine_action()
        if res == 0:
            return 0, 'P', 0 
        elif res == 1:
            # need to excute and get execution time 
            etime = 5
            speed = 1 
            start_point = base_path[0]
            # for i in range(1, len(base_path)):
            #     next_point = base_path[i]
            #     distance = np.sqrt((next_point[1] - start_point[1]) ** 2 + (next_point[0] - start_point[0]) ** 2)
            #     etime = etime + distance / speed
            #     start_point = next_point
            
            if last_refined_action + 1 == self.num_of_actions:
                return 2, 'E', etime
            else :
                return 1, 'E', etime
    

    def create_pr2(self, use_pr2_drake=True):
        connect(use_gui = False)
        set_camera_pose(camera_point=[1, -1, 6])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane = p.loadURDF("plane.urdf")        
        table_path = "/pybullet_planning/models/table_collision/table.urdf"
        wall_path = "/pybullet_planning/models/drake/objects/wall.urdf"

        wall1 = load_model(wall_path, fixed_base=False)
        set_quat(wall1, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(wall1, Pose(Point(x=9, y=-1, z=stable_z(wall1, plane))))

        table_1 = load_model(table_path, fixed_base=False)
        set_quat(table_1, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_1, Pose(Point(x=1.25, y=0.15, z=stable_z(table_1, plane))))
        #
        table_2 = load_model(table_path, fixed_base=False)
        set_quat(table_2, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_2, Pose(Point(x=1.25, y=3.0, z=stable_z(table_2, plane))))

        table_3 = load_model(table_path, fixed_base=False)
        set_quat(table_3, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_3, Pose(Point(x=4.8, y=0.4, z=stable_z(table_3, plane))))

        table_4 = load_model(table_path, fixed_base=False)
        set_quat(table_4, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_4, Pose(Point(x=5.5, y=3.25, z=stable_z(table_4, plane))))
        #
        table_5 = load_model(table_path, fixed_base=False)
        set_quat(table_5, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_5, Pose(Point(x=8.5, y=6, z=stable_z(table_5, plane))))

        table_6 = load_model(table_path, fixed_base=False)
        set_quat(table_6, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_6, Pose(Point(x=9, y=3.15, z=stable_z(table_6, plane))))

        table_7 = load_model(table_path, fixed_base=False)
        set_quat(table_7, quat_from_euler(Euler(yaw=PI / 2)))
        set_pose(table_7, Pose(Point(x=10, y=0.4, z=stable_z(table_7, plane))))

        self.obstacles = [plane, wall1, table_1, table_2, table_3, table_4, table_5, table_6, table_7]
    
        pr2_urdf = DRAKE_PR2_URDF if use_pr2_drake else PR2_URDF
        with HideOutput():
            pr2 = load_model(pr2_urdf, fixed_base=True)  # TODO: suppress warnings?
        dump_body(pr2)

        z = base_aligned_z(pr2)

        set_point(pr2, Point(z=z))
        self.base_start = (-2, 0.5, 0)
        self.base_goal = (13, 0.3, 0)

        arm_start = SIDE_HOLDING_LEFT_ARM
        # arm_start = TOP_HOLDING_LEFT_ARM
        # arm_start = REST_LEFT_ARM
        arm_goal = TOP_HOLDING_LEFT_ARM
        # arm_goal = SIDE_HOLDING_LEFT_ARM

        left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
        right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
        torso_joints = joints_from_names(pr2, PR2_GROUPS['torso'])
        set_joint_positions(pr2, left_joints, arm_start)
        set_joint_positions(pr2, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
        set_joint_positions(pr2, torso_joints, [0.2])
        open_arm(pr2, 'left')
        # test_ikfast(pr2)

        add_line(self.base_start, self.base_goal, color=RED)
        return pr2