"""
简单环境类
用于定义环境的初始化、重置、步进等操作
"""
import sys
import random
import numpy as np
import xml.etree.ElementTree as ET
from mujoco_env.mujoco_parser import MuJoCoParserClass
from mujoco_env.utils import prettify, sample_xyzs, rotation_matrix, add_title_to_img
from mujoco_env.ik import solve_ik
from mujoco_env.transforms import rpy2r, r2rpy
import os
import copy
import glfw

class SimpleEnv:
    MAX_INITIALIZATION_ATTEMPTS = 100
    INITIALIZATION_SETTLE_STEPS = 100
    MUG_XY_BOUNDS = ((0.20, 0.44), (-0.44, 0.24))
    PLATE_XY_BOUNDS = ((0.20, 0.44), (-0.44, 0.24))
    MUG_Z_BOUNDS = (0.80, 0.95)
    PLATE_Z_BOUNDS = (0.78, 0.85)

    def __init__(self,
                 xml_path,
                 action_type='eef_pose',
                 state_type='joint_angle',
                 seed=None,
                 use_viewer=True):
        """
        参数:
            xml_path: str, xml文件路径
            action_type: str, 动作空间类型，'eef_pose'、'delta_joint_angle' 或 'joint_angle'
            state_type: str, 状态空间类型，'joint_angle' 或 'ee_pose'
            seed: int, 随机数生成器的种子
            use_viewer: bool, 是否创建 MuJoCo 窗口；headless 部署时设为 False
        """
        # 加载xml文件
        self.env = MuJoCoParserClass(name='Tabletop', rel_xml_path=xml_path)
        self.action_type = action_type
        self.state_type = state_type
        self.use_viewer = use_viewer

        self.joint_names = ['shoulder_pan_joint',
                    'shoulder_lift_joint',
                    'elbow_joint',
                    'wrist_1_joint',
                    'wrist_2_joint',
                    'wrist_3_joint',]
        if self.use_viewer:
            self.init_viewer()
        self.reset(seed)

    def init_viewer(self):
        '''
        初始化查看器
        '''
        self.env.reset()
        self.env.init_viewer(
            distance          = 2.0,
            elevation         = -30, 
            transparent       = False,
            black_sky         = True,
            use_rgb_overlay = False,
            loc_rgb_overlay = 'top right',
        )

    def _has_mug_plate_contact(self):
        """检查杯子和盘子的碰撞网格是否存在直接接触。"""
        model = self.env.model
        data = self.env.data
        mug_root_id = model.body('body_obj_mug_5').id
        plate_root_id = model.body('body_obj_plate_11').id
        target_roots = {mug_root_id, plate_root_id}

        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            body1_id = model.geom_bodyid[contact.geom1]
            body2_id = model.geom_bodyid[contact.geom2]
            contact_roots = {
                int(model.body_rootid[body1_id]),
                int(model.body_rootid[body2_id]),
            }
            if contact_roots == target_roots:
                return True
        return False

    @staticmethod
    def _is_in_bounds(position, xy_bounds, z_bounds):
        """检查物体稳定后是否仍位于任务工作区。"""
        return (
            xy_bounds[0][0] <= position[0] <= xy_bounds[0][1]
            and xy_bounds[1][0] <= position[1] <= xy_bounds[1][1]
            and z_bounds[0] <= position[2] <= z_bounds[1]
        )

    def _settled_scene_error(self):
        """返回稳定后场景的拒绝原因，空字符串表示场景合法。"""
        p_mug, p_plate = self.get_obj_pose()
        if not np.isfinite(np.concatenate([p_mug, p_plate])).all():
            return 'non_finite_position'
        if self._has_mug_plate_contact():
            return 'mug_plate_contact_after_settle'
        if not self._is_in_bounds(p_mug, self.MUG_XY_BOUNDS, self.MUG_Z_BOUNDS):
            return 'mug_out_of_bounds'
        if not self._is_in_bounds(p_plate, self.PLATE_XY_BOUNDS, self.PLATE_Z_BOUNDS):
            return 'plate_out_of_bounds'
        return ''

    def reset(self, seed=None):
        '''
        重置环境
        将机器人移动到初始位置，根据种子设置物体位置
        '''
        if seed is not None:
            np.random.seed(seed=seed)

        q_zero = np.deg2rad([0, -90, 90, -90, -90, 90])
        obj_names = self.env.get_body_names(prefix='body_obj_')
        n_obj = len(obj_names)
        self.last_reset_attempts = 0
        self.last_reset_rejections = {}

        for attempt in range(1, self.MAX_INITIALIZATION_ATTEMPTS + 1):
            # 每次候选场景都从干净动力学状态开始，但 seed 只设置一次以推进确定随机序列。
            self.env.reset(step=False)
            self.env.forward(q=q_zero,joint_names=self.joint_names,increase_tick=False)

            obj_xyzs = sample_xyzs(
                n_obj,
                x_range   = [+0.24,+0.4],
                y_range   = [-0.4,+0.2],
                z_range   = [0.81,0.81],
                min_dist  = 0.15,
                xy_margin = 0.0
            )
            for obj_idx in range(n_obj):
                self.env.set_p_base_body(body_name=obj_names[obj_idx],p=obj_xyzs[obj_idx,:])
                self.env.set_R_base_body(body_name=obj_names[obj_idx],R=np.eye(3,3))
            self.env.forward(increase_tick=False)

            if self._has_mug_plate_contact():
                reason = 'mug_plate_contact_before_settle'
                self.last_reset_rejections[reason] = self.last_reset_rejections.get(reason, 0) + 1
                continue

            self.last_q = copy.deepcopy(q_zero)
            self.q = np.concatenate([q_zero, np.array([0.0]*1)])
            self.p0, self.R0 = self.env.get_pR_body(body_name='wrist_3_link')
            mug_init_pose, plate_init_pose = self.get_obj_pose()
            candidate_obj_init_pose = np.concatenate(
                [mug_init_pose, plate_init_pose],
                dtype=np.float32,
            )
            for _ in range(self.INITIALIZATION_SETTLE_STEPS):
                self.step_env()

            reason = self._settled_scene_error()
            if reason:
                self.last_reset_rejections[reason] = self.last_reset_rejections.get(reason, 0) + 1
                continue

            self.obj_init_pose = candidate_obj_init_pose
            self.last_reset_attempts = attempt
            break
        else:
            raise RuntimeError(
                f'环境初始化失败: seed={seed}, '
                f'已尝试 {self.MAX_INITIALIZATION_ATTEMPTS} 次, '
                f'拒绝原因={self.last_reset_rejections}'
            )

        if self.last_reset_attempts > 1:
            print(
                f'INITIALIZATION RESAMPLED: seed={seed}, '
                f'attempts={self.last_reset_attempts}, '
                f'rejections={self.last_reset_rejections}'
            )
        print("DONE INITIALIZATION")
        self.gripper_state = False
        self.past_chars = []

    def step(self, action):
        '''
        在环境中执行一步
        参数:
            action: np.array，形状为 (7,)，要执行的动作
        返回:
            state: np.array，执行动作后的环境状态
                - ee_pose: [px, py, pz, r, p, y] 末端执行器位姿
                - joint_angle: [j1, j2, j3, j4, j5, j6] 关节角度
        '''
        if self.action_type == 'eef_pose':
            q = self.env.get_qpos_joints(joint_names=self.joint_names)
            self.p0 += action[:3]
            self.R0 = self.R0.dot(rpy2r(action[3:6]))
            q ,ik_err_stack,ik_info = solve_ik(
                env                = self.env,
                joint_names_for_ik = self.joint_names,
                body_name_trgt     = 'wrist_3_link',
                q_init             = q,
                p_trgt             = self.p0,
                R_trgt             = self.R0,
                max_ik_tick        = 50,
                ik_stepsize        = 1.0,
                ik_eps             = 1e-2,
                ik_th              = np.radians(5.0),
                render             = False,
                verbose_warning    = False,
            )
        elif self.action_type == 'delta_joint_angle':
            q = action[:-1] + self.last_q
        elif self.action_type == 'joint_angle':
            q = action[:-1]
        else:
            raise ValueError('action_type not recognized')

        gripper_val = action[-1] * 255.0
        gripper_cmd = np.array([gripper_val], dtype=np.float32)
        self.compute_q = q
        q = np.concatenate([q, gripper_cmd])

        self.q = q
        if self.state_type == 'joint_angle':
            return self.get_joint_state()
        elif self.state_type == 'ee_pose':
            return self.get_ee_pose()
        elif self.state_type == 'delta_q' or self.action_type == 'delta_joint_angle':
            dq =  self.get_delta_q()
            return dq
        else:
            raise ValueError('state_type not recognized')

    def step_env(self):
        self.env.step(self.q)

    def grab_image(self):
        '''
        从环境中抓取图像
        返回:
            rgb_agent: np.array, 智能体视角的RGB图像
            rgb_ego: np.array, 第一人称视角的RGB图像
        '''
        self.rgb_agent = self.env.get_fixed_cam_rgb(
            cam_name='agentview')
        self.rgb_ego = self.env.get_fixed_cam_rgb(
            cam_name='d435i_rgb')
        # self.rgb_top = self.env.get_fixed_cam_rgbd_pcd(
        #     cam_name='topview')
        self.rgb_side = self.env.get_fixed_cam_rgb(
            cam_name='sideview')
        return self.rgb_agent, self.rgb_ego
        

    def render(self, teleop=False):
        '''
        渲染环境
        '''
        if not self.use_viewer:
            # headless 部署只需要固定相机离屏渲染，不需要窗口叠加层。
            return
        self.env.plot_time()
        p_current, R_current = self.env.get_pR_body(body_name='wrist_3_link')
        R_current = R_current @ np.array([[1,0,0],[0,0,1],[0,1,0 ]])
        self.env.plot_sphere(p=p_current, r=0.02, rgba=[0.95,0.05,0.05,0.5])
        self.env.plot_capsule(p=p_current, R=R_current, r=0.01, h=0.2, rgba=[0.05,0.95,0.05,0.5])
        rgb_egocentric_view = add_title_to_img(self.rgb_ego,text='Egocentric View',shape=(640,480))
        rgb_agent_view = add_title_to_img(self.rgb_agent,text='Agent View',shape=(640,480))
        
        self.env.viewer_rgb_overlay(rgb_agent_view,loc='top right')
        self.env.viewer_rgb_overlay(rgb_egocentric_view,loc='bottom right')
        if teleop:
            rgb_side_view = add_title_to_img(self.rgb_side,text='Side View',shape=(640,480))
            self.env.viewer_rgb_overlay(rgb_side_view, loc='top left')
            self.env.viewer_text_overlay(text1='Key Pressed',text2='%s'%(self.env.get_key_pressed_list()))
            self.env.viewer_text_overlay(text1='Key Repeated',text2='%s'%(self.env.get_key_repeated_list()))
            joint_angles = self.env.get_qpos_joints(joint_names=self.joint_names)
            joint_angles_deg = np.rad2deg(joint_angles)
            angle_str = ' '.join([f'{j:.0f}' for j in joint_angles_deg])
            self.env.viewer_text_overlay(text1='Joint Angles', text2=angle_str)
        self.env.render()

    def get_joint_state(self):
        '''
        获取机器人的关节状态
        返回:
            q: np.array, 机器人关节角度 + 夹爪状态（0表示打开，1表示关闭）
            [j1, j2, j3, j4, j5, j6, gripper]
        '''
        qpos = self.env.get_qpos_joints(joint_names=self.joint_names)
        gripper = self.env.get_qpos_joint('right_driver_joint')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([qpos, [gripper_cmd]],dtype=np.float32)
    
    def teleop_robot(self):
        '''
        使用键盘遥控机器人
        返回:
            action: np.array, 要执行的动作
            done: bool, 如果用户想要重置遥控则为True
        
        按键说明:
            ---------     -----------------------
               w       ->        向后移动
            s  a  d        向左   向前   向右
            ---------      -----------------------
            在 x, y 平面移动

            ---------
            R: 向上移动
            F: 向下移动
            ---------
            在 z 轴移动

            ---------
            Q: 向左倾斜
            E: 向右倾斜
            UP: 向上看
            Down: 向下看
            Right: 向右转
            Left: 向左转
            ---------
            用于旋转

            ---------
            z: 重置
            SPACEBAR: 夹爪打开/关闭
            ---------   
        '''
        # char = self.env.get_key_pressed()
        dpos = np.zeros(3)
        drot = np.eye(3)
        if self.env.is_key_pressed_repeat(key=glfw.KEY_S):
            dpos += np.array([0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_W):
            dpos += np.array([-0.007,0.0,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_A):
            dpos += np.array([0.0,-0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_D):
            dpos += np.array([0.0,0.007,0.0])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_R):
            dpos += np.array([0.0,0.0,0.007])
        if self.env.is_key_pressed_repeat(key=glfw.KEY_F):
            dpos += np.array([0.0,0.0,-0.007])
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_LEFT):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if  self.env.is_key_pressed_repeat(key=glfw.KEY_RIGHT):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 1.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_DOWN):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_UP):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[1.0, 0.0, 0.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_Q):
            drot = rotation_matrix(angle=0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_repeat(key=glfw.KEY_E):
            drot = rotation_matrix(angle=-0.1 * 0.3, direction=[0.0, 0.0, 1.0])[:3, :3]
        if self.env.is_key_pressed_once(key=glfw.KEY_Z):
            return np.zeros(7, dtype=np.float32), True
        if self.env.is_key_pressed_once(key=glfw.KEY_SPACE):
            self.gripper_state =  not  self.gripper_state
        drot = r2rpy(drot)
        action = np.concatenate([dpos, drot, np.array([self.gripper_state],dtype=np.float32)],dtype=np.float32)
        return action, False
    
    def get_delta_q(self):
        '''
        获取机器人关节角度的增量
        返回:
            delta: np.array, 机器人关节角度增量 + 夹爪状态（0表示打开，1表示关闭）
            [dj1, dj2, dj3, dj4, dj5, dj6, gripper]
        '''
        delta = self.compute_q - self.last_q
        self.last_q = copy.deepcopy(self.compute_q)
        gripper = self.env.get_qpos_joint('right_driver_joint')
        gripper_cmd = 1.0 if gripper[0] > 0.5 else 0.0
        return np.concatenate([delta, [gripper_cmd]],dtype=np.float32)

    def get_gripper_qpos(self):
        """返回夹爪关节位置，便于部署指标判断夹爪是否真正松开。"""
        gripper = self.env.get_qpos_joint('right_driver_joint')
        return float(np.asarray(gripper).reshape(-1)[0])

    def get_task_metrics(self, placement_xy_threshold=0.1, placement_z_threshold=0.1):
        """返回杯盘距离、夹爪状态和双口径成功信号，用于论文实验评估。"""
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        ee_z = float(self.env.get_p_body('wrist_3_link')[2])
        gripper_qpos = self.get_gripper_qpos()
        mug_plate_xy_dist = float(np.linalg.norm(p_mug[:2] - p_plate[:2]))
        mug_plate_z_gap = float(abs(p_mug[2] - p_plate[2]))
        # placement_success 只描述几何放置到位，不包含松爪与抬升约束。
        placement_success = (
            mug_plate_xy_dist < placement_xy_threshold
            and mug_plate_z_gap < placement_z_threshold
        )
        strict_success = (
            placement_success
            and gripper_qpos < 0.1
            and ee_z > 0.9
        )
        return {
            "mug_position": p_mug.astype(float).tolist(),
            "plate_position": p_plate.astype(float).tolist(),
            "mug_plate_xy_dist": mug_plate_xy_dist,
            "mug_plate_z_gap": mug_plate_z_gap,
            "placement_xy_threshold": placement_xy_threshold,
            "placement_z_threshold": placement_z_threshold,
            "final_gripper_qpos": gripper_qpos,
            "ee_z": ee_z,
            "placement_success": bool(placement_success),
            "strict_success": bool(strict_success),
        }

    def check_placement_success(self, placement_xy_threshold=0.1, placement_z_threshold=0.1):
        """只判断杯子是否到达盘子附近，不要求夹爪释放。"""
        return self.get_task_metrics(placement_xy_threshold, placement_z_threshold)["placement_success"]

    # 判断任务是否成功
    def check_success(self):
        '''
        ['body_obj_mug_5', 'body_obj_plate_11']
        检查杯子是否放在盘子上
        + 夹爪应该打开并且末端执行器应向上移动超过0.9高度
        '''
        return self.get_task_metrics()["strict_success"]
    
    def get_obj_pose(self):
        '''
        返回:
            p_mug: np.array, 杯子的位置
            p_plate: np.array, 盘子的位置
        '''
        p_mug = self.env.get_p_body('body_obj_mug_5')
        p_plate = self.env.get_p_body('body_obj_plate_11')
        return p_mug, p_plate

    # 设置杯子和盘子的位置和方向
    def set_obj_pose(self, p_mug, p_plate):
        '''
        设置物体姿态
        参数:
            p_mug: np.array, 杯子的位置
            p_plate: np.array, 盘子的位置
        '''
        self.env.set_p_base_body(body_name='body_obj_mug_5',p=p_mug)
        self.env.set_R_base_body(body_name='body_obj_mug_5',R=np.eye(3,3))
        self.env.set_p_base_body(body_name='body_obj_plate_11',p=p_plate)
        self.env.set_R_base_body(body_name='body_obj_plate_11',R=np.eye(3,3))
        self.step_env()

    # 获取末端执行器的位姿
    def get_ee_pose(self):
        '''
        获取机器人末端执行器的位姿 + 夹爪状态
        '''
        p, R = self.env.get_pR_body(body_name='wrist_3_link')
        rpy = r2rpy(R)
        return np.concatenate([p, rpy],dtype=np.float32)
