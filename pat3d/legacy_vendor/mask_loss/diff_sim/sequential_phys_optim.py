import argparse
import copy
from itertools import count
import math
import numpy as np
import polyscope as ps
import torch
import torch.nn as nn
import torch.optim as optim
from polyscope import imgui
import shutil
try:
    from torch.utils.tensorboard import SummaryWriter  # 新增TensorBoard支持
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_histogram(self, *args, **kwargs):
            pass

        def close(self):
            pass
import os 
from uipc import view
from uipc import Logger, Timer
from uipc import \
    Vector3, Vector3i, \
    Matrix4x4, Matrix4x4i,\
    Transform, Quaternion, AngleAxis
import trimesh as my_trimesh
from uipc import builtin
from uipc.core import *
from uipc.geometry import *
from uipc.constitution import *
from uipc.gui import *
from uipc.torch import *
from uipc.unit import GPa, MPa


def _init_polyscope_headless_safe():
    if hasattr(ps, "set_allow_headless_backends"):
        ps.set_allow_headless_backends(True)
    egl_device_index = os.environ.get("POLYSCOPE_EGL_DEVICE_INDEX", "").strip()
    if egl_device_index and hasattr(ps, "set_egl_device_index"):
        try:
            ps.set_egl_device_index(int(egl_device_index))
        except ValueError:
            pass
    ps.init()
import json
import shutil
from modules.bbox_put_utils.get_children_center import *

_UIPC_WORKSPACE_COUNTER = count()


def _make_uipc_engine_config(args):
    confige = Engine.default_config()
    confige.setdefault('gpu', {})['device'] = int(getattr(args, 'gpu_id', 0))
    extras = confige.setdefault('extras', {})
    if isinstance(extras, dict):
        gui_config = extras.setdefault('gui', {})
        if isinstance(gui_config, dict):
            gui_config['enable'] = bool(getattr(args, 'gui_flag', False))
    return confige


def _make_uipc_workspace(result_folder, role, args):
    pass_index = int(getattr(args, 'diff_sim_pass_index', 0))
    workspace = os.path.join(
        result_folder,
        'workspace',
        f'{role}_pass_{pass_index:03d}_{os.getpid()}_{next(_UIPC_WORKSPACE_COUNTER):04d}',
    )
    os.makedirs(workspace, exist_ok=True)
    return workspace


class PhysOptim(object):
    def __init__(self, args, container_name:str, mesh_dict:dict, tmesh_dict:dict):

        self.args = args
        self.container_name = container_name
        self.best_loss = -1
        self.loss_history = []

        self.init_path(mesh_dict)
        self.set_writer()
        self.set_global_parameters()

        Timer.enable_all()
        Logger.set_level(Logger.Level.Warn)

        ## fixed physical parameters
        confige = _make_uipc_engine_config(self.args)
        engine = Engine('cuda', self.workspace, confige)
        self.world = World(engine)
        config = Scene.default_config()


        ## =====================================================================================================================

        ## parameters 
        config['dt'] = self.args.time_step
        config['gravity'] = [[0.0], [-9.8], [0.0]]
        config['contact']['enable']             = True 
        # config['contact']['d_hat']              = 0.0005 ## threshold for conlision detect 
        # config['contact']['d_hat']              = 3e-4 ## threshold for conlision detect 
        config['contact']['d_hat']              = float(getattr(self.args, 'contact_d_hat', 5e-4)) ## threshold for collision detect
        config['contact']['friction']['enable'] = True ## whether to use friction ## set it to be false for now
        config['line_search']['max_iter']       = 8 ## fixed, max iteration for line search
        config['linear_system']['tol_rate']     = self.args.tol_rate ## 1e-4 is more stable, except it has some related errors
        config['contact']['eps_velocity']     = float(getattr(self.args, 'contact_eps_velocity', 1e-5))
        config['newton']['velocity_tol'] = 0.1
        # config['contact']['eps_velocity']     = 1e-4
        # config['contact']['eps_velocity']     = 1e-5
        ## =====================================================================================================================


        ## initialize the self.scene 
        self.scene = Scene(config)

        ## for loading and transform the meshes, set to be identity here for no transformation
        pre_transform = Transform.Identity()
        pre_transform.scale(1)
        ## IO for loading meshes
        self.io = SimplicialComplexIO(pre_transform)

        ## self.scene contact 
        ## friction parameter: 0.2 
        ## stiffness: 1.0 * GPa   
        self.scene.contact_tabular().default_model(0.2, 1.0 * GPa)
        default_element = self.scene.contact_tabular().default_element()

        ## create contact property for each object
        self.contact_property_dict = {}
        ## set a self for each object and assign it to each object 
        for obj_name in self.mesh_name_list:
            contact_property_item = self.scene.contact_tabular().create(obj_name)
            self.contact_property_dict[obj_name] = contact_property_item
            self.scene.contact_tabular().insert(contact_property_item, contact_property_item, 0, 0, False)

        ## define the material of rigid body 
        self.abd = AffineBodyConstitution()

        ## load the meshes, don't care about this  
        self.scene_objects = self.scene.objects().create('scene_objects')
        self.optim_samples_count = (self.args.end_frame - self.args.offset_frame) // self.args.optim_frame_interval ## the interval of computing the loss function

        ## load surface 
        self.load_meshes(mesh_dict, tmesh_dict)
        self.loss_target_ids = self._build_loss_target_ids()
        self.support_metric_target_ids = self._build_support_metric_target_ids()

        ## set the target point for each object
        self.set_target_centers()
        self.set_adaptive_end_frame_state()

        ## initial the objects in the scene
        self.set_material_contact()
        self.bind_translation_param()
        self.set_physical_properties()

        ## create the ground 
        self.set_ground()

        ## define the optimization parameters
        self.param = DiffSimParameter(scene = self.scene, size = self.param_map_index)

        ## initialize the simulation
        self.world.init(self.scene)

        ## simulation process 
        def dof_select(info:DiffSimModule.DofSelectInfo):
            if (
                info.frame() > self.args.offset_frame
                and info.frame() % self.args.optim_frame_interval == 0
                and info.frame() <= int(self.current_effective_end_frame)
            ): ## get the frame number
                sdi = torch.arange(0, self.obj_num * self.per_obj_dof_num, dtype = torch.int32) ## define dof --> (obj_num * (9+3))
                #print(f'[dof_select] sdi: {sdi}')
                return sdi
            else:
                return None

        ## define the module 
        self.diff_module = DiffSimModule(self.world, end_frame = self.args.end_frame, parm = self.param, dof_select = dof_select)

        ## set the optimizer
        optimizer_name = str(getattr(self.args, 'optimizer_name', 'adamw')).lower()
        amsgrad = bool(getattr(self.args, 'optimizer_amsgrad', False))
        optimizer_kwargs = {
            'lr': self.args.phys_lr,
            'weight_decay': float(getattr(self.args, 'weight_decay', 0.01)),
            'amsgrad': amsgrad,
        }
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(self.diff_module.parameters(), **optimizer_kwargs)
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(self.diff_module.parameters(), **optimizer_kwargs)
        else:
            raise ValueError(f'unsupported optimizer_name: {optimizer_name}')
        # self.optimizer = optim.SGD(self.diff_module.parameters(), lr = self.args.phys_lr, momentum=0.9)
        # self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.total_opt_epoch//4, eta_min=1e-6)
        # self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.args.phys_lr, total_steps=self.args.total_opt_epoch)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 20, gamma = 0.9)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 10, gamma = 0.6)
        lr = DiffSimLR(self.diff_module, self.optimizer)

        ## random initialization of the parameters
        self.jitter_param()

        self.set_JC()
                
        self.sio = SceneIO(scene = self.scene)

        self.optimize_once()
        
        self.writer.close()

    def jitter_param(self):
        # Pack xz parameters
        initial_parameter = np.zeros(2 * len(self.mesh_dict.keys())).astype(np.float64)
        for mesh_name in self.mesh_dict.keys():
            print(f"jitter param: {mesh_name}")
        for i, mesh in enumerate(self.mesh_dict.values()):
            transform = np.squeeze(view(mesh.transforms())[:])
            print(f"i: {i}, transform: ({transform[0][3]}, {transform[2][3]}), transform: {transform}")
            initial_parameter[2 * i] = transform[0][3]
            initial_parameter[2 * i + 1] = transform[2][3]


        with torch.no_grad():
            # u = torch.zeros(self.param.U().shape[0], dtype = torch.float64)
            # self.param.U().copy_(u)
            self.param.U().copy_(torch.from_numpy(initial_parameter))
            self.param.sync() ## sync the parameters for the simulation 


    def set_JC(self):

        self.JC = torch.zeros((3 * self.obj_num, self.per_obj_dof_num), dtype = torch.float64)
        for i in range(self.obj_num):

            ## set the identity matrix for the first 3 rows of each object
            self.JC[i*3 + 0, 0] = 1
            self.JC[i*3 + 1, 1] = 1
            self.JC[i*3 + 2, 2] = 1

            ## set the 3*9 matrix for each object by copying the target center of each object
            self.JC[i*3 + 0, 3] = float(self.dstX[i*3 + 0])
            self.JC[i*3 + 0, 4] = float(self.dstX[i*3 + 1])
            self.JC[i*3 + 0, 5] = float(self.dstX[i*3 + 2])
            self.JC[i*3 + 1, 6] = float(self.dstX[i*3 + 0])
            self.JC[i*3 + 1, 7] = float(self.dstX[i*3 + 1])
            self.JC[i*3 + 1, 8] = float(self.dstX[i*3 + 2])
            self.JC[i*3 + 2, 9] = float(self.dstX[i*3 + 0])
            self.JC[i*3 + 2, 10] = float(self.dstX[i*3 + 1])
            self.JC[i*3 + 2, 11] = float(self.dstX[i*3 + 2])


        self.BJs = []
        for j in range(self.obj_num):
            BJC = torch.zeros((3 * 8, self.per_obj_dof_num), dtype = torch.float64)
            i=0
            for vertice in self.BBXs[j]:

                BJC[i*3 + 0, 0] = 1
                BJC[i*3 + 1, 1] = 1
                BJC[i*3 + 2, 2] = 1
                #print(vertice)
                ## set the 3*9 matrix for each object by copying the target center of each object
                BJC[i*3 + 0, 3] = float(vertice[0])
                BJC[i*3 + 0, 4] = float(vertice[1])
                BJC[i*3 + 0, 5] = float(vertice[2])
                BJC[i*3 + 1, 6] = float(vertice[0])
                BJC[i*3 + 1, 7] = float(vertice[1])
                BJC[i*3 + 1, 8] = float(vertice[2])
                BJC[i*3 + 2, 9] = float(vertice[0])
                BJC[i*3 + 2, 10] = float(vertice[1])
                BJC[i*3 + 2, 11] = float(vertice[2])
                i = i+1
            self.BJs.append(BJC)
            

    def set_material_contact(self):

        for obj_name in self.mesh_dict.keys():

            ## apply the material to the mesh to define it as a rigid body
            self.abd.apply_to(self.mesh_dict[obj_name], 100.0*MPa, self.args.rho)


    def init_path(self, mesh_dict):

        ## create the result folder 
        self.result_folder = f'{self.args.phys_result_folder}/{self.args.exp_name}'
        #if os.path.exists(self.result_folder):
        #    shutil.rmtree(self.result_folder)
        #os.makedirs(self.result_folder)
         
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        
        ## create the tensorboard log folder
        self.log_folder = f'{self.result_folder}/logs'
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        
        ## set the folder for loading the objects 
        #self.mesh_folder = f'{self.args.low_poly_folder}/{self.args.scene_name}'
        #self.mesh_folder = f'vis/scene_low_poly/{self.args.exp_name}'
        self.mesh_folder = f'data/layer_layout/{self.args.exp_name}'


        ## get the name list for all the objects in the scene
        self.mesh_name_list = []
        for file_name in mesh_dict.keys():
            self.mesh_name_list.append(file_name)
            # if file_name.endswith('.obj'):
            #     obj_name = os.path.splitext(file_name)[0]
            #     self.mesh_name_list.append(obj_name)


        ## folder for storing the internal code logs
        self.workspace = _make_uipc_workspace(self.result_folder, 'phys_optim', self.args)

        ## folder for storing the internal geometry
        self.geo_folder = f'{self.result_folder}/internal'
        if not os.path.exists(self.geo_folder):
            os.makedirs(self.geo_folder)
        
        ## folder for storing the transform parameters (the simulated results)
        self.transform_folder = f'{self.result_folder}/transform'
        if not os.path.exists(self.transform_folder):
            os.makedirs(self.transform_folder)
        
        ## folder for storing the optimzed parameters 
        self.param_folder = f'{self.result_folder}/param'
        if not os.path.exists(self.param_folder):
            os.makedirs(self.param_folder)

        pass_index = int(getattr(self.args, 'diff_sim_pass_index', 0))
        self.loss_history_path = os.path.join(
            self.result_folder,
            f'loss_history_pass_{pass_index:03d}.json',
        )

    def set_writer(self):
        
        self.writer = SummaryWriter(log_dir = self.log_folder)
    
    def set_global_parameters(self):

        self.run = False
        self.global_step = 0
        self.per_obj_dof_num = 12
        self.optim_attempt = 0
        self.stage_best_loss = None
        self.best_selection_key = None
        self.best_selected_end_frame = None
        self.best_qualifying_frame = None
        self.best_epoch_metrics = None
        self.best_checkpoint_step = None
        self.current_epoch_metrics = None
        self.completed_opt_epoch = 0
        self.adaptive_early_stop_triggered = False
        self.adaptive_stop_reason = None
        self.forward_validation_history = []
        self.latest_forward_validation = None
        self.last_forward_validation_epoch = 0
    
    def set_target_centers(self):
        
        ## test target center list
        target_center_list = []
        #target_center_list = [[0.0, 0.0, 0.0], [0.8, 1.3, 0.0], [0.0, 3.0, 0.0], [0,0,0]]
        for obj_name in self.mesh_dict.keys():
            #print(f'obj_name: {obj_name}')
            ## get the center of the object
            obj_center = self.mesh_trimesh_dict[obj_name].bounding_box.centroid.tolist()
            #print(f'obj_name: {obj_name}, obj_center: {obj_center}')
            target_center_list.append(obj_center)

        #print(f'target_center_list: {target_center_list}')
        #exit(0)

        ## set the initial center points for each point, which is the target position for each object 
        self.dstX = torch.zeros(3 * self.obj_num, dtype = torch.float64)

        for i in range(self.obj_num):
            self.dstX[i*3 + 0] = target_center_list[i][0]
            self.dstX[i*3 + 1] = target_center_list[i][1]
            self.dstX[i*3 + 2] = target_center_list[i][2]

        #print(f'self.dstX: {self.dstX}')
        #exit(0)

        return 

    def set_adaptive_end_frame_state(self):
        self.object_names = list(self.mesh_dict.keys())
        self.object_index_by_name = {
            object_name: index for index, object_name in enumerate(self.object_names)
        }
        self.object_masses = [
            float(np.prod(self.mesh_trimesh_dict[object_name].bounding_box.extents))
            for object_name in self.object_names
        ]
        self.object_diagonal_lengths = [
            max(
                float(np.linalg.norm(self.mesh_trimesh_dict[object_name].bounding_box.extents)),
                1e-9,
            )
            for object_name in self.object_names
        ]
        self.total_object_mass = float(sum(self.object_masses))
        self.gravity_magnitude = 9.8
        self.object_heights = [
            float(self.mesh_trimesh_dict[object_name].bounding_box.extents[1])
            for object_name in self.object_names
        ]
        self.children_by_index = {index: [] for index in range(self.obj_num)}
        for child_index, parent_index in self.support_metric_target_ids.items():
            self.children_by_index.setdefault(parent_index, []).append(child_index)
        self.subtree_indices = {
            index: tuple(self._collect_descendants(index))
            for index in range(self.obj_num)
        }
        self.root_indices = tuple(
            index for index in range(self.obj_num) if index not in self.support_metric_target_ids
        )
        self.expected_center_gap_sum = float(
            sum(
                (self.object_heights[child_index] + self.object_heights[parent_index]) * 0.5
                for child_index, parent_index in self.support_metric_target_ids.items()
            )
        )
        self.summed_object_heights = float(sum(self.object_heights))
        adaptive_cap = int(
            getattr(self.args, 'adaptive_end_frame_cap', getattr(self.args, 'end_frame', 300))
        )
        self.adaptive_end_frame_enabled = bool(
            getattr(self.args, 'adaptive_end_frame_enabled', False)
        )
        self.adaptive_controller = str(
            getattr(self.args, 'adaptive_controller', 'sampled_criteria') or 'sampled_criteria'
        )
        self.adaptive_end_frame_cap = max(
            1,
            min(int(getattr(self.args, 'end_frame', adaptive_cap)), adaptive_cap),
        )
        self.args.end_frame = self.adaptive_end_frame_cap
        self.optim_samples_count = max(
            1,
            (self.args.end_frame - self.args.offset_frame) // self.args.optim_frame_interval,
        )
        self.sample_frame_numbers = self._build_sample_frame_numbers()
        self.adaptive_end_frame_min = min(
            self.adaptive_end_frame_cap,
            max(
                1,
                int(
                    getattr(
                        self.args,
                        'adaptive_end_frame_min',
                        self.adaptive_end_frame_cap,
                    )
                ),
            ),
        )
        self.current_effective_end_frame = self.adaptive_end_frame_cap
        self.selected_end_frame = self.adaptive_end_frame_cap
        self.adaptive_warmup_epochs = int(
            getattr(self.args, 'adaptive_warmup_epochs', 10)
        )
        self.adaptive_required_consecutive_epochs = int(
            getattr(self.args, 'adaptive_required_consecutive_epochs', 3)
        )
        self.adaptive_required_stable_epochs = int(
            getattr(self.args, 'adaptive_required_stable_epochs', 5)
        )
        self.adaptive_acceleration_residual_threshold = float(
            getattr(self.args, 'adaptive_acceleration_residual_threshold', 0.1)
        )
        self.adaptive_force_residual_threshold = float(
            getattr(self.args, 'adaptive_force_residual_threshold', 0.4)
        )
        self.adaptive_velocity_residual_threshold = float(
            getattr(self.args, 'adaptive_velocity_residual_threshold', 0.003)
        )
        self.adaptive_forward_validation_min_frame = max(
            1,
            int(getattr(self.args, 'adaptive_forward_validation_min_frame', 50)),
        )
        self.adaptive_forward_validation_static_window = max(
            1,
            int(getattr(self.args, 'adaptive_forward_validation_static_window', 100)),
        )
        self.adaptive_forward_validation_refresh_epochs = max(
            1,
            int(getattr(self.args, 'adaptive_forward_validation_refresh_epochs', 5)),
        )
        self.adaptive_forward_validation_end_frame = max(
            1,
            int(
                getattr(
                    self.args,
                    'adaptive_forward_validation_end_frame',
                    self.adaptive_end_frame_cap,
                )
            ),
        )
        self.adaptive_forward_validation_cushion_frames = max(
            0,
            int(
                getattr(
                    self.args,
                    'adaptive_forward_validation_cushion_frames',
                    50,
                )
            ),
        )
        self.adaptive_forward_validation_align_multiple = max(
            1,
            int(
                getattr(
                    self.args,
                    'adaptive_forward_validation_align_multiple',
                    10,
                )
            ),
        )
        self.adaptive_require_velocity = bool(
            getattr(self.args, 'adaptive_require_velocity', True)
        )
        self.adaptive_require_force = bool(
            getattr(self.args, 'adaptive_require_force', True)
        )
        self.adaptive_min_stop_frame = int(
            getattr(self.args, 'adaptive_min_stop_frame', 200)
        )
        self.adaptive_overlap_threshold = float(
            getattr(self.args, 'adaptive_overlap_threshold', 0.95)
        )
        self.adaptive_height_ratio_threshold = float(
            getattr(self.args, 'adaptive_height_ratio_threshold', 0.90)
        )
        self.adaptive_require_min_stop_frame = bool(
            getattr(self.args, 'adaptive_require_min_stop_frame', False)
        )
        self.adaptive_require_order = bool(
            getattr(self.args, 'adaptive_require_order', False)
        )
        self.adaptive_require_overlap = bool(
            getattr(self.args, 'adaptive_require_overlap', False)
        )
        self.adaptive_require_height_ratio = bool(
            getattr(self.args, 'adaptive_require_height_ratio', False)
        )
        if self.adaptive_controller == 'sampled_criteria':
            self.adaptive_required_criteria = ('acceleration',)
        else:
            self.adaptive_required_criteria = ()
        self.adaptive_epoch_history = []
        self._adaptive_should_stop = False

    def _collect_descendants(self, index: int) -> list[int]:
        descendants = [index]
        for child_index in self.children_by_index.get(index, []):
            descendants.extend(self._collect_descendants(child_index))
        return descendants

    def _build_sample_frame_numbers(self):
        frames = []
        for frame in range(self.args.offset_frame + 1, self.args.end_frame + 1):
            if frame > self.args.offset_frame and frame % self.args.optim_frame_interval == 0:
                frames.append(frame)
        if not frames:
            frames.append(int(self.args.end_frame))
        return frames

    def _active_sample_frame_numbers(self, end_frame: int | None = None):
        if end_frame is None:
            end_frame = int(self.current_effective_end_frame)
        frame_limit = max(int(self.args.offset_frame), int(end_frame))
        frames = [frame for frame in self.sample_frame_numbers if frame <= frame_limit]
        if not frames and frame_limit > int(self.args.offset_frame):
            frames.append(frame_limit)
        return frames

    def _tensor_scalar(self, value) -> float:
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu().item())
        return float(value)

    def _xz_overlap_ratio(self, lower_min, lower_max, upper_min, upper_max) -> float:
        overlap_x = max(0.0, min(lower_max[0], upper_max[0]) - max(lower_min[0], upper_min[0]))
        overlap_z = max(0.0, min(lower_max[2], upper_max[2]) - max(lower_min[2], upper_min[2]))
        overlap_area = overlap_x * overlap_z
        area_lower = max(0.0, lower_max[0] - lower_min[0]) * max(0.0, lower_max[2] - lower_min[2])
        area_upper = max(0.0, upper_max[0] - upper_min[0]) * max(0.0, upper_max[2] - upper_min[2])
        min_area = min(area_lower, area_upper)
        if min_area <= 0.0:
            return 0.0
        return float(overlap_area / min_area)

    def _sample_support_metrics(self, center_history, min_history, max_history, sample_index: int, sample_frame: int):
        center_values = center_history[sample_index]
        min_values = min_history[sample_index]
        max_values = max_history[sample_index]
        overlap_ratios = []
        direct_gaps = []
        expected_direct_gaps = []
        order_matches_expected = True
        for child_index, parent_index in sorted(self.support_metric_target_ids.items()):
            overlap_ratio = self._xz_overlap_ratio(
                min_values[parent_index],
                max_values[parent_index],
                min_values[child_index],
                max_values[child_index],
            )
            overlap_ratios.append(overlap_ratio)
            direct_gap = float(center_values[child_index][1] - center_values[parent_index][1])
            expected_gap = float(
                (self.object_heights[child_index] + self.object_heights[parent_index]) * 0.5
            )
            direct_gaps.append(max(direct_gap, 0.0))
            expected_direct_gaps.append(expected_gap)
            if direct_gap <= 0.0:
                order_matches_expected = False

        xz_points = np.asarray(
            [[center[0], center[2]] for center in center_values],
            dtype=np.float64,
        )
        xz_mean = xz_points.mean(axis=0) if len(xz_points) else np.zeros(2, dtype=np.float64)
        xz_radii = np.linalg.norm(xz_points - xz_mean, axis=1) if len(xz_points) else np.zeros(1, dtype=np.float64)
        y_values = np.asarray([center[1] for center in center_values], dtype=np.float64)
        y_range = float(y_values.max() - y_values.min()) if len(y_values) else 0.0
        summed_direct_y_gaps = float(sum(direct_gaps))
        expected_center_gap_sum = float(sum(expected_direct_gaps))
        y_range_ratio = (
            float(y_range / expected_center_gap_sum)
            if expected_center_gap_sum > 1e-9
            else 0.0
        )
        direct_gap_ratio = (
            float(summed_direct_y_gaps / expected_center_gap_sum)
            if expected_center_gap_sum > 1e-9
            else 0.0
        )
        retained_height_ratio = min(y_range_ratio, direct_gap_ratio)
        avg_overlap = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0
        max_radius = float(np.max(xz_radii)) if len(xz_radii) else 0.0
        return {
            'sample_frame': int(sample_frame),
            'order_matches_expected': bool(order_matches_expected),
            'avg_adjacent_xz_overlap_ratio': avg_overlap,
            'max_center_xz_radius': max_radius,
            'y_range': y_range,
            'summed_direct_y_gaps': summed_direct_y_gaps,
            'expected_center_gap_sum': expected_center_gap_sum,
            'summed_object_heights': float(self.summed_object_heights),
            'y_range_ratio': y_range_ratio,
            'direct_gap_ratio': direct_gap_ratio,
            'retained_height_ratio': retained_height_ratio,
        }

    def _transform_matrix_to_dof_vector(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        return np.asarray(
            [
                matrix[0, 3],
                matrix[1, 3],
                matrix[2, 3],
                matrix[0, 0],
                matrix[0, 1],
                matrix[0, 2],
                matrix[1, 0],
                matrix[1, 1],
                matrix[1, 2],
                matrix[2, 0],
                matrix[2, 1],
                matrix[2, 2],
            ],
            dtype=np.float64,
        )

    def _sample_acceleration_residual(self, transform_history, sample_index: int):
        if sample_index <= 0 or sample_index >= len(transform_history) - 1:
            return {
                'acceleration_residual_max': float('inf'),
                'acceleration_residual_mean': float('inf'),
                'acceleration_residual_valid': False,
            }

        acceleration_dt = max(
            float(self.args.time_step) * max(int(self.args.optim_frame_interval), 1),
            1e-9,
        )
        prev_transforms = transform_history[sample_index - 1]
        current_transforms = transform_history[sample_index]
        next_transforms = transform_history[sample_index + 1]
        object_acceleration_norms = []
        for object_index, object_name in enumerate(self.object_names):
            prev_vector = self._transform_matrix_to_dof_vector(prev_transforms[object_name])
            current_vector = self._transform_matrix_to_dof_vector(current_transforms[object_name])
            next_vector = self._transform_matrix_to_dof_vector(next_transforms[object_name])
            acceleration_vector = (
                next_vector
                - (2.0 * current_vector)
                + prev_vector
            ) / (acceleration_dt ** 2)
            acceleration_vector[3:] *= float(self.object_diagonal_lengths[object_index])
            object_acceleration_norms.append(float(np.max(np.abs(acceleration_vector))))
        return {
            'acceleration_residual_max': float(max(object_acceleration_norms, default=float('inf'))),
            'acceleration_residual_mean': (
                float(np.mean(object_acceleration_norms))
                if object_acceleration_norms
                else float('inf')
            ),
            'acceleration_residual_valid': bool(object_acceleration_norms),
        }

    def _select_epoch_metrics(self, sample_metrics):
        if not sample_metrics:
            return None
        if self.adaptive_controller == 'sampled_criteria':
            valid_metrics = [
                (sample_index, metrics)
                for sample_index, metrics in enumerate(sample_metrics)
                if bool(metrics.get('acceleration_residual_valid'))
            ]
            last_large_index = None
            last_large_frame = None
            for sample_index, metrics in valid_metrics:
                if not bool(metrics.get('qualifies_adaptive_stop')):
                    last_large_index = int(sample_index)
                    last_large_frame = int(metrics.get('sample_frame'))

            if last_large_index is not None:
                for sample_index, metrics in valid_metrics:
                    if sample_index <= last_large_index:
                        continue
                    if metrics.get('qualifies_adaptive_stop'):
                        selected = dict(metrics)
                        selected['qualifying_frame'] = int(selected['sample_frame'])
                        selected['selected_end_frame'] = int(
                            min(
                                self.adaptive_end_frame_cap,
                                int(selected['sample_frame']) + int(self.args.optim_frame_interval),
                            )
                        )
                        selected['qualifies_adaptive_stop'] = True
                        selected['last_large_acceleration_frame'] = last_large_frame
                        return selected

            return {
                'qualifying_frame': None,
                'selected_end_frame': int(self.current_effective_end_frame),
                'qualifies_adaptive_stop': False,
                'adaptive_required_criteria': list(self.adaptive_required_criteria),
                'last_large_acceleration_frame': last_large_frame,
            }

        qualifying = [
            metrics for metrics in sample_metrics
            if metrics.get('qualifies_adaptive_stop')
        ]
        if qualifying:
            selected = dict(min(qualifying, key=lambda item: int(item['sample_frame'])))
            selected['qualifying_frame'] = int(selected['sample_frame'])
            selected['selected_end_frame'] = int(
                min(
                    self.adaptive_end_frame_cap,
                    int(selected['sample_frame']) + int(self.args.optim_frame_interval),
                )
            )
            selected['qualifies_adaptive_stop'] = True
            return selected
        return {
            'qualifying_frame': None,
            'selected_end_frame': int(self.current_effective_end_frame),
            'qualifies_adaptive_stop': False,
            'adaptive_required_criteria': list(self.adaptive_required_criteria),
        }

    def _use_forward_static_window_controller(self):
        return (
            bool(self.adaptive_end_frame_enabled)
            and str(self.adaptive_controller) == 'forward_static_window'
        )

    def _load_validation_meshes(self, transformations):
        pre_transform = Transform.Identity()
        pre_transform.scale(1)
        io = SimplicialComplexIO(pre_transform)
        validation_mesh_root = os.path.join(self.result_folder, '_forward_validation_meshes')
        if not os.path.exists(validation_mesh_root):
            os.makedirs(validation_mesh_root)
        mesh_dict = {}
        for object_name in self.object_names:
            mesh_path = os.path.join(validation_mesh_root, f'{object_name}.obj')
            if not os.path.exists(mesh_path):
                self.mesh_trimesh_dict[object_name].export(mesh_path)
            mesh = io.read(mesh_path)
            matrix = transformations.get(object_name)
            if matrix is not None:
                view(mesh.transforms())[:] = np.asarray(matrix, dtype=np.float64)
            mesh_dict[object_name] = mesh
        return mesh_dict

    def _make_forward_validation_args(self):
        validation_args = copy.deepcopy(self.args)
        validation_args.exp_name = f'adaptive_validation_epoch_{int(self.completed_opt_epoch):03d}'
        validation_args.scene_name = str(getattr(self.args, 'scene_name', self.args.exp_name))
        validation_args.phys_result_folder = self.result_folder
        validation_args.end_frame = int(self.adaptive_forward_validation_end_frame)
        validation_args.end_frame_upper_bound = int(self.adaptive_forward_validation_end_frame)
        validation_args.stop_when_static = True
        validation_args.static_consecutive_frames = int(
            self.adaptive_forward_validation_static_window
        )
        validation_args.min_static_start_frame = int(
            self.adaptive_forward_validation_min_frame
        )
        validation_args.gui_flag = False
        return validation_args

    def _make_sampled_observation_args(self):
        observation_args = copy.deepcopy(self.args)
        observation_args.exp_name = f'adaptive_observation_epoch_{int(self.completed_opt_epoch):03d}'
        observation_args.scene_name = str(getattr(self.args, 'scene_name', self.args.exp_name))
        observation_args.phys_result_folder = self.result_folder
        observation_args.end_frame = int(self.adaptive_end_frame_cap)
        observation_args.end_frame_upper_bound = int(self.adaptive_end_frame_cap)
        observation_args.stop_when_static = False
        observation_args.gui_flag = False
        return observation_args

    def _bbox_histories_from_transform_history(self, transform_history):
        center_history = []
        min_history = []
        max_history = []
        for frame_transform in transform_history:
            frame_centers = []
            frame_mins = []
            frame_maxs = []
            for object_name in self.object_names:
                matrix = np.asarray(frame_transform[object_name], dtype=np.float64)
                bbox_vertices = np.asarray(
                    self.mesh_trimesh_dict[object_name].bounding_box.vertices,
                    dtype=np.float64,
                )
                hom_vertices = np.concatenate(
                    [bbox_vertices, np.ones((bbox_vertices.shape[0], 1), dtype=np.float64)],
                    axis=1,
                )
                transformed_vertices = hom_vertices @ matrix.T
                transformed_vertices = transformed_vertices[:, :3]
                min_values = transformed_vertices.min(axis=0)
                max_values = transformed_vertices.max(axis=0)
                frame_mins.append(min_values)
                frame_maxs.append(max_values)
                frame_centers.append((min_values + max_values) * 0.5)
            center_history.append(frame_centers)
            min_history.append(frame_mins)
            max_history.append(frame_maxs)
        return center_history, min_history, max_history

    def _run_sampled_observation_rollout(self):
        if not (
            self.adaptive_end_frame_enabled
            and str(self.adaptive_controller) == 'sampled_criteria'
        ):
            return None

        checkpoint = {
            object_name: np.asarray(matrix, dtype=np.float64).copy()
            for object_name, matrix in self._current_param_transformations().items()
        }
        observation_args = self._make_sampled_observation_args()
        observation_mesh_dict = self._load_validation_meshes(checkpoint)
        simulator = PhysSimulator(
            observation_args,
            observation_mesh_dict,
            int(self.adaptive_end_frame_cap),
            False,
        )
        transform_history = list(getattr(simulator, 'sampled_transform_history', []) or [])
        sample_frames = list(getattr(simulator, 'sampled_frame_numbers_recorded', []) or [])
        if not transform_history or not sample_frames:
            return {
                'qualifying_frame': None,
                'selected_end_frame': int(self.current_effective_end_frame),
                'qualifies_adaptive_stop': False,
                'adaptive_required_criteria': list(self.adaptive_required_criteria),
                'observation_end_frame': int(self.adaptive_end_frame_cap),
                'observation_final_frame': int(getattr(simulator, 'final_frame', 0)),
            }

        center_history, min_history, max_history = self._bbox_histories_from_transform_history(
            transform_history
        )
        sample_metrics = []
        for sample_index, sample_frame in enumerate(sample_frames[:len(center_history)]):
            metrics = self._sample_support_metrics(
                center_history=center_history,
                min_history=min_history,
                max_history=max_history,
                sample_index=sample_index,
                sample_frame=sample_frame,
            )
            metrics.update(self._sample_acceleration_residual(transform_history, sample_index))
            metrics['adaptive_required_criteria'] = list(self.adaptive_required_criteria)
            metrics['qualifies_adaptive_stop'] = self._sample_qualifies_adaptive_stop(
                int(sample_frame),
                metrics,
            )
            sample_metrics.append(metrics)

        selected = self._select_epoch_metrics(sample_metrics)
        if selected is None:
            selected = {
                'qualifying_frame': None,
                'selected_end_frame': int(self.current_effective_end_frame),
                'qualifies_adaptive_stop': False,
                'adaptive_required_criteria': list(self.adaptive_required_criteria),
            }
        selected['observation_end_frame'] = int(self.adaptive_end_frame_cap)
        selected['observation_final_frame'] = int(getattr(simulator, 'final_frame', 0))
        return selected

    def _reset_forward_static_window_stage_tracking(self):
        self.stage_best_loss = None
        self.best_selection_key = None
        self.best_checkpoint_step = None
        self.best_epoch_metrics = None

    def _attach_forward_validation_to_latest_point(self, record):
        if not self.loss_history:
            return
        point = self.loss_history[-1]
        point.update(
            {
                'last_forward_validation_epoch': int(record['validation_epoch']),
                'last_forward_validation_found': bool(record['static_window_found']),
                'last_forward_validation_start_frame': record['static_window_start_frame'],
                'last_forward_validation_end_frame': record['static_window_end_frame'],
                'last_forward_validation_target_frame': record['static_window_target_frame'],
                'last_forward_validation_final_frame': int(record['validation_final_frame']),
                'effective_end_frame_after_validation': int(record['effective_end_frame_after']),
            }
        )

    def _forward_validation_horizon_step(self):
        sample_interval = max(int(getattr(self.args, 'optim_frame_interval', 1)), 1)
        return max(
            1,
            math.lcm(sample_interval, int(self.adaptive_forward_validation_align_multiple)),
        )

    def _forward_validation_target_frame(self, static_window_start):
        sample_interval = max(int(getattr(self.args, 'optim_frame_interval', 1)), 1)
        target_frame = int(static_window_start)
        target_frame += int(self.adaptive_forward_validation_cushion_frames)
        # Keep one extra sampled step beyond the static cushion so the shortened
        # horizon still retains a sampled static tail in the diff-sim loss.
        target_frame += sample_interval
        horizon_step = self._forward_validation_horizon_step()
        target_frame = ((target_frame + horizon_step - 1) // horizon_step) * horizon_step
        return max(1, min(int(self.adaptive_end_frame_cap), int(target_frame)))

    def _run_forward_validation_refresh(self, force=False):
        if not self._use_forward_static_window_controller():
            return
        if not isinstance(self.transformation_parameter, dict) or not self.transformation_parameter:
            return
        if self.completed_opt_epoch <= 0:
            return
        if force:
            if self.last_forward_validation_epoch == int(self.completed_opt_epoch):
                return
        elif (
            int(self.completed_opt_epoch) % int(self.adaptive_forward_validation_refresh_epochs) != 0
            or self.last_forward_validation_epoch == int(self.completed_opt_epoch)
        ):
            return

        checkpoint = {
            object_name: np.asarray(matrix, dtype=np.float64).copy()
            for object_name, matrix in self.transformation_parameter.items()
        }
        validation_args = self._make_forward_validation_args()
        validation_mesh_dict = self._load_validation_meshes(checkpoint)
        effective_before = int(self.current_effective_end_frame)
        simulator = PhysSimulator(
            validation_args,
            validation_mesh_dict,
            int(self.adaptive_forward_validation_end_frame),
            False,
        )
        static_window_start = getattr(simulator, 'first_static_window_start_frame', None)
        static_window_end = getattr(simulator, 'first_static_window_end_frame', None)
        static_window_found = static_window_start is not None
        static_window_target = None
        if static_window_found:
            new_end_frame = self._forward_validation_target_frame(static_window_start)
            static_window_target = int(new_end_frame)
            self.current_effective_end_frame = max(1, new_end_frame)
            self.selected_end_frame = int(self.current_effective_end_frame)
            self.best_selected_end_frame = int(self.current_effective_end_frame)
            self.best_qualifying_frame = int(self.current_effective_end_frame)
        effective_after = int(self.current_effective_end_frame)
        record = {
            'validation_epoch': int(self.completed_opt_epoch),
            'checkpoint_loss': (
                None if self.best_loss == -1 else float(self.best_loss)
            ),
            'checkpoint_step': (
                None if self.best_checkpoint_step is None else int(self.best_checkpoint_step)
            ),
            'validation_end_frame': int(self.adaptive_forward_validation_end_frame),
            'validation_final_frame': int(getattr(simulator, 'final_frame', 0)),
            'stopped_because_static': bool(getattr(simulator, 'stopped_because_static', False)),
            'static_window_found': bool(static_window_found),
            'static_window_start_frame': (
                None if static_window_start is None else int(static_window_start)
            ),
            'static_window_end_frame': (
                None if static_window_end is None else int(static_window_end)
            ),
            'static_window_target_frame': (
                None if static_window_target is None else int(static_window_target)
            ),
            'effective_end_frame_before': int(effective_before),
            'effective_end_frame_after': int(effective_after),
        }
        if static_window_found and effective_after != effective_before:
            self._reset_forward_static_window_stage_tracking()
        self.latest_forward_validation = record
        self.forward_validation_history.append(record)
        self.last_forward_validation_epoch = int(self.completed_opt_epoch)
        self._attach_forward_validation_to_latest_point(record)
        self.write_loss_history()

    def _current_selection_key(self):
        if self.current_epoch_metrics is None:
            return None
        metrics = self.current_epoch_metrics
        return (
            0 if metrics.get('qualifies_adaptive_stop') else 1,
            int(metrics.get('selected_end_frame', self.current_effective_end_frame)),
            *self._adaptive_metric_priority(metrics),
            float(self.loss.item()) if isinstance(self.loss, torch.Tensor) else float(self.loss),
        )

    def _sample_satisfies_required_validity(self, metrics):
        if not bool(metrics.get('acceleration_residual_valid')):
            return False
        return True

    def _adaptive_metric_priority(self, metrics):
        priority = [
            float(metrics.get('acceleration_residual_max', float('inf'))),
            float(metrics.get('acceleration_residual_mean', float('inf'))),
        ]
        if self.adaptive_require_overlap:
            priority.append(-float(metrics.get('avg_adjacent_xz_overlap_ratio', 0.0)))
        if self.adaptive_require_height_ratio:
            priority.append(-float(metrics.get('retained_height_ratio', 0.0)))
        if self.adaptive_require_order:
            priority.append(0.0 if metrics.get('order_matches_expected') else 1.0)
        if priority:
            return tuple(priority)
        return (0.0,)

    def _sample_qualifies_adaptive_stop(self, sample_frame: int, metrics):
        del sample_frame
        return bool(metrics.get('acceleration_residual_valid')) and (
            float(metrics.get('acceleration_residual_max', float('inf')))
            <= float(self.adaptive_acceleration_residual_threshold)
        )

    def _update_adaptive_controller(self):
        if self.current_epoch_metrics is None:
            return

        epoch_index = int(self.global_step)
        if self._use_forward_static_window_controller():
            self.selected_end_frame = int(self.current_effective_end_frame)
            history_item = {
                'epoch': epoch_index,
                'effective_end_frame': int(self.current_effective_end_frame),
                **self.current_epoch_metrics,
            }
            self.adaptive_epoch_history.append(history_item)
            return

        proposed_end_frame = int(self.current_epoch_metrics.get('selected_end_frame', self.current_effective_end_frame))
        qualifies = bool(self.current_epoch_metrics.get('qualifies_adaptive_stop'))
        if (
            self.adaptive_end_frame_enabled
            and (epoch_index + 1) > self.adaptive_warmup_epochs
            and qualifies
        ):
            self.current_effective_end_frame = proposed_end_frame

        self.selected_end_frame = int(self.current_effective_end_frame)
        history_item = {
            'epoch': epoch_index,
            'effective_end_frame': int(self.current_effective_end_frame),
            **self.current_epoch_metrics,
        }
        self.adaptive_epoch_history.append(history_item)

    def _loss_history_point(self):
        point = {
            'step': int(self.global_step),
            'epoch': int(self.global_step),
            'loss': float(self.loss.item()) if isinstance(self.loss, torch.Tensor) else float(self.loss),
            'adaptive_controller': str(self.adaptive_controller),
            'effective_end_frame': int(self.current_effective_end_frame),
            'selected_end_frame': int(self.selected_end_frame),
            'adaptive_enabled': bool(self.adaptive_end_frame_enabled),
            'adaptive_required_criteria': list(self.adaptive_required_criteria),
            'adaptive_forward_validation_min_frame': int(self.adaptive_forward_validation_min_frame),
            'adaptive_forward_validation_static_window': int(self.adaptive_forward_validation_static_window),
            'adaptive_forward_validation_refresh_epochs': int(self.adaptive_forward_validation_refresh_epochs),
            'adaptive_forward_validation_end_frame': int(self.adaptive_forward_validation_end_frame),
            'adaptive_forward_validation_cushion_frames': int(self.adaptive_forward_validation_cushion_frames),
            'adaptive_forward_validation_align_multiple': int(self.adaptive_forward_validation_align_multiple),
        }
        if self.current_epoch_metrics is not None:
            point.update(
                {
                    'qualifying_frame': self.current_epoch_metrics.get('qualifying_frame'),
                    'last_large_acceleration_frame': self.current_epoch_metrics.get('last_large_acceleration_frame'),
                    'acceleration_residual_max': float(self.current_epoch_metrics.get('acceleration_residual_max', float('inf'))),
                    'acceleration_residual_mean': float(self.current_epoch_metrics.get('acceleration_residual_mean', float('inf'))),
                    'acceleration_residual_valid': bool(self.current_epoch_metrics.get('acceleration_residual_valid')),
                    'force_residual_total': float(self.current_epoch_metrics.get('force_residual_total', float('inf'))),
                    'force_residual_max': float(self.current_epoch_metrics.get('force_residual_max', float('inf'))),
                    'velocity_residual_max': float(self.current_epoch_metrics.get('velocity_residual_max', float('inf'))),
                    'velocity_residual_mean': float(self.current_epoch_metrics.get('velocity_residual_mean', float('inf'))),
                    'support_force_residual_total': float(self.current_epoch_metrics.get('support_force_residual_total', float('inf'))),
                    'avg_adjacent_xz_overlap_ratio': float(self.current_epoch_metrics.get('avg_adjacent_xz_overlap_ratio', 0.0)),
                    'max_center_xz_radius': float(self.current_epoch_metrics.get('max_center_xz_radius', 0.0)),
                    'y_range': float(self.current_epoch_metrics.get('y_range', 0.0)),
                    'retained_height_ratio': float(self.current_epoch_metrics.get('retained_height_ratio', 0.0)),
                    'qualifies_adaptive_stop': bool(self.current_epoch_metrics.get('qualifies_adaptive_stop')),
                }
            )
        if isinstance(self.latest_forward_validation, dict):
            point.update(
                {
                    'last_forward_validation_epoch': int(self.latest_forward_validation['validation_epoch']),
                    'last_forward_validation_found': bool(self.latest_forward_validation['static_window_found']),
                    'last_forward_validation_start_frame': self.latest_forward_validation['static_window_start_frame'],
                    'last_forward_validation_end_frame': self.latest_forward_validation['static_window_end_frame'],
                    'last_forward_validation_final_frame': int(self.latest_forward_validation['validation_final_frame']),
                }
            )
        return point

    def recover(self):
        self.world.recover(0)
        self.param.sync()

    def bind_translation_param(self):
        
        self.param_map_index = 0
        self.current_mesh_status_dict = {}
        self.initial_mesh_status_dict = {}
        self.param_map_back = {}

        for obj_name in self.mesh_dict.keys():
            #if obj_name == 'sofa1':
                #continue
            ## define the initial transformation optimization matrix mask for each object 
            diff_tran_index0 = -1 * Matrix4x4i.Ones() ## this is a mask, value>=0 means enable the optimization, -1 measn disable the optimization
            diff_tran0 = self.mesh_dict[obj_name].instances().create('diff/transform', diff_tran_index0)

            ## enable the parameter that should be optimized
            diff_tran_index0[0, 3] = self.param_map_index ## map to parms
            self.param_map_back[self.param_map_index] = [obj_name, 0, 3]
            self.param_map_index += 1

            #diff_tran_index0[1, 3] = self.param_map_index
            #self.param_map_back[self.param_map_index] = [obj_name, 1, 3]
            #self.param_map_index += 1
            
            diff_tran_index0[2, 3] = self.param_map_index
            self.param_map_back[self.param_map_index] = [obj_name, 2, 3]
            self.param_map_index += 1

            ## bind the new mask to the mesh
            view(diff_tran0)[0] = diff_tran_index0


    def save_transform_parameters(self):
        
        if self.world.frame() == self.current_effective_end_frame:
            transform_dict = {}
            for obj_name in self.current_mesh_status_dict.keys():
                transform_param_piece = view(self.current_mesh_status_dict[obj_name].geometry().transforms())[0]
                current_frame_num = self.world.frame()
                transform_dict[obj_name] = transform_param_piece
            cur_optim_save_folder = os.path.join(self.transform_folder, str(self.optim_attempt))
            if not os.path.exists(cur_optim_save_folder):
                os.makedirs(cur_optim_save_folder)
            cur_optim_save_path = os.path.join(self.transform_folder, str(self.optim_attempt), f'frame_{self.world.frame():05d}.npz')
            np.savez(cur_optim_save_path, **transform_dict)
                


    def set_physical_properties(self):

        for obj_name in self.mesh_dict.keys():

            ## fixed, apply for each object (0 --> no inertance)
            is_dynamic = self.mesh_dict[obj_name].instances().find(builtin.is_dynamic)
            view(is_dynamic)[0] = 0

            ## create a new instance in the self.scene based on the mesh
            ## current_mesh_status is the current property of the mesh [s_n] --> would be updated in the simulation
            ## initial_mesh_status is the initial property of the mesh [s_0]
            current_mesh_status, initial_mesh_status = self.scene_objects.geometries().create(self.mesh_dict[obj_name])
            self.current_mesh_status_dict[obj_name] = current_mesh_status
            self.initial_mesh_status_dict[obj_name] = initial_mesh_status



    def load_meshes(self, mesh_dict:dict, tmesh_dict:dict):

        self.mesh_dict = mesh_dict
        self.mesh_trimesh_dict = tmesh_dict

        self.BBXs = []
        ## load the meshes in the self.scene
        print(f"Mesh dict: {mesh_dict.keys()}")
        for obj_name in mesh_dict.keys():
            print(f"Optimizer reading in object name: {obj_name}")
            
            ## read the self.mesh_dict['cube1']
            
            bbox_vertices = self.mesh_trimesh_dict[obj_name].bounding_box.vertices.tolist()
            self.BBXs.append(bbox_vertices)
            ## apply the contact property to the mesh 
            self.contact_property_dict[obj_name].apply_to(self.mesh_dict[obj_name]) 
            
            ## label the self.mesh_dict['cube1']
            label_surface(self.mesh_dict[obj_name])

        ## get the total number of the objects in the scene
        self.obj_num = len(self.mesh_dict.keys())
        return self.mesh_dict

    def _build_loss_target_ids(self):
        object_names = list(self.mesh_dict.keys())
        name_to_index = {name: index for index, name in enumerate(object_names)}
        if isinstance(self.container_name, dict):
            target_ids = {}
            for object_name, container_name in self.container_name.items():
                object_index = name_to_index.get(object_name)
                target_index = name_to_index.get(container_name)
                if object_index is None or target_index is None:
                    continue
                if object_index == target_index:
                    continue
                target_ids[object_index] = target_index
            return target_ids

        target_index = name_to_index.get(self.container_name)
        if target_index is None:
            raise KeyError(f'container_name {self.container_name} not found in mesh_dict')
        return {
            index: target_index
            for index in range(len(object_names))
            if index != target_index
        }

    def _build_support_metric_target_ids(self):
        object_names = list(self.mesh_dict.keys())
        name_to_index = {name: index for index, name in enumerate(object_names)}
        support_order = str(getattr(self.args, 'adaptive_support_order', '') or '').strip()
        if support_order:
            ordered_names = [
                object_name.strip()
                for object_name in support_order.split(',')
                if object_name.strip()
            ]
            target_ids = {}
            for parent_name, child_name in zip(ordered_names, ordered_names[1:]):
                parent_index = name_to_index.get(parent_name)
                child_index = name_to_index.get(child_name)
                if parent_index is None or child_index is None or parent_index == child_index:
                    continue
                target_ids[child_index] = parent_index
            if target_ids:
                return target_ids
        return dict(self.loss_target_ids)

    def set_ground(self):  

        self.ground_obj = self.scene.objects().create('ground')
        g = ground(self.args.ground_y_value) ## y axis of the ground
        self.ground_obj.geometries().create(g) ## create the ground instance 

    def optimize_once(self):
        
        for i in range(self.args.total_opt_epoch):
            self.iterate()
            self.optimizer.step()
            #smoothed_loss = ema_loss.update(self.loss)
            self.scheduler.step()
            self.completed_opt_epoch = i + 1
            self._run_forward_validation_refresh(force=False)
            if self._adaptive_should_stop:
                break
        self._run_forward_validation_refresh(force=True)
        self.optim_attempt += 1
        Timer.report()


    def _current_param_transformations(self):
        save_param_dict = {}
        for obj_name in self.mesh_dict.keys():
            save_param_dict[obj_name] = np.zeros((4, 4), dtype=np.float64)
            save_param_dict[obj_name][:, :] = np.eye(4, dtype=np.float64)
        for index in self.param_map_back.keys():
            fetched_param = self.param.U()[index]
            obj_name, row, col = self.param_map_back[index]
            save_param_dict[obj_name][row, col] = fetched_param
        return save_param_dict

    def save_optimized_param(self):
        #print('param u:', self.param.U())
        save_param_path = os.path.join(self.param_folder, f'optim_{self.global_step}.npz')
        save_param_dict = self._current_param_transformations()
        print(f'save_param_dict: {save_param_dict}')
        np.savez(save_param_path, **save_param_dict)

        current_loss = float(self.loss.item())
        if self._use_forward_static_window_controller():
            if self.stage_best_loss is None or current_loss < float(self.stage_best_loss):
                self.stage_best_loss = current_loss
                self.best_loss = current_loss
                self.best_checkpoint_step = int(self.global_step)
                self.best_selected_end_frame = int(self.current_effective_end_frame)
                self.best_qualifying_frame = int(self.current_effective_end_frame)
                self.best_epoch_metrics = (
                    None if self.current_epoch_metrics is None else dict(self.current_epoch_metrics)
                )
                print(
                    f"Saving parameter of model on forward-validated stage loss "
                    f"end_frame={self.best_selected_end_frame} loss={self.best_loss}."
                )
                self.transformation_parameter = save_param_dict
        elif self.best_loss == -1 or self.best_loss > current_loss:
            self.best_loss = current_loss
            self.best_checkpoint_step = int(self.global_step)
            self.best_selected_end_frame = int(self.current_effective_end_frame)
            self.best_qualifying_frame = (
                None
                if self.current_epoch_metrics is None
                else self.current_epoch_metrics.get('qualifying_frame')
            )
            self.best_epoch_metrics = (
                None if self.current_epoch_metrics is None else dict(self.current_epoch_metrics)
            )
            print(f"Saving parameter of model on best loss {self.best_loss}.")
            self.transformation_parameter = save_param_dict
        self.write_loss_history()

    def write_loss_history(self):
        payload = {
            'available': True,
            'mode': 'optimize_then_forward',
            'scene_id': self.args.exp_name,
            'pass_index': int(getattr(self.args, 'diff_sim_pass_index', 0)),
            'label': str(getattr(self.args, 'diff_sim_pass_label', f'Pass {int(getattr(self.args, "diff_sim_pass_index", 0)) + 1}')),
            'best_loss': float(self.best_loss) if self.best_loss != -1 else None,
            'adaptive_end_frame_enabled': bool(self.adaptive_end_frame_enabled),
            'adaptive_controller': str(self.adaptive_controller),
            'adaptive_end_frame_cap': int(self.adaptive_end_frame_cap),
            'adaptive_required_criteria': list(self.adaptive_required_criteria),
            'adaptive_forward_validation_min_frame': int(self.adaptive_forward_validation_min_frame),
            'adaptive_forward_validation_static_window': int(self.adaptive_forward_validation_static_window),
            'adaptive_forward_validation_refresh_epochs': int(self.adaptive_forward_validation_refresh_epochs),
            'adaptive_forward_validation_end_frame': int(self.adaptive_forward_validation_end_frame),
            'adaptive_forward_validation_cushion_frames': int(self.adaptive_forward_validation_cushion_frames),
            'adaptive_forward_validation_align_multiple': int(self.adaptive_forward_validation_align_multiple),
            'best_selected_end_frame': (
                int(self.best_selected_end_frame)
                if self.best_selected_end_frame is not None
                else None
            ),
            'best_qualifying_frame': (
                int(self.best_qualifying_frame)
                if self.best_qualifying_frame is not None
                else None
            ),
            'completed_opt_epoch': int(self.completed_opt_epoch),
            'adaptive_early_stop_triggered': bool(self.adaptive_early_stop_triggered),
            'adaptive_stop_reason': self.adaptive_stop_reason,
            'latest_forward_validation': self.latest_forward_validation,
            'forward_validation_history': self.forward_validation_history,
            'points': self.loss_history,
        }
        with open(self.loss_history_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write('\n')


    def iterate(self)->torch.Tensor:

        self.optimizer.zero_grad(set_to_none=True)

        ## get the parameters for optimization
        X = self.diff_module()
        self.loss = 0.0
        losses = []
        active_sample_frames = self._active_sample_frame_numbers()
        ## for the frames to be sampled 
        for i, sample_frame in enumerate(active_sample_frames):
            
            weight = 1.0
            #U = self.param.U()
            
            min_Boxs = []
            max_Boxs = []
            
            for t in range(self.obj_num):
                verts = []
                for j in range(8):
                    #print(self.BJC[j*3:j*3 + 3])
                    new_vertices = self.BJs[t][j*3:j*3 + 3].matmul(X[i*self.per_obj_dof_num * self.obj_num + (t)*self.per_obj_dof_num:i*self.per_obj_dof_num * self.obj_num + (t)*self.per_obj_dof_num + self.per_obj_dof_num])
                    verts.append(new_vertices)
                
                verts_array = torch.stack(verts)
                #verts_array_np = verts_array.numpy()
                minv = torch.min(verts_array, axis=0)
                maxv = torch.max(verts_array, axis=0)
                min_Boxs.append(minv)
                max_Boxs.append(maxv)
            #print(min_xyz)
            #print(max_xyz)
            ## compute the center loss for each object
            #for j in range(self.obj_num):
                #current_center = self.JC[j*3:j*3 + 3].matmul(X[i*self.per_obj_dof_num * self.obj_num + j*self.per_obj_dof_num:i*self.per_obj_dof_num * self.obj_num + j*self.per_obj_dof_num + self.per_obj_dof_num])
                #target_center = self.dstX[j*3:j*3 + 3]
                #y_loss = self.args.weight_y * (current_center[1] - target_center[1]).square().sum()
                #x_loss = self.args.weight_x * (current_center[0] - target_center[0]).square().sum()
                #z_loss = self.args.weight_z * (current_center[2] - target_center[2]).square().sum()
                #self.loss += (y_loss + x_loss + z_loss)
            for jj in range(self.obj_num):
                targetId = self.loss_target_ids.get(jj)
                if targetId is None:
                    continue
                min_xyz = min_Boxs[targetId]
                max_xyz = max_Boxs[targetId]
                container_bbox_scale_xz = float(getattr(self.args, 'container_bbox_scale_xz', 1.0))
                min_x = min_xyz.values[0]
                max_x = max_xyz.values[0]
                min_z = min_xyz.values[2]
                max_z = max_xyz.values[2]
                if container_bbox_scale_xz != 1.0:
                    center_x = 0.5 * (min_x + max_x)
                    center_z = 0.5 * (min_z + max_z)
                    half_x = 0.5 * (max_x - min_x) * container_bbox_scale_xz
                    half_z = 0.5 * (max_z - min_z) * container_bbox_scale_xz
                    min_x = center_x - half_x
                    max_x = center_x + half_x
                    min_z = center_z - half_z
                    max_z = center_z + half_z

                min = min_Boxs[jj]
                max = max_Boxs[jj]
                if min.values[0] < min_x:
                    xmiloss = (min.values[0] - min_x)**2
                    losses.append(xmiloss)
                if min.values[0] > max_x:
                    xmaloss = (min.values[0] - max_x)**2
                    losses.append(xmaloss)
                if min.values[2] < min_z:
                    zmiloss = (min.values[2] - min_z)**2
                    losses.append(zmiloss)
                if min.values[2] > max_z:
                    zmaloss = (min.values[2] - max_z)**2
                    losses.append(zmaloss)

                if max.values[0] < min_x:
                    xmiloss = (max.values[0] - min_x)**2
                    losses.append(xmiloss)
                if max.values[0] > max_x:
                    xmaloss = (max.values[0] - max_x)**2
                    losses.append(xmaloss)
                if max.values[2] < min_z:
                    zmiloss = (max.values[2] - min_z)**2
                    losses.append(zmiloss)
                if max.values[2] > max_z:
                    zmaloss = (max.values[2] - max_z)**2
                    losses.append(zmaloss)
        self.current_epoch_metrics = self._run_sampled_observation_rollout()
        if self.current_epoch_metrics is None:
            self.current_epoch_metrics = {
                'qualifying_frame': None,
                'selected_end_frame': int(self.current_effective_end_frame),
                'qualifies_adaptive_stop': False,
                'adaptive_required_criteria': list(self.adaptive_required_criteria),
            }
        self._update_adaptive_controller()
        if not losses:
            self.loss = torch.tensor(0.0, dtype = torch.float64)
            self.loss_history.append(self._loss_history_point())
            self.save_optimized_param()
            self.update_writer()
            self.global_step += 1 
            return
        self.loss = torch.stack(losses).sum()
        print(f"Loss {self.loss}")
        self.loss_history.append(self._loss_history_point())
        

        self.save_optimized_param()
        self.update_writer()
        self.loss.backward()
        grad_clip_norm = float(getattr(self.args, 'grad_clip_norm', 0.0))
        if grad_clip_norm > 0.0:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.diff_module.parameters(),
                grad_clip_norm,
                error_if_nonfinite=True,
            )
            self.writer.add_scalar('grad/total_norm_before_clip', float(total_norm), self.global_step)
        self.global_step += 1 
        return self.loss


    def update_writer(self):

        self.writer.add_scalar('loss/train', self.loss.item(), self.global_step)
        self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        self.writer.add_histogram('parameters', self.param.U(), self.global_step)




class PhysSimulator(object):
    def __init__(self, args, mesh_dict:dict, end_frame:int=1000, store_all_transformation:bool=False, transformation_store_root_dir:str|None=None):

        self.args = args
        self.end_frame = end_frame
        self.store_all_transformation = store_all_transformation
        self.transformation_store_root_dir = transformation_store_root_dir

        self.init_path()
        self.set_global_parameters()

        Timer.enable_all()
        Logger.set_level(Logger.Level.Warn)


   ## =====================================================================================================================

        ## parameters 
        ## fixed physical parameters

        confige = _make_uipc_engine_config(self.args)
        engine = Engine('cuda', self.workspace, confige)
        self.world = World(engine)
        config = Scene.default_config()



        # config['contact']['eps_velocity'] = 1e-5
        config['contact']['eps_velocity'] = float(getattr(self.args, 'contact_eps_velocity', 1e-5))
        config['newton']['velocity_tol'] = 0.1
        # config['contact']['eps_velocity'] = 1e-4
        config['dt'] = self.args.time_step
        config['gravity'] = [[0.0], [-9.8], [0.0]]
        config['contact']['enable']             = True 
        # config['contact']['d_hat']              = 0.0005 ## threshold for conlision detect 
        # config['contact']['d_hat']              = 3e-4 ## threshold for conlision detect 
        config['contact']['d_hat']              = float(getattr(self.args, 'contact_d_hat', 5e-4)) ## threshold for collision detect
        config['contact']['friction']['enable'] = True ## whether to use friction ## set it to be false for now
        config['line_search']['max_iter']       = 8 ## fixed, max iteration for line search
        config['linear_system']['tol_rate']     = self.args.tol_rate ## 1e-4 is more stable, except it has some related errors

        ## =====================================================================================================================


        ## initialize the self.scene 
        self.scene = Scene(config)

        ## self.scene contact 
        ## friction parameter: 0.2 
        ## stiffness: 1.0 * GPa   
        self.scene.contact_tabular().default_model(0.2, 1.0 * GPa)
        default_element = self.scene.contact_tabular().default_element()

        ## create contact property for each object
        self.contact_property_dict = {}
        ## set a self for each object and assign it to each object 
        for obj_name in mesh_dict.keys():
            contact_property_item = self.scene.contact_tabular().create(obj_name)
            self.contact_property_dict[obj_name] = contact_property_item
            self.scene.contact_tabular().insert(contact_property_item, contact_property_item, 0, 0, False)

        ## define the material of rigid body 
        self.abd = AffineBodyConstitution()

        ## load the meshes, don't care about this  
        self.bunny = self.scene.objects().create('bunny')
        self.optim_samples_count = (self.args.end_frame - self.args.offset_frame) // self.args.optim_frame_interval ## the interval of computing the loss function

        ## load surface 
        self.load_meshes(mesh_dict)

        ## initial the objects in the scene
        self.set_material_contact()
        self.set_physical_properties()

        ## create the ground 
        self.set_ground()

        ## define the optimization parameters
        self.param = DiffSimParameter(scene = self.scene, size = 1)

        ## initialize the simulation
        self.world.init(self.scene)

        self.sio = SceneIO(scene = self.scene)

        if GUIInfo.enabled():
            _init_polyscope_headless_safe()
            self.sgui = SceneGUI(self.scene)
            self.sgui.register_ground(self.ground_obj)
            self.sgui.register()

            if hasattr(ps, "is_headless") and ps.is_headless():
                for _ in range(self.args.end_frame):
                    self.update()
            else:
                ps.set_user_callback(self.update)
                ps.show()
        else:
            for _ in range(self.args.end_frame):
                self.update()
            

    def set_material_contact(self):

        for obj_name in self.mesh_dict.keys():

            ## apply the material to the mesh to define it as a rigid body
            self.abd.apply_to(self.mesh_dict[obj_name], 100.0*MPa, self.args.rho)


    def init_path(self):

        ## create the result folder 
        self.result_folder = f'{self.args.phys_result_folder}/{self.args.exp_name}'
        #if os.path.exists(self.result_folder):
        #    shutil.rmtree(self.result_folder)
        #os.makedirs(self.result_folder)
         
        if not os.path.exists(self.result_folder):
            os.makedirs(self.result_folder)
        
        ## create the tensorboard log folder
        self.log_folder = f'{self.result_folder}/logs'
        if not os.path.exists(self.log_folder):
            os.makedirs(self.log_folder)
        
        ## set the folder for loading the objects 
        #self.mesh_folder = f'{self.args.low_poly_folder}/{self.args.scene_name}'
        #self.mesh_folder = f'vis/scene_low_poly/{self.args.exp_name}'
        self.mesh_folder = f'data/layer_layout/{self.args.exp_name}'


        ## get the name list for all the objects in the scene


        ## folder for storing the internal code logs
        self.workspace = _make_uipc_workspace(self.result_folder, 'phys_sim', self.args)

        ## folder for storing the internal geometry
        self.geo_folder = f'{self.result_folder}/internal'
        if not os.path.exists(self.geo_folder):
            os.makedirs(self.geo_folder)
        
        ## folder for storing the transform parameters (the simulated results)
        self.transform_folder = f'{self.result_folder}/transform'
        if not os.path.exists(self.transform_folder):
            os.makedirs(self.transform_folder)
        
        ## folder for storing the optimzed parameters 
        self.param_folder = f'{self.result_folder}/param'
        if not os.path.exists(self.param_folder):
            os.makedirs(self.param_folder)

    
    def set_global_parameters(self):

        self.run = True
        self.global_step = 0
        self.per_obj_dof_num = 12
        self.optim_attempt = 0
        self.stop_when_static = bool(getattr(self.args, "stop_when_static", True))
        self.static_translation_tol = float(getattr(self.args, "static_translation_tol", 1e-4))
        self.static_rotation_tol = float(getattr(self.args, "static_rotation_tol", 1e-4))
        self.static_consecutive_frames = int(getattr(self.args, "static_consecutive_frames", 10))
        self.min_static_start_frame = int(
            getattr(
                self.args,
                "min_static_start_frame",
                max(1, int(getattr(self.args, "offset_frame", 0))),
            )
        )
        self.static_frame_count = 0
        self.prev_transform_dict = None
        self.stopped_because_static = False
        self.final_frame = 0
        self.first_static_window_start_frame = None
        self.first_static_window_end_frame = None
        self.sample_frame_numbers = [
            frame
            for frame in range(int(getattr(self.args, "offset_frame", 0)) + 1, int(self.end_frame) + 1)
            if frame > int(getattr(self.args, "offset_frame", 0))
            and frame % int(getattr(self.args, "optim_frame_interval", 1)) == 0
        ]
        if not self.sample_frame_numbers and int(self.end_frame) > int(getattr(self.args, "offset_frame", 0)):
            self.sample_frame_numbers = [int(self.end_frame)]
        self._sample_frame_lookup = set(self.sample_frame_numbers)
        self.sampled_transform_history = []
        self.sampled_frame_numbers_recorded = []
    

    def recover(self):
        self.world.recover(0)
        self.param.sync()

    def update(self):
        if GUIInfo.enabled():
            imgui.Text(f'frame: {self.world.frame()}')
            if(imgui.Button('Run & Stop')):
                self.run = not self.run
        
        if self.run:
            if(self.world.frame() < self.end_frame):
                self.world.advance()
                self.world.retrieve()
                if hasattr(self, "sgui"):
                    self.sgui.update()
                #self.sio.write_surface(f'{self.geo_folder}/out/step_{self.world.frame()}.obj')
                #print("--------------")
                #print(view(self.current_mesh_status_dict['cube1'].geometry().transforms())[0])
                self.save_transform_parameters()
                self._update_static_state(self.transform_dict)
                #print("--------------")
            else:
                self.run = False
                # ps.unshow() 
                pass

    def save_transform_parameters(self):
        
        #if self.world.frame() == self.args.end_frame:
        transform_dict = {}
        for obj_name in self.current_mesh_status_dict.keys():
            transform_param_piece = view(self.current_mesh_status_dict[obj_name].geometry().transforms())[0]
            current_frame_num = self.world.frame()
            transform_dict[obj_name] = transform_param_piece

        # cur_optim_save_folder = os.path.join(self.transform_folder, str(self.optim_attempt))
        # if not os.path.exists(cur_optim_save_folder):
        #     os.makedirs(cur_optim_save_folder)
        if self.store_all_transformation:
            frame_standard_id = str(self.world.frame()).zfill(5)
            transform_path = os.path.join(self.transformation_store_root_dir, f'frame_{frame_standard_id}.npz')
            np.savez(transform_path, **transform_dict)
        self.transform_dict = transform_dict
        current_frame = int(self.world.frame())
        if current_frame in self._sample_frame_lookup:
            self.sampled_frame_numbers_recorded.append(current_frame)
            self.sampled_transform_history.append(self._copy_transform_dict(transform_dict))

    def get_transform_parameters(self):
        return self.transform_dict

    def _copy_transform_dict(self, transform_dict):
        return {
            obj_name: np.asarray(matrix, dtype=np.float64).copy()
            for obj_name, matrix in transform_dict.items()
        }

    def _is_static_step(self, transform_dict):
        if self.prev_transform_dict is None:
            return False

        max_translation_delta = 0.0
        max_rotation_delta = 0.0
        for obj_name, matrix in transform_dict.items():
            previous = self.prev_transform_dict.get(obj_name)
            if previous is None:
                return False
            current = np.asarray(matrix, dtype=np.float64)
            previous = np.asarray(previous, dtype=np.float64)
            max_translation_delta = max(
                max_translation_delta,
                float(np.max(np.abs(current[:3, 3] - previous[:3, 3]))),
            )
            max_rotation_delta = max(
                max_rotation_delta,
                float(np.max(np.abs(current[:3, :3] - previous[:3, :3]))),
            )
        return (
            max_translation_delta <= self.static_translation_tol
            and max_rotation_delta <= self.static_rotation_tol
        )

    def _update_static_state(self, transform_dict):
        self.final_frame = int(self.world.frame())
        if not self.stop_when_static:
            self.prev_transform_dict = self._copy_transform_dict(transform_dict)
            return
        if self.final_frame < self.min_static_start_frame:
            self.static_frame_count = 0
            self.prev_transform_dict = self._copy_transform_dict(transform_dict)
            return
        if self._is_static_step(transform_dict):
            self.static_frame_count += 1
        else:
            self.static_frame_count = 0
        self.prev_transform_dict = self._copy_transform_dict(transform_dict)
        if self.static_frame_count >= self.static_consecutive_frames:
            self.first_static_window_start_frame = int(
                max(self.min_static_start_frame, self.final_frame - self.static_consecutive_frames)
            )
            self.first_static_window_end_frame = int(self.final_frame)
            self.run = False
            self.stopped_because_static = True
            

    def set_physical_properties(self):
        self.current_mesh_status_dict = {}
        self.initial_mesh_status_dict = {}

        for obj_name in self.mesh_dict.keys():

            ## fixed, apply for each object (0 --> no inertance)
            is_dynamic = self.mesh_dict[obj_name].instances().find(builtin.is_dynamic)
            view(is_dynamic)[0] = 0

            ## create a new instance in the self.scene based on the mesh
            ## current_mesh_status is the current property of the mesh [s_n] --> would be updated in the simulation
            ## initial_mesh_status is the initial property of the mesh [s_0]
            current_mesh_status, initial_mesh_status = self.bunny.geometries().create(self.mesh_dict[obj_name])
            self.current_mesh_status_dict[obj_name] = current_mesh_status
            self.initial_mesh_status_dict[obj_name] = initial_mesh_status




    def load_meshes(self, mesh_dict):

        self.mesh_dict = mesh_dict
        self.mesh_name_list = mesh_dict.keys()
        ## load the meshes in the self.scene
        for obj_name in mesh_dict.keys():
                
            ## apply the contact property to the mesh 
            self.contact_property_dict[obj_name].apply_to(self.mesh_dict[obj_name]) 
            
            ## label the self.mesh_dict['cube1']
            label_surface(self.mesh_dict[obj_name])

        ## get the total number of the objects in the scene
        self.obj_num = len(self.mesh_dict.keys())

        return self.mesh_dict

    def set_ground(self):  

        self.ground_obj = self.scene.objects().create('ground')
        g = ground(self.args.ground_y_value) ## y axis of the ground
        self.ground_obj.geometries().create(g) ## create the ground instance 

    





class ContainerNode:
    def __init__(self, name:str, mesh=None, tmesh:my_trimesh.Trimesh|None=None,father=None,children:list|None=None, is_root_of_single_link:bool=False):
        self._name = name
        self._children = children if children is not None else []
        self._mesh = mesh
        self._tmesh = tmesh
        self._transformation = np.eye(4)
        self._father = father
        self._depth = 0
        self._is_root_of_single_link = is_root_of_single_link


    def append_child(self, child):
        self._children.append(child)

    @property
    def is_root_of_single_link(self):
        return self._is_root_of_single_link

    @is_root_of_single_link.setter
    def is_root_of_single_link(self, value):
        self._is_root_of_single_link = value

    @property
    def father(self):
        return self._father

    @father.setter
    def father(self, value):
        self._father = value
        self._depth = self._father.depth + 1

    @property
    def depth(self):
        return self._depth

    @property
    def name(self) -> str:
        return self._name

    @property
    def children(self) -> dict:
        return self._children

    @children.setter
    def children(self, value):
        self._children = value

    @property
    def mesh_bounding_box(self):
        return self._mesh.bounds

    @property
    def transformation(self):
        return self._transformation

    @transformation.setter
    def transformation(self, value):
        self._transformation = value

    def set_transformation_xz(self, x_value, z_value):
        self._transformation[0][3] = x_value
        self._transformation[2][3] = z_value

    def get_bounding_box(self):
        bounding_box = {}
        if self.name != "ground":
            transformed_tmesh = self.transformed_tmesh
            bounding_box["x_min"] = transformed_tmesh.bounds[0, 0]
            bounding_box["x_max"] = transformed_tmesh.bounds[1, 0]

            bounding_box["y_min"] = transformed_tmesh.bounds[0, 1]
            bounding_box["y_max"] = transformed_tmesh.bounds[1, 1]

            bounding_box["z_min"] = transformed_tmesh.bounds[0, 2]
            bounding_box["z_max"] = transformed_tmesh.bounds[1, 2]
        else:
            bounding_box =  {'x_min': -100, 'x_max': 100, 'y_max': -1.1, 'z_min': -100, 'z_max': 100}
        return bounding_box

    @property
    def mesh(self):
        return self._mesh.copy()

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def transformed_mesh(self):
        mesh = self._mesh.copy()
        view(mesh.transforms())[:] = self._transformation
        np.set_printoptions(precision=3)
        return mesh

    @property
    def tmesh(self) -> my_trimesh.Trimesh:
        return self._tmesh

    @tmesh.setter
    def tmesh(self, value):
        self._tmesh = value

    @property
    def transformed_tmesh(self) -> my_trimesh.Trimesh:
        return self._tmesh.copy().apply_transform(self._transformation)


class LayerTreeOptimizer:
    def __init__(self):
        self._nodes = {"ground" : ContainerNode("ground")}
        self._root = self._nodes["ground"]
        self._gapy = 0.005
        self.progressive_pass = 0

    @property
    def root(self) -> ContainerNode:
        return self._root


    def init_tree(self, args, mesh_folder_path:str, tree_config_path:str):
        self._args = args

        self._mesh_folder_path = mesh_folder_path
        pre_transform_np = np.eye(4)
        pre_transform_np[0][0] = 1
        pre_transform_np[1][1] = 1


        # pre_transform_np[0][0] = 1
        # pre_transform_np[1][1] = 1
        tmp_dir = f"/tmp/mesh_fix"
        os.makedirs(tmp_dir, exist_ok=True)

        # Fix x,y
        for filename in os.listdir(mesh_folder_path):
            print(f"Loading {filename}")
            if not filename.lower().endswith(".obj"):
                continue

            full_filepath = os.path.join(mesh_folder_path, filename)
            if not os.path.isfile(full_filepath):
                continue

            mesh_name = os.path.splitext(filename)[0]

            tmesh = my_trimesh.load(full_filepath).apply_transform(pre_transform_np)
            tmesh.export(f"{tmp_dir}/{mesh_name}.obj")


        mesh_dict = {}
        trimesh_dict = {}
        pre_transform = Transform.Identity()
        pre_transform.scale(1)
        io = SimplicialComplexIO(pre_transform)
        for filename in os.listdir(mesh_folder_path):
            if not filename.lower().endswith(".obj"):
                continue

            full_filepath = os.path.join(tmp_dir, filename)
            if not os.path.isfile(full_filepath):
                continue

            mesh_name = os.path.splitext(filename)[0]

            mesh_dict[mesh_name] = io.read(full_filepath)
            trimesh_dict[mesh_name] = my_trimesh.load(full_filepath)
            print(f"Trimesh bounds: {trimesh_dict[mesh_name].bounds}")

            self._nodes[mesh_name] = ContainerNode(mesh_name, mesh_dict[mesh_name], trimesh_dict[mesh_name])


        with open(tree_config_path, encoding="utf-8") as f:
            tree_config = json.load(f)
        self.tree_config = tree_config

        for k in sorted(tree_config, key=int):
            layer = tree_config[k]
            for father, children in layer.items():
                for child in children:
                    self._nodes[father].append_child(self._nodes[child])
                    self._nodes[child].father = self._nodes[father]


        # Scan the entire tree and mark single list
        container_queue = [self._root]
        while len(container_queue) > 0:
            current_container = container_queue[0]
            container_queue.pop(0)

            if len(current_container.children) == 0:
                continue
            # Note that this computation for single link is NOT a correct computation of an arbitrary tree
            # We just assume the input data is properly formatted
            elif current_container.name != "ground" and len(current_container.children) == 1:
                current_container.is_root_of_single_link = True
                print(f"Marking node {current_container.name} as root of single link.")
            else:
                for child_node in current_container.children:
                    container_queue.append(child_node)




    def bfs_optimize(self, args, tolerance_rate:float=1e-4):

        container_queue = [self._root]
        finished_container = []

        while len(container_queue) > 0:

            current_container = container_queue[0] 
            container_queue.pop(0)
            print(f"Current container {current_container.name}\n\t Bounding Box {current_container.get_bounding_box()}")

            if len(current_container.children) == 0:
                continue


            processing_root_of_single_link = False
            progressive_pass_start_children_name = []
            if current_container.is_root_of_single_link:
                tmp_container = current_container
                while len(tmp_container.children) > 0:
                    children_container_old_center_dict = {}
                    current_bbox =  tmp_container.get_bounding_box()
                    children_container_old_center_dict[tmp_container.children[0].name] = np.sum(tmp_container.children[0].tmesh.bounds, axis=0) / 2
                    progressive_pass_start_children_name.append(tmp_container.children[0].name)
                    print(f"Append single link node: {tmp_container.children[0].name}")
                    # children_container_new_center_dict = get_children_center(current_bbox, tmp_container.name, self._mesh_folder_path, self.tree_config[str(tmp_container.depth)], self._gapy)
                    # for name, obj_center in children_container_old_center_dict.items():
                    #     center_translation = children_container_new_center_dict[name] - children_container_old_center_dict[name]
                    #     print(f"Translating node: {name} by {center_translation}")
                    #     current_transformation = self._nodes[name].transformation
                    #     new_transformation = np.eye(4)
                        # new_transformation[:3,3] = center_translation
                        # self._nodes[name].transformation = new_transformation @ current_transformation

                    tmp_container = tmp_container.children[0]

                processing_root_of_single_link = True
            else:
                children_container_old_center_dict = {}
                for child_node in current_container.children:
                    progressive_pass_start_children_name.append(child_node.name)
                    container_queue.append(child_node)
                #     children_container_old_center_dict[child_node.name] = np.sum(child_node.tmesh.bounds, axis=0) / 2
                    # print(f"Child Node: {child_node.name}\n\tBounding Box: {child_node.get_bounding_box()}")


                # Use X-Z optimization
                # TODO: call Guying's X-Z optimization
                current_bbox = current_container.get_bounding_box()
                # children_container_new_center_dict = get_children_center(current_bbox, current_container.name, self._mesh_folder_path, self.tree_config[str(current_container.depth)], self._gapy)

                # for name, obj_center in children_container_old_center_dict.items():
                #     center_translation = children_container_new_center_dict[name] - children_container_old_center_dict[name]
                #     print(f"Translating node: {name} by {center_translation}")
                #     current_transformation = self._nodes[name].transformation
                #     new_transformation = np.eye(4)
                #     # new_transformation[:3,3] = center_translation
                #     # self._nodes[name].transformation = new_transformation @ current_transformation
                #     print(f"Translated matrix {name}\n\t{self._nodes[name].transformation}")

            if self._args.export_render_transformation:
                export_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_start/transform/0")
                os.makedirs(export_dir, exist_ok=True)
                export_data = {}
                for container in finished_container:
                    export_data[container.name] = self._nodes[container.name].transformation

                for name in progressive_pass_start_children_name:
                    export_data[name] = self._nodes[name].transformation
                    print(f"Exporting Pre-Optimized Transform: {name}, {export_data[name]}")
                        
                np.savez(os.path.join(export_dir, "transformation.npz"), **export_data)


            # Dictionary of objects in the container 
            print("============================Begin Optimization============================")
            # if current_container.name != "ground":
            if self._args.preoptimized_transform_dir is not None and current_container.name == "cup1":

                export_optimize_start_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_optimized_simulation_start/transform/0")
                export_optimize_end_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_optimized_simulation_end/transform/0")
                export_optimize_success_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_optimized_simulation_success/transform/0")
                export_optimize_init_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_optimized_simulation_init/transform/0")
                os.makedirs(export_optimize_start_dir, exist_ok=True)
                os.makedirs(export_optimize_end_dir, exist_ok=True)
                os.makedirs(export_optimize_success_dir, exist_ok=True)
                os.makedirs(export_optimize_init_dir, exist_ok=True)

                npz_files = sorted(f for f in os.listdir(self._args.preoptimized_transform_dir)  if f.lower().endswith(".npz") and os.path.isfile(os.path.join(self._args.preoptimized_transform_dir, f))) 
                total_file_count = len(npz_files)
                for optimize_iteration in range(self._args.successful_optim_frame + 1):
                    # optimize_iteration = int(os.path.splitext(fname)[0].rsplit('_', 1)[-1]) 
                    fname = f"optim_{optimize_iteration}.npz"
                    print(f"Optimization file name {fname}, iteration {optimize_iteration}")
                    fpath = os.path.join(self._args.preoptimized_transform_dir, fname)
                    data = np.load(fpath, allow_pickle=False)

                    for name in data.keys():
                        preoptimized_x = data[name][0][3]
                        preoptimized_z = data[name][2][3]
                        self._nodes[name].set_transformation_xz(preoptimized_x, preoptimized_z)
                        print(f"Iteration {optimize_iteration} | Setting xz-translation to ({preoptimized_x}, {preoptimized_z}) with pre-optimized {name} node.")

                    # Reload the transformation

                    for name in progressive_pass_start_children_name:
                        export_data[name] = self._nodes[container.name].transformation

                    np.savez(os.path.join(export_optimize_start_dir, f"optim_transformation_{optimize_iteration:05d}.npz"), **export_data)
                    simulator_mesh_dict = {}
                    for name in export_data.keys():
                        simulator_mesh_dict[name] = self._nodes[name].transformed_mesh

                    if optimize_iteration == 0:
                        simulator = PhysSimulator(args, simulator_mesh_dict, 1000, True, export_optimize_init_dir)
                    elif optimize_iteration == self._args.successful_optim_frame:
                        simulator = PhysSimulator(args, simulator_mesh_dict, 1000, True, export_optimize_success_dir)
                    else:
                        simulator = PhysSimulator(args, simulator_mesh_dict, 1000, False)

                    transformations = simulator.get_transform_parameters()
                    print(f"Optim End Transform parameter keys: {transformations.keys()}")

                    for name in transformations.keys():
                        export_data[name] = transformations[name]
                        print(f"Optim End: export name :{name}, value {export_data[name]}")

                    np.savez(os.path.join(export_optimize_end_dir, f"optim_transformation_{optimize_iteration:05d}.npz"), **export_data)

            else:
                if current_container.name == "sofa1":
                    print(f"Current container {current_container.name}")

                    optimizer_mesh_dict = { current_container.name : current_container.transformed_mesh }
                    optimizer_tmesh_dict = { current_container.name : current_container.tmesh }

                    if processing_root_of_single_link:
                        # dfs
                        tmp_node = current_container
                        while len(tmp_node.children) > 0:
                            tmp_node = tmp_node.children[0]
                            optimizer_mesh_dict[tmp_node.name] = tmp_node.transformed_mesh
                            optimizer_tmesh_dict[tmp_node.name] = tmp_node.tmesh
                    else:
                        # bfs
                        for child in current_container.children:
                            optimizer_mesh_dict[child.name] = child.transformed_mesh
                            optimizer_tmesh_dict[child.name] = child.tmesh

                    for name in optimizer_mesh_dict.keys():
                        self._nodes[name].transformed_tmesh.export(f"/tmp/largestudy_export/{name}.obj")

                    # optimize
                    print(f"Current container {current_container.name} {current_container.get_bounding_box()}")
                    print(f"Current container father {current_container.father.name} {current_container.father.get_bounding_box()}")
                    # args.ground_y_value = current_container.get_bounding_box()["y_min"] - 0.1
                    # simulator = PhysSimulator(args, optimizer_mesh_dict)
                    layer_optimizer = PhysOptim(args, current_container.name, optimizer_mesh_dict, optimizer_tmesh_dict)
                    # current_trans = current_container.transformation
                    optimized_transformation = layer_optimizer.transformation_parameter
                    for container_name, transformation in optimized_transformation.items():
                        # self._nodes[container_name].transformation[1, 3] += 0.05
                        self._nodes[container_name].set_transformation_xz(transformation[0][3], transformation[2][3])
                    # args.ground_y_value = -1.1
            print("============================End Optimization============================")


            
            simulator_mesh_dict = {}

            if processing_root_of_single_link:
                tmp_node = current_container
                while len(tmp_node.children) > 0:
                    tmp_node = tmp_node.children[0]
                    simulator_mesh_dict[tmp_node.name] = tmp_node.transformed_mesh
            else:
                for child in current_container.children:
                    simulator_mesh_dict[child.name] = child.transformed_mesh

            for container in finished_container:
                simulator_mesh_dict[container.name] = container.transformed_mesh
            print(f"Simulator mesh dict: {simulator_mesh_dict.keys()}")

            print("============================Begin Simulation============================")
            # run till static
            if self._args.export_render_root_dir is not None:
                simulation_store_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_full/transform/0")
                os.makedirs(simulation_store_dir, exist_ok=True)
                simulator = PhysSimulator(args, simulator_mesh_dict, 1000, True, simulation_store_dir)
            else:
                print("here")
                simulator = PhysSimulator(args, simulator_mesh_dict, 1000, False)
            transformations = simulator.get_transform_parameters()



            if processing_root_of_single_link:
                tmp_node = current_container
                while len(tmp_node.children) > 0:
                    tmp_node = tmp_node.children[0]
                    finished_container.append(tmp_node)
            else:
                for child in current_container.children:
                    finished_container.append(child)


            for container in finished_container:
                container.transformation = transformations[container.name]

            if self._args.export_render_transformation:
                export_dir = os.path.join(self._args.export_render_root_dir, f"progressive_pass_{self.progressive_pass}_end/transform/0")
                os.makedirs(export_dir, exist_ok=True)
                export_data = {}
                for container in finished_container:
                    export_data[container.name] = transformations[container.name]
                np.savez(os.path.join(export_dir, "transformation.npz"), **export_data)
            print("============================End Simulation============================")
            self.progressive_pass += 1





def parse_options(return_parser=False):
    # New CLI parser
    parser = argparse.ArgumentParser(description='Train 3D layout for physics-grounded scene.')
    
    # Global arguments
    global_group = parser.add_argument_group('global')
    global_group.add_argument('--exp_name', type = str,
                              help='Experiment name.')
    global_group.add_argument('--scene_name', type = str,
                              help='The name for the scene.')
    global_group.add_argument('--preprocess', action='store_true',
                              help='Preprocess to get the initial scene.')
    global_group.add_argument('--mask_optim', action='store_true',
                              help='Use mask optimization.')
    global_group.add_argument('--phys_optim', action='store_true',
                              help='Use physical optimization.')
    global_group.add_argument('--bbox_put', action='store_true',
                              help='Use bounding box placement.')
    global_group.add_argument('--visualize', action='store_true',
                              help='Visualize the results after optimization.')
    global_group.add_argument('--gpu_id', type=int, default=0,
                              help='GPU ID to use for training. Default is 0.')
    
    

    ## parameter for preprocess
    parser.add_argument('--text_prompt', type=str,
                        help='Text prompt for the scene.')
    parser.add_argument('--img_prompt_folder', type=str,
                        default='modules/preprocess_utils/img_utils',
                        help='Folder of the template text prompt for the scene or the objects.')
    parser.add_argument('--items', type=str, nargs='+',
                        help='List of items to be included in the scene.')
    parser.add_argument('--ref_image_folder', type=str, default='data/ref_img',
                        help='Folder containing reference images for the scene.')
    parser.add_argument('--ref_image_candidate_folder', type=str, default='data/ref_img_candidate',
                        help='Folder containing reference images candidate for the scene.')
    parser.add_argument('--ref_image_num', type=int, default = 1,
                        help='Number of reference images to generate for the scene.')
    parser.add_argument('--vqas_env_path', type=str, default='/home/guyinglin/miniconda3/envs/t2v/bin/python3',
                        help='Path to the vqas environment python executable.')
    parser.add_argument('--vqa_score_folder', type=str, default='data/vqa_score',
                        help='Folder to save the original vqa scores for the images.')
    parser.add_argument('--depth_folder', type=str, default='data/depth',
                        help='Folder to save and load the depth data.')
    parser.add_argument('--seg_folder', type=str, default='data/seg',
                        help='Folder to save and load the segmentation data.')
    parser.add_argument('--items_folder', type=str, default='data/items',
                        help='Folder to save and load the item data in the scene.')
    parser.add_argument('--gpt_prompt_folder', type=str, default='modules/preprocess_utils/gpt_utils',
                        help='Folder containing the GPT prompt files.')
    parser.add_argument('--gpt_apikey_path', type=str, default='modules/preprocess_utils/gpt_utils/apikey.txt',
                        help='Path to the GPT API key file.')
    parser.add_argument('--descrip_folder', type=str, default='data/descrip',
                        help='Folder to save and load the object description data.')
    parser.add_argument('--ref_image_obj_folder', type=str, default='data/ref_img_obj',
                        help='Folder to save and load the reference image for object data.')
    parser.add_argument('--ref_image_obj_candidate_folder', type=str, default='data/ref_img_obj_candidate',
                        help='Folder to save and load the reference image candidate for object data.')
    parser.add_argument('--raw_obj_folder', type=str, default='data/raw_obj',
                        help='Folder to save and load the raw object data.')
    parser.add_argument('--organized_obj_folder', type=str, default='data/organized_obj',
                        help='Folder to save and load the organized object data.')
    parser.add_argument('--layout_front_num', type = int, default = 5, 
                        help = 'When computing the layout, the number of points to compute the front mean position')
    parser.add_argument('--layout_folder', type = str, default = 'data/layout',
                        help='Folder to save and load the layout data.')
    parser.add_argument('--low_poly_folder', type=str, default='data/low_poly',
                        help='Folder to save and load the low poly object data.')
    parser.add_argument('--low_poly_psample', type=int, default = 300000,
                        help='Number of points to sample for reconstructing low poly objects.')
    parser.add_argument('--low_poly_fnum', type=int, default = 2000,
                        help='Number of target faces for the low poly objects.')
    parser.add_argument('--size_folder', type=str, default='data/size',
                        help='Folder to save and load the size data.')
    parser.add_argument('--contain_folder', type=str, default='data/contain',
                        help='Folder to save and load the contain information in the scene.')
    parser.add_argument('--contain_on_folder', type=str, default='data/contain_on',
                        help='Folder to save and load the contain on information in the scene.')
    parser.add_argument('--manifold_code_path', type=str, default='extern/Manifold/build',
                        help='Path to the manifold code folder.')

    ## bbox arranger
    parser.add_argument('--bbox_put_folder', type=str, default='data/bbox_put',
                        help='Folder to save and load the bounding box placement data.')
    parser.add_argument('--max_layer_num', type = int, default = 6,
                        help='The maximum number of layers for the bounding box placement.')
    parser.add_argument('--xyz_order_folder', type=str, default='data/xyz_order',
                        help='Folder to save and load the xyz order data.')
    parser.add_argument('--layer_folder', type=str, default='data/layer',
                        help='Folder to save and load the layer data.')
    parser.add_argument('--layer_layout_folder', type=str, default='data/layer_layout',
                        help='Folder to save and load the layer layout data.')

    ## mask loss
    parser.add_argument('--outdir', help = 'specify output directory', default = 'cube_pose')
    parser.add_argument('--mp4save-interval', type = int, default = 50)
    parser.add_argument('--para-save-interval', type = int, default = 200)
    parser.add_argument('--cluster-save-interval', type = int, default = 200)
    parser.add_argument('--max-iter', type = int, default = 10000)
    parser.add_argument('--pick-num-per-object', type = int, default = 2000,
                        help = 'The number of points to downsample for each object when computing the intersection loss.')
    parser.add_argument('--log-interval', type = int, default = 50)
    parser.add_argument('--log-fn', default = 'log.txt')
    parser.add_argument('--lr-base', type = float, default = 0.01)
    parser.add_argument('--lr-falloff', type = float, default = 1.0)
    parser.add_argument('--mask-weight', type = float, default = 0.5)
    parser.add_argument('--depth-weight', type = float, default = 5.0)
    parser.add_argument('--floor-weight', type = float, default = 100.0)
    parser.add_argument('--inter-weight', type = float, default = 300.0)
    parser.add_argument('--size-weight', type = float, default = 1)
    parser.add_argument('--contain-weight', type = float, default = 10)
    parser.add_argument('--nr-base', type = float, default = 1.0)
    parser.add_argument('--nr-falloff', type = float, default = 1e-4)
    parser.add_argument('--grad-phase-start', type = float, default = 0.8)
    parser.add_argument('--noise-scale-factor', type = float, default = 1.0)
    parser.add_argument('--mp4save-fn', default = 'progress.mp4')
    parser.add_argument('--mask_flag', action='store_true',
                        help='Use mask loss.')
    parser.add_argument('--depth_flag', action='store_true',
                        help='Use depth loss.')
    parser.add_argument('--floor_flag', action='store_true',
                        help='Use floor loss.')
    parser.add_argument('--inter_flag', action='store_true',
                        help='Use intersection loss.')
    parser.add_argument('--contain_flag', action='store_true',
                        help='Use contain loss.')
    parser.add_argument('--size_flag', action='store_true',
                        help='Use size loss.')
    parser.add_argument('--opt_scale', action='store_true',
                        help='Optimize scale.')
    parser.add_argument('--opt_translation', action='store_true',
                        help='Optimize translation.')
    parser.add_argument('--opt_glo_rotation', action='store_true',
                        help='Optimize the global rotation.')
    parser.add_argument('--opt_rotation', action='store_true',
                        help='Optimize the per object rotation.')
    parser.add_argument('--resume', type = str, default = None,
                        help='The path to the parameters to resume from.')
                        
    ## physical optimization
    parser.add_argument('--gui_flag', action='store_true',
                        help='Use GUI for visualization during optimization.')
    parser.add_argument('--total_random_attempt', type = int, default = 3,
                        help='Total number of random attempts for physical optimization.')
    parser.add_argument('--total_opt_epoch', type = int, default = 50,
                        help='Total number of epochs for physical optimization.')
    parser.add_argument('--phys_result_folder', type = str, default = '_phys_result',
                        help='Folder to save the results of physical optimization.')
    parser.add_argument('--time_step', type = float, default = 0.03,
                        help='Timestep for the physical simulation.')
    parser.add_argument('--tol_rate', type = float, default = 1e-4,
                        help='Tolerance rate for the linear system.')
    parser.add_argument('--contact_d_hat', type = float, default = 5e-4,
                        help='Contact distance threshold. Paper stacked-block setting.')
    parser.add_argument('--contact_eps_velocity', type = float, default = 1e-5,
                        help='Friction velocity threshold. Paper stacked-block setting.')
    parser.add_argument('--rho', type = float, default = 1000.0)
    parser.add_argument('--ground_y_value', type = float, default = -1.1,
                        help='The y value of the ground plane.')
    parser.add_argument('--end_frame', type = int, default = 300,
                        help='The end frame for the simulation.')
    parser.add_argument('--offset_frame', type = int, default = 0,
                        help='The offset frame for the optimization. This is the number of frames to be skipped before the optimization starts.')
    parser.add_argument('--optim_frame_interval', type = int, default = 10,
                        help='The interval frames for the optimization. This is the number of frames to be skipped between each optimization step.')
    parser.add_argument('--weight_y', type = float, default = 20.0,
                        help='The weight for the y axis. This is used to control the weight of difference on the y axis in the optimization.')
    parser.add_argument('--weight_x', type = float, default = 1.0,
                        help='The weight for the x axis. This is used to control the weight of difference on the x axis in the optimization.')
    parser.add_argument('--weight_z', type = float, default = 1.0,
                        help='The weight for the z axis. This is used to control the weight of difference on the z axis in the optimization.')
    parser.add_argument('--phys_lr', type = float, default = 0.001,
                        help='Learning rate for the physical optimization.')
    parser.add_argument('--weight_decay', type = float, default = 0.01,
                        help='Weight decay used by the diff-sim optimizer.')
    parser.add_argument('--optimizer_name', type=str, default='adamw',
                        choices=('adam', 'adamw'),
                        help='Optimizer used by the diff-sim stage.')
    parser.add_argument('--optimizer_amsgrad', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable AMSGrad for Adam/AdamW.')
    parser.add_argument('--grad_clip_norm', type=float, default=0.0,
                        help='Gradient norm clipping threshold. Set <= 0 to disable clipping.')
    parser.add_argument('--adaptive_end_frame_enabled', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable adaptive diff-sim horizon selection up to the configured cap.')
    parser.add_argument('--adaptive_controller', type=str, default='sampled_criteria',
                        choices=('sampled_criteria', 'forward_static_window'),
                        help='Adaptive diff-sim horizon controller.')
    parser.add_argument('--adaptive_end_frame_cap', type = int, default = 300,
                        help='Maximum diff-sim horizon when adaptive end-frame selection is enabled.')
    parser.add_argument('--adaptive_end_frame_min', type = int, default = 200,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_warmup_epochs', type = int, default = 10,
                        help='Number of optimization epochs to observe before sampled horizon updates are allowed.')
    parser.add_argument('--adaptive_required_consecutive_epochs', type = int, default = 3,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_required_stable_epochs', type = int, default = 5,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_acceleration_residual_threshold', type = float, default = 0.1,
                        help='Maximum per-object acceleration-style inf norm for a sampled frame to qualify for adaptive early stop.')
    parser.add_argument('--adaptive_force_residual_threshold', type = float, default = 0.4,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the acceleration-only controller.')
    parser.add_argument('--adaptive_velocity_residual_threshold', type = float, default = 0.003,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the acceleration-only controller.')
    parser.add_argument('--adaptive_min_stop_frame', type = int, default = 200,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_support_order', type = str, default = '',
                        help='Optional comma-separated object order used for adaptive support metrics, independent of the optimization loss target.')
    parser.add_argument('--adaptive_overlap_threshold', type = float, default = 0.95,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_height_ratio_threshold', type = float, default = 0.90,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_require_min_stop_frame', action=argparse.BooleanOptionalAction, default=False,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_require_order', action=argparse.BooleanOptionalAction, default=False,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_require_overlap', action=argparse.BooleanOptionalAction, default=False,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_require_height_ratio', action=argparse.BooleanOptionalAction, default=False,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the simplified controller.')
    parser.add_argument('--adaptive_require_velocity', action=argparse.BooleanOptionalAction, default=True,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the acceleration-only controller.')
    parser.add_argument('--adaptive_require_force', action=argparse.BooleanOptionalAction, default=True,
                        help='Deprecated compatibility option for sampled_criteria; ignored by the acceleration-only controller.')

    
    ## visualization 
    parser.add_argument('--end_frame_upper_bound', type = int, default = 300,
                        help='The upper bound for the end frame. This is used to control the maximum number of frames for the simulation.')
    parser.add_argument('--low_poly', action='store_true',
                        help='Use low poly version of the geometry.')
    parser.add_argument('--semantic_optim_output_folder', type=str, default='vis/scene_high_poly',
                        help='Folder to save the semantic optimization output.')
    parser.add_argument('--simulation_output_folder', type=str, default='vis/simulated_scene',
                        help='Folder to save the forward simulation output.')
    parser.add_argument('--optim_attempt_num', type = int, default = 0,
                        help='The attempt number of the optimization, 0 means the initial one without optimization.')
    parser.add_argument('--optim_initial_scene_folder', type=str, default='visualize/optim_s0',
                        help='Folder to save the optimized initial scene.')
    
    # Architecture for network
    net_group = parser.add_argument_group('net')

    # Arguments for dataset
    data_group = parser.add_argument_group('dataset')
    data_group.add_argument('--simu_save_dir', type = str, default = 'output',
                            help = 'The directory for saving the simulation results.')
    data_group.add_argument('--max_intersection_iter', type = int, default = 30,
                            help = 'The max interation number for solving the intersections.')
    data_group.add_argument('--object_sample_num', type = int, default = 10000,
                            help = 'The sampling number for each object in order to get the sdf query function.')
    data_group.add_argument('--protect_strip_value', type = float, default = 0.008,
                            help = 'The protection strip for more efficient simulation.')
    data_group.add_argument('--global_min_x', type = float, default = -3,
                            help = 'The global min x value for the bounding box.')
    data_group.add_argument('--global_min_y', type = float, default = -1,
                            help = 'The global min y value for the bounding box.')  
    data_group.add_argument('--global_min_z', type = float, default = -3,
                            help = 'The global min z value for the bounding box.')
    data_group.add_argument('--global_max_x', type = float, default = 3,
                            help = 'The global max x value for the bounding box.')
    data_group.add_argument('--global_max_y', type = float, default = 4,
                            help = 'The global max y value for the bounding box.')
    data_group.add_argument('--global_max_z', type = float, default = 3,
                            help = 'The global max z value for the bounding box.')


    ## Arguments for render
    render_group = parser.add_argument_group('render')
    render_group.add_argument('--camera_view_num', type = int, default = 1,
                              help='Number of the camera views.')
    render_group.add_argument('--resolution', type = int, default = 512,
                              help='Resolution for the rendered images.')

    # Arguments for optimizer
    optim_group = parser.add_argument_group('optimizer')
    optim_group.add_argument('--lr', type = float, default = 0.01, 
                             help = 'Learning rate.')
    optim_group.add_argument('--lr_step_size', type = int, default = 1, 
                             help = 'Step size for learning rate decay.')
    optim_group.add_argument('--lr_step_gamma', type = float, default = 0.999, 
                             help = 'Gamma for learning rate decay.')
    
    # Arguments for training
    train_group = parser.add_argument_group('trainer')             
    train_group.add_argument('--epochs', type = int, default = 100,                                        
                             help='Number of epochs to run the training.')    
    train_group.add_argument('--batch_size', type = int, default = 5,                                        
                             help='Number of rendered images for training each epoch.')      

    # Arguments for loss
    loss_group = parser.add_argument_group('loss')
    loss_group.add_argument('--stable_weight', type = float, default = 3.0)
    loss_group.add_argument('--bbox_weight', type = float, default = 0)    
    loss_group.add_argument('--step_size', type = float, default = 0.005, 
                            help = 'The step size for the numerical optimization for translation.')
    loss_group.add_argument('--max_iter', type = int, default = 100,
                            help = 'The max iteration number for the numerical optimization for translation.')
    loss_group.add_argument('--bbox_tolerance', type = float, default = 0.2,
                            help = 'The tolerance for the bbox loss.')
    loss_group.add_argument('--ccd_steps', type = int, default = 5)
        
    ## Argument for post epoch 
    post_group = parser.add_argument_group('post_epoch')
    post_group.add_argument('--save_layout_interval', type = int, default = 20,
                            help = 'The interval for saving the layout parameters.')
    post_group.add_argument('--save_img_interval', type = int, default = 1,
                            help = 'The interval for saving the rendered images.')
    post_group.add_argument('--save_cluster_interval', type = int, default = 2,
                            help = 'The interval for saving the cluster meshed.')
    post_group.add_argument('--result_folder', type = str, default = '_result',
                            help = 'The folder for saving the results.')


    # Parse and run
    if return_parser:
        return parser
    else:
        return argparse_to_str(parser)


                

if __name__ == "__main__":
    parser = parse_options(True)
    parser.add_argument("--mesh_folder_path", required=True, type=str, help="The path for mesh folder.")
    parser.add_argument("--layout_path", required=True, type=str, help="The path for layout file.")
    parser.add_argument("--preoptimized_transform_dir", type=str, default=None)
    parser.add_argument("--export_render_transformation",default=False, action="store_true")
    parser.add_argument("--export_render_root_dir", type=str, default=None)
    parser.add_argument("--successful_optim_frame", type=int, default=0)


    args = parser.parse_args()


    tree_optimizer = LayerTreeOptimizer()
    tree_optimizer.init_tree(args, args.mesh_folder_path, args.layout_path)

    tree_optimizer.bfs_optimize(args)

    
