import numpy as np
import polyscope as ps
import torch
import torch.nn as nn
import torch.optim as optim
import trimesh as my_trimesh
from polyscope import imgui
import shutil
from torch.utils.tensorboard import SummaryWriter  # 新增TensorBoard支持
import os 
from uipc import view
from uipc import Logger, Timer
from uipc import \
    Vector3, Vector3i, \
    Matrix4x4, Matrix4x4i,\
    Transform, Quaternion, AngleAxis
from uipc import builtin
from uipc.core import *
from uipc.geometry import *
from uipc.constitution import *
from uipc.gui import *
from uipc.torch import *
from uipc.unit import GPa, MPa


## TODO: add some constraints in loss terms to make the rotation matrix only contain the rotation information of the object 
## TODO: set some loss terms to constrain the object to start from the sky 
## TODO: why would sample between frames work? 

class EMA:
    def __init__(self, alpha=0.9):
        self.alpha = alpha
        self.smoothed_loss = None
    
    def update(self, loss):
        if self.smoothed_loss is None:
            self.smoothed_loss = loss.item()
        else:
            self.smoothed_loss = self.alpha * self.smoothed_loss + (1 - self.alpha) * loss.item()
        return self.smoothed_loss

ema_loss = EMA(alpha=0.9)  # alpha越大平滑效果越强（0.9~0.99）



class PhysOptim(object):
    def __init__(self, args):

        self.args = args

        self.init_path()
        self.set_writer()
        self.set_global_parameters()

        Timer.enable_all()
        Logger.set_level(Logger.Level.Warn)


   ## =====================================================================================================================

        ## parameters 
        ## fixed physical parameters

        confige = Engine.default_config() 
        confige['gpu']['device'] = 1
        engine = Engine('cuda', self.workspace, confige)
        self.world = World(engine)
        config = Scene.default_config()



        config['contact']['eps_velocity'] = 0.0001
        config['dt'] = self.args.time_step
        config['gravity'] = [[0.0], [-9.8], [0.0]]
        config['contact']['enable']             = True 
        config['contact']['d_hat']              = 0.008 ## threshold for conlision detect 
        config['contact']['friction']['enable'] = True ## whether to use friction ## set it to be false for now
        config['line_search']['max_iter']       = 8 ## fixed, max iteration for line search
        config['linear_system']['tol_rate']     = self.args.tol_rate ## 1e-4 is more stable, except it has some related errors

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
        self.scene.contact_tabular().default_model(0.1, 1.0 * GPa)
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
        self.bunny = self.scene.objects().create('bunny')
        self.optim_samples_count = (self.args.end_frame - self.args.offset_frame) // self.args.optim_frame_interval ## the interval of computing the loss function

        ## load surface 
        self.load_meshes()

        ## set the target point for each object
        self.set_target_centers()

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
            if info.frame() > self.args.offset_frame and info.frame() % self.args.optim_frame_interval == 0: ## get the frame number
                sdi = torch.arange(0, self.obj_num * self.per_obj_dof_num, dtype = torch.int32) ## define dof --> (obj_num * (9+3))
                #print(f'[dof_select] sdi: {sdi}')
                return sdi
            else:
                return None

        ## define the module 
        self.diff_module = DiffSimModule(self.world, end_frame = self.args.end_frame, parm = self.param, dof_select = dof_select)

        ## set the optimizer
        self.optimizer = optim.Adam(self.diff_module.parameters(), lr = self.args.phys_lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size = 30, gamma = 0.8)
        lr = DiffSimLR(self.diff_module, self.optimizer)

        ## random initialization of the parameters
        self.jitter_param()

        self.set_JC()
    
        self.sio = SceneIO(scene = self.scene)


        for epoch in range(self.args.end_frame):
            self.update()
        

    def jitter_param(self):

        with torch.no_grad():
            u = torch.zeros(self.param.U().shape[0], dtype = torch.float64)
            self.param.U().copy_(u)
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
        #self.mesh_folder = f'{self.args.forward_sim_layer_layout_folder}/{self.args.exp_name}'
        self.mesh_folder = f'{self.args.diff_sim_layer_layout_folder}/{self.args.exp_name}'

        ## get the name list for all the objects in the scene
        self.mesh_name_list = []
        for file_name in os.listdir(self.mesh_folder):
            if file_name.endswith('.obj'):
                obj_name = os.path.splitext(file_name)[0]
                self.mesh_name_list.append(obj_name)


        ## folder for storing the internal code logs
        self.workspace = f'{self.result_folder}/workspace'
        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace)

        ## folder for storing the internal geometry
        self.geo_folder = f'{self.result_folder}/internal'
        if not os.path.exists(self.geo_folder):
            os.makedirs(self.geo_folder)
        
        ## folder for storing the transform parameters (the simulated results)
        self.transform_folder = f'{self.result_folder}/transform'
        if os.path.exists(self.transform_folder):
            shutil.rmtree(self.transform_folder)
        os.makedirs(self.transform_folder)
        
        
        ## folder for storing the optimzed parameters 
        self.param_folder = f'{self.result_folder}/param'
        if not os.path.exists(self.param_folder):
            os.makedirs(self.param_folder)

    def set_writer(self):
        
        self.writer = SummaryWriter(log_dir = self.log_folder)
    
    def set_global_parameters(self):

        self.run = False
        self.global_step = 0
        self.per_obj_dof_num = 12
        self.optim_attempt = 0
    
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

    def recover(self):
        self.world.recover(0)
        self.param.sync()

    def bind_translation_param(self):
        
        self.param_map_index = 0
        self.current_mesh_status_dict = {}
        self.initial_mesh_status_dict = {}
        self.param_map_back = {}

        for obj_name in self.mesh_dict.keys():
         
            ## define the initial transformation optimization matrix mask for each object 
            diff_tran_index0 = -1 * Matrix4x4i.Ones() ## this is a mask, value>=0 means enable the optimization, -1 measn disable the optimization
            diff_tran0 = self.mesh_dict[obj_name].instances().create('diff/transform', diff_tran_index0)

            ## enable the parameter that should be optimized
            diff_tran_index0[0, 3] = self.param_map_index ## map to parms
            self.param_map_back[self.param_map_index] = [obj_name, 0, 3]
            self.param_map_index += 1

            diff_tran_index0[1, 3] = self.param_map_index
            self.param_map_back[self.param_map_index] = [obj_name, 1, 3]
            self.param_map_index += 1
            
            diff_tran_index0[2, 3] = self.param_map_index
            self.param_map_back[self.param_map_index] = [obj_name, 2, 3]
            self.param_map_index += 1

            ## bind the new mask to the mesh
            view(diff_tran0)[0] = diff_tran_index0


    def update(self):
        
        if(self.world.frame() < self.args.end_frame):
            self.world.advance()
            self.world.retrieve()
            #self.sgui.update()
            #self.sio.write_surface(f'{self.geo_folder}/out/step_{self.world.frame()}.obj')
            #print("--------------")
            #print(view(self.current_mesh_status_dict['cup1'].geometry().transforms())[0])
            self.save_transform_parameters()
            print(f'frame: {self.world.frame()}')
            #print("--------------")


    def save_transform_parameters(self):
        
        #if self.world.frame() == self.args.end_frame:
        transform_dict = {}
        for obj_name in self.current_mesh_status_dict.keys():
            transform_param_piece = view(self.current_mesh_status_dict[obj_name].geometry().transforms())[0]
            current_frame_num = self.world.frame()
            transform_dict[obj_name] = transform_param_piece
        cur_optim_save_folder = os.path.join(self.transform_folder, str(self.optim_attempt))
        if not os.path.exists(cur_optim_save_folder):
            os.makedirs(cur_optim_save_folder)
        frame_standard_id = str(self.world.frame()).zfill(5)
        cur_optim_save_path = os.path.join(self.transform_folder, str(self.optim_attempt), f'frame_{frame_standard_id}.npz')
        np.savez(cur_optim_save_path, **transform_dict)
            

    def set_physical_properties(self):

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



    def load_meshes(self):

        self.mesh_dict = {}
        self.mesh_trimesh_dict = {}
        ## load the meshes in the self.scene
        for file_name in os.listdir(self.mesh_folder):

            if file_name.endswith('.obj'):

                obj_name = os.path.splitext(file_name)[0]
                mesh_path = os.path.join(self.mesh_folder, file_name)
                print(f'Loading mesh_from {mesh_path}')
                
                ## read the self.mesh_dict['cube1']
                self.mesh_dict[obj_name] = self.io.read(mesh_path)
                self.mesh_trimesh_dict[obj_name] = my_trimesh.load(mesh_path)
                
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

    def optimize_once(self):
        
        for i in range(self.args.total_opt_epoch):
            self.optimizer.step(self.iterate)
            self.scheduler.step()
        self.optim_attempt += 1
        Timer.report()
        self.save_optimized_param()

    def save_optimized_param(self):

        #print('param u:', self.param.U())
        save_param_path = os.path.join(self.param_folder, f'optim_{self.optim_attempt}.npz')

        save_param_dict = {}

        ## for each object, initialize a 4 * 4 identity numpy matrix 
        for obj_name in self.mesh_dict.keys():
            save_param_dict[obj_name] = np.zeros((4, 4), dtype = np.float64)
            save_param_dict[obj_name][:, :] = np.eye(4, dtype = np.float64)

        ## recover the parameters to the transform matrix 
        for index in (self.param_map_back.keys()):
            fetched_param = self.param.U()[index]
            obj_name, row, col = self.param_map_back[index]
            save_param_dict[obj_name][row, col] = fetched_param
        
        #print(f'save_param_dict: {save_param_dict}')
        ## save the parameters to the npz file
        np.savez(save_param_path, **save_param_dict)


    def iterate(self)->torch.Tensor:

        self.optimizer.zero_grad()

        ## get the parameters for optimization
        X = self.diff_module() ## the transform of the objects [i*12:(i+1)*12]  --> get svd --> u*v*sigma [012 T 3-11 R] 3/4/5 6/7/8 9/10/11
        self.loss = 0.0

        ## for the frames to be sampled 
        for i in range(self.optim_samples_count):
            
            weight = 1.0
            U = self.param.U()
            
            ## compute the center loss for each object
            for j in range(self.obj_num):
                current_center = self.JC[j*3:j*3 + 3].matmul(X[i*self.per_obj_dof_num * self.obj_num + j*self.per_obj_dof_num:i*self.per_obj_dof_num * self.obj_num + j*self.per_obj_dof_num + self.per_obj_dof_num])
                target_center = self.dstX[j*3:j*3 + 3]
                y_loss = self.args.weight_y * (current_center[1] - target_center[1]).square().sum()
                x_loss = self.args.weight_x * (current_center[0] - target_center[0]).square().sum()
                z_loss = self.args.weight_z * (current_center[2] - target_center[2]).square().sum()
                self.loss += (y_loss + x_loss + z_loss)

        
        self.update_writer()
        self.loss.backward()
        self.global_step += 1  


    def update_writer(self):

        self.writer.add_scalar('loss/train', self.loss.item(), self.global_step)
        self.writer.add_scalar('learning rate', self.optimizer.param_groups[0]['lr'], self.global_step)
        self.writer.add_histogram('parameters', self.param.U(), self.global_step)
        
    

