import shutil
import sys
import os
import fnmatch
import torch, gc
import torch.nn as nn
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pdb


# 设置模型运行的设备
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class pyOMT_raw():	
    '''This class is designed to compute the semi-discrete Optimal Transport (OT) problem. 
    Specifically, within the unit cube [0,1]^n of the n-dim Euclidean space,
    given a source continuous distribution mu, and a discrete target distribution nu = \sum nu_i * \delta(P_i),
    where \delta(x) is the Dirac function at x \in [0,1]^n, compute the Optimal Transport map pushing forward mu to nu.

    The method is based on the variational principle of solving semi-discrete OT, (See e.g.
    Gu, Xianfeng, et al. "Variational principles for Minkowski type problems, discrete optimal transport, and discrete Monge-Ampere equations." Asian Journal of Mathematics 20.2 (2016): 383-398.)
    where a convex energy is minimized to obtain the OT map. 

    Adam gradient descent method is used here to perform the optimization, and Monte-Carlo integration method is used to calculate the energy.
    '''

    def __init__ (self, h_P, num_P, dim, max_iter, lr, bat_size_P, bat_size_n, result_root_path):
        '''Parameters to compute semi-discrete Optimal Transport (OT)
        Args:
            h_P: Host vector (i.e. CPU vector) storing locations of target points with float type and of shape (num_P, dim).
            num_P: A positive interger indicating the number of target points (i.e. points the target discrete measure concentrates on).
            dim: A positive integer indicating the ambient dimension of OT problem.
            max_iter: A positive integer indicating the maximum steps the gradient descent would iterate.
            lr: A positive float number indicating the step length (i.e. learning rate) of the gradient descent algorithm.
            bat_size_P: Size of mini-batch of h_P that feeds to device (i.e. GPU). Positive integer.
            bat_size_n: Size of mini-batch of Monte-Carlo samples on device. The total number of MC samples used in each iteration is batch_size_n * num_bat.
        '''
        self.h_P = h_P
        self.num_P = num_P
        self.dim = dim
        self.max_iter = max_iter
        self.lr = lr
        self.bat_size_P = bat_size_P
        self.bat_size_n = bat_size_n
        self.result_root_path = result_root_path

        if num_P % bat_size_P != 0:
            sys.exit('Error: (num_P) is not a multiple of (bat_size_P)')
        
        self.num_bat_P = num_P // bat_size_P
        #!internal variables
        self.d_G_z = torch.empty(self.bat_size_n*self.dim, dtype=torch.float, device=device)
        self.d_volP = torch.empty((self.bat_size_n, self.dim), dtype=torch.float, device=device)
        self.d_h = torch.zeros(self.num_P, dtype=torch.float, device=device)
        self.d_delta_h = torch.zeros(self.num_P, dtype=torch.float, device=device)
        self.d_ind = torch.empty(self.bat_size_n, dtype=torch.long, device=device)
        self.d_ind_val = torch.empty(self.bat_size_n, dtype=torch.float, device=device)
        
        self.d_ind_val_argmax = torch.empty(self.bat_size_n, dtype=torch.long, device=device)
        self.d_tot_ind = torch.empty(self.bat_size_n, dtype=torch.long, device=device)
        self.d_tot_ind_val = torch.empty(self.bat_size_n, dtype=torch.float, device=device)
        self.d_g = torch.zeros(self.num_P, dtype=torch.float, device=device)
        self.d_g_sum = torch.zeros(self.num_P, dtype=torch.float, device=device)
        self.d_adam_m = torch.zeros(self.num_P, dtype=torch.float, device=device)
        self.d_adam_v = torch.zeros(self.num_P, dtype=torch.float, device=device)

        #!temp variables
        self.d_U = torch.empty((self.bat_size_P, self.bat_size_n), dtype=torch.float, device=device)
        self.d_temp_h = torch.empty(self.bat_size_P, dtype=torch.float, device=device)
        self.d_temp_P = torch.empty((self.bat_size_P, self.dim), dtype=torch.float, device=device)

        #!random number generator
        self.qrng = torch.quasirandom.SobolEngine(dimension=self.dim)

        # print('Allocated GPU memory: {}MB'.format(torch.cuda.memory_allocated()/1e6))
        # print('Cached memory: {}MB'.format(torch.cuda.memory_cached()/1e6))


    def pre_cal(self,count):
        '''Monte-Carlo sample generator.
        Args: 
            count: Index of MC mini-batch to generate in the current iteration step. Used to set the state of random number generator.
        Returns:
            self.d_volP: Generated mini-batch of MC samples on device (i.e. GPU) of shape (self.bat_size_n, dim).
        '''
        self.qrng.draw(self.bat_size_n,out=self.d_volP)
        self.d_volP.add_(-0.5)

    def cal_measure(self):
        '''Calculate the pushed-forward measure of current step. 
        '''
        self.d_tot_ind_val.fill_(-1e30)
        self.d_tot_ind.fill_(-1)
        i = 0     
        while i < self.num_P // self.bat_size_P:
            temp_P = self.h_P[i*self.bat_size_P:(i+1)*self.bat_size_P]
            temp_P = temp_P.view(temp_P.shape[0], -1)	
                
            '''U=PX+H'''
            # print(self.d_temp_P.shape, temp_P.shape)
            self.d_temp_h = self.d_h[i*self.bat_size_P:(i+1)*self.bat_size_P]
            self.d_temp_P.copy_(temp_P)
            torch.mm(self.d_temp_P, self.d_volP.t(),out=self.d_U)
            torch.add(self.d_U, self.d_temp_h.expand([self.bat_size_n, -1]).t(), out=self.d_U)
            '''compute max'''
            torch.max(self.d_U, 0, out=(self.d_ind_val, self.d_ind))
            '''add P id offset'''
            self.d_ind.add_(i*self.bat_size_P)
            '''store best value'''
            torch.max(torch.stack((self.d_tot_ind_val, self.d_ind_val)), 0, out=(self.d_tot_ind_val, self.d_ind_val_argmax))
            self.d_tot_ind = torch.stack((self.d_tot_ind, self.d_ind))[self.d_ind_val_argmax, torch.arange(self.bat_size_n)] 
            '''add step'''
            i = i+1
            
        '''calculate histogram'''
        self.d_g.copy_(torch.bincount(self.d_tot_ind, minlength=self.num_P))
        self.d_g.div_(self.bat_size_n)
        

    def update_h(self):
        '''Calculate the update step based on gradient'''
        self.d_g -= 1./self.num_P
        self.d_adam_m *= 0.9
        self.d_adam_m += 0.1*self.d_g
        self.d_adam_v *= 0.999
        self.d_adam_v += 0.001*torch.mul(self.d_g,self.d_g)
        torch.mul(torch.div(self.d_adam_m, torch.add(torch.sqrt(self.d_adam_v),1e-8)),-self.lr,out=self.d_delta_h)
        torch.add(self.d_h, self.d_delta_h, out=self.d_h)
        '''normalize h'''
        self.d_h -= torch.mean(self.d_h)


    def run_gd(self, last_step=0, num_bat=1):
        '''Gradient descent method. Update self.d_h to the optimal solution.
        Args:
            last_step: Iteration performed before the calling. Used when resuming the training. Default [0].
            num_bat: Starting number of mini-batch of Monte-Carlo samples. Value of num_bat will increase during iteration. Default [1].
                     total number of MC samples used in each iteration = self.batch_size_n * num_bat
        Returns:
            self.d_h: Optimal value of h (the variable to be optimized of the variational Energy).
        '''
        g_ratio = 1e20
        best_g_norm = 1e20
        curr_best_g_norm = 1e20
        steps = 0
        count_bad = 0
        dyn_num_bat_n = num_bat
        h_file_list = []
        m_file_list = []
        v_file_list = []

        while(steps <= self.max_iter):
            self.qrng.reset()
            self.d_g_sum.fill_(0.)
            for count in range(dyn_num_bat_n):
                self.pre_cal(count)
                self.cal_measure()
                torch.add(self.d_g_sum, self.d_g, out=self.d_g_sum)

            torch.div(self.d_g_sum, dyn_num_bat_n, out=self.d_g)			
            self.update_h()

            g_norm = torch.sqrt(torch.sum(torch.mul(self.d_g,self.d_g)))
            num_zero = torch.sum(self.d_g == -1./self.num_P)

            torch.abs(self.d_g, out=self.d_g)
            g_ratio = torch.max(self.d_g)*self.num_P
            
            # print('[{0}/{1}] Max absolute error ratio: {2:.3f}. g norm: {3:.6f}. num zero: {4:d}'.format(
            #     steps, self.max_iter, g_ratio, g_norm, num_zero))

            if g_norm < 2e-3:
                # torch.save(self.d_h, './h_final.pt')
                return

            h_id, h_file = load_last_file(self.result_root_path+'/h', '.pt')
            adam_m_id, m_file = load_last_file(self.result_root_path+'/adam_m', '.pt')
            adam_v_id, v_file = load_last_file(self.result_root_path+'/adam_v', '.pt')
            h_tmp, m_tmp, v_tmp = torch.load(h_file), torch.load(m_file), torch.load(v_file)
            h_tmp[:self.d_h.shape[0]] = self.d_h
            m_tmp[:self.d_adam_m.shape[0]] = self.d_adam_m
            v_tmp[:self.d_adam_v.shape[0]] = self.d_adam_v
            torch.save(h_tmp, self.result_root_path+'/h/{}.pt'.format(steps+last_step))
            torch.save(m_tmp, self.result_root_path+'/adam_m/{}.pt'.format(steps+last_step))
            torch.save(v_tmp, self.result_root_path+'/adam_v/{}.pt'.format(steps+last_step))
            h_file_list.append(self.result_root_path+'/h/{}.pt'.format(steps+last_step))
            m_file_list.append(self.result_root_path+'/adam_m/{}.pt'.format(steps+last_step))
            v_file_list.append(self.result_root_path+'/adam_v/{}.pt'.format(steps+last_step))
            if len(h_file_list)>5:
                if os.path.exists(h_file_list[0]):
                    os.remove(h_file_list[0])
                h_file_list.pop(0)
                if os.path.exists(v_file_list[0]):
                    os.remove(v_file_list[0])
                v_file_list.pop(0)
                if os.path.exists(m_file_list[0]):
                    os.remove(m_file_list[0])
                m_file_list.pop(0)

            if g_norm <= curr_best_g_norm:
                curr_best_g_norm = g_norm
                count_bad = 0
            else:
                count_bad += 1
            if count_bad > 30:
                dyn_num_bat_n *= 2
                # print('bat_size_n has increased to {}'.format(dyn_num_bat_n*self.bat_size_n))
                count_bad = 0
                curr_best_g_norm = 1e20

            steps += 1


    def set_h(self, h_tensor):
        # print(self.d_h.shape, h_tensor.shape)
        self.d_h.copy_(h_tensor[:self.d_h.shape[0]])

    def set_adam_m(self, adam_m_tensor):
        self.d_adam_m.copy_(adam_m_tensor[:self.d_adam_m.shape[0]])

    def set_adam_v(self, adam_v_tensor):
        self.d_adam_v.copy_(adam_v_tensor[:self.d_adam_v.shape[0]])
        
    def train_omt(self, num_bat=1):
        last_step = 0
        '''load last trained model parameters and last omt parameters'''
        h_id, h_file = load_last_file(self.result_root_path+'/h', '.pt')
        adam_m_id, m_file = load_last_file(self.result_root_path+'/adam_m', '.pt')
        adam_v_id, v_file = load_last_file(self.result_root_path+'/adam_v', '.pt')
        if h_id != None:
            if h_id != adam_m_id or h_id!= adam_v_id:
                sys.exit('Error: h, adam_m, adam_v file log does not match')
            elif h_id != None and adam_m_id != None and adam_v_id != None:
                last_step = h_id
                self.set_h(torch.load(h_file))
                self.set_adam_m(torch.load(m_file))
                self.set_adam_v(torch.load(v_file))

        '''run gradient descent'''
        self.run_gd(last_step=last_step, num_bat=num_bat)

        '''record result'''
        # np.savetxt('./h_final.csv',p_s.d_h.cpu().numpy(), delimiter=',')
        
    def gen_P(self, numX, output_P_gen, thresh=-1, topk=5, dissim=0.75, max_gen_samples=None):
        I_all = -torch.ones([topk, numX], dtype=torch.long).to(device)
        num_bat_x = numX//self.bat_size_n
        bat_size_x = min(numX, self.bat_size_n)
        for ii in range(max(num_bat_x, 1)):
            self.pre_cal(ii)
            self.cal_measure()
            _, I = torch.topk(self.d_U, topk, dim=0)
            for k in range(topk):
                I_all[k, ii*bat_size_x:(ii+1)*bat_size_x].copy_(I[k, 0:bat_size_x])
        I_all_2 = -torch.ones([2, (topk-1) * numX], dtype=torch.long).to(device)
        for ii in range(topk-1):
            I_all_2[0, ii * numX:(ii+1) * numX] = I_all[0,:]
            I_all_2[1, ii * numX:(ii+1) * numX] = I_all[ii + 1, :]
        I_all = I_all_2
        
        
        
        if torch.sum(I_all < 0) > 0:
            print('Error: numX is not a multiple of bat_size_n')

        '''compute angles'''
        P = self.h_P      
        nm = torch.cat([P, torch.ones(self.num_P,1).to(device)], dim=1)
        nm /= torch.norm(nm,dim=1).view(-1,1)
        cs = torch.sum(nm[I_all[0,:],:] * nm[I_all[1,:],:], 1) #element-wise multiplication
        cs = torch.min(torch.ones([cs.shape[0]]).to(device), cs)
        theta = torch.acos(cs)
        # pdb.set_trace()


        '''filter out generated samples with theta larger than threshold'''
        I_gen = I_all[:, theta <= thresh]
        I_gen, _ = torch.sort(I_gen, dim=0)
        # _, uni_gen_id = np.unique(I_gen.numpy(), return_index=True, axis=1)
        _, uni_gen_id = np.unique(I_gen[0,:].cpu().numpy(), return_index=True)
        np.random.shuffle(uni_gen_id)
        I_gen = I_gen[:, torch.from_numpy(uni_gen_id)]
        # pdb.set_trace()
        
        numGen = I_gen.shape[1]
        if max_gen_samples is not None:
            numGen = min(numGen, max_gen_samples)
        I_gen = I_gen[:,:numGen]
        print('OT successfully generated {} samples'.format(
            numGen))
        
        '''generate new features'''
        # rand_w = torch.rand([numGen,1])    
        rand_w = dissim * torch.ones([numGen,1]).to(device)
        P_gen = torch.mul(P[I_gen[0,:],:], 1 - rand_w) + torch.mul(P[I_gen[1,:],:], rand_w)


        if P_gen.shape[0] > 0:
            P.cpu().numpy()[I_gen[0,:],:] = P_gen.cpu().numpy()
            

        id_gen = I_gen[0,:].squeeze().cpu().numpy().astype(int)
        # print(f"P_gen:{P_gen.shape}, I_gen:{I_gen.shape}, P:{P.shape}, theta:{theta}")

        sio.savemat(output_P_gen, {'features':P.cpu().numpy(), 'ids':id_gen})
        

    
        
class OTBlock(nn.Module):
    def __init__(self, result_root_path="./ot_result",
                 max_iter=400, ot_lr=5e-2, topk=20, angle_threshold=0.7, rec_gen_distance=0.75,
                 num_gen_x_bat=3,
                #  max_gen_samples=32
                 ):
        super().__init__()
        self.max_iter = max_iter
        self.ot_lr = ot_lr
        '''args for generation'''
        self.topk = topk
        self.num_gen_x_bat = num_gen_x_bat #a multiple num of bat_size_n
        # self.max_gen_samples = max_gen_samples #max number of generated samples. Used to avoid out of memory error.
        self.angle_threshold = angle_threshold #angle threshold of OT generator ranging from [0,1]. See paper for details.
        self.rec_gen_distance = rec_gen_distance #dis-similarity between reconstructed samples and generated samples, ranging from [0,1] with smaller meaning more similar
        
        '''args for save'''
        os.makedirs(result_root_path, exist_ok=True)
        self.result_root_path = result_root_path
        self.selected_ot_model_path = os.path.join(result_root_path, 'h.pt')
        self.feature_save_path = os.path.join(result_root_path, 'features.pt') 
        self.gen_feature_path = os.path.join(result_root_path, 'output_P_gen.mat')

        
    def compute_ot(self, input_P, output_h, output_P_gen, mode='train'):
        '''args for omt'''
        TRAIN = False
        GENERATE = False
        if mode=='train':
            TRAIN = True
        elif mode=='generate':
            GENERATE = True
        else:
            assert(True, 'unrecogonized OT computation action: ' + mode)
            
        h_P = torch.load(input_P)
        h_P = h_P.view(h_P.shape[0], -1)	
        num_P = h_P.shape[0]
        dim_y = h_P.shape[1]
        bat_size_P = num_P
        if not TRAIN:
            self.maxIter = 0


        #crop h_P to fit bat_size_P
        h_P = h_P[0:num_P//bat_size_P*bat_size_P,:]
        num_P = h_P.shape[0]


        p_s = pyOMT_raw(h_P=h_P, num_P=num_P, dim=dim_y, max_iter=self.max_iter, lr=self.ot_lr, 
                        bat_size_P=bat_size_P, bat_size_n=bat_size_P, result_root_path=self.result_root_path)
        '''train omt'''
        if TRAIN:
            p_s.train_omt(bat_size_P)
            torch.save(p_s.d_h, output_h)
        else:
            p_s.set_h(torch.load(output_h))

        if GENERATE:
            '''generate new samples'''
            p_s.gen_P(self.num_gen_x_bat*bat_size_P, output_P_gen, thresh=self.angle_threshold, topk=self.topk, dissim=self.rec_gen_distance, max_gen_samples=bat_size_P)
        
        return True
            
        
    def process_feature(self, z):
        #extract feature
        feature_shape = z.shape
        features = z.squeeze().detach()
        torch.save(features, self.feature_save_path)
        
        #train OT
        if self.training:
            self.compute_ot(self.feature_save_path, self.selected_ot_model_path, self.gen_feature_path, mode='train')
  
        #generate feature via OT
        ot_model_load_path = self.selected_ot_model_path
        if not os.path.exists(self.selected_ot_model_path):
            for file in os.listdir(self.result_root_path+'/h/'):
                if fnmatch.fnmatch(file, '*.pt'):
                    ot_model_load_path = os.path.join(self.result_root_path+'/h/',file)
                    print('Successfully loaded OT model ' + ot_model_load_path)

        # print('Generating features with OT solver...')
        self.compute_ot(self.feature_save_path, self.selected_ot_model_path, self.gen_feature_path, mode='generate')
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 
        
        feature_dict = sio.loadmat(self.gen_feature_path)
        features = feature_dict['features']
        num_feature = features.shape[0]
            
        z = torch.from_numpy(features).to(device)
        z = z.view(num_feature, feature_shape[1], feature_shape[2], feature_shape[3])
        
        return z

            
    def forward(self, x):
        z = self.process_feature(x).type_as(x)
        z = x + (z - x).detach()
        return z

        
      

def clear_file_in_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def clear_temp_data():
    folder_names = ['./adam_m', './adam_v', 'h']
    for folder in folder_names:
        clear_file_in_folder(folder)

def load_last_file(path, file_ext):
    if not os.path.exists(path):
        os.makedirs(path)
        return None, None
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    file_ids = [(int(f.split('.')[0]), os.path.join(path,f)) for f in files]
    if not file_ids:
        return None, None
    else:
        last_f_id, last_f = max(file_ids, key=lambda item:item[0])
        # print('Last' + path + ': ', last_f_id)
        return last_f_id, last_f

