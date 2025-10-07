#---------------------------------------------------
# Imports
#---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
from collections import OrderedDict
from matplotlib import pyplot as plt
import copy

cfg = {
    'VGG5' : [64, 'A', 128, 128, 'A'],
    'VGG9':  [64, 'A', 128, 256, 'A', 256, 512, 'A', 512, 'A', 512],
    'VGG11': [64, 'A', 128, 256, 256, 'A', 512, 512, 512, 'A', 512, 512],
    'VGG13': [64, 64, 'A', 128, 128, 'A', 256, 256, 'A', 512, 512, 512, 'A', 512],
    'VGG16': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 'A', 512, 512, 512, 'A', 512, 512, 512],
    'VGG19': [64, 64, 'A', 128, 128, 'A', 256, 256, 256, 256, 'A', 512, 512, 512, 512, 'A', 512, 512, 512, 512]
}
class PoissonGenerator(nn.Module):
	
	def __init__(self):
		super().__init__()

	def forward(self,input):
		
		out = torch.mul(torch.le(torch.rand_like(input), torch.abs(input)*1.0).float(),torch.sign(input))
		return out

class signed_if_encoder(torch.nn.Module):
    def __init__(self, step_num, max_rate = 1.0, threshold=1.0, reset_mode='soft'):
        '''
        :param step_num:
        :param max_rate:
        :param threshold:
        :param reset_mode: 'soft' or 'hard'
        '''

        super().__init__()
        self.step_num = step_num
        self.reset_mode = reset_mode
        self.max_rate = max_rate
        self.threshold = threshold

        self.threshold = torch.nn.Parameter(torch.tensor(1.0))

        self.threshold.requires_grad = False

    def forward(self, x):
        """
        :param x: [batch, c, h, w], assume image is scaled in range [0,1]
        :return: shape [b,c,h,w,step_num]
        """

        spikes = []

        v = torch.zeros_like(x)
        for i in range(self.step_num):

            v = v + x*self.max_rate

            spike = torch.zeros_like(v)

            positive = v >= self.threshold
            negative = v <= -self.threshold

            spike[positive] = 1.0
            spike[negative] = -1.0

            if self.reset_mode == 'soft':
                v[positive] = v[positive] - self.threshold
                v[negative] = v[negative] + self.threshold
            else:
                v[positive] = 0.0
                v[negative] = 0.0

            spikes += [spike]

        return torch.stack(spikes,dim=-1)


class threshold_rect(torch.autograd.Function):
	"""
    """

	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return input.gt(1.0).float()

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		temp = abs(input - 1.0) < 0.5
		return grad_input * temp.float()


class threshold_logistic(torch.autograd.Function):
	"""
    heaviside step threshold function
    """

	@staticmethod
	def forward(ctx, input):
		"""
        """
		# a asjusts the max value of gradient and sharpness
		# b moves gradient to left (positive) or right (negative)
		a = 4  # set to 4 as it sets max value to 1
		b = -1
		ctx.save_for_backward(input)
		ctx.a = a
		ctx.b = b

		output = input.gt(1.0).float()

		return output

	@staticmethod
	def backward(ctx, grad_output):
		"""
        """

		# a = time.time()
		input, = ctx.saved_tensors
		a = ctx.a
		b = ctx.b

		x = input
		logictic_ = a * torch.exp(-a * (b + x)) / ((torch.exp(-a * (b + x)) + 1) ** 2)

		grad = logictic_ * grad_output

		return grad, None


class if_encoder(torch.nn.Module):
	def __init__(self, step_num, max_rate=1.0, threshold=1.0, reset_mode='soft', forward_choice=1):
		'''
        :param step_num:
        :param max_rate:
        :param threshold:
        :param reset_mode: 'soft' or 'hard'
        :param forward_choice: select different forward functions
        '''

		super().__init__()
		self.step_num = step_num
		self.reset_mode = reset_mode
		self.max_rate = max_rate
		self.threshold = threshold

		self.threshold = torch.nn.Parameter(torch.tensor(1.0))

		self.threshold.requires_grad = False

		# 1 does not support bp
		# 2 and 3 support bp
		if forward_choice == 1:
			self.forward_func = self.forward_1
		elif forward_choice == 2:
			self.forward_func = self.forward_2
		elif forward_choice == 3:
			self.forward_func = self.forward_3

	def forward(self, x):

		return self.forward_func(x)

	def forward_1(self, x):
		"""
        no gradient approximation
        :param x: [batch, c, h, w], assume image is scaled in range [0,1]
        :return: shape [b,c,h,w,step_num]
        """

		spikes = []

		v = torch.zeros_like(x)
		for i in range(self.step_num):

			v = v + x * self.max_rate

			spike = v.clone()

			spike[spike < self.threshold] = 0.0
			spike[spike >= self.threshold] = 1.0

			if self.reset_mode == 'soft':
				v[v >= self.threshold] = v[v >= self.threshold] - self.threshold
			else:
				v[v >= self.threshold] = 0.0

			spikes += [spike]

		return torch.stack(spikes, dim=-1)

	def forward_2(self, x):
		"""
        gradient approximation same as stbp
        """
		spikes = []

		v = torch.zeros_like(x)
		spike = torch.zeros_like(x)

		for i in range(self.step_num):

			if self.reset_mode == 'soft':
				v = v + x * self.max_rate - spike
			else:
				v = (1 - spike) * v + x * self.max_rate

			threshold_function = threshold_rect.apply
			spike = threshold_function(v)

			spikes += [spike]

		return torch.stack(spikes, dim=-1)

	def forward_3(self, x):
		"""
        use logistic function to approximate gradient
        :param x: [batch, c, h, w], assume image is scaled in range [0,1]
        :return: shape [b,c,h,w,step_num]
        """

		spikes = []

		v = torch.zeros_like(x)
		for i in range(self.step_num):

			v = v + x * self.max_rate

			threshold_function = threshold_logistic.apply
			spike = threshold_function(v)

			if self.reset_mode == 'soft':
				v = v - spike
			else:
				v = (1 - spike) * v

			spikes += [spike]

		return torch.stack(spikes, dim=-1)


# class if_encoder(torch.nn.Module):
#     def __init__(self, step_num, max_rate = 1.0, threshold=1.0, reset_mode='soft'):
#         '''
#         :param step_num:
#         :param max_rate:
#         :param threshold:
#         :param reset_mode: 'soft' or 'hard'
#         '''
#
#         super().__init__()
#         self.step_num = step_num
#         self.reset_mode = reset_mode
#         self.max_rate = max_rate
#         self.threshold = threshold
#
#         self.threshold = torch.nn.Parameter(torch.tensor(1.0))
#
#         self.threshold.requires_grad = True
#
#     def forward(self, x):
#         """
#         :param x: [batch, c, h, w], assume image is scaled in range [0,1]
#         :return: shape [b,c,h,w,step_num]
#         """
#
#         spikes = []
#
#         v = torch.zeros_like(x)
#         for i in range(self.step_num):
#
#             v += x
#
#             spike = v.clone()
#
#             spike[spike < self.threshold] = 0.0
#             spike[spike >= self.threshold] = 1.0
#
#             if self.reset_mode == 'soft':
#                 v[v >= self.threshold] = v[v >= self.threshold] - self.threshold
#             else:
#                 v[v >= self.threshold] = 0.0
#
#             spikes += [spike]
#
#         return torch.stack(spikes,dim=-1)


class STDB(torch.autograd.Function):

	alpha 	= ''
	beta 	= ''
    
	@staticmethod
	def forward(ctx, input, last_spike, alpha, beta):
        
		ctx.save_for_backward(last_spike,alpha,beta)
		out = torch.zeros_like(input).cuda()
		out[input > 0] = 1.0
		return out

	@staticmethod
	def backward(ctx, grad_output):
	    		
		last_spike, alpha, beta = ctx.saved_tensors
		# last_spike = ctx.saved_tensors
		grad_input = grad_output.clone()
		# grad = STDB.alpha * torch.exp(-1*last_spike)**STDB.beta
		grad = alpha * torch.exp(-1*last_spike)**beta
		return grad*grad_input, None, alpha, beta

class LinearSpike(torch.autograd.Function):
    """
    Here we use the piecewise-linear surrogate gradient as was done
    in Bellec et al. (2018).
    """
    gamma = 0.3 # Controls the dampening of the piecewise-linear surrogate gradient

    @staticmethod
    def forward(ctx, input, last_spike):
        
        ctx.save_for_backward(input)
        out = torch.zeros_like(input).cuda()
        out[input > 0] = 1.0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        
        input,     = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad       = LinearSpike.gamma*F.threshold(1.0-torch.abs(input), 0, 0)
        return grad*grad_input, None

class VGG_SNN_STDB(nn.Module):

	def __init__(self, vgg_name, activation='Linear', labels=10, timesteps=100, leak=1.0, default_threshold = 1.0, alpha=0.3, beta=0.01, dropout=0.2, kernel_size=3, dataset='CIFAR10'):
		super().__init__()
		
		self.vgg_name 		= vgg_name
		if activation == 'Linear':
			self.act_func 	= LinearSpike.apply
		elif activation == 'STDB':
			self.act_func	= STDB.apply
		self.labels 		= labels
		self.timesteps 		= timesteps
		self.leak 	 		= torch.tensor(leak)
		# STDB.alpha 		 	= alpha
		self.alpha 		 	= nn.Parameter(torch.tensor(alpha))
		# STDB.beta 			= beta
		self.beta 			= nn.Parameter(torch.tensor(beta))
		self.dropout 		= dropout
		self.kernel_size 	= kernel_size
		self.dataset 		= dataset
		# self.input_layer 	= PoissonGenerator()
		# self.encoder = signed_if_encoder(timesteps, 1.0, 1.0, 'soft')
		self.encoder = if_encoder(timesteps, 1.0, 1.0, 'soft', forward_choice=2)
		self.threshold 		= {}
		self.mem 			= {}
		self.mask 			= {}
		self.spike 			= {}
		
		self.features, self.classifier = self._make_layers(cfg[self.vgg_name])
		
		self._initialize_weights2()

		for l in range(len(self.features)):
			if isinstance(self.features[l], nn.Conv2d):
				self.threshold[l] 	= torch.tensor(default_threshold)
				
		prev = len(self.features)
		for l in range(len(self.classifier)-1):
			if isinstance(self.classifier[l], nn.Linear):
				self.threshold[prev+l] 	= torch.tensor(default_threshold)

	def _initialize_weights2(self):
		for m in self.modules():
            
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				n = m.weight.size(1)
				m.weight.data.normal_(0, 0.01)
				if m.bias is not None:
					m.bias.data.zero_()

	def threshold_update(self, scaling_factor=1.0, thresholds=[]):

		# Initialize thresholds
		self.scaling_factor = scaling_factor
		
		for pos in range(len(self.features)):
			if isinstance(self.features[pos], nn.Conv2d):
				if thresholds:
					self.threshold[pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
				#print('\t Layer{} : {:.2f}'.format(pos, self.threshold[pos]))

		prev = len(self.features)

		for pos in range(len(self.classifier)-1):
			if isinstance(self.classifier[pos], nn.Linear):
				if thresholds:
					self.threshold[prev+pos] = torch.tensor(thresholds.pop(0)*self.scaling_factor)
				#print('\t Layer{} : {:.2f}'.format(prev+pos, self.threshold[prev+pos]))

	def _make_layers(self, cfg):
		layers 		= []
		if self.dataset =='MNIST':
			in_channels = 1
		else:
			in_channels = 3

		for x in (cfg):
			stride = 1
						
			if x == 'A':
				layers.pop()
				layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
			
			else:
				layers += [nn.Conv2d(in_channels, x, kernel_size=self.kernel_size, padding=(self.kernel_size-1)//2, stride=stride, bias=False),
							nn.ReLU(inplace=True)
							]
				layers += [nn.Dropout(self.dropout)]
				in_channels = x
		
		features = nn.Sequential(*layers)
		
		layers = []
		if self.vgg_name == 'VGG11' and self.dataset == 'CIFAR100':
			layers += [nn.Linear(8192, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024, self.labels, bias=False)]
		elif self.vgg_name == 'VGG5' and self.dataset != 'MNIST':
			# layers += [nn.Linear(512*4*4, 4096, bias=False)]
			# layers += [nn.ReLU(inplace=True)]
			# layers += [nn.Dropout(0.5)]
			# layers += [nn.Linear(4096, 4096, bias=False)]
			# layers += [nn.ReLU(inplace=True)]
			# layers += [nn.Dropout(0.5)]
			# layers += [nn.Linear(4096, self.labels, bias=False)]
			layers += [nn.Linear(512*4*4, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024, 1024, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(1024, self.labels, bias=False)]
		
		elif self.vgg_name != 'VGG5' and self.dataset != 'MNIST':
			layers += [nn.Linear(512*2*2, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
		
		elif self.vgg_name == 'VGG5' and self.dataset == 'MNIST':
			layers += [nn.Linear(128*7*7, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, self.labels, bias=False)]

		elif self.vgg_name != 'VGG5' and self.dataset == 'MNIST':
			layers += [nn.Linear(512*1*1, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, 4096, bias=False)]
			layers += [nn.ReLU(inplace=True)]
			layers += [nn.Dropout(0.5)]
			layers += [nn.Linear(4096, self.labels, bias=False)]
	

		classifer = nn.Sequential(*layers)
		return (features, classifer)

	def network_update(self, timesteps, leak):
		self.timesteps 	= timesteps
		self.leak 	 	= torch.tensor(leak)
	
	def neuron_init(self, x):
		self.batch_size = x.size(0)
		self.width 		= x.size(2)
		self.height 	= x.size(3)

		self.mem 		= {}
		self.mask 		= {}
		self.spike 		= {}			
				
		for l in range(len(self.features)):
								
			if isinstance(self.features[l], nn.Conv2d):
				self.mem[l] 		= torch.zeros(self.batch_size, self.features[l].out_channels, self.width, self.height).cuda()
			
			elif isinstance(self.features[l], nn.Dropout):
				self.mask[l] = self.features[l](torch.ones(self.mem[l-2].shape)).cuda()

			elif isinstance(self.features[l], nn.AvgPool2d):
				self.width = self.width//self.features[l].kernel_size
				self.height = self.height//self.features[l].kernel_size
		
		prev = len(self.features)

		for l in range(len(self.classifier)):
			
			if isinstance(self.classifier[l], nn.Linear):
				self.mem[prev+l] 		= torch.zeros(self.batch_size, self.classifier[l].out_features).cuda()
			
			elif isinstance(self.classifier[l], nn.Dropout):
				self.mask[prev+l] = self.classifier[l](torch.ones(self.mem[prev+l-2].shape)).cuda()
				
		self.spike = copy.deepcopy(self.mem)
		for key, values in self.spike.items():
			for value in values:
				value.fill_(-1000)

	def forward(self, x, find_max_mem=False, max_mem_layer=0):
		
		self.neuron_init(x)
		# self.cuda()
		max_mem=0.0
		input_spike_train = self.encoder(x)
		input_spike_train_unbind = input_spike_train.unbind(dim=-1)
		for t in range(self.timesteps):
			# out_prev = self.input_layer(x)
			out_prev = input_spike_train_unbind[t]

			for l in range(len(self.features)):
				
				if isinstance(self.features[l], (nn.Conv2d)):
					
					if find_max_mem and l==max_mem_layer:
						if (self.features[l](out_prev)).max()>max_mem:
							max_mem = (self.features[l](out_prev)).max()
						break

					mem_thr 		= (self.mem[l]/self.threshold[l]) - 1.0
					out 			= self.act_func(mem_thr, (t-1-self.spike[l]),self.alpha,self.beta)
					rst 			= self.threshold[l]* (mem_thr>0).float()
					self.spike[l] 	= self.spike[l].masked_fill(out.bool(),t-1)
					self.mem[l] 	= self.leak*self.mem[l] + self.features[l](out_prev) - rst
					out_prev  		= out.clone()

				elif isinstance(self.features[l], nn.AvgPool2d):
					out_prev 		= self.features[l](out_prev)
				
				elif isinstance(self.features[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[l]
			
			if find_max_mem and max_mem_layer<len(self.features):
				continue

			out_prev       	= out_prev.reshape(self.batch_size, -1)
			prev = len(self.features)
			
			for l in range(len(self.classifier)-1):
													
				if isinstance(self.classifier[l], (nn.Linear)):
					
					if find_max_mem and (prev+l)==max_mem_layer:
						if (self.classifier[l](out_prev)).max()>max_mem:
							max_mem = (self.classifier[l](out_prev)).max()
						break

					mem_thr 			= (self.mem[prev+l]/self.threshold[prev+l]) - 1.0
					out 				= self.act_func(mem_thr, (t-1-self.spike[prev+l]),self.alpha,self.beta)
					rst 				= self.threshold[prev+l] * (mem_thr>0).float()
					self.spike[prev+l] 	= self.spike[prev+l].masked_fill(out.bool(),t-1)
					
					self.mem[prev+l] 	= self.leak*self.mem[prev+l] + self.classifier[l](out_prev) - rst
					out_prev  		= out.clone()

				elif isinstance(self.classifier[l], nn.Dropout):
					out_prev 		= out_prev * self.mask[prev+l]
			
			# Compute the classification layer outputs
			if not find_max_mem:
				self.mem[prev+l+1] 		= self.mem[prev+l+1] + self.classifier[l+1](out_prev)
		if find_max_mem:
			return max_mem

		return self.mem[prev+l+1].softmax(1)
		# return self.mem[prev+l+1]



