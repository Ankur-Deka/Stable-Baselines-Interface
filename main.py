# -------------------- generic -------------------- #
import argparse, os, sys, time, numpy as np
# -------------------- gym -------------------- #
import gym
# -------------------- stable_baselines 3 -------------------- #
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback


class RunnerClass():
	def __init__(self, args):
		self.envName = args.envName
		self.numEnvs = args.numEnvs
		self.algo = args.algo
		self.timeSteps = args.timeSteps
		self.saveDir = args.saveDir

		# if self.algo in ['PPO','A2C']:
		self.env = make_vec_env(self.envName, n_envs=self.numEnvs)
		# else:
		# 	self.env = gym.make(self.envName)

	def train(self):
		# -------------------- walk through exisiting directories -------------------- #
		rootDir = self.saveDir
		if not os.path.exists(rootDir):
			os.makedirs(rootDir)
			runID = 0
		else:
			runs = [int(d.split('_')[-1]) for d in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir,d))]
			runID = max(runs)+1 if len(runs)>0 else 0

		# -------------------- save config and tensorboard -------------------- #
		path = os.path.join(rootDir,'run_{}'.format(runID))
		os.makedirs(path)
		config = vars(args)
		config['runID'] = runID 
		f = open(os.path.join(path,'config.txt'),'w')
		f.write(str(config))

		# -------------------- save config to common file -------------------- #
		f = open(os.path.join(rootDir,'all_config.txt'),'a')
		f.write('\n\n'+str(config))
		f.close()
		f.close()

		# -------------------- Instantiate the agent -------------------- #
		if self.algo == 'A2C':
			self.model = A2C('MlpPolicy', self.env, tensorboard_log=path, verbose=1)
		elif self.algo == 'PPO':
			self.model = PPO('MlpPolicy', self.env, tensorboard_log=path, verbose=1)
		elif self.algo == 'SAC':
			self.model = SAC('MlpPolicy', self.env, tensorboard_log=path, verbose=1)
		elif self.algo == 'TD3':
			self.model = TD3('MlpPolicy', self.env, tensorboard_log=path, verbose=1)


		# --------------------Train -------------------- #
		print('Starting Training')
		tStart = time.time()
		# Train the agent
		# self.model.learn(total_timesteps=int(self.timeSteps), callback=self.callback)

		self.model.learn(total_timesteps=int(self.timeSteps))
		
		# Save the agent
		saveDir = os.path.join(path, 'trained_model')
		self.model.save(saveDir)

		tEnd = time.time()
		print('Training finished in {} seconds'.format(tEnd-tStart))

	# def test(self):
	# 	rootDir = self.saveDir
	# 	print('Loading Run {}, Checkpoint {}'.format(self.loadRun, self.loadCkpt))
	# 	path = os.path.join(rootDir, 'run_'+str(self.loadRun), 'epoch_'+str(self.loadCkpt)) 
	# 	self.adversaryModel.load_state_dict(torch.load(path))
	# 	self.adversaryModel.eval()

	# 	print('Starting Testing')
	# 	tStart = time.time()
	# 	loss, testLoss, c = 0, 0, 0
	# 	for X,Y in self.testLoader:
	# 		with torch.no_grad():
	# 			outputs = self.adversaryModel(X.to(self.device))
	# 			loss = self.lossFunc(outputs.to(self.device), Y.to(self.device), self.criterion)
	# 			testLoss += loss.data
	# 			c += 1
	# 	testLoss /= c
	# 	print('Test loss {}'.format(testLoss))
		
	# 	tEnd = time.time()
	# 	print('Testing finished in {} seconds'.format(tEnd-tStart))

	# 	# -------------------- visualize -------------------- #
	# 	if self.visualize:
	# 		figDir = os.path.join(rootDir, 'run_'+str(self.loadRun), 'figures')
	# 		paths = [os.path.join(figDir, path) for path in ['uniform_viz', 'leader_viz']]
	# 		for path in paths:
	# 			if not os.path.exists(path):
	# 				os.makedirs(path)
			
	# 		for i, (traj, leader_id) in enumerate(self.testDataset):
	# 			leader_id = leader_id[0].numpy()
	# 			with torch.no_grad():
	# 				out = self.adversaryModel(traj.to(self.device).view(1,-1,self.obsDim*self.nAgents)).detach().cpu().numpy().reshape(-1,self.nAgents)
	# 				out = np.argmax(out, axis = 1)
					
	# 			plot_trajectories(traj, leader_id, initSteps = self.initSteps, pred = out, leader_viz = False, fname = os.path.join(paths[0],'uniform_viz_{}'.format(i)))
	# 			plot_trajectories(traj, leader_id, initSteps = self.initSteps, pred = out, leader_viz = True, fname = os.path.join(paths[1],'leader_viz_{}'.format(i)))


if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Arguments for adversary training')
	# -------------------- environment configuration -------------------- #
	parser.add_argument('--mode', type=str, default='train', help='{Train, Test}')
	parser.add_argument('--envName', type=str, default='CartPole-v0', help='Gym env ID')
	parser.add_argument('--numEnvs', type=int, default=1, help='No. of parallel environments')
	parser.add_argument('--algo', type=str, default='PPO', help='RL algorithm')
	parser.add_argument('--timeSteps', type=int, default=1e6, help='Env interaction time steps')
	parser.add_argument('--saveDir', type=str, default='./runs', help='Save directory')
	# -------------------- parse all the arguments -------------------- #
	args = parser.parse_args()
	# -------------------- train/test -------------------- #
	myRunner = RunnerClass(args)
	if args.mode == 'train':
		myRunner.train()
	# elif args.mode == 'test':
	# 	myRunner.test()


