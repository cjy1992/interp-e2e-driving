# Interpretable End-to-end Autonomous Driving
[[Project webpage]](https://sites.google.com/berkeley.edu/interp-e2e/) [[Paper]](https://arxiv.org/abs/2001.08726)

This repo contains code for [Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning](https://arxiv.org/abs/2001.08726). This work introduces an end-to-end autonomous driving approach which is able to handle complex urban scenarios, and at the same time generates a semantic birdeye mask interpreting how the learned agents reasons about the environment. This repo also provides implementation of popular model-free reinforcement learning algorithms (DQN, DDPG, TD3, SAC) on the urban autonomous driving problem in CARLA simulator. All of the algorithms take raw camera and lidar sensor inputs.

## System Requirements
- Ubuntu 16.04
- NVIDIA GPU with CUDA 10. See [GPU guide](https://www.tensorflow.org/install/gpu) for TensorFlow.

## Installation
1. Setup conda environment
```
$ conda create -n env_name python=3.6
$ conda activate env_name
```

2. Install the gym-carla wrapper following the installation steps 2-4 in [https://github.com/cjy1992/gym-carla](https://github.com/cjy1992/gym-carla).

3. Clone this git repo to an appropriate folder
```
$ git clone https://github.com/cjy1992/interp-e2e-driving.git
```

4. Enter the root folder of this repo and install the packages:
```
$ pip install -r requirements.txt
$ pip install -e .
```

## Usage
1. Enter the CARLA simulator folder and launch the CARLA server by:
```
$ ./CarlaUE4.sh -windowed -carla-port=2000
```
You can use ```Alt+F1``` to get back your mouse control.
Or you can run in non-display mode by:
```
$ DISPLAY= ./CarlaUE4.sh -opengl -carla-port=2000
```
It might take several seconds to finish launching the simulator.

2. Enter the root folder of this repo and run:
```
$ ./run_train_eval.sh
```
It will then connect to the CARLA simulator, collect exploration data, train and evaluate the agent. Parameters are stored in ```params.gin```. Set train_eval.agent_name from ['latent_sac', 'dqn', 'ddpg', 'td3', 'sac'] to choose the reinforcement learning algorithm.

3. Run `tensorboard --logdir logs` and open http://localhost:6006 to view training and evaluation information.

## Trouble Shootings
1. If out of system memory, change the parameter ```replay_buffer_capacity``` and ```initial_collect_steps``` the function ```tran_eval``` smaller.

2. If out of CUDA memory, set parameter ```model_batch_size``` or ```sequence_length``` of the function ```tran_eval``` smaller.

## Citation
If you find this useful for your research, please use the following.

```
@article{chen2020interpretable,
  title={Interpretable End-to-end Urban Autonomous Driving with Latent Deep Reinforcement Learning},
  author={Chen, Jianyu and Li, Shengbo Eben and Tomizuka, Masayoshi},
  journal={arXiv preprint arXiv:2001.08726},
  year={2020}
}
```
