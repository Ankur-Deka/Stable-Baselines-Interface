This is a template script to run Stable-Baselines3 on different environments

## Requirements
1. Install stable-baselines3 from here: https://github.com/Ankur-Deka/stable-baselines3
   This is because I added tensorboard logging feature. I will likely remove this requirement once the main stable-baselines3 repo integrates tensorboard.

## Features
1. Support for tensorboard
1. Easy interface to change configuration of env, algorithm, etc
1. Automatically creates new folder for every run and stores configurations in a single file