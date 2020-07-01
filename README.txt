We recommend Ubuntu or MacOS as the operating systems for all projects in this class. However, the following installation instructions "should" work for Windows as well.

1. Go to (https://www.anaconda.com/download/) and install the Python 3 version of Anaconda. 

2. Open a new terminal and run the following commands to create a new conda environment (https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html):

conda create -n cs348e-p5 python=3.6

3. Activate & enter the new environment you just creared:

conda activate cs348e-p5

4. Inside the new environment (you should see "(cs348e-p5)" preceeding your ternimal prompt), and inside the project directory: 

pip install -r requirements.txt

5. Install pytorch (should work with or without cuda):

#conda install pytorch -c soumith
conda install pytorch cudatoolkit=10.2 -c pytorch

6. go to "baselines" subdirectory: 

cd baselines

7. Install baselines for RL utilities: 

pip install -e .

8. cd ..

10. See if this runs without problem (You might see EndOfFile Error, that is normal when the program exits):

python main.py --env-name "HumanoidSwimmerEnv-v1" --algo ppo --use-gae --log-interval 2 --num-steps 4800 --num-processes 4  --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 16 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 19600 --use-linear-lr-decay --use-proper-time-limits --clip-param 0.2 --save-dir ./tmp --seed 20000

11. After finishing the project, to deactivate an active environment, use:

conda deactivate
# conda env remove -n ENV_NAME
