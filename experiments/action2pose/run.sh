
# export LD_LIBRARY_PATH="/home_nfs/haziq/cenvs/ptg/lib/:$LD_LIBRARY_PATH"

# # # # # # # # #
# kit mocap     # 
# - generation  #
# # # # # # # # #

# train
CUDA_VISIBLE_DEVICES=1 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_mocap/generation/combined.ini"

# test
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/generation/combined.ini" seed=0 batch_size=2 teacher_force_ratio=0.0

# # # # # # # # # #
# kit mocap       # 
# - segmentation  #
# # # # # # # # # #

# train
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/grnn.ini" &\
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/gtcn.ini" &\
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/rnn.ini" &\
wait

# test on ground truth
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/grnn.ini" &\
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/gtcn.ini" &\
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/rnn.ini" &\
wait

# test on generated data
# - The generative models were trained at time_step_size=0.1 whereas segmentation models at time_step_size=0.2. The step size here should therefore be 2
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/grnn.ini" \
data_loader="dataloaders.action2pose.kit_mocap_generative" \
data_root="~/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results/action2pose/kit_mocap/generation" \
data_name="combined" \
result_name="kit_mocap/segmentation/fake_grnn" \
time_step_size=2 \
batch_size=2

CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/gtcn.ini" \
data_loader="dataloaders.action2pose.kit_mocap_generative" \
data_root="~/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results/action2pose/kit_mocap/generation" \
data_name="combined" \
result_name="kit_mocap/segmentation/fake_gtcn" \
time_step_size=2 \
batch_size=2

CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_mocap/segmentation/rnn.ini" \
data_loader="dataloaders.action2pose.kit_mocap_generative" \
data_root="~/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results/action2pose/kit_mocap/generation" \
data_name="combined" \
result_name="kit_mocap/segmentation/fake_rnn" \
time_step_size=2 \
batch_size=2

# # # # # # # # #
# kit rgbd      #
# - generation  #
# # # # # # # # #

# train
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_rgbd/generation/combined.ini"

# test
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/generation/combined.ini" seed=0 batch_size=2 teacher_force_ratio=0.0

# # # # # # # # # #
# kit rgbd        #
# - segmentation  #
# # # # # # # # # #

# train
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/grnn.ini"
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/gtcn.ini"
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/train.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/rnn.ini"

# test on ground truth
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/grnn.ini"
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/gtcn.ini"
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/rnn.ini"

# test on generated data
# - The generative models were trained at time_step_size=5 whereas segmentation models at time_step_size=10. The step size here should therefore be 2
CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/grnn.ini" \
data_loader="dataloaders.action2pose.kit_rgbd_generative" \
data_root="~/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results/action2pose/kit_rgbd/generation" \
data_name="combined" \
result_name="kit_rgbd/segmentation/fake_grnn" \
time_step_size=2 \
batch_size=2

CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/gtcn.ini" \
data_loader="dataloaders.action2pose.kit_rgbd_generative" \
data_root="~/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results/action2pose/kit_rgbd/generation" \
data_name="combined" \
result_name="kit_rgbd/segmentation/fake_gtcn" \
time_step_size=2 \
batch_size=2

CUDA_VISIBLE_DEVICES=0 python "$HOME/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/test.py" --args="args.action2pose" --config_file="kit_rgbd/segmentation/rnn.ini" \
data_loader="dataloaders.action2pose.kit_rgbd_generative" \
data_root="~/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/results/action2pose/kit_rgbd/generation" \
data_name="combined" \
result_name="kit_rgbd/segmentation/fake_rnn" \
time_step_size=2 \
batch_size=2