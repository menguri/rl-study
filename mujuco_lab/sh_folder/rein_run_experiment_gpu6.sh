#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu6 
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --job-name=UBAIJOB    
#SBATCH -o ./jupyter_output/jupyter.%N.%j.out  # STDOUT ( Standard Output ) 
#SBATCH -e ./jupyter_output/jupyter.%N.%j.err  # STDERR ( Standard Error ) 

echo "start at:" `date` # 접속한 날짜 표기
echo "node: $HOSTNAME" # 접속한 노드 번호 표기 
echo "jobid: $alsrn5051" # jobid 표기 

# GPU 환경을 이용하고 싶은 경우에만 해당! 그렇지 않은 경우 해당 명령어들은 지우셔도 무관합니다.
# module avail CUDA # CUDA 어떤 버전들이 설치되어있는지 확인하는 방법 
# module unload CUDA/11.2.2 # 기본적으로 탑재되어있는 쿠다버전은 unload. 
# module load cuda/11.8.0 # GPU를 사용하는 경우 CUDA 버전을 지정해 줄 수 있으며, UBAI 최신 CUDA 버전은 12.2.1입니다.
module python/3.11.2 # GPU를 사용하는 경우 CUDA 버전을 지정해 줄 수 있으며, UBAI 최신 CUDA 버전은 12.2.1입니다.

#!/bin/bash
# run_experiment.sh

# 프로젝트 루트 폴더로 이동 (이 스크립트가 sh_folder 안에 있다고 가정)
cd ..

# 사용할 config 파일 경로 (configs 폴더 내 default.yaml 사용)
CONFIG_PATH="./configs/REINFORCE.yaml"

# wandb 로그를 위해 현재 날짜와 시간 정보를 추가할 수도 있음
EXP_NAME="halfcheetah_REINFORCE_experiment_$(date +%Y%m%d_%H%M%S)"

echo "Starting training with config: $CONFIG_PATH"
python train.py --config $CONFIG_PATH --name $EXP_NAME


# echo "Training completed. Starting testing..."
# python test.py --config $CONFIG_PATH --name $EXP_NAME

echo "Experiment finished."
