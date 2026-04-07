#
# run for seed 0, 10, 20
# for env_name in\
#       OfflinePointPush1Gymnasium-v0\
#       OfflinePointPush2Gymnasium-v0\
#       OfflineHopperVelocityGymnasium-v1\
#       OfflinePointCircle2Gymnasium-v0\
#       OfflineCarButton1Gymnasium-v0\
#       OfflineCarButton2Gymnasium-v0\
#       OfflinePointButton1Gymnasium-v0\
#       OfflinePointButton2Gymnasium-v0\
#       OfflineHalfCheetahVelocityGymnasium-v1\
#       OfflineWalker2dVelocityGymnasium-v1\
#       OfflineAntVelocityGymnasium-v1\
#       OfflinePointCircle1Gymnasium-v0\
#       OfflinePointGoal1Gymnasium-v0\
#       OfflinePointGoal2Gymnasium-v0\
#       OfflineCarCircle1Gymnasium-v0\
#       OfflineCarCircle2Gymnasium-v0\
#       OfflineCarGoal1Gymnasium-v0\
#       OfflineCarGoal2Gymnasium-v0\
#       OfflineCarPush1Gymnasium-v0\
#       OfflineCarPush2Gymnasium-v0\
#       OfflineSwimmerVelocityGymnasium-v1\
#     ; do
#   for seed in 0 10 20; do 
#     echo $env_name seed $seed
#     python ./bcrl_stochastic.py env_name=$env_name seed=$seed
#   done
# done
#
#
# for env_name in\
#       OfflineCarRun-v0\
#       OfflineBallRun-v0\
#       OfflineDroneRun-v0\
#       OfflineAntRun-v0\
#       OfflineBallCircle-v0\
#       OfflineCarCircle-v0\
#       OfflineDroneCircle-v0\
#       OfflineAntCircle-v0\
#     ; do
#   for seed in 0 10 20; do 
#     echo $env_name seed $seed
#     python ./bcrl_stochastic.py env_name=$env_name seed=$seed
#   done
# done

for env_name in\
       OfflineMetadrive-hardmean-v0\
    ; do
  for seed in 0 10 20; do 
    echo $env_name seed $seed
    USE_GYMNASIUM=0 python ./bcrl_det.py env_name=$env_name seed=$seed
  done
done
