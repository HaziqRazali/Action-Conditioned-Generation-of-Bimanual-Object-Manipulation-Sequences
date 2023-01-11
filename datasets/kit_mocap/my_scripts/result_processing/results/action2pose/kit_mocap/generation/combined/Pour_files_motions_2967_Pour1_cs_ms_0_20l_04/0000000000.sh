/home/haziq/MMMTools/build/bin/MMMMotionConverter --inputMotion "/home/haziq/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_mocap/my_scripts/result_processing/results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.xml" --motionName pred_1480 --converterConfigFile "/home/haziq/MMMTools/data/Model/Winter/NloptConverterVicon2MMM_WinterConfig.xml" --outputModelFile "/home/haziq/MMMTools/data/Model/Winter/mmm.xml" --outputModelProcessorConfigFile "/home/haziq/MMMTools/data/Model/Winter/config/1480.xml" --outputMotion "/home/haziq/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_mocap/my_scripts/result_processing/results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.xml" &&\
sleep 5 &&\
/home/haziq/MMMTools/build/bin/MMMMotionConverter --inputMotion "/home/haziq/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_mocap/my_scripts/result_processing/results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.xml" --motionName pred_cup_small --converterConfigFile "/home/haziq/MMMTools/data/Model/Objects/cup_small/NloptConverterVicon2MMM_CupSmallConfig.xml" --outputModelFile "/home/haziq/MMMTools/data/Model/Objects/cup_small/cup_small.xml" --outputMotion "/home/haziq/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_mocap/my_scripts/result_processing/results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.xml" &&\
sleep 2 &&\
/home/haziq/MMMTools/build/bin/MMMMotionConverter --inputMotion "/home/haziq/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_mocap/my_scripts/result_processing/results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.xml" --motionName pred_milk_small --converterConfigFile "/home/haziq/MMMTools/data/Model/Objects/milk_small/NloptConverterVicon2MMM_MilkSmallConfig.xml" --outputModelFile "/home/haziq/MMMTools/data/Model/Objects/milk_small/milk_small.xml" --outputMotion "/home/haziq/Action-Conditioned-Generation-of-Bimanual-Object-Manipulation/datasets/kit_mocap/my_scripts/result_processing/results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.xml" &&\
sleep 2 &&\
echo "results//action2pose/kit_mocap/generation/combined/Pour_files_motions_2967_Pour1_cs_ms_0_20l_04/0000000000.sh DONE"