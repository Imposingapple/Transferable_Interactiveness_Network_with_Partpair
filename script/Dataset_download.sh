#!/bin/bash

# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

## ---------------V-COCO Dataset------------------
#echo "Downloading V-COCO Dataset"
#git clone --recursive https://github.com/s-gupta/v-coco.git Data/v-coco/
#cd Data/v-coco/coco
#
#URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
#URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip
#URL_2014_Test_images=http://images.cocodataset.org/zips/test2014.zip
#URL_2014_Trainval_annotation=http://images.cocodataset.org/annotations/annotations_trainval2014.zip
#
#wget -N $URL_2014_Train_images
#wget -N $URL_2014_Val_images
#wget -N $URL_2014_Test_images
#wget -N $URL_2014_Trainval_annotation
#
#mkdir images
#
#unzip train2014.zip -d images/
#unzip val2014.zip -d images/
#unzip test2014.zip -d images/
#unzip annotations_trainval2014.zip
#
#rm train2014.zip
#rm val2014.zip
#rm test2014.zip
#rm annotations_trainval2014
#
#cd ../
#python script_pick_annotations.py coco/annotations
#
## Build V-COCO
#echo "Building"
#cd coco/PythonAPI/ && make install
#cd ../../ && make
#cd ../../

# ---------------HICO-DET Dataset------------------
#echo "Downloading HICO-DET Dataset"
#
#URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz
#
#wget -N $URL_HICO_DET -P Data/
#tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
#rm Data/hico_20160224_det.tar.gz


echo "Downloading training data..."
python script/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
python script/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
python script/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
python script/Download_data.py 12-ZFl2_AwRkVpRe5sqOJRJrGzBeXtWLm Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl
python script/Download_data.py 1CCfHNBMs3NpjgKMVtLfdsI4OPHTGgikK Data/my_Trainval_GT.pkl
python script/Download_data.py 1VEYk-PR31U5UPkSaNF4v6FQkZ6uDBuCc Data/my_Trainval_Neg.pkl
python script/Download_data.py 1mTT0sa57GKT-fYHz65YlUIc60zMikq0- Data/my_Test.pkl
python script/Download_data.py 1sjV6e916NIPcYYqbGwhKM6Vhl7SY6WqD Results/80000_TIN_D_noS.pkl
python script/Download_data.py 1Ymot1WBdQ4Fdxb5DZ9oiBfSmcOYNxmVT Results/merge_binary_and_partpair.py
python script/Download_data.py 1mRAwnSQOIx-BY4OA2MSccu93ES7WW4KF Results/merge.py
python script/Download_data.py 1j1GFB9sAEIJsuVi8mpoBCvLnOxcc-fjC Results/1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part1.py
python script/Download_data.py 1bI-8fAbtUAQartNw482Bt5tiB1t5wg4m Results/1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part2.py
python script/Download_data.py 1Fd-sygrsZ6fqlNa57QNab1frBuHF2Cc4 Results/1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part3.py
python script/Download_data.py 1ApHPbECoCRDjrpEN6Fdjh2IRLr9JPK-I Results/1700000_TIN_HICO_0.8_0.3new_pair_selection_H2O2_part4.py
python script/Download_data.py 17DPWNZzeJ6kFts-vpWn9uEALKXBjlVGQ Results/700000_partpair_0.8_0.3_part1.py
python script/Download_data.py 1AtbhpRlBRxWLF_KmcpaFT-zNgGQD8Nue Results/700000_partpair_0.8_0.3_part2.py
python script/Download_data.py 1zFalVXukYwUvJQOoC2gqtb_q8gAELMHv Results/700000_partpair_0.8_0.3_part3.py
python script/Download_data.py 1A2_yT2x9eXSSleCp4tu73aw-J9QcIlh9 Results/700000_partpair_0.8_0.3_part4.py


echo "Downloading HICO-DET Evaluation Code"
cd Data/
git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp script/Generate_detection.m Data/ho-rcnn/
cp script/save_mat.m Data/ho-rcnn/
cp script/load_mat.m Data/ho-rcnn/

mkdir Data/ho-rcnn/data/hico_20160224_det/
cp HICO-DET_Benchmark/data/hico_20160224_det/anno_bbox.mat Data/ho-rcnn/data/hico_20160224_det/
cp HICO-DET_Benchmark/data/hico_20160224_det/anno.mat Data/ho-rcnn/data/hico_20160224_det/

mkdir -Results/
echo "Downloading HICO-DET Dataset"
python script/Download_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz