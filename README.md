# Semi-Supervised-Model-with-Spatio-Temporal-Data-andApplied-in-PM2.5-sensor-anomaly-detection
## Introduction
The PM2.5 issue has drawn much attention in Taiwan, and manyinexpensive sensors have been deployed in recent years. However, thesesensors are fragile and susceptible to environmental factors. In addition,the large number of sensors results in low maintenance frequency, so themonitored values returned by a single sensor are unreliable.This thesis compares supervised, unsupervised, and semi-supervisedmethods to identify the problematic sensors.  We prepared the trainingdata by converting monitored values into images, integrated data, and se-quential data to incorporate the spatio-temporal information of the sensors.We obtained sensors’status (normal or abnormal) based on the inspec-tion records provided by the Industrial Technology Research Institute. Weexplored how the ratio of labeled data to unlabeled data influences the per-formance of the semi-supervised models. Experimental results show thatour studied methods outperform the current inspection strategy (randominspection).

Paper Links: [結合時空資料的半監督模型並應用於 PM2.5 空污感測器的異常偵測](https://drive.google.com/file/d/1-s-bMio5FEWzdtcqI2hUwjwoNREOh_nf/view?usp=sharing)
## Run
### Deep SAD
main.py
```python
python main.py 
<dataset name/choose from['description_data','heatmap','description_plus_timeseries_data','line_chart_all_data']>
<net name/choose from['description_mlp','heatmap_mlp','description_plus_timeseries_mlp','line_chart_all_mlp']>
<log path>
<data path> 
--ratio_known_outlier   --ratio_pollution  --lr  --n_epochs  --lr_milestone  --batch_size  --weight_decay --pretrain  --ae_lr  --ae_n_epochs --ae_batch_size --ae_weight_decay  --normal_class  --known_outlier_class --n_known_outlier_classes --seed--modelname_and_numbers  
```
Example:
```python 
python main.py heatmap heatmap_mlp ../log/DeepSAD ../src/pm2_5_data/Last_use_data --ratio_known_outlier 0.15  --ratio_pollution 0 --lr 0.0001 --n_epochs 150 --lr_milestone 50 --batch_size 128 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001 --ae_n_epochs 150 --ae_batch_size 128 --ae_weight_decay 0.5e-3 --normal_class 0 --known_outlier_class 1 --n_known_outlier_classes 1 --seed 64 --modelname_and_numbers  model_Adjust_A
```
### SSDO
main.py
```python
python main.py 
<dataset name/choose from['description_data','heatmap','description_plus_timeseries_data','line_chart_all_data']>
<net name/choose from['IsolationForest','SSDO_with_iforest','SSDO_with_COP-Kmeans']>
--n_labeled_normal --n_labeled_anomaly --u_unlabeled 
```

Example:
```python 
python main.py description_data SSDO_with_COP-Kmeans --n_labeled_normal 100 --n_labeled_anomaly 100 --n_unlabeled 9660
```

### Data
Due to the large data <heatmap> and <line chart data>, I placed them separately in Releases

### Reference
https://github.com/Vincent-Vercruyssen/anomatools
https://github.com/lukasruff/Deep-SAD-PyTorch
