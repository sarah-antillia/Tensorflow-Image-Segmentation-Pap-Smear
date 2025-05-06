<h2>Tensorflow-Image-Segmentation-Pap-Smear (Updated: 2025/05/07)</h2>

Sarah T. Arai<br>
Software Laboratory antillia.com<br><br>

<li>2025/05/07: Updated to use the latest Tensorflow-Image-Segmentation-API</li>
<br>
This is the first experiment of Image Segmentation for Pap-Smear 
 based on 
the latest <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>, 
and  <a href="https://drive.google.com/file/d/1s6TYPD8nSto8X_M6u3ectf-RdXFVk-1M/view?usp=sharing">Smear2005-Seamless-ImageMask-Dataset-V2.zip</a>
, which was derived by us from <a href="https://mde-lab.aegean.gr/index.php/downloads/">
PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES<br>
Part II : smear2005.zip [85.17 MB] New Pap-smear Database (images)
</a>
<br>
<br>
Please see also our first experiment <a href="https://github.com/sarah-antillia/Image-Segmentation-Pap-Smear">Image-Segmentation-Pap-Smear</a>.<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512pixels</b><br>
As shown below, the inferred masks look similar to the ground truth masks. <br>

<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/149056410-149056423-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/149056410-149056423-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/149056410-149056423-001.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/149096927-149096943-002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/149096927-149096943-002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/149096927-149096943-002.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/flipped_149058262-149058309-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/flipped_149058262-149058309-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/flipped_149058262-149058309-001.jpg" width="320" height="auto"></td>
</tr>
</table>

<hr>
<br>
In this experiment, we used the simple UNet Model 
<a href="./src/TensorflowUNet.py">TensorflowSlightlyFlexibleUNet</a> for this Pap-SmearSegmentation Model.<br>
As shown in <a href="https://github.com/sarah-antillia/Tensorflow-Image-Segmentation-API">Tensorflow-Image-Segmentation-API</a>.
you may try other Tensorflow UNet Models:<br>

<li><a href="./src/TensorflowSwinUNet.py">TensorflowSwinUNet.py</a></li>
<li><a href="./src/TensorflowMultiResUNet.py">TensorflowMultiResUNet.py</a></li>
<li><a href="./src/TensorflowAttentionUNet.py">TensorflowAttentionUNet.py</a></li>
<li><a href="./src/TensorflowEfficientUNet.py">TensorflowEfficientUNet.py</a></li>
<li><a href="./src/TensorflowUNet3Plus.py">TensorflowUNet3Plus.py</a></li>
<li><a href="./src/TensorflowDeepLabV3Plus.py">TensorflowDeepLabV3Plus.py</a></li>

<br>

<h3>1. Dataset Citation</h3>
The image dataset used here has been taken from the following web site.<br>
<br>
<a href="https://mde-lab.aegean.gr/index.php/downloads/">
PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES
<br>
Part II : smear2005.zip [85.17 MB] New Pap-smear Database (images)
</a>
<br><br>
This is the new website that hosts the DTU/Herlev Pap Smear Databases, as well as selected studies and papers <br>
related to these data. For more than 10 years, Dr Jan Jantzen works on pap-smear data acquired from images of <br>
healthy & cancerous smears coming from the Herlev University Hospital (Denmark), thanks to Dr MD Beth Bjerregaard.<br>
The Old Pap Smear Database was formed in the late 90’s while the New Pap Smear Database (improved) was formed <br>
within 2005. The analysis of these databases was made through several Master Theses most of which where elaborated <br>
in Denmark, under the supervision of Dr Jantzen, while he was joining DTU, Dept. of Automation (Denmark) and also <br>
through collaboration to other researchers from around the world, many of which were made with G.Dounias and his<br>
research team of the MDE-Lab, University of the Aegean. During the last years, Dr Jantzen collaborates with the <br>
University of the Aegean, Dept. of Financial and Management Engineering (FME) as teaching associate of the <br>
Postgraduate Program of the FME-Dept. and as research associate of the MDE-Lab. The site will be continuously<br> 
updated with new papers, studies, theses and citations related to the hosted pap-smear databases.<br>

In case you use material from this site, please cite the current link and related studies.<br>


<br>
<h3>
<a id="2">
2 Pap-Smear ImageMask Dataset
</a>
</h3>
 If you would like to train this Pap-Smear Segmentation model by yourself,
 please download our 512x512 pixels dataset from the google drive  
 If you would like to train this Pap-Smear Segmentation model by yourself,
please download the ImageMask-Dataset-Pap-Smear-V2 created by us from the google drive
<a href="https://drive.google.com/file/d/1s6TYPD8nSto8X_M6u3ectf-RdXFVk-1M/view?usp=sharing">Smear2005-Seamless-ImageMask-Dataset-V2.zip</a>
<br>, 
expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
 └─Pap-Smear
     └─severe_dysplastic
        ├─test
        │   ├─images
        │   └─masks
        ├─train
        │   ├─images
        │   └─masks
        └─valid
            ├─images
            └─masks
</pre>
<br>
<b>Pap-Smear Statistics</b><br>
<img src ="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/train_images_sample.png" width="1024" height="auto">
<br>
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowUNet Model
</h3>
 We trained Pap-Smear TensorflowUNet Model by using the following
<a href="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorflowSlightlyFlexibleUNet/Pap-Smearand run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorflowUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters</b> and large <b>base_kernels</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dilation       = (3,3)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Enabled our online augmentation tool. 
<pre>
[model]
model         = "TensorflowUNet"
generator     = True
</pre>

<b>Loss and metrics functions</b><br>
Specified "bce_dice_loss" and "dice_coef".<br>
<pre>
[model]
loss           = "bce_dice_loss"
metrics        = ["dice_coef"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>


<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>

By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/epoch_change_infer_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (38,39,40)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/epoch_change_infer_end.png" width="1024" height="auto"><br>
<br>

In this experiment, the training process was stopped at epoch 40  by EarlyStopping Callback.<br><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/train_console_output_at_epoch_40.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/eval/train_losses.png" width="520" height="auto"><br>
<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear</b> folder,<br>
and run the following bat file to evaluate TensorflowUNet model for Pap-Smear.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorflowUNetEvaluator.py ./train_eval_infer_aug.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/evaluate_console_output_at_epoch_40.png" width="720" height="auto">
<br><br>Image-Segmentation-Pap-Smear

<a href="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/evaluation.csv">evaluation.csv</a><br>
The loss (bce_dice_loss) to this Pap-Smear/test was low, and dice_coef high as shown below.
<br>
<pre>
loss,0.0691
dice_coef,0.9389
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowUNet model for Pap-Smear.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorflowUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/asset/mini_test_output.png" width="1024" height="auto"><br>

<br>
<hr>
<b>Enlarged images and masks (512x512 pixels)</b><br>

<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/149056410-149056444-003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/149056410-149056444-003.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/149056410-149056444-003.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/149096927-149096943-002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/149096927-149096943-002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/149096927-149096943-002.jpg" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/153655016-153655039-004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/153655016-153655039-004.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/153655016-153655039-004.jpg" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/flipped_149058262-149058309-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/flipped_149058262-149058309-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/flipped_149058262-149058309-001.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/flipped_149096854-149096870-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/flipped_149096854-149096870-001.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/flipped_149096854-149096870-001.jpg" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/images/mirrored_149316754-149316795-002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test/masks/mirrored_149316754-149316795-002.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorflowSlightlyFlexibleUNet/Pap-Smear/mini_test_output/mirrored_149316754-149316795-002.jpg" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>


<h3>
References
</h3>
<b>1. PAP-SMEAR (DTU/HERLEV) DATABASES & RELATED STUDIES</b><br>
<pre>
https://mde-lab.aegean.gr/index.php/downloads/
</pre>

<b>2. Liquid based cytology pap smear images for multi-class diagnosis of cervical cancer</b><br>
<pre>
https://data.mendeley.com/datasets/zddtpgzv63/4
</pre>

<b>3. Pap-smear Benchmark Data For Pattern Classiﬁcation<br></b>
Jan Jantzen, Jonas Norup , George Dounias , Beth Bjerregaard<br>
<pre>
https://www.researchgate.net/publication/265873515_Pap-smear_Benchmark_Data_For_Pattern_Classification
</pre>
<b>4. Deep Convolution Neural Network for Malignancy Detection and Classification in Microscopic Uterine Cervix Cell Images</b><br>
Shanthi P B,1 Faraz Faruqi, Hareesha K S, and Ranjini Kudva<br>
<pre>
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7062987/
</pre>

<b>5. DeepCyto: a hybrid framework for cervical cancer classification by using deep feature fusion of cytology images</b><br>
Swati Shinde, Madhura Kalbhor, Pankaj Wajire<br>
<pre>
https://www.aimspress.com/article/doi/10.3934/mbe.2022301?viewType=HTML#b40
</pre>

<b>6. EfficientNet-Pap-Smear</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/atlan-antillia/EfficientNet-Pap-Smear
</pre>

<b>7. ImageMask-Dataset-Pap-Smear</b><br>
Toshiyuki Arai @antillia.com<br>
<pre>
https://github.com/sarah-antillia/ImageMask-Dataset-Pap-Smear
</pre>

