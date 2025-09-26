# Introduction
 
 The problem consisted in training a model in order to perform building data extraction through satelite images, whilst balancing computational cost (speed eficiency) with model performance.

 The submited files consist of a script used to prepare data ready to be treated in PyTorch (prepare_dataset.py); a script that converts the images and masks from the traing and validation groups into tensors (dataset.py); two training scripts: a "failed" attempt at training (trainv1.py), and the training script that produced the model used for testing (train_final_version.py). Finally, this repository also contains a script used to test the model and produce the submission results (test.py).

 The training of the model was conducted in Google Colab for computational reasons.

 # Choice of the model

 The model chose was U-Net, a CNN. The two main reasons behind the choice of this model are the fact that this is a semantic segmentation model, attributes a class to each pixel. In this case, each pixel is one of two things: building or background, which makes this model fitting for our work. The reason has to do with the fact that U-Net is a model that combines computation eficciency and accuracy.

 # Architecture
 
 The first attempt at training the model (trainv1.py) used a simpler U-Net structure, with 3 downsampling blocks and 2 upsampling blocks. The 3 encoders allow for the extraction of finer features and 2 decoders cause lower memory usage in trade-off for the loss of some detail. This was noted in poor metrics: The IoU was alway smaller than 0.4 and the f1-score was never greater than 0.5. Also, the predicted masks had some blurriness to them, this was perhaps consequence of the assymetric structure of the U-Net.

 The loss was computed trough a combination of BCE loss and F1-score, giving equal weight to both metrics, this was later changed.

 The optimizer employed was Adam wich adds weight directly into the gradient update.

 The script was made in order to reduce the learning rate after the loss hit a plateau. This would be great for longer training with a lot of epochs, however, for computationl reasons, training was only carried out until o epochs.

 The second and final attempt employed a more complex U-Net model, employing ResNet-34 as the encoder, and 4 decoder blocks, this is a more efficient model that was expected to deliver better results. In order order to balance out the extra memory usage, torch.amp.autocast and GradScaler were used, this two tools allow for some data to be converted to float 16 instead of float 32, preserving memory usage. In the end each epoch ended up lasting for about 5 minutes, wich was acceptable.

 The loss was now computed with a bigger emphasys towards F1-Score, altough recall precision and IoU were also monitored.

 The optimizer is now AdamW wich allows for better generalization, because weight decay is decoupled from the gradient update.

 The learning followed a pre-defined cycle, wich allows for a more "agressive" convergence, usual for limited epochs.

 With this model, I was able to reach a IoU of 0.7833

# Metrics used

The main metrics used were IoU, BCE loss and F1-score, altough others were alo computed. BCE loss encoureges pixel wise accuracy while F1 and IoU focus on mask alignment (the shape of the mask). A visual representation of one batch's predicted masks was also coputed in order to perform a visual check.

# Inference

The testing script (test.py) loads the previous model weights and runs inference on the set of 1000 images, resizing them first. After the inference process, tiny speckles may be removed from the images. Aditionally the binary masks are saved for visual inspection and a csv file is created for kaggle submission.

# Conclusions

The second training model obtained much better results then the first one, mainly because of the higher complexity of the CNN, but the added focus to F1-score during training may have helped. In order to increase computational efficiency, the script could be modified in order to not compute as many metrics or not to perform any visual check. Longer training cycles with a higher number of epochs could achieve better results, specially if the learning rate is updated when the metrics plateau.

When subitting to kaggle, the score obtained was around 0.26, which is odd, the main suspicion is that it has something to do with the way that the masks are transformed into polygons coordinates, however, I wasn't able to spot the error. restructuring of this part of the test.py file may be able to up the score.
 
