# cat-ml-plugin
just a plugin
but with more features
and more independent
work-in-progress

cat_trainer.py fine-tunes a pretrained efficientNetB0 (Google) image recognition model into a binary classification role to identify/differentiate a unique individual cat from other cats or animals. 

Training dataset includes high resolution images of the unique individual cat and 9K assorted cat images from kaggle: https://www.kaggle.com/datasets/crawford/cat-dataset/data
As is routine, a percentage of the dataset was omitted from training to serve as a validation test

It is efficient and fast but can only return a boolean result and can only recognize the one cat it was trained on. (but sometimes that's all you care about?)

cat_eval.py evaluates new images in the new_cat_images directory and can be used as a preliminary weed-out of obvious non-matches, while flagging the ones that it recognizes for further human investigation.