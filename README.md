# Project: Weakly supervised learning-- label noise and correction


### [Full Project Description](doc/project3_desc.md)

Term: Spring 2022

+ Team 04
+ Team members
	+ Varchasvi Vedula vvv2108@columbia.edu
	+ Weixun Qian wq2157@columbia.edu
	+ Ran Zhang rz2568@columbia.edu
	+ Jiazheng Chen jc5656@columbia.edu
	+ Sharon Meng zm2380@columbia.edu


+ Project summary: In this project, we carried out model evaluation and selection for predictive analytics on image data with noisy labels using semi-supervised learning techniques. We created two architectures, CNN and Transfer learning. For both CNN and Transfer Learning, model I was built on noisy dataset, and model II was built using label-correction dataset.

	
**Contribution statement**: ([default](doc/a_note_on_contributions.md)) Everyone in the group picked  different classifiers to try and we ultimately picked the best performing one as a team. The contributions were as follows:

+ Varchasvi Vedula - Wrote all of [main.ipynb](doc/main.ipynb) (code + explanations) except the code is section 2.0. Implemented transfer learning-based model (74% accuracy) with cross validation and hyperparameter turning,  model evaluation, and final day model testing. Tried a sample weight based weak-supervised learning method.
+ Weixun Qian - Presenter. Trained a CNN-based model, deviced and used our optimal weak-supervised learning technique to make use of noisy labels.
+ Ran Zhang - Trained a linear SVM model and a CNN-based model. Edited most description in Github repo.
+ Jiazheng Chen - Trained a CNN-based model and tried Weixun's weak-supervised learning technique.
+ Sharon Meng - Initiated group organization. Trained a linear SVM model and a CNN-based model and tried Weixun's weak-supervised learning technique.


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
