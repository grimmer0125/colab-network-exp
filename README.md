## colab note

ref: https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d

### mount google drive in colab VM

p.s. this mounting may be accesable by other session but is has expirtation time

```
from google.colab import drive
drive.mount('/content/drive/') 

cd /content/drive/My\ Drive/Colab\ Notebooks/
!git clone https://github.com/grimmer0125/network-exp
cd network-exp
```
### upload data and trained weight via google drive web UI

Alternative way: 

```
## from google.colab import files
## uploaded = files.upload() 
```

But you need to move the files to the mounted google drive folder, otherwise the data is only accessable in the VM's lifetime. 

### enable GPU
```
Edit > Notebook settings or Runtime>Change runtime type and # select GPU as Hardware accelerator.
```

### start training
```
run -i classifier_from_little_data_script_3 
```
or
```
import fine_tune_model from classifier_from_little_data_script_3
fine_tune_model ()
```

