## colab note

ref: https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d

### mount google drive in colab VM

p.s. this mounting may be accessible by other session but is has expiration time

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
from google.colab import files
uploaded = files.upload() 
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
### How to clone your git private repository from GitLab or Bitbucket

step1: upload and setup your private ssl key on colab, follow https://stackoverflow.com/a/49933595/7354486, summary
1. `! ssh-keyscan gitlab.com >> /root/.ssh/known_hosts` or `! ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts`, then `! chmod 644 /root/.ssh/known_hosts`. Use `!ssh -T hg@bitbucket.org` to test if it is ok or not.
2. upload `KEY_FILE_NAME` using `uploaded = files.upload(), then `!chmod 600 KEY_FILE_NAME`,  ~and move it to `~/.ssh`~  

step2: setup your public ssl key

- GitLab: Settings > Repository section by expanding the Deploy Key
- Bitbucket: Settings->Access keys

step3:

Then you can use `!GIT_SSH_COMMAND="ssh -i KEY_FILE_NAME -F /dev/null" git clone git@gitlab.com:USER_NAME/REPO_NAME.git` to download your code.

~p.s. after testing for Bitbucket, the private key file should be located "~/.ssh/id_rsa" on colab, then it works. Somehow other name does not work.~ 
