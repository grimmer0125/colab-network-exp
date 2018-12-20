## colab note

ref: https://medium.com/deep-learning-turkey/google-colab-free-gpu-tutorial-e113627b9f5d

### mount google drive in colab VM and download public git repo code

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

#### Step1: Generate a ssh key pair locally

Follow the guide: https://docs.gitlab.com/ee/ssh/. Assume `KEY_FILE_NAME` is the private key file name.

#### step2: setup your public ssl key on GitLab or bitbucket

- GitLab: Settings > Repository section by expanding the Deploy Key
- Bitbucket: Settings->Access keys

#### Step3: Prepare to git clone on colab 

**3-1: setup ssh hosts**

create the folder 

```
cd /root
mkdir .ssh
```

For Gitlab:

```
! ssh-keyscan gitlab.com >> /root/.ssh/known_hosts 
```

For Bicbucket:

`! ssh-keyscan bitbucket.org >> /root/.ssh/known_hosts`

Use `!ssh -T hg@gitlab.com` to test if it is ok or not.

**3-2: Upload the ssh private key to colab**

Either use Google drive to upload ssh private key file to the mounted colab folder or using `uploaded = files.upload()` on colab.

Then `!chmod 600 KEY_FILE_NAME`

#### Step4: Use Git command to download the code on colab

`!GIT_SSH_COMMAND="ssh -i KEY_FILE_NAME -F /dev/null" git clone git@gitlab.com:USER_NAME/REPO_NAME.git`

#### Step5: Use Git command to update code on colab

cd /content/drive/My\ Drive/Colab\ Notebooks/YOUR_GIT_PROJECT
!GIT_SSH_COMMAND="ssh -i ../id_ed25519 -F /dev/null" git fetch --all
!GIT_SSH_COMMAND="ssh -i ../id_ed25519 -F /dev/null" git reset --hard origin/develop
