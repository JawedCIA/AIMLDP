Set-ExecutionPolicy RemoteSigned -scope CurrentUser # Allow scripts to run
#install Scoop. 
irm get.scoop.sh | iex
#add the extras bucket, which contains Miniconda:
scoop bucket add extras

#Finally, install Miniconda with:
scoop install miniconda3

conda --version