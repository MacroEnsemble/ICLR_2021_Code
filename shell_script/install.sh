apt-get update
apt-get -y install openssh-server
apt-get -y install sshfs
apt-get -y install tmux htop
apt-get -y install python3.5
apt-get -y install python3-pip
apt-get -y install libsm6 libxext6 libxrender-dev 
apt-get -y install libopenmpi-dev
apt-get -y install sysstat
pip3 install virtualenv 
pip3 install -r requirements.txt

echo "cd $PWD/lib/" >> skill_search.sh
cp skill_search.sh ~/
# PWD=~/neural_skill_search
# mkdir -p ~/neural_skill_search/
# mkdir ~/envs
# virtualenv -p python3 ~/envs/skill_search
# source ~/envs/skill_search/bin/activate
# pip install -r $PWD/lib/requirements.txt
# cp $PWD/lib/shell_script/skill_search.sh ~/

