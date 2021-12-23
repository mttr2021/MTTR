# prepare Refer-YouTube-VOS
wget https://github.com/JerryX1110/awesome-rvos/blob/main/down_YTVOS_w_refer.py
python down_YTVOS_w_refer.py

# prepare a2d_sentences
mkdir a2d_sentences
cd a2d_sentences
  wget https://web.eecs.umich.edu/~jjcorso/bigshare/A2D_main_1_0.tar.bz
  tar jxvf A2D_main_1_0.tar.bz
  mkdir text_annotations
  cd text_annotations
    wget https://kgavrilyuk.github.io/actor_action/a2d_annotation.txt
    wget https://kgavrilyuk.github.io/actor_action/a2d_missed_videos.txt
    wget https://github.com/JerryX1110/awesome-rvos/blob/main/down_a2d_annotation_with_instances.py
    python down_a2d_annotation_with_instances.py
    unzip a2d_annotation_with_instances.zip
    #rm a2d_annotation_with_instances.zip
  cd ..
cd ..

# prepare jhmdb_sentences
mkdir jhmdb_sentences
cd jhmdb_sentences
  wget http://files.is.tue.mpg.de/jhmdb/Rename_Images.tar.gz
  wget https://kgavrilyuk.github.io/actor_action/jhmdb_annotation.txt
  wget http://files.is.tue.mpg.de/jhmdb/puppet_mask.zip
  tar -xzvf  Rename_Images.tar.gz
  unzip puppet_mask.zip
cd ..


echo "FINISHED!!"
