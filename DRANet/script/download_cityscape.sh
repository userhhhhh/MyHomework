cd /home/featurize/DRANet/data

# 解压（假设你下载后的 zip 文件位于 ~/Downloads）
unzip /home/featurize/cityscape_zip/leftImg8bit_trainvaltest.zip
unzip /home/featurize/cityscape_zip/gtFine_trainvaltest.zip

# 创建你的目标目录结构
mkdir -p Cityscapes/Images
mkdir -p Cityscapes/GT

# 移动图像数据
mv leftImg8bit/train Cityscapes/Images/train
mv leftImg8bit/val Cityscapes/Images/val
mv leftImg8bit/test Cityscapes/Images/test

# 移动标签数据
mv gtFine/train Cityscapes/GT/train
mv gtFine/val Cityscapes/GT/val
mv gtFine/test Cityscapes/GT/test

# 清理
rm -r leftImg8bit gtFine
