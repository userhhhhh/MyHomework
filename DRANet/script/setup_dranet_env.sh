# #!/bin/bash

# echo "🚀 开始配置 DRANet 所需环境..."

# # Step 1: 检查是否已安装 torch，并提示版本
# echo "🔍 检查当前 PyTorch 版本..."
# python3 -c "import torch; print('当前 PyTorch 版本:', torch.__version__)" 2>/dev/null || echo "当前未安装 PyTorch"

# # Step 2: 卸载现有 PyTorch 及相关组件
# echo "📦 卸载现有 PyTorch..."
# pip uninstall -y torch torchvision torchaudio

# # Step 3: 安装 PyTorch 1.8.0 + cu111
# echo "📥 安装 PyTorch 1.8.0（CUDA 11.1）..."
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# # Step 4: 安装 numpy 和 scipy 指定版本
# echo "🔧 安装 numpy==1.21.0 和 scipy==1.7.1..."
# pip install numpy==1.21.0 scipy==1.7.1 --force-reinstall

# # Step 5: 安装其他依赖
# echo "📘 安装 tensorboardX 和 prettytable..."
# pip install tensorboardX prettytable

# # Step 6: 检查 GPU 可用性和 PyTorch 是否配置成功
# echo "✅ 最后检查 PyTorch 和 GPU 是否配置成功："
# python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU 可用:', torch.cuda.is_available())"

# echo "🎉 环境配置完成！你现在可以运行 DRANet 项目了。"

#!/bin/bash

echo "🚀 开始配置 DRANet 所需环境..."

# Step 1: 检查是否已安装 torch，并提示版本
echo "🔍 检查当前 PyTorch 版本..."
python3 -c "import torch; print('当前 PyTorch 版本:', torch.__version__)" 2>/dev/null || echo "当前未安装 PyTorch"

# Step 2: 卸载现有 PyTorch 及相关组件
echo "📦 卸载现有 PyTorch..."
pip uninstall -y torch torchvision torchaudio

# Step 3: 安装 PyTorch 1.8.0 + cu111（不使用清华源）
echo "📥 安装 PyTorch 1.8.0（CUDA 11.1）..."
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html \
  -i https://pypi.org/simple

# Step 4: 安装 numpy 和 scipy 指定版本
echo "🔧 安装 numpy==1.21.0 和 scipy==1.7.1..."
pip install numpy==1.21.0 scipy==1.7.1 --force-reinstall

# Step 5: 安装其他依赖
echo "📘 安装 tensorboardX 和 prettytable..."
pip install tensorboardX prettytable

# Step 6: 检查 GPU 可用性和 PyTorch 是否配置成功
echo "✅ 最后检查 PyTorch 和 GPU 是否配置成功："
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU 可用:', torch.cuda.is_available())"

echo "🎉 环境配置完成！你现在可以运行 DRANet 项目了。"

