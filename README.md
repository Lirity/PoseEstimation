Setup env:

```
conda create -n pose python=3.9
conda activate pose
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

cd lib/pointnet2/
pip install .
cd ../sphericalmap_utils/
pip install .
```