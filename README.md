# luau

# Packages

```
pip install stable-baselines3[extra]
pip install swig
pip install gymnasium[box2d]
pip install minigrid
pip install ffio
pip install wandb
pip install scikit-image
pip install h5py
pip install seaborn
```

# Making a video from image frames

```
ffmpeg -framerate 25 -i frame_%06d.png -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p ../output.mp4
```