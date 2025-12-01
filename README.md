conda activate fast

python full2.py --weights-file fsrcnn_x2.pth --image-file data/frame001_750.png --scale 2 --device cuda --tiles 5 --overlap 20

python full.py --weights-file fsrcnn_x2.pth --image-file data/frame001_750.png --scale 2 --device cpu --tiles 5