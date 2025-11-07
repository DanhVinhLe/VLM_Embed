mkdir -p ./vlm2vec_train/MMEB-train/images/
wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/N24News.zip
unzip N24News.zip -d ./vlm2vec_train/MMEB-train/images/
rm N24News.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/NIGHTS.zip
unzip NIGHTS.zip -d ./vlm2vec_train/MMEB-train/images/
rm NIGHTS.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/OK-VQA.zip
unzip OK-VQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm OK-VQA.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/SUN397.zip
unzip SUN397.zip -d ./vlm2vec_train/MMEB-train/images/
rm SUN397.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VOC2007.zip
unzip VOC2007.zip -d ./vlm2vec_train/MMEB-train/images/
rm VOC2007.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VisDial.zip
unzip VisDial.zip -d ./vlm2vec_train/MMEB-train/images/
rm VisDial.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/Visual7W.zip
unzip Visual7W.zip -d ./vlm2vec_train/MMEB-train/images/
rm Visual7W.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VisualNews_i2t.zip
unzip VisualNews_i2t.zip -d ./vlm2vec_train/MMEB-train/images/
rm VisualNews_i2t.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/VisualNews_t2i.zip
unzip VisualNews_t2i.zip -d ./vlm2vec_train/MMEB-train/images/
rm VisualNews_t2i.zip

wget https://huggingface.co/datasets/TIGER-Lab/MMEB-train/resolve/main/images_zip/WebQA.zip
unzip WebQA.zip -d ./vlm2vec_train/MMEB-train/images/
rm WebQA.zip