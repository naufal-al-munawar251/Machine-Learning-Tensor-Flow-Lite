# Training-Machine-Learning
Untuk membuat model pelatihan machine learning TensorFlow yang dapat di gunakan untuk deteksi yang nantinya dapat di gunakan
sebelum proses pelatihan machine learning _Tensor Flow_

silahkan install terlebih dahulu untuk memulai pemasangan
````
pip install tflite-model-maker
````

Jika Anda ingin menginstal nightly version _tflite-model-maker-nightly_, ikuti perintah:
````
pip install tflite-model-maker-nightly
````
atau anda juga bisa langsung cloning installan menggunakan github
````
git clone https://github.com/tensorflow/examples
cd examples/tensorflow_examples/lite/model_maker/pip_package
pip install -e .
````

Di bawah sini ada contoh kode dari tensorflow
````
from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

data = DataLoader.from_folder('flower_photos/')
model = image_classifier.create(data)
loss, accuracy = model.evaluate()
model.export(export_dir='/tmp/')
````

untuk gambar sample yang bakal di latih di usahakan formatnya _jpg_ atau _png_ ( _JPG_).


di sini ada full codingan dari tensorflow

````
import tensorflow as tf

data_path = tf.keras.utils.get_file(
      'flower_photos',
      'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
      untar=True)


from tflite_model_maker import image_classifier
from tflite_model_maker.image_classifier import DataLoader

# Load input data specific to an on-device ML app.
data = DataLoader.from_folder(data_path)
train_data, test_data = data.split(0.9)

# Customize the TensorFlow model.
model = image_classifier.create(train_data)

# Evaluate the model.
loss, accuracy = model.evaluate(test_data)

# Export to Tensorflow Lite model and label file in `export_dir`.
model.export(export_dir='/tmp/')
````
