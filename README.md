# speed-limit-signs
Identifying Speed Limits using CNNS

Big Kudos to

https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6#.i728o84ib

for providing the initial idea and many of the functions used to prepare and display the images

## Local Installation (Works on Mac, Linux, and Ubuntu Subsystem for Windows 10)
* Install 3.x Anaconda https://www.continuum.io/downloads
* Update to latest version of sklearn `conda install --name root scikit-learn`
* Install TensorFlow: https://www.tensorflow.org/install/ `conda install --name root -c conda-forge tensorflow`
* Install Keras: https://keras.io/#installation `pip install keras`
* `cd notebooks/local`
* `jupyter notebook --no-browser`

## Troubleshooting
* If you see `Intel MKL FATAL ERROR: Cannot load libmkl_avx2.so or libmkl_def.so.` try `conda install nomkl`
* on a Windows Ubuntu Subsystem you might see `Invalid argument (src/tcp_address.cpp:190)` then install `conda install -c jzuhone zeromq=4.1.dev` as indicated here https://github.com/Microsoft/BashOnWindows/issues/185

## Keras
* https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
* https://github.com/fchollet/keras/blob/master/README.md#getting-started-30-seconds-to-keras
* https://keras.io/getting-started/sequential-model-guide/
* https://github.com/fchollet/keras/tree/master/examples
* http://keras.io/getting-started/functional-api-guide
* https://github.com/fchollet/keras-resources
