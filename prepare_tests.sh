# Run this script from the root folder (parent of ./tests/)
mkdir tests/fixture_setup/
wget -O "./tests/fixture_setup/plaza.jpg" https://unsplash.com/photos/F7I6sexmIS8/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjUwMTI3ODIw&force=true
wget -O "./tests/fixture_setup/efficientdet_lite0_320_ptq.tflite" https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite0_320_ptq.tflite
wget -O "./tests/fixture_setup/efficientdet_lite0_320_ptq_edgetpu.tflite" https://raw.githubusercontent.com/google-coral/test_data/master/efficientdet_lite0_320_ptq_edgetpu.tflite
wget -O "./tests/fixture_setup/coco_labels.txt" https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt