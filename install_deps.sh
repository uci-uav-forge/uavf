###  Script to install package dependencies
TEMP_DIR=./temp_install

# create a temporary directory
echo "Creating temp directory in $TEMP_DIR"
mkdir $TEMP_DIR
cd $TEMP_DIR

# run the installation from setup.py
echo "Installing AUVSI Interop Client from https://github.com/auvsi-suas/interop"
git clone https://github.com/auvsi-suas/interop
cd interop/client
echo "Installing Requirements from requirements.txt..."
pip install -qr requirements.txt
echo "Moving proto directory..."
cp -r ../proto ./auvsi_suas/
echo "Installing Client from setup.py..."
pip install .
echo "Done!"

# remove temp directory
echo "Removing temp directory"
cd ../../../
rm -rf $TEMP_DIR
echo "Done!"

echo "Installing UAVF python package..."
pip install -U git+https://github.com/uci-uav-forge/uavf.git
echo "Done!"