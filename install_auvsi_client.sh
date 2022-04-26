###  Script to install AUVSI Client
TEMP_DIR=./temp_install

# create a temporary directory
echo "Creating temp directory in {$TEMP_DIR}"
mkdir $TEMP_DIR
cd $TEMP_DIR

# run the installation from setup.py
echo "Installing AUVSI Interop Client from https://github.com/auvsi-suas/interop"
git clone https://github.com/auvsi-suas/interop
cd interop/client
echo "Installing Requirements from requirements.txt"
pip install -r requirements.txt
echo "Installing Client from setup.py"
python "./setup.py" install

# remove temp directory
echo "Removing temp directory"
cd ../../../
rm -rf $TEMP_DIR