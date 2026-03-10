echo "Retrieving global data"

# Define environment variables
MARS_FILE=mars_request.txt

# Generate mars request
$RETRIEVE_CMD $CONFIG_PATH --output $MARS_FILE

# Overwrite metadata error. TODO: fix metadata instead
sed -i 's/origin=4/origin=se-al-ec/g' $MARS_FILE

# Get data from mars
$MARS_CMD $MARS_FILE

# Copy files to output directory
mkdir -p $OUTPUT_PATH
mv *.grib $OUTPUT_PATH/global.grib

################################################################################

echo "Retrieving lam data"

# Define environment variables
MARS_FILE=mars_request.txt

# Generate mars request
$RETRIEVE_CMD $CONFIG_PATH --output $MARS_FILE

# Overwrite metadata error. TODO: fix metadata instead
sed -i 's/origin=4/origin=se-al-ec/g' $MARS_FILE

# Get data from mars
$MARS_CMD $MARS_FILE

# Copy files to output directory
mkdir -p $OUTPUT_PATH
mv *.grib $OUTPUT_PATH/lam.grib
