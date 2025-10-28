# Define environment variables
MARS_FILE=mars_request.txt

# Generate mars request
$RETRIEVE_CMD $CONFIG_PATH --output $MARS_FILE

# Get data from mars
$MARS_CMD $MARS_FILE

# Copy files to output directory
mkdir -p $OUTPUT_PATH
mv *.grib $OUTPUT_PATH/
