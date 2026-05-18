if [[ ! -d "$OUTPUT_PATH" ]]; then
  echo "❌ Output directory not found: $OUTPUT_PATH"
  exit 1
fi
cd "$OUTPUT_PATH"

if [[ ! -f output.grib  && ! -f output.nc ]]; then
  echo "❌ Output file not found: output.grib/output.nc"
  exit 1
fi
echo "✅ Inference output file found: output.grib/output.nc"
