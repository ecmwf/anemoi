mkdir -p $OUTPUT_PATH

########################

echo "Retrieving global data (ERA5 analysis, O96)"
cat << 'EOF' > global_request.txt
# Surface fields
retrieve,
   class=ea,
   expver=0001,
   type=an,
   levtype=sfc,
   param=10u/10v/2d/2t,
   grid=O96,
   date=20241231/20250101,
   time=1800/0000,
   target=global.grib

# Pressure-level fields
retrieve,
   class=ea,
   expver=0001,
   type=an,
   levtype=pl,
   param=q/t,
   level=50/100,
   grid=O96,
   date=20241231/20250101,
   time=1800/0000,
   target=global.grib

# Accumulations (cp, tp)
retrieve,
   class=ea,
   expver=0001,
   levtype=sfc,
   param=cp/tp,
   grid=O96,
   date=20241231/20250101,
   time=0000/0600/1200/1800,
   step=0/6,
   target=global.grib
EOF

$MARS_CMD global_request.txt
mv global.grib $OUTPUT_PATH/global.grib

################################################################################

echo "Retrieving LAM data (CERRA reanalysis, native 5.5km grid)"

cat << 'EOF' > lam_request.txt
# Pressure-level fields
retrieve,
   class=rr,
   type=an,
   stream=oper,
   origin=se-al-ec,
   levtype=pl,
   param=r/t,
   level=50/100,
   date=20241231/20250101,
   time=1800/0000,
   target=lam.grib

# Surface fields
retrieve,
   class=rr,
   type=an,
   stream=oper,
   origin=se-al-ec,
   levtype=sfc,
   param=sp/msl,
   date=20241231/20250101,
   time=1800/0000,
   target=lam.grib

# Accumulations (tp, sf) — these need step-based retrieval
# for 6h accumulations (the dataset config uses accumulate with period=6h)
retrieve,
   class=rr,
   stream=oper,
   origin=se-al-ec,
   levtype=sfc,
   param=tp/sf,
   date=20241231/20250101,
   time=0000/0600/1200/1800,
   step=0/6,
   target=lam.grib
EOF

$MARS_CMD lam_request.txt
mv lam.grib $OUTPUT_PATH/lam.grib

echo "Retrieval complete"
