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
   date=20161231/20170101,
   time=1800/0000,
   target=input.grib

# Pressure-level fields
retrieve,
   class=ea,
   expver=0001,
   type=an,
   levtype=pl,
   param=q/t,
   level=50/100,
   grid=O96,
   date=20161231/20170101,
   time=1800/0000,
   target=input.grib

# Accumulations (cp, tp)
retrieve,
   class=ea,
   expver=0001,
   type=fc,
   levtype=sfc,
   param=cp/tp,
   grid=O96,
   date=20161231/20170101,
   time=0600/1800,
   step=6,
   target=input.grib
EOF

$MARS_CMD global_request.txt
ls -lha
mv --verbose input.grib $OUTPUT_PATH/global.grib

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
   date=20161231/20170101,
   time=1800/0000,
   target=input.grib

# Surface fields
retrieve,
   class=rr,
   type=an,
   stream=oper,
   origin=se-al-ec,
   levtype=sfc,
   param=sp/msl,
   date=20161231/20170101,
   time=1800/0000,
   target=input.grib

# Accumulations (tp, sf) — these need step-based retrieval
# for 6h accumulations (the dataset config uses accumulate with period=6h)
retrieve,
   class=rr,
    type=fc,
   stream=oper,
   origin=se-al-ec,
   levtype=sfc,
   param=tp/sf,
   date=20161231/20170101,
   time=0000/0600/1200/1800,
   step=6,
   target=input.grib
EOF

$MARS_CMD lam_request.txt
mv --verbose input.grib $OUTPUT_PATH/lam.grib

echo "Retrieval complete"
