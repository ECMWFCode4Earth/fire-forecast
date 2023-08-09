#!/bin/bash
# 20230809 kaiser@satfire.org

echo 'retrieve_meteo_fire: Moin!'

module reset
module load ecmwf-toolbox/new

MARS_CMD=mars
GRIB2NC_CMD=grib_to_netcdf

DATADIR_0p5=/ec/res4/scratch/cyjk/code4earth/0p5
mkdir -p $DATADIR_0p5
cd $DATADIR_0p5

for year in {2022..2023}; do
    
    day1=${year}0501
    if [ $year != 2023 ]; then
	dayX=${year}1031
    else
	dayX=${year}0731
    fi # year
    # only for testing:
    dayX=${year}0502

    echo "Processing meteo in time range $day1 to $dayX ..."

    for levtype in pl sfc; do
	
	if [ $levtype == 'sfc' ]; then
	    # previous choice of parameters:
	    #param='260048/27.128/28.128/29.128/30.128/66.128/67.128/129.128/134.128/151.128/159.128/160.128/161.128/162.128/163.128/165.128/166.128/167.128/168.128/246.228/247.228'
	    # Johannes's choice of parameters (10 metre U wind component, 10 metre V wind component,
	    # Skin temperature, Total precipitation rate, Volumetric soil water layer 1):
            param='39.128/165.128/166.128/235.128/260048'
	elif [ $levtype == 'pl' ]; then
	    # relative humidity on lowest pressure level
	    param='157.128'
	else
	    echo "ERROR: unknown levtype $levtype"
	fi # levtype 

	echo "Processing parameters $param from $levtype ..."
	
	$MARS_CMD << EOF
retrieve,
class=od,
date=${day1}/to/${dayX},
expver=1,
levtype=${levtype},
levelist=1,
param=${param},
stream=oper,
time=00/12,
type=fc,
step=0/1/2/3/4/5/6/7/8/9/10/11,
area=70/-180/30/180,
grid=0.5/0.5,
target="meteo_${levtype}_0p5_${day1}_${dayX}.grib"
EOF

    $GRIB2NC_CMD -D NC_FLOAT -o meteo_${levtype}_0p5_${day1}_${dayX}.nc -k 4 -d 6 meteo_sfc_05p_${day1}_${dayX}.grib

    done # levtype

    echo "Processing fire in time range $day1 to $dayX ..."

    param='210097/210099'
    
    echo "Processing parameters $param ..."
	
    $MARS_CMD << EOF
retrieve,
class=mc,
date=${day1}/to/${dayX},
expver=0010,
levtype=sfc,
param=$param,
step=0-1,
stream=gfas,
time=00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00,
type=ga,
target="fire_gfas_0p5_${day1}_${dayX}.grib",
interpolation="--interpolation=grid-box-average",
area=70/-180/30/180,
grid=0.5/0.5
EOF

    $GRIB2NC_CMD -D NC_FLOAT -o fire_gfas_0p5_${day1}_${dayX}.nc -k 4 -d 6 fire_gfas_0p5_${day1}_${dayX}.grib
    
done # year
echo 'retrieve_meteo_fire: Bye!'
