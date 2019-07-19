#!/bin/bash

export SUBJECTS_DIR=/fs/pool/pool-ruben/Maria/curvature/freesurfer/curv_test
cd $SUBJECTS_DIR

for SUBJECT in noisy_sphere  # is a folder
do
	cd $SUBJECT
	mkdir surf
	for file in *.vtk
	do
		if [[ $file == *"r10"* ]]
		then
			echo $file
			BASE=${file%".surface.vtk"}

			# Convert VTK surface file to FreeSurfer surface:
			mris_convert $file surf/$BASE.smoothwm
			# "smoothwm" extension for the surface is assumed for some reason next

			for A in 0  {0..10}
			do
				# Calculate curvatures, including principal curvatures:
				mris_curvature_stats -a $A -m -o ${BASE}_a${A}.stats -G --signedPrincipals --writeCurvatureFiles -c 1 $SUBJECT $BASE surface

				# Convert the desired output curvatures to VTK:
				mris_convert -c surf/unknown.$BASE.smoothwm.K1.crv surf/$BASE.smoothwm $BASE.smoothwm.K1.vtk
				mris_convert -c surf/unknown.$BASE.smoothwm.K2.crv surf/$BASE.smoothwm $BASE.smoothwm.K2.vtk
				mris_convert -c surf/unknown.$BASE.smoothwm.H.crv surf/$BASE.smoothwm $BASE.smoothwm.H.vtk

				# Remove "unknown." and replace "smoothwm." by "a${A}." in all the output files:
				for out_file in surf/unknown*
				do
					new_out_file=`echo $out_file | sed -e "s/unknown\.//" | sed -e "s/smoothwm\./a${A}\./"`
					mv $out_file $new_out_file
				done
			done
			# Rename FreeSurfer surface:
			mv surf/$BASE.smoothwm surf/$BASE.surface
		fi
	done
	cd ..
done
