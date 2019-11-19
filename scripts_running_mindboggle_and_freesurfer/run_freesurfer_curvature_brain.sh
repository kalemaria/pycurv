#!/bin/bash


# A bash script running FreeSurfer on the brain (pial cortex) surface. Requires
# FreeSurfer to be installed and loaded on your system.
#
# Author: Maria Kalemanov (Max Planck Institute for Biochemistry)


export SUBJECTS_DIR=/fs/pool/pool-ruben/Maria/curvature/freesurfer/brain
cd $SUBJECTS_DIR

SUBJECT=arno  # is a folder
cd $SUBJECT
for file in *.pial
do
	echo $file
	BASE=${file%".pial"}

	# Copy surface file into subfolder "surf" with "smoothwm" extension:
	cp $file surf/$BASE.smoothwm

	# Calculate curvatures, including principal curvatures:
	mris_curvature_stats -m -o ${BASE}.stats -G --signedPrincipals --writeCurvatureFiles -c 1 $SUBJECT $BASE surface

	# Convert the desired output curvatures to VTK:
	mris_convert -c surf/$BASE.smoothwm.K1.crv surf/$BASE.smoothwm $BASE.smoothwm.K1.vtk
	mris_convert -c surf/$BASE.smoothwm.K2.crv surf/$BASE.smoothwm $BASE.smoothwm.K2.vtk
	mris_convert -c surf/$BASE.smoothwm.H.crv surf/$BASE.smoothwm $BASE.smoothwm.H.vtk
	mris_convert -c surf/$BASE.smoothwm.K.crv surf/$BASE.smoothwm $BASE.smoothwm.K.vtk

	# Remove "smoothwm." in all the output files:
	for out_file in surf/*smoothwm\.*
	do
		new_out_file=`echo $out_file | sed -e "s/smoothwm\.//"`
		mv $out_file $new_out_file
	done
done
