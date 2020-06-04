=======================================================================
	
														Medical physics group

								Carl-von-Ossietzky University of Oldenburg, Institute of Physics 
									Carl-von-Ossietzky Str. 9-11, 26111 Oldenburg, Germany 
													http://medi.uni-oldenburg.de/

								  COPYRIGHT University of Oldenburg 2009. All rights reserved.  
									
												Supported by the DFG (SFB/TR 31) 
																	and 
									the European Commission under the integrated project 
						DIRAC (Detection and Identification of Rare Audio-visual Cues, IST-027787).
									


=======================================================================

							Database of multichannel in-ear and behind-the-ear head-related
												and binaural room impulse responses
												
												
Hendrik Kayser, Stephan Ewert, Jörn Anemüller,																		     June 2009
Thomas Rohdenburg, Volker Hohmann, Birger Kollmeier

Contact: hendrik.kayser@uni-oldenburg.de
--------------------------------------------------------------------------------------------------------------------------------

An eight-channel database of head-related impulse responses (HRIR) and binaural room impulse responses 
(BRIR) is introduced. The impulse responses (IR) were measured with three-channel behind-the-ear (BTE) 
hearing aids and an in-ear microphone at both ears of a human head and torso simulator. The database aims 
at providing a tool for the evaluation of multichannel hearing aid algorithms in hearing aid research. In addition 
to the HRIRs derived from measurements in an anechoic chamber, sets of BRIRs for multiple, realistic head and 
sound-source positions in four natural environments reflecting daily-life communication situations with different 
reverberation times are provided.
The scenes' natural acoustic background was also recorded in each of the realworld environments for all eight 
channels. Overall, the present database allows for a realistic construction of simulated sound fields for hearing 
instrument research and, consequently, for a realistic evaluation of hearing instrument algorithms.

--------------------------------------------------------------------------------------------------------------------------------

This directory contains documentation files for the HRIR database.

README.txt         			: this file
coordinate_system.pdf	: coordinate system for angle specification (Anechoic and Office I)
office_I.pdf					: sketch of the Office I environment
office_II.pdf					: sketch of the Office II environment
cafeteria.pdf					: sketch of the Cafeteria environment
courtyard.pdf					: sketch of the Courtyard environment 
dummy_ear.pdf    	        : picture of the right ear of the artificial head with distances between microphones
                                      and labeling of the microphones

--------------------------------------------------------------------------------------------------------------------------------


GENERAL DOCUMENTATION
==================

All data is available with a sampling rate of 48kHz and a resolution of 32 bit.
The HRIR data is provided in MatLab and .wav format, the ambient sound recordings in .wav format.
The sketches of the environments are true to scale.


Impulse responses: length
-------------------------------

All impulse responses are faded in and out using a hanning window at the beginning and the end. 
The window at the beginning ranges from the first sample to 24 samples before the main peak of the IR has 
reached 15% of its maximum. The IRs are faded out with a window of 2400 samples length.
The length of the IRs measured in the realistic environments are 22000 samples, 
the anechoic IRs are 4800 samples long.


Impulse responses: normalization
---------------------------------------

In each of the environments the impulse responses were normalized. For this purpose, the transferfunction with 
the highest average gain between 150 and 200 Hz was determined and normalized to -20dB at this frequency 
range. All remaining IRs from the same environment were rescaled using  the same factor.


Ambient sound recordings: normalization
-----------------------------------------------

The recordings were normalized such that a fully amplified sine corresponds to a a RMS level of 110dB (SPL).


===========
Loading the data
===========

IRs in MatLab
----------------

For the database in MatLab format, a function named "loadHRIR.m" is available. The input arguments are the 
name of the desired environment and the correspondig parameters of the source position. Type 'help loadHRIR'
in the MatLab command window for details.
By default, the function must be in the same directory with the 'hrir' directory. If you want to adapt the function 
to work from any directory, you can edit the script and replace the relative path by your absolute path.


IRs in .wav format
----------------------

Each set of impulse responses is stored in a 8-channel .wav-file (6-channel for Office I). The first to channels 
are the left and the rigth in-ear IRs (not available for Office I) and the six last channels are the BTE-IRs from 
left to rigth and front to back, such that odds channel numbers refer always to the left side of the head and the
highest number (8 or 6 in case of Office I) is always the right rear hearing aid microphone.
The database in .wav format is stored with the following file naming:

Anechoic
----------
/hrir/anechoic/anechoic_distcm_<distance>_el_<elevation>_az_<azimuth>.wav
distance = {80, 300}
elevation = {-10 : 10 : 20}
azimuth = {-180 : 5 : 180}

Office_I
---------
/hrir/office_I/office_I_distcm_100_el_0_az_<azimuth>.wav
azimuth = {-90 : 5 : 90}

Office_II, Cafeteria, Courtyard
-----------------------------------
/hrir/<environment>/<environment>_<headorientation>_<sourceposition>.wav
environment = {office_II, cafeteria, courtyard}
headorientation = {1,2}
sourceposition = {A,B,C,D,E,F} and {A,B,C,D} for Office_II (cp. sketches of the environments)


--------------------------
IRs of the loudspeaker
--------------------------

The IRs of the loudspeaker were measured in the anechoic room for distances of 80 cm and 300 cm using a 
probe microphone. Two versions of for both distances are available in the 'speaker' directory of the database in 
MatLab and .wav format: 
short: 4800 samples length, corresponding to the length all IRs from the anechoic room,
long: 22000 samples length, corresponding to the length all IRs from all realistic environments 

The loudspeaker IRs were measured with the same equipment as the whole database and the same procedure 
was used to cut and fade the IRs.

NOTE: For the Office I environment another type of loudspeaker was used and the impulse response is not
		  available. 



Ambient sounds
==========

The recodings are stored in 2-channel .wav files for each microphone pair with the following file naming:

Office II
----------
/<event>/office_II_<eventdescription>_<headorientation>_<duration>_min_<microphonepair>.wav
event = {door, telephone, typing, ventilation}
eventdescription = {door, telephone_ringing, typing_<position>, ventilation},	position ={k1,k2,k3} (cp. sketch)
headorientation = {1,2}
duration --> duration of recording
microphonepair = {in-ear, bte_front, bte_middle, bte_rear}


Cafeteria
-----------
/cafeteria_<eventdescription>_<headorientation>_<duration>_min_<microphonepair>.wav
eventdescription = {[], babble}
headorientation = {1}
duration --> duration of recording
microphonepair = {in-ear, bte_front, bte_middle, bte_rear}


Courtyard
------------
courtyard_<headorientation>_<duration>_min_<microphonepair>.wav
headorientation = {1}
duration --> duration of recording
microphonepair = {in-ear, bte_front, bte_middle, bte_rear}


=====
Contact
=====

If you have any questions or comments, please contact hendrik.kayser@uni-oldenburg.de.

 





