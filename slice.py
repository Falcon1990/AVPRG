import os
import glob
from pydub import AudioSegment
import argparse
import audioread
import time

out_format = '.wav'
channels = 1
sample_width = 2
sample_rate = 44100
sample_slice_length_ms = 30000
window_slide_ms = 15000
infiles = glob.glob('C:\PyScripts\CNN_MFCC_Audio_Classification\data\*.flac')

def slice_audio(files, channels, outformat, width, rate, slice_length, slide):
	outformat = outformat.replace('.','').lower()
	#Allow the user to see their x-bit selection with this dictionary.
	width_translator = {1:'8-bit', 2:'16-bit', 4:'32-bit'}
	#For every file in the input list do processing.
	for file in files:
		fileName, fileExtension = os.path.splitext(file)
		#Print to screen the processing parameters.
		with audioread.audio_open(file) as f:
			print ('\nConverting '+fileName+' from:')
			print (fileExtension+' to .'+outformat+';')
			print (str(f.channels)+' channel(s) to '+str(channels)+' channel(s);')
			print (str(f.samplerate)+' Hz to '+str(rate)+' Hz;')
			print ('Slicing '+str(f.duration*1000)+' ms file into '+str(slice_length)+' ms slices with a window slide of '+str(slide)+' ms;')
		#Store the file in RAM.
		sound = AudioSegment.from_file(file, fileExtension.replace('.','').lower())
		#Print the 'x-bit' conversion parameters.
		print (width_translator[sound.sample_width]+' to '+width_translator[int(width)]+'.\n')
		#Implement the user-selected or default (if nothing selected) parameters for processing.
		sound = sound.set_frame_rate(int(rate))
		sound = sound.set_sample_width(int(width))
		sound = sound.set_channels(int(channels))
		length_sound_ms = len(sound)
		length_slice_ms = int(slice_length)
		slice_start = 0
		#create audiosegment object
		notes_reversed = sound[0:1].reverse()
		#Begin slicing at the start of the file.
		while slice_start + length_slice_ms < length_sound_ms:
			sound_slice = sound[slice_start:slice_start+length_slice_ms]
			backwards = sound_slice.reverse()
			notes_reversed += backwards
			sound_slice.export(fileName+'.slice'+str(slice_start/1000)+'SecsTo'+str((slice_start+length_slice_ms)/1000)+'Secs.'+outformat, format=outformat)
			slice_start += int(slide)
		#When the slice is abutting the end of the file, output that slice too.'
		if slice_start + length_slice_ms >= length_sound_ms:
			sound_slice = sound[slice_start:length_sound_ms]
			backwards = sound_slice.reverse()
			notes_reversed += backwards
			sound_slice.export(fileName+'.slice'+str(slice_start/1000)+'SecsToEndFileAt'+str((length_sound_ms)/1000)+'Secs.'+outformat, format=outformat)
			
if __name__ == "__main__":
    
#Execute the slice_audio function.
    slice_audio(infiles, channels, out_format, sample_width, sample_rate, sample_slice_length_ms, window_slide_ms)
