# import what we need to import
import bioread
import neurodsp
import scipy.stats
# Import spectral power functions
from neurodsp.spectral import compute_spectrum, rotate_powerlaw

# Import utilities for loading and plotting data
from neurodsp.utils import create_times
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series
import bioread
import numpy as np
import matplotlib.pyplot as plt
import mne
import neurodsp
# Import spectral power functions
from neurodsp.spectral import compute_spectrum, rotate_powerlaw

# Import utilities for loading and plotting data
from neurodsp.utils import create_times
from neurodsp.utils.download import load_ndsp_data
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series
# This file is included in bioread
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
# clean this up later
import json
import pandas as pd

#### define some functions
#function for reading in the file and making it an MNE structure
def read_fileACQ(aFileNameandPath):
	dataFile= bioread.read_file(aFileNameandPath)

	dataFile.channels
	dataFile.channels[2].name
	np.shape(dataFile.channels[2].data)



	sfreq = 2000  # Sampling frequency
	times = dataFile.channels[2].time_index

	chNames= [dataFile.channels[2].name,
			  dataFile.channels[3].name,
			  dataFile.channels[4].name,
			  dataFile.channels[7].name,
			  dataFile.channels[8].name,
			  dataFile.channels[9].name]

	chNames= ['Fz','O1', 'F3', 'O2', 'F4', 'Pz']


	eegMat =  np.array([dataFile.channels[2].data,
						dataFile.channels[3].data,
						dataFile.channels[4].data,
						dataFile.channels[7].data,
						dataFile.channels[8].data,
						dataFile.channels[9].data])

	#print(np.shape(eegMat))

	# start MNE stuff
	chTypes = ['eeg' for x in chNames]# python is ugly beautiful.
	info = mne.create_info(ch_names=chNames, sfreq=sfreq, ch_types=chTypes)
	info.set_montage('standard_1020')


	raw = mne.io.RawArray(eegMat, info)

	########################
	#### GETTING EVENTS ####
	########################

	myEvents = np.zeros((8, eegMat.shape[1]),dtype=int) #np.array([data.channels[ch].data for ch in range(13,21)])
	for evChan in range(np.shape(myEvents)[0]):
		chanEvs = np.array(np.nonzero(dataFile.channels[evChan+13].data)[0])
		#print(chanEvs)
		#print(np.shape(chanEvs))
		#print('')

		myEvents[evChan, chanEvs//1] = 1 << evChan # floor division gets us ints


	# Combine all event tracks into a single byte by summing
	byteEvents = np.sum(myEvents, axis=0)
	#print(np.shape(byteEvents))
	#print(np.transpose(np.unique(byteEvents, return_counts=True)))
	#info.set_montage('standard_1020')
	#get missed events from log file

	missEvs = np.empty((3,),dtype = float)
	logFile =aFileNameandPath[0:-3]
	logFile= logFile+'log'
	print(logFile)
	srchForPhrase = "The following output port codes were not sent because of a conflict on the port."


	try:
		with open(logFile) as f:
			f = f.readlines()

		foundSection = 0
		for line in f:

			if foundSection == 1:
				#only works on lines with three numbers
				try:
					#split and convert to float
					line_arr = np.array(line.split())
					line_val_arr = line_arr.astype(np.float)

					if np.size(line_val_arr) == 3:
						#print(line_val_arr)
						missEvs = np.append(missEvs, line_val_arr)
				except:
					#print("not a line with numbers that we seek")
					a=1+1 #python parsing forces me to put something here, can't have an empty except

			if srchForPhrase in line:
				#start reading in lines next iteration
				foundSection = 1


		missEvs = np.reshape(missEvs, (-1, 3))
		missEvs = missEvs[1:][:]
		#print(missEvs)
		#combine the missed events with byteEvents
		byteEvents_wMissed = byteEvents

		for missEv in missEvs:
			code = int(missEv[1])
			latency = int(missEv[2])

			byteEvents_wMissed[latency] = code

		#plt.plot(byteEvents_wMissed)
		print(np.transpose(np.unique(byteEvents_wMissed, return_counts=True)))


		t_len = len(byteEvents_wMissed)/sfreq
		print("initial (continuous) eeg epoch is "+str(t_len)+" seconds long!")


	except:
		print("no log file found looking for "+logFile)
		byteEvents_wMissed = byteEvents

	# filtering here.
	raw_filter = raw.copy().filter(l_freq=1, h_freq=20)
	# let's hack a sloppy edge detector here. no debounce...
	# loop through, if new is different than old keep new, if new is same as old set new to 0
	oldInd = 0
	cleanEvents = byteEvents_wMissed.copy()
	for i in range(len(cleanEvents)):
		newEv  =cleanEvents[i].copy()
		if newEv == oldInd:
			cleanEvents[i] = 0 # if we leave this alone we'll only get the first occurance
		if newEv != oldInd:
			oldInd = newEv	
		# the best solution is usually the easiest
	eventsArray = (np.array([range(len(raw.times)), cleanEvents]))
	softArray = eventsArray[:,cleanEvents != 0]
	softArray.shape[1]
	evb = "blah"
	eventCode = 0

	#4 bins
	# Paper go correct 24->95
	# Paper go incorrect 24 ->3
	# neutral correct 22 -> 95
	# neutral incorrect 22-> 3

	#soft_arr_name = logFile+'_softarr.csv'
	#np.savetxt(soft_arr_name, softArray.T, delimiter=",")

	eventsArray = []
	for aRowIndex in range(softArray.shape[1]):
		targTime = -1
		if softArray[1,aRowIndex] == 22 or softArray[1,aRowIndex] == 24:
			targTime = softArray[0,aRowIndex]
			#print('target')
		if softArray[1,aRowIndex] == 187:
			#resting
			theREsponseCode = softArray[1,aRowIndex]
			thisEventTime = softArray[0,aRowIndex]
			eventCode = 5
			eventsArray.append([thisEventTime,0,eventCode])
			#break # this is OK because we only enter the 2nd loop for responses
		if softArray[1,aRowIndex] == 3 or softArray[1,aRowIndex] == 95:
			#print(softArray[:,aRowIndex])
			thisResponse = softArray[:,aRowIndex]
			theREsponseCode = softArray[1,aRowIndex]
			thisEventTime = softArray[0,aRowIndex]
			# now find the 22 or 24 before it
			for anotherRowIndex in range(softArray.shape[1]):
				if softArray[0,anotherRowIndex] > softArray[0,aRowIndex]:
					break
				if softArray[1,anotherRowIndex] == 22 or softArray[1,anotherRowIndex] == 24:
					eventBefore = softArray[1,anotherRowIndex]# we'll make sure that we're not going to
					evb = softArray[:,anotherRowIndex]
					responseTime = thisEventTime
					if theREsponseCode == 95:
						if eventBefore == 22:
							eventCode = 1
							#targTime = softArray[0,anotherRowIndex]
							#eventsArray.append([thisEventTime, 0,eventCode])
						if eventBefore == 24:
							eventCode = 2
							#targTime = softArray[0,anotherRowIndex]
							#eventsArray.append([thisEventTime,0,eventCode])
					if theREsponseCode == 3:
						if eventBefore == 22:
							eventCode = 3
							#targTime = softArray[0,anotherRowIndex]
							#eventsArray.append([thisEventTime,0,eventCode])
						if eventBefore == 24:
							eventCode = 4
							#targTime = softArray[0,anotherRowIndex]
							#eventsArray.append([thisEventTime,0,eventCode])
			eventsArray.append([thisEventTime,0,eventCode])
		if targTime > 0:
			eventsArray.append([targTime,0,6])

		# print(evb) # remember these fuckers are in half millisecond samples
		# print(eventCode + "," + str(thisEventTime))

	earrray = np.array(eventsArray)

	#events_arr_name = logFile+'_evarr.csv'
	#np.savetxt(events_arr_name, earrray, delimiter=",")
	

#print(earrray)
	
	event_dict = {'neutralCorrect': 1, 'paperCorrect': 2, 'neutralError': 3,
	              'paperError': 4}
	picks = mne.pick_types(info, meg=False, eeg=True, misc=False)
	pochs = mne.Epochs(raw_filter, earrray, tmin=-0.5 , tmax=0.35,event_id=event_dict, preload=True, baseline=(-0.5,-0.1))
	vent_dict = {'resting': 5}
	restingPochs = mne.Epochs(raw_filter, earrray, tmin=-0.0 , tmax=10,event_id=vent_dict, preload=True, baseline=(0,0))
	#raw.set_montage('standard_1020')
	reject_criteria = dict(eeg= 125)  # 200 µV
	_ = pochs.drop_bad(reject=reject_criteria)
	_ = restingPochs.drop_bad(reject=reject_criteria)
	lookingFor = [5]  # start with resting rembember we recoaded these earlier
	print(eventsArray)
	
	timesforsegs =[]
	tagsforsegs =[]
	endtimes = []
	segType='resting'
	for i in range(0,len(eventsArray)):
		if eventsArray[i][2] in lookingFor:
			print(eventsArray[i])
			print(eventsArray[i][0]/2000)
			timesforsegs.append(eventsArray[i][0]/2000)
			tagsforsegs.append(segType)
			if segType=='resting':
				if len(endtimes) >= 1:
					endtimes.append(eventsArray[i-1][0]/2000)
					print(eventsArray[i-1])
					print(eventsArray[i-1][0]/2000)
				endtimes.append(eventsArray[i][0]/2000 + 10)
			if 5 in lookingFor:
				lookingFor= [1,2,3,4]
				segType='task'
			else:
				lookingFor = [5]
				segType='resting'
	segDF= pd.DataFrame(
		{'segType' : tagsforsegs,
		'segTime' : timesforsegs,
		'endTime' : endtimes}
	)
	print(segDF)
		
	#for index, row in eventsArray.iterrows():
	#	print(row['c1'], row['c2'])
	
	
	
	return pochs,restingPochs, segDF


#read_fileACQ(testFile)

def pochs2json(someEpochs):
	chname = ['Fz','O1', 'F3', 'O2', 'F4', 'Pz']
	# this is ugly but we need those locations
	
	sds = mne.channels.make_standard_montage(kind = "standard_1020")
	sdsPOS = sds.get_positions()
	chanLocs = [sdsPOS['ch_pos'][x].tolist() for x in chname]
	chanLocs = []
	newTimes =someEpochs.times.tolist()
	newTimes2 =[x*1000 for x in newTimes]
	newDict ={
		"chans" : someEpochs.ch_names,
		"chanlocs" :chanLocs,
		"times" : newTimes2,
		"sampleRate" : 2000, # prob should pull from structure, but so it goes
		"bins": [
			{"name":"neutralCorrect",
			 "bad":1,
			 "good":someEpochs['neutralCorrect'].get_data().shape[0],
			 "data": someEpochs['neutralCorrect'].average().data.tolist()
			 },
			{"name":"neutralError",
			 "bad":1,
			 "good":someEpochs['neutralError'].get_data().shape[0],
			 "data":someEpochs['neutralError'].average().data.tolist()
			 },
			{"name":"paperCorrect",
			 "bad":1,
			 "good":someEpochs['paperCorrect'].get_data().shape[0],
			 "data":someEpochs['paperCorrect'].average().data.tolist()
			 },
			{"name":"paperError",
			 "bad":1,
			 "good":someEpochs['paperError'].get_data().shape[0],
			 "data":someEpochs['paperError'].average().data.tolist()
			 }]
	}
	return(newDict)

##
dataDir = "/Users/diogo/Desktop/HoardingProc/new_subj"
import glob, os


arr = os.listdir(dataDir)
os.chdir(dataDir)
for file in glob.glob("*.acq"):
	
	try:
		print(file)
		aaa, bbb, segTimes = read_fileACQ(dataDir +"/"+file)
		smpledct = pochs2json(aaa)
		jsonName = file+".json"
		psdName = file+"_PSD.csv"
		alphaName =  file+"_PSDalpha.csv"
		segTimeName  =  file+"_segtimes.csv"
		with open(jsonName, 'w') as outfile:
			json.dump(smpledct, outfile)
		# now we need to dump the PSDs
		psds, freqs = psd_multitaper(bbb, fmin=2, fmax=40, n_jobs=1)
		psds = np.log10(psds)
		psds_mean = np.transpose(psds.mean(0)) # n channels nephochs nfreqs
		df = pd.DataFrame(psds_mean, columns = ['Fz','O1', 'F3', 'O2', 'F4', 'Pz'])
		df['freq']= freqs
		df.to_csv(psdName)
		alphaDF = df[(df['freq'] > 7.8) & (df['freq'] < 12.5)]
		#print(df)
		#average em
		alphaMean = alphaDF.mean(axis = 0) # got our alpha for each electrode now
		alphaMean.to_csv(alphaName)
		segTimes.to_csv(segTimeName)
		
	except Exception as e:
		print(e)
		print("something failed")