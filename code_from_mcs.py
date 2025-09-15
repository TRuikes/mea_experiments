from MCStream import MCStreamWrapper, MCSObject2ClassWrapper
from GeneratedMcStream import *

# Path to datafile
file_to_open = r"C:\test_read_mcd\20250513_davis_OD1_poly600_2000ms_5-100_noblocker.mcd"

stream = MCStreamWrapper()

# Open file to read data from
stream.OpenFileEx(file_to_open)

# Read and print timestamps
startTimeStamp: IMCSTimeStamp = MCSObject2ClassWrapper(IMCSTimeStamp, stream.GetStartTime())
print(startTimeStamp.GetYear(), startTimeStamp.GetMonth(), startTimeStamp.GetDay())

# Digital data in stream 0
# digiStream : IMCSStream = MCSObject2ClassWrapper(IMCSStream, stream.GetStream(0))
# sRate = digiStream.GetSampleRate()
# print(f'sample rate: {sRate:.0f} ')
#
# for i in range(3):
#     chName = digiStream.GetChannelName(i)
#     print(f'channel name: {chName}')

"""
console output:
sample rate: 20000 
channel name: D1
channel name: \
channel name: \
"""

# Electrode data in stream 1
ecStream : IMCSStream = MCSObject2ClassWrapper(IMCSStream, stream.GetStream(1))
# Then there is the read raw data method:
Data = ecStream.GetRawData(pData=0, tsFrom=0, tsTo=10) # what should pData be? when calling this
    # the process exists with an undefined code


for i in range(3):
    chName = ecStream.GetChannelName(i)
    print(f'channel name: {chName}')

"""
console output:
channel name: 01
channel name: 02
channel name: 03
"""

# Untill here all good, but now the question, how do I load data from for example channel 5
# I found the readChunk methods
Chunk = ecStream.GetFirstChunk()
wrappedChunk = MCSObject2ClassWrapper(IMCSChunk, Chunk)
for i in range(3):
    data = wrappedChunk.ReadData(i) # Unclear what keywords to pass (no typehints)
    print(data)


# This data seems to be empty
"""
console output:
(0, 0)
(0, 1)
(0, 2)
(0, 3)
"""

# Then there is the read raw data method:
Data = ecStream.GetRawData(pData=0, tsFrom=0, tsTo=10) # what should pData be? when calling this
    # the process exists with an undefined code

