# -*- coding: cp1252 -*-
# Created by makepy.py version 0.5.01
# By python version 3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)]
# From type library 'MCStream.dll'
# On Thu Jul 24 11:15:23 2025
''
makepy_version = '0.5.01'
python_version = 0x30d00f0

import win32com.client.CLSIDToClass, pythoncom, pywintypes
import win32com.client.util
from pywintypes import IID
from win32com.client import Dispatch

# The following 3 lines may need tweaking for the particular server
# Candidates are pythoncom.Missing, .Empty and .ArgNotFound
defaultNamedOptArg=pythoncom.Empty
defaultNamedNotOptArg=pythoncom.Empty
defaultUnnamedArg=pythoncom.Empty

CLSID = IID('{8FB22B36-D3FE-11D3-8F42-008048B000C7}')
MajorVersion = 3
MinorVersion = 2
LibraryFlags = 8
LCID = 0x0

from win32com.client import DispatchBaseClass
class IMCSChannel(DispatchBaseClass):
	CLSID = IID('{F68B2B00-40BC-11D1-B441-0080C6FF1BCF}')
	coclass_clsid = IID('{F68B2B01-40BC-11D1-B441-0080C6FF1BCF}')

	def GetGroupID(self):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (3, 0), (),)

	def GetRefBufferID(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (3, 0), ((3, 0),),Index
			)

	def GetRefChannelID(self, Index=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (3, 0), ((3, 0),),Index
			)

	_prop_map_get_ = {
		"Comment": (5, 2, (8, 0), (), "Comment", None),
		"DecoratedName": (10, 2, (8, 0), (), "DecoratedName", None),
		"Diameter": (17, 2, (3, 0), (), "Diameter", None),
		"DisplayExtendX": (20, 2, (3, 0), (), "DisplayExtendX", None),
		"DisplayExtendY": (21, 2, (3, 0), (), "DisplayExtendY", None),
		"DisplayX": (18, 2, (3, 0), (), "DisplayX", None),
		"DisplayY": (19, 2, (3, 0), (), "DisplayY", None),
		"ExtendX": (15, 2, (3, 0), (), "ExtendX", None),
		"ExtendY": (16, 2, (3, 0), (), "ExtendY", None),
		"Gain": (11, 2, (3, 0), (), "Gain", None),
		"HWID": (6, 2, (3, 0), (), "HWID", None),
		"HeaderVersion": (2, 2, (2, 0), (), "HeaderVersion", None),
		"HighF": (23, 2, (3, 0), (), "HighF", None),
		"ID": (3, 2, (3, 0), (), "ID", None),
		"LowF": (22, 2, (3, 0), (), "LowF", None),
		"Name": (4, 2, (8, 0), (), "Name", None),
		"PosX": (12, 2, (3, 0), (), "PosX", None),
		"PosY": (13, 2, (3, 0), (), "PosY", None),
		"PosZ": (14, 2, (3, 0), (), "PosZ", None),
		"ReferenceCount": (1, 2, (3, 0), (), "ReferenceCount", None),
	}
	_prop_map_put_ = {
		"Comment" : ((5, LCID, 4, 0),()),
		"DecoratedName" : ((10, LCID, 4, 0),()),
		"Diameter" : ((17, LCID, 4, 0),()),
		"DisplayExtendX" : ((20, LCID, 4, 0),()),
		"DisplayExtendY" : ((21, LCID, 4, 0),()),
		"DisplayX" : ((18, LCID, 4, 0),()),
		"DisplayY" : ((19, LCID, 4, 0),()),
		"ExtendX" : ((15, LCID, 4, 0),()),
		"ExtendY" : ((16, LCID, 4, 0),()),
		"Gain" : ((11, LCID, 4, 0),()),
		"HWID" : ((6, LCID, 4, 0),()),
		"HeaderVersion" : ((2, LCID, 4, 0),()),
		"HighF" : ((23, LCID, 4, 0),()),
		"ID" : ((3, LCID, 4, 0),()),
		"LowF" : ((22, LCID, 4, 0),()),
		"Name" : ((4, LCID, 4, 0),()),
		"PosX" : ((12, LCID, 4, 0),()),
		"PosY" : ((13, LCID, 4, 0),()),
		"PosZ" : ((14, LCID, 4, 0),()),
		"ReferenceCount" : ((1, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSChannelLayout(DispatchBaseClass):
	CLSID = IID('{D528F251-3971-45BF-8B0C-2959ED502FC2}')
	coclass_clsid = IID('{BC6A0146-FAE1-450E-BC78-FFCF76E3C6B2}')

	_prop_map_get_ = {
		"AmplifierName": (22, 2, (8, 0), (), "AmplifierName", None),
		"DX": (17, 2, (3, 0), (), "DX", None),
		"DXExtend": (19, 2, (3, 0), (), "DXExtend", None),
		"DY": (18, 2, (3, 0), (), "DY", None),
		"DYExtend": (20, 2, (3, 0), (), "DYExtend", None),
		"DecoratedName": (7, 2, (8, 0), (), "DecoratedName", None),
		"Diameter": (16, 2, (3, 0), (), "Diameter", None),
		"ElecType": (10, 2, (3, 0), (), "ElecType", None),
		"Gain": (3, 2, (3, 0), (), "Gain", None),
		"HWID": (1, 2, (3, 0), (), "HWID", None),
		"HighF": (5, 2, (3, 0), (), "HighF", None),
		"LowF": (4, 2, (3, 0), (), "LowF", None),
		"MeaName": (9, 2, (8, 0), (), "MeaName", None),
		"MeaNumber": (8, 2, (3, 0), (), "MeaNumber", None),
		"MeaSubType": (2, 2, (3, 0), (), "MeaSubType", None),
		"Name": (6, 2, (8, 0), (), "Name", None),
		"PinName": (21, 2, (8, 0), (), "PinName", None),
		"X": (11, 2, (3, 0), (), "X", None),
		"XExtend": (14, 2, (3, 0), (), "XExtend", None),
		"Y": (12, 2, (3, 0), (), "Y", None),
		"YExtend": (15, 2, (3, 0), (), "YExtend", None),
		"Z": (13, 2, (3, 0), (), "Z", None),
	}
	_prop_map_put_ = {
		"AmplifierName" : ((22, LCID, 4, 0),()),
		"DX" : ((17, LCID, 4, 0),()),
		"DXExtend" : ((19, LCID, 4, 0),()),
		"DY" : ((18, LCID, 4, 0),()),
		"DYExtend" : ((20, LCID, 4, 0),()),
		"DecoratedName" : ((7, LCID, 4, 0),()),
		"Diameter" : ((16, LCID, 4, 0),()),
		"ElecType" : ((10, LCID, 4, 0),()),
		"Gain" : ((3, LCID, 4, 0),()),
		"HWID" : ((1, LCID, 4, 0),()),
		"HighF" : ((5, LCID, 4, 0),()),
		"LowF" : ((4, LCID, 4, 0),()),
		"MeaName" : ((9, LCID, 4, 0),()),
		"MeaNumber" : ((8, LCID, 4, 0),()),
		"MeaSubType" : ((2, LCID, 4, 0),()),
		"Name" : ((6, LCID, 4, 0),()),
		"PinName" : ((21, LCID, 4, 0),()),
		"X" : ((11, LCID, 4, 0),()),
		"XExtend" : ((14, LCID, 4, 0),()),
		"Y" : ((12, LCID, 4, 0),()),
		"YExtend" : ((15, LCID, 4, 0),()),
		"Z" : ((13, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSChunk(DispatchBaseClass):
	CLSID = IID('{904180C0-C09F-11D1-AC75-E498383B5A44}')
	coclass_clsid = IID('{904180C1-C09F-11D1-AC75-E498383B5A44}')

	def GetFirstEvent(self):
		ret = self._oleobj_.InvokeTypes(9, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetFirstEvent', None)
		return ret

	def ReadData(self, pBuffer=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (3, 0), ((16386, 0),),pBuffer
			)

	_prop_map_get_ = {
		"FromHigh": (1, 2, (3, 0), (), "FromHigh", None),
		"FromLow": (2, 2, (3, 0), (), "FromLow", None),
		"Size": (5, 2, (3, 0), (), "Size", None),
		"TimeStampFrom": (7, 2, (9, 0), (), "TimeStampFrom", None),
		"TimeStampTo": (6, 2, (9, 0), (), "TimeStampTo", None),
		"ToHigh": (3, 2, (3, 0), (), "ToHigh", None),
		"ToLow": (4, 2, (3, 0), (), "ToLow", None),
	}
	_prop_map_put_ = {
		"FromHigh" : ((1, LCID, 4, 0),()),
		"FromLow" : ((2, LCID, 4, 0),()),
		"Size" : ((5, LCID, 4, 0),()),
		"TimeStampFrom" : ((7, LCID, 4, 0),()),
		"TimeStampTo" : ((6, LCID, 4, 0),()),
		"ToHigh" : ((3, LCID, 4, 0),()),
		"ToLow" : ((4, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSEvent(DispatchBaseClass):
	CLSID = IID('{904180C2-C09F-11D1-AC75-E498383B5A44}')
	coclass_clsid = IID('{904180C3-C09F-11D1-AC75-E498383B5A44}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(16, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetRawTimeStamp(self):
		'method GetRawTimeStamp'
		return self._oleobj_.InvokeTypes(18, LCID, 1, (21, 0), (),)

	def GetSize(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (3, 0), (),)

	def Next(self):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
		"Day": (3, 2, (2, 0), (), "Day", None),
		"Hour": (4, 2, (2, 0), (), "Hour", None),
		"Microsecond": (8, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (12, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (7, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (11, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (5, 2, (2, 0), (), "Minute", None),
		"Month": (2, 2, (2, 0), (), "Month", None),
		"Nanosecond": (9, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (13, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (6, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (10, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (14, 2, (9, 0), (), "TimeStamp", None),
		"Year": (1, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Day" : ((3, LCID, 4, 0),()),
		"Hour" : ((4, LCID, 4, 0),()),
		"Microsecond" : ((8, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((12, LCID, 4, 0),()),
		"Millisecond" : ((7, LCID, 4, 0),()),
		"MillisecondFromStart" : ((11, LCID, 4, 0),()),
		"Minute" : ((5, LCID, 4, 0),()),
		"Month" : ((2, LCID, 4, 0),()),
		"Nanosecond" : ((9, LCID, 4, 0),()),
		"NanosecondFromStart" : ((13, LCID, 4, 0),()),
		"Second" : ((6, LCID, 4, 0),()),
		"SecondFromStart" : ((10, LCID, 4, 0),()),
		"TimeStamp" : ((14, LCID, 4, 0),()),
		"Year" : ((1, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSEvtAverage(DispatchBaseClass):
	CLSID = IID('{B93B64BF-0CB4-4D71-8B1A-779F897491DD}')
	coclass_clsid = IID('{012F99E0-D08C-4793-AC6E-928057B2F694}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetADData(self, iIndex=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), ((3, 0),),iIndex
			)

	def GetADDataArray(self, buf=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (11, 0), ((16386, 0),),buf
			)

	def GetHWID(self):
		'method GetHWID'
		return self._oleobj_.InvokeTypes(6, LCID, 1, (2, 0), (),)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
		"Channel": (2, 2, (2, 0), (), "Channel", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"TimeWindowCount": (3, 2, (3, 0), (), "TimeWindowCount", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Channel" : ((2, LCID, 4, 0),()),
		"Count" : ((1, LCID, 4, 0),()),
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"TimeWindowCount" : ((3, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __bool__(self):
		return True

class IMCSEvtBurstParameter(DispatchBaseClass):
	CLSID = IID('{C7D985C4-0403-4229-833F-7A9325E2F86B}')
	coclass_clsid = IID('{55CAAEE3-E497-4724-ADF9-72561ED007AF}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetHWID(self):
		'method GetHWID'
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), (),)

	def GetTimeStampOfBurst(self):
		'method GetTimeStampOfBurst'
		return self._oleobj_.InvokeTypes(5, LCID, 1, (21, 0), (),)

	def GetUnitID(self):
		'method GetUnitId'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (2, 0), (),)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	def Parameter(self, ParameterId=defaultNamedNotOptArg):
		'method Parameter'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (4, 0), ((19, 0),),ParameterId
			)

	_prop_map_get_ = {
		"Channel": (1, 2, (2, 0), (), "Channel", None),
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Channel" : ((1, LCID, 4, 0),()),
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSEvtParam(DispatchBaseClass):
	CLSID = IID('{06746628-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{06746629-DA9C-11D1-8EB8-0000B4552050}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetAmplitude(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetArea(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetDataArray(self, iChannel=defaultNamedNotOptArg, buf=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (11, 0), ((2, 0), (16388, 0)),iChannel
			, buf)

	def GetHeight(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetMax(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetMin(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetNumber(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetRate(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetSlope(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetTmax(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetTmin(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def GetWidth(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (4, 0), ((2, 0),),iChannel
			)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSEvtRaw(DispatchBaseClass):
	CLSID = IID('{06746620-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{06746621-DA9C-11D1-8EB8-0000B4552050}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetADData(self, iChannel=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (2, 0), ((2, 0),),iChannel
			)

	def GetADDataArray(self, buf=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (11, 0), ((16386, 0),),buf
			)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSEvtSpike(DispatchBaseClass):
	CLSID = IID('{06746622-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{06746623-DA9C-11D1-8EB8-0000B4552050}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetADData(self, iIndex=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), ((3, 0),),iIndex
			)

	def GetADDataArray(self, buf=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (11, 0), ((16386, 0),),buf
			)

	def GetHWID(self):
		'method GetHWID'
		return self._oleobj_.InvokeTypes(7, LCID, 1, (2, 0), (),)

	def GetUnitID(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (2, 0), (),)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
		"Channel": (3, 2, (2, 0), (), "Channel", None),
		"Count": (1, 2, (3, 0), (), "Count", None),
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"PreEvent": (2, 2, (3, 0), (), "PreEvent", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Channel" : ((3, LCID, 4, 0),()),
		"Count" : ((1, LCID, 4, 0),()),
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"PreEvent" : ((2, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)
	#This class has Count() property - allow len(ob) to provide this
	def __len__(self):
		return self._ApplyTypes_(*(1, 2, (3, 0), (), "Count", None))
	#This class has a __len__ - this is needed so 'if object:' always returns TRUE.
	def __bool__(self):
		return True

class IMCSEvtSpikeParameter(DispatchBaseClass):
	CLSID = IID('{12C1F607-3162-425D-B116-044C937334E2}')
	coclass_clsid = IID('{4AFCF788-E915-47B2-8BFC-479914F63892}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetHWID(self):
		'method GetHWID'
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), (),)

	def GetUnitID(self):
		'method GetUnitId'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (2, 0), (),)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	def Parameter(self, ParameterId=defaultNamedNotOptArg):
		'method Parameter'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (4, 0), ((19, 0),),ParameterId
			)

	_prop_map_get_ = {
		"Channel": (1, 2, (2, 0), (), "Channel", None),
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Channel" : ((1, LCID, 4, 0),()),
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSEvtTrigger(DispatchBaseClass):
	CLSID = IID('{06746624-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{06746625-DA9C-11D1-8EB8-0000B4552050}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(65552, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetADData(self):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (2, 0), (),)

	def GetManualTriggerCount(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (3, 0), (),)

	def GetStimulusNumber(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (3, 0), (),)

	def GetTrialNumber(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (3, 0), (),)

	def Next(self):
		return self._oleobj_.InvokeTypes(65551, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
		"Day": (65539, 2, (2, 0), (), "Day", None),
		"Hour": (65540, 2, (2, 0), (), "Hour", None),
		"Microsecond": (65544, 2, (2, 0), (), "Microsecond", None),
		"MicrosecondFromStart": (65548, 2, (2, 0), (), "MicrosecondFromStart", None),
		"Millisecond": (65543, 2, (2, 0), (), "Millisecond", None),
		"MillisecondFromStart": (65547, 2, (2, 0), (), "MillisecondFromStart", None),
		"Minute": (65541, 2, (2, 0), (), "Minute", None),
		"Month": (65538, 2, (2, 0), (), "Month", None),
		"Nanosecond": (65545, 2, (2, 0), (), "Nanosecond", None),
		"NanosecondFromStart": (65549, 2, (2, 0), (), "NanosecondFromStart", None),
		"Second": (65542, 2, (2, 0), (), "Second", None),
		"SecondFromStart": (65546, 2, (3, 0), (), "SecondFromStart", None),
		"TimeStamp": (65550, 2, (9, 0), (), "TimeStamp", None),
		"Year": (65537, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Day" : ((65539, LCID, 4, 0),()),
		"Hour" : ((65540, LCID, 4, 0),()),
		"Microsecond" : ((65544, LCID, 4, 0),()),
		"MicrosecondFromStart" : ((65548, LCID, 4, 0),()),
		"Millisecond" : ((65543, LCID, 4, 0),()),
		"MillisecondFromStart" : ((65547, LCID, 4, 0),()),
		"Minute" : ((65541, LCID, 4, 0),()),
		"Month" : ((65538, LCID, 4, 0),()),
		"Nanosecond" : ((65545, LCID, 4, 0),()),
		"NanosecondFromStart" : ((65549, LCID, 4, 0),()),
		"Second" : ((65542, LCID, 4, 0),()),
		"SecondFromStart" : ((65546, LCID, 4, 0),()),
		"TimeStamp" : ((65550, LCID, 4, 0),()),
		"Year" : ((65537, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoAverage(DispatchBaseClass):
	CLSID = IID('{A8F8C595-7B79-49F2-9805-8AF57F5EF4CB}')
	coclass_clsid = IID('{A3073478-3666-40D7-B1C0-99552C1B8E99}')

	def GetMaxTimeWindowCount(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), (),)

	def GetTimeWindowStartTime(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (4, 0), (),)

	def GetTimeWindowWindowExtend(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (4, 0), (),)

	def GetTriggerBufferID(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1, LCID, 1, (8, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoBurstParameter(DispatchBaseClass):
	CLSID = IID('{02408658-BB5F-48E9-8E30-B9B1520785C6}')
	coclass_clsid = IID('{FA067E74-9370-4DBF-91B5-3166527A4EA5}')

	def InputBufferName(self):
		'method InputBufferName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1, LCID, 1, (8, 0), (),)

	def ParameterCount(self):
		'method ParameterCount'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (3, 0), (),)

	def ParameterExponent(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterExponent'
		return self._oleobj_.InvokeTypes(6, LCID, 1, (3, 0), ((19, 0),),ParameterId
			)

	def ParameterFactor(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterFactor'
		return self._oleobj_.InvokeTypes(5, LCID, 1, (5, 0), ((19, 0),),ParameterId
			)

	def ParameterName(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(4, LCID, 1, (8, 0), ((19, 0),),ParameterId
			)

	def ParameterSelected(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterSelected'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (11, 0), ((19, 0),),ParameterId
			)

	def ParameterUnit(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterUnit'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(7, LCID, 1, (8, 0), ((19, 0),),ParameterId
			)

	def UnitCount(self):
		'method UnitCount'
		return self._oleobj_.InvokeTypes(8, LCID, 1, (3, 0), (),)

	def UnitSelected(self, UnitId=defaultNamedNotOptArg):
		'method UnitSelected'
		return self._oleobj_.InvokeTypes(9, LCID, 1, (11, 0), ((19, 0),),UnitId
			)

	def UnitSortMethod(self):
		'method UnitSortMethod'
		return self._oleobj_.InvokeTypes(10, LCID, 1, (3, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoChannelTool(DispatchBaseClass):
	CLSID = IID('{DF7EDC7A-C505-4D57-9F94-902637AF2DCD}')
	coclass_clsid = IID('{731B89DA-6D16-4D18-9DB2-10BC51F1B78B}')

	def InputBufferName(self):
		'method InputBufferName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1, LCID, 1, (8, 0), (),)

	def RefChannelName(self):
		'method RefChannelName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(2, LCID, 1, (8, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoFilter(DispatchBaseClass):
	CLSID = IID('{26BB09B1-6E95-11D4-8FFB-008048B000C7}')
	coclass_clsid = IID('{26BB09B3-6E95-11D4-8FFB-008048B000C7}')

	def GetCenterFrequency(self):
		'method GetCenterFrequency'
		return self._oleobj_.InvokeTypes(9, LCID, 1, (5, 0), (),)

	def GetCutoff(self):
		'method GetCutoff'
		return self._oleobj_.InvokeTypes(13, LCID, 1, (3, 0), (),)

	def GetDownsamplingFrequency(self):
		'method GetDownsamplingFrequency'
		return self._oleobj_.InvokeTypes(12, LCID, 1, (3, 0), (),)

	def GetFilterName(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1, LCID, 1, (8, 0), (),)

	def GetFilterType(self):
		'get filter type, see docu for explanation of the return value'
		return self._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), (),)

	def GetLowerCutoff(self):
		'method GetLowerCutoff'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (3, 0), (),)

	def GetPassType(self):
		'method GetPassType'
		return self._oleobj_.InvokeTypes(5, LCID, 1, (3, 0), (),)

	def GetPassTypeAsString(self):
		'method GetPassTypeAsString'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(4, LCID, 1, (8, 0), (),)

	def GetQFactor(self):
		'Get Q factor of Savitzky-Golay filter'
		return self._oleobj_.InvokeTypes(10, LCID, 1, (5, 0), (),)

	def GetSGNumDataPointsLeft(self):
		'method GetSGNumDataPointsLeft'
		return self._oleobj_.InvokeTypes(8, LCID, 1, (3, 0), (),)

	def GetSGNumSamples(self):
		'method GetSGNumSamples'
		return self._oleobj_.InvokeTypes(7, LCID, 1, (3, 0), (),)

	def GetSGOrder(self):
		'method GetSGOrder'
		return self._oleobj_.InvokeTypes(6, LCID, 1, (3, 0), (),)

	def GetUpperCutoff(self):
		'method GetUpperCutoff'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (3, 0), (),)

	def IsDownsamplingEnabled(self):
		'method IsDownsamplingEnabled'
		return self._oleobj_.InvokeTypes(11, LCID, 1, (11, 0), (),)

	def IsSGFilter(self):
		'returns true if filter is a Savitzky-Golay filter'
		return self._oleobj_.InvokeTypes(14, LCID, 1, (11, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoParam(DispatchBaseClass):
	CLSID = IID('{06746630-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{06746631-DA9C-11D1-8EB8-0000B4552050}')

	def AmplitudePos(self):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (2, 0), (),)

	def AreaPos(self):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (2, 0), (),)

	def HeightPos(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (2, 0), (),)

	def InputBufferName(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(12, LCID, 1, (8, 0), (),)

	def MaxPos(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (2, 0), (),)

	def MinPos(self):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (2, 0), (),)

	def NumTimeWindows(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (2, 0), (),)

	def NumberPos(self):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (2, 0), (),)

	def RatePos(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (2, 0), (),)

	def SlopePos(self):
		return self._oleobj_.InvokeTypes(18, LCID, 1, (2, 0), (),)

	def TMaxPos(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), (),)

	def TMinPos(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (2, 0), (),)

	def TimeWindowChoice(self):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (2, 0), (),)

	def TimeWindowStartTriggerID(self):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (2, 0), (),)

	def TimeWindowStopTriggerID(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (2, 0), (),)

	def TimeWindowTime1(self):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (4, 0), (),)

	def TimeWindowTime2(self):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (4, 0), (),)

	def WidthPos(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (2, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoRaw(DispatchBaseClass):
	CLSID = IID('{0674662A-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{0674662B-DA9C-11D1-8EB8-0000B4552050}')

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoSpike(DispatchBaseClass):
	CLSID = IID('{0674662C-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{0674662D-DA9C-11D1-8EB8-0000B4552050}')

	def GetChannelGroupID(self, iChHWID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (2, 0), ((2, 0),),iChHWID
			)

	def GetDeadTime(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (4, 0), (),)

	def GetDeadTime_us(self):
		'dead time in microseconds'
		return self._oleobj_.InvokeTypes(20, LCID, 1, (19, 0), (),)

	def GetDetectMethod(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (2, 0), (),)

	def GetInputBufferID(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1, LCID, 1, (8, 0), (),)

	def GetPostTrigger(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (4, 0), (),)

	def GetPostTrigger_us(self):
		'post trigger interval in micro seconds'
		return self._oleobj_.InvokeTypes(19, LCID, 1, (3, 0), (),)

	def GetPreTrigger(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (4, 0), (),)

	def GetPreTrigger_us(self):
		'pre trigger interval in microseconds'
		return self._oleobj_.InvokeTypes(18, LCID, 1, (3, 0), (),)

	def GetSlopeDeltaV(self, iChHWID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (4, 0), ((2, 0),),iChHWID
			)

	def GetSlopeMax(self, iChHWID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (4, 0), ((2, 0),),iChHWID
			)

	def GetSlopeMin(self, iChHWID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (4, 0), ((2, 0),),iChHWID
			)

	def GetSortMethod(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (2, 0), (),)

	def GetSpikeUnitCount(self):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (2, 0), (),)

	def GetSpikeUnitID(self, iIndex=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (2, 0), ((2, 0),),iIndex
			)

	def GetSpikeUnitWndMax(self, iChHWID=defaultNamedNotOptArg, iUnitID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (4, 0), ((2, 0), (2, 0)),iChHWID
			, iUnitID)

	def GetSpikeUnitWndMin(self, iChHWID=defaultNamedNotOptArg, iUnitID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (4, 0), ((2, 0), (2, 0)),iChHWID
			, iUnitID)

	def GetSpikeUnitWndTime(self, iChHWID=defaultNamedNotOptArg, iUnitID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (4, 0), ((2, 0), (2, 0)),iChHWID
			, iUnitID)

	def GetThresholdLevel(self, iChHWID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (4, 0), ((2, 0),),iChHWID
			)

	def GetThresholdSlope(self, iChHWID=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (2, 0), ((2, 0),),iChHWID
			)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoSpikeParameter(DispatchBaseClass):
	CLSID = IID('{4DBE71EE-0635-4F56-B89B-8B5BC384DC4C}')
	coclass_clsid = IID('{6EFEC165-D03D-40ED-9C02-24302793225C}')

	def InputBufferName(self):
		'method InputBufferName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(1, LCID, 1, (8, 0), (),)

	def ParameterCount(self):
		'method ParameterCount'
		return self._oleobj_.InvokeTypes(2, LCID, 1, (3, 0), (),)

	def ParameterExponent(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterExponent'
		return self._oleobj_.InvokeTypes(6, LCID, 1, (3, 0), ((19, 0),),ParameterId
			)

	def ParameterFactor(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterFactor'
		return self._oleobj_.InvokeTypes(5, LCID, 1, (5, 0), ((19, 0),),ParameterId
			)

	def ParameterName(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(4, LCID, 1, (8, 0), ((19, 0),),ParameterId
			)

	def ParameterSelected(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterSelected'
		return self._oleobj_.InvokeTypes(3, LCID, 1, (11, 0), ((19, 0),),ParameterId
			)

	def ParameterUnit(self, ParameterId=defaultNamedNotOptArg):
		'method ParameterUnit'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(7, LCID, 1, (8, 0), ((19, 0),),ParameterId
			)

	def UnitCount(self):
		'method UnitCount'
		return self._oleobj_.InvokeTypes(8, LCID, 1, (3, 0), (),)

	def UnitSelected(self, UnitId=defaultNamedNotOptArg):
		'method UnitSelected'
		return self._oleobj_.InvokeTypes(9, LCID, 1, (11, 0), ((19, 0),),UnitId
			)

	def UnitSortMethod(self):
		'method UnitSortMethod'
		return self._oleobj_.InvokeTypes(10, LCID, 1, (3, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSInfoTrigger(DispatchBaseClass):
	CLSID = IID('{0345A2C0-A4EF-11D2-98D5-444553540000}')
	coclass_clsid = IID('{0345A2C1-A4EF-11D2-98D5-444553540000}')

	def GetChannel(self):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (3, 0), (),)

	def GetDeadTime(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (3, 0), (),)

	def GetDigitalTriggerMask(self):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (3, 0), (),)

	def GetDigitalTriggerType(self):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (3, 0), (),)

	def GetDigitalTriggerValue(self):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (3, 0), (),)

	def GetInputBufferID(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(10, LCID, 1, (8, 0), (),)

	def GetLevel(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (3, 0), (),)

	def GetParameter(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(12, LCID, 1, (8, 0), (),)

	def GetParameterUnit(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(13, LCID, 1, (8, 0), (),)

	def GetSlope(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (3, 0), (),)

	def GetTriggeredStreamId(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (3, 0), (),)

	def IsDigitalTrigger(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (11, 0), (),)

	def IsParameterTrigger(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (11, 0), (),)

	def TrialSynchronization(self):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (11, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSLayout(DispatchBaseClass):
	CLSID = IID('{D52FFF63-73C3-471B-8227-222B0F5615E1}')
	coclass_clsid = IID('{4A5ABA66-BADC-4438-8B7D-B9ED24863290}')

	def GetAmplifierNameFromHWID(self, HWID=defaultNamedNotOptArg):
		'get name of MEA for the electrode with HWID'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(12, LCID, 1, (8, 0), ((3, 0),),HWID
			)

	def GetChannelLayout(self, Index=defaultNamedNotOptArg):
		'method GetChannelLayout'
		ret = self._oleobj_.InvokeTypes(1, LCID, 1, (9, 0), ((3, 0),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetChannelLayout', None)
		return ret

	def GetChannelTypeFromHWID(self, HWID=defaultNamedNotOptArg):
		'method GetChannelTypeFromHWID'
		return self._oleobj_.InvokeTypes(14, LCID, 1, (3, 0), ((3, 0),),HWID
			)

	def GetMEAIndexFromHWID(self, HWID=defaultNamedNotOptArg):
		'get MEA index (1 to 4)'
		return self._oleobj_.InvokeTypes(13, LCID, 1, (3, 0), ((3, 0),),HWID
			)

	def GetMEANameFromHWID(self, HWID=defaultNamedNotOptArg):
		'get name of MEA for the electrode with HWID'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(11, LCID, 1, (8, 0), ((3, 0),),HWID
			)

	def GetNameOfConfigAmp(self, Index=defaultNamedNotOptArg):
		'get name of amplifier with index index'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(8, LCID, 1, (8, 0), ((3, 0),),Index
			)

	def GetNameOfConfigMEA(self, Index=defaultNamedNotOptArg):
		'get name of configured MEA with index index'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(10, LCID, 1, (8, 0), ((3, 0),),Index
			)

	def GetNumberOfConfigAmps(self):
		'get number of configured amplifiers'
		return self._oleobj_.InvokeTypes(7, LCID, 1, (3, 0), (),)

	def GetNumberOfConfigMEAs(self):
		'get number of configures MEAs'
		return self._oleobj_.InvokeTypes(9, LCID, 1, (3, 0), (),)

	def GetRelativeChannelPosX(self, MEAIndex=defaultNamedNotOptArg, HWID=defaultNamedNotOptArg):
		'method GetRelativeChannelPosX'
		return self._oleobj_.InvokeTypes(15, LCID, 1, (3, 0), ((3, 0), (3, 0)),MEAIndex
			, HWID)

	def GetRelativeChannelPosY(self, MEAIndex=defaultNamedNotOptArg, HWID=defaultNamedNotOptArg):
		'method GetRelativeChannelPosY'
		return self._oleobj_.InvokeTypes(16, LCID, 1, (3, 0), ((3, 0), (3, 0)),MEAIndex
			, HWID)

	_prop_map_get_ = {
		"LayoutType": (2, 2, (3, 0), (), "LayoutType", None),
		"NAnlg": (5, 2, (3, 0), (), "NAnlg", None),
		"NDigi": (6, 2, (3, 0), (), "NDigi", None),
		"NElec": (4, 2, (3, 0), (), "NElec", None),
		"NTotal": (3, 2, (3, 0), (), "NTotal", None),
	}
	_prop_map_put_ = {
		"LayoutType" : ((2, LCID, 4, 0),()),
		"NAnlg" : ((5, LCID, 4, 0),()),
		"NDigi" : ((6, LCID, 4, 0),()),
		"NElec" : ((4, LCID, 4, 0),()),
		"NTotal" : ((3, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSMea21StimulatorProperty(DispatchBaseClass):
	CLSID = IID('{BB528DF6-FC0E-4B0B-A593-521EEBA14FAE}')
	coclass_clsid = IID('{F50409D9-9982-40C8-AF80-181B8E11F23E}')

	def GetNumberOfStreams(self):
		'method GetNumberOfStreams'
		return self._oleobj_.InvokeTypes(1, LCID, 1, (3, 0), (),)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSStream(DispatchBaseClass):
	CLSID = IID('{F5B4BFA0-40A5-11D1-B441-0080C6FF1BCF}')
	coclass_clsid = IID('{F5B4BFA1-40A5-11D1-B441-0080C6FF1BCF}')

	def EventCountFromTo(self, dispFrom=defaultNamedNotOptArg, dispTo=defaultNamedNotOptArg, arrEventCount=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(26, LCID, 1, (24, 0), ((9, 0), (9, 0), (16387, 0)),dispFrom
			, dispTo, arrEventCount)

	def GetBufferID(self):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(28, LCID, 1, (8, 0), (),)

	def GetChannel(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(16, LCID, 1, (9, 0), ((2, 0),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetChannel', None)
		return ret

	def GetChannelName(self, Index=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(23, LCID, 1, (8, 0), ((2, 0),),Index
			)

	def GetChunkNextTo(self, ts=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(24, LCID, 1, (9, 0), ((9, 0),),ts
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetChunkNextTo', None)
		return ret

	def GetEventNextTo(self, ts=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(25, LCID, 1, (9, 0), ((9, 0),),ts
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetEventNextTo', None)
		return ret

	def GetFirstChunk(self):
		ret = self._oleobj_.InvokeTypes(18, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetFirstChunk', None)
		return ret

	def GetFirstEvent(self):
		ret = self._oleobj_.InvokeTypes(20, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetFirstEvent', None)
		return ret

	def GetInfo(self):
		ret = self._oleobj_.InvokeTypes(27, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetInfo', None)
		return ret

	def GetNextChunk(self, pCurrentChunk=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(19, LCID, 1, (9, 0), ((9, 0),),pCurrentChunk
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetNextChunk', None)
		return ret

	def GetNextEvent(self, pCurrentEvent=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(21, LCID, 1, (9, 0), ((9, 0),),pCurrentEvent
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetNextEvent', None)
		return ret

	def GetRawData(self, pData=defaultNamedNotOptArg, tsFrom=defaultNamedNotOptArg, tsTo=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(33, LCID, 1, (3, 0), ((16386, 0), (9, 0), (9, 0)),pData
			, tsFrom, tsTo)

	def GetRawDataBufferSize(self, tsFrom=defaultNamedNotOptArg, tsTo=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(34, LCID, 1, (3, 0), ((9, 0), (9, 0)),tsFrom
			, tsTo)

	def GetRawDataBufferSizeOfChannel(self):
		return self._oleobj_.InvokeTypes(31, LCID, 1, (3, 0), (),)

	def GetRawDataOfChannel(self, pData=defaultNamedNotOptArg, lIndex=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(32, LCID, 1, (3, 0), ((16386, 0), (3, 0)),pData
			, lIndex)

	def GetSampleRate(self):
		'get sample rate in Hz'
		return self._oleobj_.InvokeTypes(37, LCID, 1, (3, 0), (),)

	def GetStreamFormatPrivate(self, pData=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (3, 0), ((16387, 0),),pData
			)

	def GetSweepRawData(self, pData=defaultNamedNotOptArg, lSweepIndex=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(36, LCID, 1, (3, 0), ((16386, 0), (3, 0)),pData
			, lSweepIndex)

	def GetSweepRawDataBufferSize(self, lSweepIndex=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(35, LCID, 1, (3, 0), ((3, 0),),lSweepIndex
			)

	def HasContinuousData(self):
		return self._oleobj_.InvokeTypes(30, LCID, 1, (11, 0), (),)

	def HasRawData(self):
		return self._oleobj_.InvokeTypes(29, LCID, 1, (11, 0), (),)

	def Reset(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), (),)

	_prop_map_get_ = {
		"ADBits": (10, 2, (2, 0), (), "ADBits", None),
		"ADZero": (11, 2, (3, 0), (), "ADZero", None),
		"BytesPerChannel": (13, 2, (2, 0), (), "BytesPerChannel", None),
		"ChannelCount": (1, 2, (3, 0), (), "ChannelCount", None),
		"Comment": (5, 2, (8, 0), (), "Comment", None),
		"DataType": (3, 2, (8, 0), (), "DataType", None),
		"DefaultSamplesPerSegment": (14, 2, (2, 0), (), "DefaultSamplesPerSegment", None),
		"DefaultSegmentCount": (15, 2, (2, 0), (), "DefaultSegmentCount", None),
		"FormatVersion": (8, 2, (2, 0), (), "FormatVersion", None),
		"HeaderVersion": (2, 2, (3, 0), (), "HeaderVersion", None),
		"ID": (6, 2, (2, 0), (), "ID", None),
		"MillisamplesPerSecond": (7, 2, (3, 0), (), "MillisamplesPerSecond", None),
		"Name": (4, 2, (8, 0), (), "Name", None),
		"UnitSign": (9, 2, (2, 0), (), "UnitSign", None),
		"UnitsPerAD": (12, 2, (5, 0), (), "UnitsPerAD", None),
	}
	_prop_map_put_ = {
		"ADBits" : ((10, LCID, 4, 0),()),
		"ADZero" : ((11, LCID, 4, 0),()),
		"BytesPerChannel" : ((13, LCID, 4, 0),()),
		"ChannelCount" : ((1, LCID, 4, 0),()),
		"Comment" : ((5, LCID, 4, 0),()),
		"DataType" : ((3, LCID, 4, 0),()),
		"DefaultSamplesPerSegment" : ((14, LCID, 4, 0),()),
		"DefaultSegmentCount" : ((15, LCID, 4, 0),()),
		"FormatVersion" : ((8, LCID, 4, 0),()),
		"HeaderVersion" : ((2, LCID, 4, 0),()),
		"ID" : ((6, LCID, 4, 0),()),
		"MillisamplesPerSecond" : ((7, LCID, 4, 0),()),
		"Name" : ((4, LCID, 4, 0),()),
		"UnitSign" : ((9, LCID, 4, 0),()),
		"UnitsPerAD" : ((12, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSStreamFile(DispatchBaseClass):
	CLSID = IID('{E27A2D84-400F-11D1-B517-52544C190838}')
	coclass_clsid = IID('{E27A2D85-400F-11D1-B517-52544C190838}')

	def CloseFile(self):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (2, 0), (),)

	def GetAnalogChannelOffset1(self):
		return self._oleobj_.InvokeTypes(47, LCID, 1, (2, 0), (),)

	def GetAnalogChannelOffset2(self):
		return self._oleobj_.InvokeTypes(48, LCID, 1, (2, 0), (),)

	def GetAnalogChannels(self):
		return self._oleobj_.InvokeTypes(33, LCID, 1, (2, 0), (),)

	def GetAnalogChannels2(self):
		return self._oleobj_.InvokeTypes(44, LCID, 1, (2, 0), (),)

	def GetBusType(self):
		'method GetBusType'
		return self._oleobj_.InvokeTypes(55, LCID, 1, (3, 0), (),)

	def GetDataSourceName(self):
		'method GetDataSourceName'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(53, LCID, 1, (8, 0), (),)

	def GetDigitalChannels(self):
		return self._oleobj_.InvokeTypes(34, LCID, 1, (2, 0), (),)

	def GetDriverVersionMajor(self):
		return self._oleobj_.InvokeTypes(36, LCID, 1, (3, 0), (),)

	def GetDriverVersionMinor(self):
		return self._oleobj_.InvokeTypes(37, LCID, 1, (3, 0), (),)

	def GetElectrodeChannelOffset1(self):
		return self._oleobj_.InvokeTypes(45, LCID, 1, (2, 0), (),)

	def GetElectrodeChannelOffset2(self):
		return self._oleobj_.InvokeTypes(46, LCID, 1, (2, 0), (),)

	def GetElectrodeChannels(self):
		return self._oleobj_.InvokeTypes(32, LCID, 1, (2, 0), (),)

	def GetElectrodeChannels2(self):
		return self._oleobj_.InvokeTypes(43, LCID, 1, (2, 0), (),)

	def GetImageData(self,
					 imageIndex=defaultNamedNotOptArg,
					 selectedElec1=defaultNamedNotOptArg,
					 selectedElec2=defaultNamedNotOptArg,
					 xLayoutPosElec1=defaultNamedNotOptArg
					 , yLayoutPosElec1=defaultNamedNotOptArg,
					 xLayoutPosElec2=defaultNamedNotOptArg,
					 yLayoutPosElec2=defaultNamedNotOptArg,
					 xImagePosElec1=defaultNamedNotOptArg,
					 yImagePosElec1=defaultNamedNotOptArg
			, xImagePosElec2=defaultNamedNotOptArg,
					 yImagePosElec2=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(42, LCID, 1, (24, 0), ((3, 0), (16387, 0), (16387, 0), (16387, 0), (16387, 0), (16387, 0), (16387, 0), (16389, 0), (16389, 0), (16389, 0), (16389, 0)),imageIndex
			, selectedElec1, selectedElec2, xLayoutPosElec1, yLayoutPosElec1, xLayoutPosElec2
			, yLayoutPosElec2, xImagePosElec1, yImagePosElec1, xImagePosElec2, yImagePosElec2
			)

	def GetImageFilePathName(self, imageIndex=defaultNamedNotOptArg):
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(41, LCID, 1, (8, 0), ((3, 0),),imageIndex
			)

	def GetLayout(self):
		'method GetLayout'
		ret = self._oleobj_.InvokeTypes(51, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetLayout', None)
		return ret

	def GetMea21StimulatorProperty(self):
		'method GetMea21StimulatorProperty'
		ret = self._oleobj_.InvokeTypes(58, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetMea21StimulatorProperty', None)
		return ret

	def GetPostTriggerTime(self):
		return self._oleobj_.InvokeTypes(24, LCID, 1, (3, 0), (),)

	def GetPreTriggerTime(self):
		return self._oleobj_.InvokeTypes(23, LCID, 1, (3, 0), (),)

	def GetProductId(self):
		'method GetProductId'
		return self._oleobj_.InvokeTypes(57, LCID, 1, (3, 0), (),)

	def GetSegmentTime(self):
		return self._oleobj_.InvokeTypes(31, LCID, 1, (3, 0), (),)

	def GetSerialNumber(self):
		'method GetSerialNumber'
		# Result is a Unicode object
		return self._oleobj_.InvokeTypes(54, LCID, 1, (8, 0), (),)

	def GetSoftwareVersionMajor(self):
		return self._oleobj_.InvokeTypes(29, LCID, 1, (3, 0), (),)

	def GetSoftwareVersionMinor(self):
		return self._oleobj_.InvokeTypes(30, LCID, 1, (3, 0), (),)

	def GetSourceType(self):
		return self._oleobj_.InvokeTypes(26, LCID, 1, (2, 0), (),)

	def GetStartTime(self):
		ret = self._oleobj_.InvokeTypes(39, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetStartTime', None)
		return ret

	def GetStartTriggerStreamID(self):
		return self._oleobj_.InvokeTypes(21, LCID, 1, (3, 0), (),)

	def GetStopTime(self):
		ret = self._oleobj_.InvokeTypes(27, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetStopTime', None)
		return ret

	def GetStopTriggerStreamID(self):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (3, 0), (),)

	def GetStream(self, Index=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(16, LCID, 1, (9, 0), ((3, 0),),Index
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetStream', None)
		return ret

	def GetSweepCount(self):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (3, 0), (),)

	def GetSweepEndTimeAt(self, lSweep=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(20, LCID, 1, (9, 0), ((3, 0),),lSweep
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetSweepEndTimeAt', None)
		return ret

	def GetSweepLength(self):
		return self._oleobj_.InvokeTypes(18, LCID, 1, (3, 0), (),)

	def GetSweepRawStartTimeAt(self, sweepIndex=defaultNamedNotOptArg):
		'method GetSweepRawStartTimeAt'
		return self._oleobj_.InvokeTypes(50, LCID, 1, (21, 0), ((3, 0),),sweepIndex
			)

	def GetSweepStartTimeAt(self, lSweep=defaultNamedNotOptArg):
		ret = self._oleobj_.InvokeTypes(19, LCID, 1, (9, 0), ((3, 0),),lSweep
			)
		if ret is not None:
			ret = Dispatch(ret, 'GetSweepStartTimeAt', None)
		return ret

	def GetTimeStamp(self):
		ret = self._oleobj_.InvokeTypes(25, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetTimeStamp', None)
		return ret

	def GetTimeStampStartPressed(self):
		ret = self._oleobj_.InvokeTypes(38, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'GetTimeStampStartPressed', None)
		return ret

	def GetTotalChannels(self):
		return self._oleobj_.InvokeTypes(35, LCID, 1, (2, 0), (),)

	def GetVendorId(self):
		'method GetVendorId'
		return self._oleobj_.InvokeTypes(56, LCID, 1, (3, 0), (),)

	def GetVoltageRange(self):
		'method GetVoltageRange'
		return self._oleobj_.InvokeTypes(49, LCID, 1, (3, 0), (),)

	def GetWithIndex(self):
		return self._oleobj_.InvokeTypes(28, LCID, 1, (11, 0), (),)

	def HasStimulation(self):
		return self._oleobj_.InvokeTypes(59, LCID, 1, (11, 0), (),)

	def MCRack_DataSourceType(self):
		'method MCRack_DataSourceType'
		return self._oleobj_.InvokeTypes(52, LCID, 1, (3, 0), (),)

	def OpenFile(self, szFileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (2, 0), ((8, 0),),szFileName
			)

	def OpenFileEx(self, szFileName=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(40, LCID, 1, (2, 0), ((8, 0),),szFileName
			)

	_prop_map_get_ = {
		"Comment": (4, 2, (8, 0), (), "Comment", None),
		"Day": (7, 2, (2, 0), (), "Day", None),
		"HeaderVersion": (1, 2, (2, 0), (), "HeaderVersion", None),
		"Hour": (8, 2, (2, 0), (), "Hour", None),
		"Microsecond": (12, 2, (2, 0), (), "Microsecond", None),
		"MillisamplesPerSecond": (3, 2, (3, 0), (), "MillisamplesPerSecond", None),
		"Millisecond": (11, 2, (2, 0), (), "Millisecond", None),
		"Minute": (9, 2, (2, 0), (), "Minute", None),
		"Month": (6, 2, (2, 0), (), "Month", None),
		"Nanosecond": (13, 2, (2, 0), (), "Nanosecond", None),
		"Second": (10, 2, (2, 0), (), "Second", None),
		"StreamCount": (2, 2, (3, 0), (), "StreamCount", None),
		"Year": (5, 2, (2, 0), (), "Year", None),
	}
	_prop_map_put_ = {
		"Comment" : ((4, LCID, 4, 0),()),
		"Day" : ((7, LCID, 4, 0),()),
		"HeaderVersion" : ((1, LCID, 4, 0),()),
		"Hour" : ((8, LCID, 4, 0),()),
		"Microsecond" : ((12, LCID, 4, 0),()),
		"MillisamplesPerSecond" : ((3, LCID, 4, 0),()),
		"Millisecond" : ((11, LCID, 4, 0),()),
		"Minute" : ((9, LCID, 4, 0),()),
		"Month" : ((6, LCID, 4, 0),()),
		"Nanosecond" : ((13, LCID, 4, 0),()),
		"Second" : ((10, LCID, 4, 0),()),
		"StreamCount" : ((2, LCID, 4, 0),()),
		"Year" : ((5, LCID, 4, 0),()),
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

class IMCSTimeStamp(DispatchBaseClass):
	CLSID = IID('{06746632-DA9C-11D1-8EB8-0000B4552050}')
	coclass_clsid = IID('{06746633-DA9C-11D1-8EB8-0000B4552050}')

	def Clone(self):
		ret = self._oleobj_.InvokeTypes(18, LCID, 1, (9, 0), (),)
		if ret is not None:
			ret = Dispatch(ret, 'Clone', None)
		return ret

	def GetDay(self):
		return self._oleobj_.InvokeTypes(3, LCID, 1, (2, 0), (),)

	def GetHour(self):
		return self._oleobj_.InvokeTypes(4, LCID, 1, (2, 0), (),)

	def GetMicrosecond(self):
		return self._oleobj_.InvokeTypes(8, LCID, 1, (2, 0), (),)

	def GetMicrosecondFromStart(self):
		return self._oleobj_.InvokeTypes(12, LCID, 1, (2, 0), (),)

	def GetMillisecond(self):
		return self._oleobj_.InvokeTypes(7, LCID, 1, (2, 0), (),)

	def GetMillisecondFromStart(self):
		return self._oleobj_.InvokeTypes(11, LCID, 1, (2, 0), (),)

	def GetMinute(self):
		return self._oleobj_.InvokeTypes(5, LCID, 1, (2, 0), (),)

	def GetMonth(self):
		return self._oleobj_.InvokeTypes(2, LCID, 1, (2, 0), (),)

	def GetNanosecond(self):
		return self._oleobj_.InvokeTypes(9, LCID, 1, (2, 0), (),)

	def GetNanosecondFromStart(self):
		return self._oleobj_.InvokeTypes(13, LCID, 1, (2, 0), (),)

	def GetSecond(self):
		return self._oleobj_.InvokeTypes(6, LCID, 1, (2, 0), (),)

	def GetSecondFromStart(self):
		return self._oleobj_.InvokeTypes(10, LCID, 1, (3, 0), (),)

	def GetStartTimeMajor(self):
		return self._oleobj_.InvokeTypes(21, LCID, 1, (3, 0), (),)

	def GetStartTimeMinor(self):
		return self._oleobj_.InvokeTypes(22, LCID, 1, (3, 0), (),)

	def GetTimeStampMajor(self):
		return self._oleobj_.InvokeTypes(20, LCID, 1, (3, 0), (),)

	def GetTimeStampMinor(self):
		return self._oleobj_.InvokeTypes(19, LCID, 1, (3, 0), (),)

	def GetYear(self):
		return self._oleobj_.InvokeTypes(1, LCID, 1, (2, 0), (),)

	def SetMicrosecondFromStart(self, sMicro=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(15, LCID, 1, (24, 0), ((2, 0),),sMicro
			)

	def SetMillisecondFromStart(self, sMilli=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(14, LCID, 1, (24, 0), ((2, 0),),sMilli
			)

	def SetNanosecondFromStart(self, sNano=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(16, LCID, 1, (24, 0), ((2, 0),),sNano
			)

	def SetSecondFromStart(self, sSecond=defaultNamedNotOptArg):
		return self._oleobj_.InvokeTypes(17, LCID, 1, (24, 0), ((3, 0),),sSecond
			)

	_prop_map_get_ = {
	}
	_prop_map_put_ = {
	}
	def __iter__(self):
		"Return a Python iterator for this object"
		try:
			ob = self._oleobj_.InvokeTypes(-4,LCID,3,(13, 10),())
		except pythoncom.error:
			raise TypeError("This object does not support enumeration")
		return win32com.client.util.Iterator(ob, None)

from win32com.client import CoClassBaseClass
class MCSCHANNEL(CoClassBaseClass): # A CoClass
	CLSID = IID('{F68B2B01-40BC-11D1-B441-0080C6FF1BCF}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSChannel,
	]
	default_interface = IMCSChannel

class MCSCHUNK(CoClassBaseClass): # A CoClass
	CLSID = IID('{904180C1-C09F-11D1-AC75-E498383B5A44}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSChunk,
	]
	default_interface = IMCSChunk

class MCSChannelLayout(CoClassBaseClass): # A CoClass
	CLSID = IID('{BC6A0146-FAE1-450E-BC78-FFCF76E3C6B2}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSChannelLayout,
	]
	default_interface = IMCSChannelLayout

class MCSEVENT(CoClassBaseClass): # A CoClass
	CLSID = IID('{904180C3-C09F-11D1-AC75-E498383B5A44}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvent,
	]
	default_interface = IMCSEvent

class MCSEVTPARAM(CoClassBaseClass): # A CoClass
	CLSID = IID('{06746629-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtParam,
	]
	default_interface = IMCSEvtParam

class MCSEVTRAW(CoClassBaseClass): # A CoClass
	CLSID = IID('{06746621-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtRaw,
	]
	default_interface = IMCSEvtRaw

class MCSEVTSPIKE(CoClassBaseClass): # A CoClass
	CLSID = IID('{06746623-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtSpike,
	]
	default_interface = IMCSEvtSpike

class MCSEVTSPIKEPARAMETER(CoClassBaseClass): # A CoClass
	CLSID = IID('{4AFCF788-E915-47B2-8BFC-479914F63892}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtSpikeParameter,
	]
	default_interface = IMCSEvtSpikeParameter

class MCSEVTTRIGGER(CoClassBaseClass): # A CoClass
	CLSID = IID('{06746625-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtTrigger,
	]
	default_interface = IMCSEvtTrigger

class MCSEvtAverage(CoClassBaseClass): # A CoClass
	CLSID = IID('{012F99E0-D08C-4793-AC6E-928057B2F694}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtAverage,
	]
	default_interface = IMCSEvtAverage

class MCSEvtBurstParameter(CoClassBaseClass): # A CoClass
	CLSID = IID('{55CAAEE3-E497-4724-ADF9-72561ED007AF}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSEvtBurstParameter,
	]
	default_interface = IMCSEvtBurstParameter

class MCSINFOPARAM(CoClassBaseClass): # A CoClass
	CLSID = IID('{06746631-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoParam,
	]
	default_interface = IMCSInfoParam

class MCSINFORAW(CoClassBaseClass): # A CoClass
	CLSID = IID('{0674662B-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoRaw,
	]
	default_interface = IMCSInfoRaw

class MCSINFOSPIKE(CoClassBaseClass): # A CoClass
	CLSID = IID('{0674662D-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoSpike,
	]
	default_interface = IMCSInfoSpike

class MCSINFOSPIKEPARAMETER(CoClassBaseClass): # A CoClass
	CLSID = IID('{6EFEC165-D03D-40ED-9C02-24302793225C}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoSpikeParameter,
	]
	default_interface = IMCSInfoSpikeParameter

class MCSINFOTRIGGER(CoClassBaseClass): # A CoClass
	CLSID = IID('{0345A2C1-A4EF-11D2-98D5-444553540000}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoTrigger,
	]
	default_interface = IMCSInfoTrigger

class MCSInfoAverage(CoClassBaseClass): # A CoClass
	CLSID = IID('{A3073478-3666-40D7-B1C0-99552C1B8E99}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoAverage,
	]
	default_interface = IMCSInfoAverage

class MCSInfoBurstParameter(CoClassBaseClass): # A CoClass
	CLSID = IID('{FA067E74-9370-4DBF-91B5-3166527A4EA5}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoBurstParameter,
	]
	default_interface = IMCSInfoBurstParameter

class MCSInfoChannelTool(CoClassBaseClass): # A CoClass
	CLSID = IID('{731B89DA-6D16-4D18-9DB2-10BC51F1B78B}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoChannelTool,
	]
	default_interface = IMCSInfoChannelTool

class MCSInfoFilter(CoClassBaseClass): # A CoClass
	CLSID = IID('{26BB09B3-6E95-11D4-8FFB-008048B000C7}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSInfoFilter,
	]
	default_interface = IMCSInfoFilter

class MCSLayout(CoClassBaseClass): # A CoClass
	CLSID = IID('{4A5ABA66-BADC-4438-8B7D-B9ED24863290}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSLayout,
	]
	default_interface = IMCSLayout

class MCSMea21StimulatorProperty(CoClassBaseClass): # A CoClass
	CLSID = IID('{F50409D9-9982-40C8-AF80-181B8E11F23E}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSMea21StimulatorProperty,
	]
	default_interface = IMCSMea21StimulatorProperty

class MCSSTREAM(CoClassBaseClass): # A CoClass
	CLSID = IID('{F5B4BFA1-40A5-11D1-B441-0080C6FF1BCF}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSStream,
	]
	default_interface = IMCSStream

# This CoClass is known by the name 'MCSTREAM.MCSSTRM'
class MCSSTRM(CoClassBaseClass): # A CoClass
	CLSID = IID('{E27A2D85-400F-11D1-B517-52544C190838}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSStreamFile,
	]
	default_interface = IMCSStreamFile

class MCSTIMESTAMP(CoClassBaseClass): # A CoClass
	CLSID = IID('{06746633-DA9C-11D1-8EB8-0000B4552050}')
	coclass_sources = [
	]
	coclass_interfaces = [
		IMCSTimeStamp,
	]
	default_interface = IMCSTimeStamp

RecordMap = {
}

CLSIDToClassMap = {
	'{E27A2D84-400F-11D1-B517-52544C190838}' : IMCSStreamFile,
	'{E27A2D85-400F-11D1-B517-52544C190838}' : MCSSTRM,
	'{F5B4BFA0-40A5-11D1-B441-0080C6FF1BCF}' : IMCSStream,
	'{F5B4BFA1-40A5-11D1-B441-0080C6FF1BCF}' : MCSSTREAM,
	'{F68B2B00-40BC-11D1-B441-0080C6FF1BCF}' : IMCSChannel,
	'{F68B2B01-40BC-11D1-B441-0080C6FF1BCF}' : MCSCHANNEL,
	'{904180C0-C09F-11D1-AC75-E498383B5A44}' : IMCSChunk,
	'{904180C1-C09F-11D1-AC75-E498383B5A44}' : MCSCHUNK,
	'{904180C2-C09F-11D1-AC75-E498383B5A44}' : IMCSEvent,
	'{904180C3-C09F-11D1-AC75-E498383B5A44}' : MCSEVENT,
	'{06746620-DA9C-11D1-8EB8-0000B4552050}' : IMCSEvtRaw,
	'{06746621-DA9C-11D1-8EB8-0000B4552050}' : MCSEVTRAW,
	'{06746622-DA9C-11D1-8EB8-0000B4552050}' : IMCSEvtSpike,
	'{06746623-DA9C-11D1-8EB8-0000B4552050}' : MCSEVTSPIKE,
	'{06746624-DA9C-11D1-8EB8-0000B4552050}' : IMCSEvtTrigger,
	'{06746625-DA9C-11D1-8EB8-0000B4552050}' : MCSEVTTRIGGER,
	'{06746628-DA9C-11D1-8EB8-0000B4552050}' : IMCSEvtParam,
	'{06746629-DA9C-11D1-8EB8-0000B4552050}' : MCSEVTPARAM,
	'{0674662A-DA9C-11D1-8EB8-0000B4552050}' : IMCSInfoRaw,
	'{0674662B-DA9C-11D1-8EB8-0000B4552050}' : MCSINFORAW,
	'{0674662C-DA9C-11D1-8EB8-0000B4552050}' : IMCSInfoSpike,
	'{0674662D-DA9C-11D1-8EB8-0000B4552050}' : MCSINFOSPIKE,
	'{06746630-DA9C-11D1-8EB8-0000B4552050}' : IMCSInfoParam,
	'{06746631-DA9C-11D1-8EB8-0000B4552050}' : MCSINFOPARAM,
	'{06746632-DA9C-11D1-8EB8-0000B4552050}' : IMCSTimeStamp,
	'{06746633-DA9C-11D1-8EB8-0000B4552050}' : MCSTIMESTAMP,
	'{0345A2C0-A4EF-11D2-98D5-444553540000}' : IMCSInfoTrigger,
	'{0345A2C1-A4EF-11D2-98D5-444553540000}' : MCSINFOTRIGGER,
	'{26BB09B1-6E95-11D4-8FFB-008048B000C7}' : IMCSInfoFilter,
	'{26BB09B3-6E95-11D4-8FFB-008048B000C7}' : MCSInfoFilter,
	'{A8F8C595-7B79-49F2-9805-8AF57F5EF4CB}' : IMCSInfoAverage,
	'{A3073478-3666-40D7-B1C0-99552C1B8E99}' : MCSInfoAverage,
	'{B93B64BF-0CB4-4D71-8B1A-779F897491DD}' : IMCSEvtAverage,
	'{012F99E0-D08C-4793-AC6E-928057B2F694}' : MCSEvtAverage,
	'{12C1F607-3162-425D-B116-044C937334E2}' : IMCSEvtSpikeParameter,
	'{4AFCF788-E915-47B2-8BFC-479914F63892}' : MCSEVTSPIKEPARAMETER,
	'{4DBE71EE-0635-4F56-B89B-8B5BC384DC4C}' : IMCSInfoSpikeParameter,
	'{6EFEC165-D03D-40ED-9C02-24302793225C}' : MCSINFOSPIKEPARAMETER,
	'{D52FFF63-73C3-471B-8227-222B0F5615E1}' : IMCSLayout,
	'{4A5ABA66-BADC-4438-8B7D-B9ED24863290}' : MCSLayout,
	'{D528F251-3971-45BF-8B0C-2959ED502FC2}' : IMCSChannelLayout,
	'{BC6A0146-FAE1-450E-BC78-FFCF76E3C6B2}' : MCSChannelLayout,
	'{C7D985C4-0403-4229-833F-7A9325E2F86B}' : IMCSEvtBurstParameter,
	'{55CAAEE3-E497-4724-ADF9-72561ED007AF}' : MCSEvtBurstParameter,
	'{02408658-BB5F-48E9-8E30-B9B1520785C6}' : IMCSInfoBurstParameter,
	'{FA067E74-9370-4DBF-91B5-3166527A4EA5}' : MCSInfoBurstParameter,
	'{DF7EDC7A-C505-4D57-9F94-902637AF2DCD}' : IMCSInfoChannelTool,
	'{731B89DA-6D16-4D18-9DB2-10BC51F1B78B}' : MCSInfoChannelTool,
	'{BB528DF6-FC0E-4B0B-A593-521EEBA14FAE}' : IMCSMea21StimulatorProperty,
	'{F50409D9-9982-40C8-AF80-181B8E11F23E}' : MCSMea21StimulatorProperty,
}
CLSIDToPackageMap = {}
win32com.client.CLSIDToClass.RegisterCLSIDsFromDict( CLSIDToClassMap )
VTablesToPackageMap = {}
VTablesToClassMap = {
}


NamesToIIDMap = {
	'IMCSStreamFile' : '{E27A2D84-400F-11D1-B517-52544C190838}',
	'IMCSStream' : '{F5B4BFA0-40A5-11D1-B441-0080C6FF1BCF}',
	'IMCSChannel' : '{F68B2B00-40BC-11D1-B441-0080C6FF1BCF}',
	'IMCSChunk' : '{904180C0-C09F-11D1-AC75-E498383B5A44}',
	'IMCSEvent' : '{904180C2-C09F-11D1-AC75-E498383B5A44}',
	'IMCSEvtRaw' : '{06746620-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSEvtSpike' : '{06746622-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSEvtTrigger' : '{06746624-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSEvtParam' : '{06746628-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSInfoRaw' : '{0674662A-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSInfoSpike' : '{0674662C-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSInfoParam' : '{06746630-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSTimeStamp' : '{06746632-DA9C-11D1-8EB8-0000B4552050}',
	'IMCSInfoTrigger' : '{0345A2C0-A4EF-11D2-98D5-444553540000}',
	'IMCSInfoFilter' : '{26BB09B1-6E95-11D4-8FFB-008048B000C7}',
	'IMCSInfoAverage' : '{A8F8C595-7B79-49F2-9805-8AF57F5EF4CB}',
	'IMCSEvtAverage' : '{B93B64BF-0CB4-4D71-8B1A-779F897491DD}',
	'IMCSEvtSpikeParameter' : '{12C1F607-3162-425D-B116-044C937334E2}',
	'IMCSInfoSpikeParameter' : '{4DBE71EE-0635-4F56-B89B-8B5BC384DC4C}',
	'IMCSLayout' : '{D52FFF63-73C3-471B-8227-222B0F5615E1}',
	'IMCSChannelLayout' : '{D528F251-3971-45BF-8B0C-2959ED502FC2}',
	'IMCSEvtBurstParameter' : '{C7D985C4-0403-4229-833F-7A9325E2F86B}',
	'IMCSInfoBurstParameter' : '{02408658-BB5F-48E9-8E30-B9B1520785C6}',
	'IMCSInfoChannelTool' : '{DF7EDC7A-C505-4D57-9F94-902637AF2DCD}',
	'IMCSMea21StimulatorProperty' : '{BB528DF6-FC0E-4B0B-A593-521EEBA14FAE}',
}


