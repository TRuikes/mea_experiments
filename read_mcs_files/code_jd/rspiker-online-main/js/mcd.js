/*
== A FRAME ==
find "digi0" in unicode
32 dead bytes
4000 bytes of digital channel data (2000 points x 2bytes -- 16bits)
"elec0" in unicode lies here
32 dead bytes
256 * 4000 bytes of electrodes data (256 electrodes x 2000 points x 2bytes -- 16 bits)
## 16bits is specifically unsigned short int in this case
## The sequence is one data point per electrode : 2bytes for El#1, 2bytes for El#2, 2bytes for El#3 ... 2bytes for El#256 , 2bytes for El#1 ...
== this represent 100ms at 20kHz ==

*/

class MCDFile{
    constructor(file, callback){
        this.file = file
        this.start = 0
        this.dataArray = null
        this.MCDReader = null
        callback(this)
    }

    loadFile(callback){
        let reader = new FileReader()
        reader.onload = (ev) => {
            const data = ev.target.result
            this.dataArray = new Uint8Array(data)
            let stream = FS.open("experiment.mcd",'w+')
            FS.write(stream,this.dataArray,0,this.dataArray.length,0)
            FS.close(stream)

            this.MCDReader = Module.initMCDFile()
            callback()
        }
        reader.readAsArrayBuffer(this.file)
    }

    loadFrames(firstFrame, size){
        Module.loadFrames(firstFrame,size)
    }

    get filename(){
        return this.file.name
    }

    get basename(){
        return this.file.name.split('.mcd')[0]
    }

    get filesize(){
        return this.MCDReader.filesize
    }

    get framesCount(){
        return this.MCDReader.framesCount
    }

    get header(){
        return {
            timeStart: fileTimeToDate(this.MCDReader.getTimeStart()),
            timeStop: fileTimeToDate(this.MCDReader.getTimeStop()),
            timePush: fileTimeToDate(this.MCDReader.getTimePush()),
        }
    }

    get duration(){
        return Math.floor((this.header.timeStop - this.header.timeStart) /100) / 10
    }
}

