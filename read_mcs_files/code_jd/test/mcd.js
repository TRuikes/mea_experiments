class MCDFile{
    constructor(file, callback){
        this.file = file
        this.MCDReader = null
        callback(this)
    }

    loadFile(){
        let reader = new FileReader()
        reader.onload = (ev) => {
            const data = ev.target.result
            let dataArray = new Uint8Array(data)
            let stream = FS.open("experiment.mcd",'w+')
            FS.write(stream,dataArray,0,dataArray.length,0)
            FS.close(stream)

            // this.MCDReader = Module.initMCDFile()
        }
        reader.readAsArrayBuffer(this.file)
    }
}