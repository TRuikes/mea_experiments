import fs from 'fs'

//The header of the MCD contains file information, such as stimulation start, stimulation end
const headerSize = 287864
//A frame is a 100ms piece of data containing, in order, the digital electrodes and the 256 analog electrodes
const frameSize = 1028064

//We check if there is enough parameters, else the script stops
if(process.argv.length < 4) {
    console.error('Usage: node splitter.mjs pathToMCDFile chunkDuration')
    process.exit(1)
}
console.log('Axorus Splitter 0.11')
//The path to the mcd file is the first parameter, thus the third argument, thus the #2 in the array (which starts at 0)
const path = process.argv[2]
//Same with the chunk duration as the second parameter
const chunkDuration = process.argv[3] //in sec
//We check if the file has a ".mcd" extension, else exit
const fileExtension = path.substring(path.length-4)
if(fileExtension != '.mcd'){
    console.error('Expected MCD file')
    process.exit(1)
}

const fileSize = fs.statSync(path).size
//To count the total number of frames, we divide the data part of the file (file size - header size) by the frame size. And take the integer part
const framesCount = Math.floor((fileSize - headerSize) / frameSize)
//To count the total number of chunks, we divide the number of frames (of 100ms) by the duration of a single chunk (in sec converted in 100ms)
const totalChunks = Math.floor(framesCount / (chunkDuration * 10))

//We open the source mcd file in read-only mode
const fd = fs.openSync(path,'r')
//We set two buffers respectively to the size of the header and the size of a single chunk
let headerBuffer = Buffer.alloc(headerSize)
let chunkBuffer = Buffer.alloc(frameSize * chunkDuration * 10)

//We read, once and for all, the header into its buffer
fs.readSync(fd, headerBuffer)

//We initiate an iterator i and a blank chunk path (aka the chunk file name)
let i = 0
let chunkPath = ""

//We read the first chunk into its buffer and iterate until the end of the file
while(fs.readSync(fd, chunkBuffer)){
    //We build the chunk file name
    chunkPath = path.substring(0,path.length-4)+'.chunk'+i+fileExtension
    //For each chunk file we copy the source file header and the chunk data
    fs.appendFileSync(chunkPath,headerBuffer)
    fs.appendFileSync(chunkPath,chunkBuffer)
    //We print the progress status on the console the progressbar
    process.stdout.clearLine(0)
    process.stdout.cursorTo(0)
    process.stdout.write(`Writing chunk [${"=".repeat(Math.round(i/totalChunks*40))}${"-".repeat(Math.round((totalChunks-i)/totalChunks*40))}] ${i} / ${totalChunks}`)
    //We increase i to build the next chunk file name
    i++
}

//We truncate the last chunk to its actual size instead of the default chunk buffer size
const lastChunkSize = (fileSize - headerSize) % (frameSize * chunkDuration * 10) + headerSize
fs.truncateSync(chunkPath,lastChunkSize)