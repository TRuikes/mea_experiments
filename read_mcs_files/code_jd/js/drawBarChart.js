class PlotBarChart{
    constructor(id){
        this.canvas = document.getElementById(id)
        this.ctx = this.canvas.getContext('2d')
        this.controls = new Map()
        this.ctx.transform(1, 0, 0, -1, 0, this.canvas.height)
    }

    addControl(id){
        const domElement = document.getElementById(id)
        domElement.onchange = ()=> {
            this.plot()
        }
        this.controls.set(id,domElement)
    }

    addButton(id){
        const domElement = document.getElementById(id)
        domElement.onclick = ()=> {
            this.plot()
        }
    }

    plot(){
        if(!controls.params.checkboxRaster) return console.error('Compute rasterplot first')
        const divSingleElectrode = document.getElementById('tile-Single')
        if(!divSingleElectrode.checkVisibility()) return console.error('Select electrode first')

        //MCD data consts
        const n = Number(divSingleElectrode.getAttribute('ax-electrode'))
        const size = Module.getRasteredSize()

        //User params
        const timeWidth = controls.params.inputTimeWindowWidth
        const lineWidthInFrames = timeWidth * 10
        const lineWidth = lineWidthInFrames * 2000
        const firstPoint = controls.params.inputOffset / 100 * 2000
        const barWidth = this.controls.get('inputBarWidth').value
        const barsCount = Math.ceil(timeWidth * 1000 / barWidth)
        const barPointsWidth = lineWidth/barsCount
        const barStart = this.controls.get('inputBarStart').value
        const barEnd = this.controls.get('inputBarEnd').value

        let bars = new Array(barsCount).fill(0)
        let barsStim = new Array(barsCount).fill(0)
        let barsTotal = new Array(barsCount).fill(0)

        for(let i=firstPoint; i<size;i++){
            if(i < barStart/1000 * 20000 || (barEnd !=0 && i > barEnd/1000 * 20000)) continue
            const v = Module.getRastered(n,i)
            if(v==false) continue
            const j = i - firstPoint
            const x = j % lineWidth
            barsTotal[Math.floor(x/barPointsWidth)]++
            if(!Module.getStimFromCache(n,i)) bars[Math.floor(x/barPointsWidth)]++
            else barsStim[Math.floor(x/barPointsWidth)]++
        }

        const xStep = this.canvas.width / (timeWidth * 1000 / barWidth)
        const yStep = this.canvas.height / Math.max(...barsTotal)
        this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height)
        let exportBars = `time;stack;stackOFF;stackON\n`
        for(let b = 0; b<barsCount; b++){
            this.ctx.fillStyle = config.color.raster
            this.ctx.fillRect(b*xStep,0,xStep,bars[b] * yStep)
            this.ctx.fillStyle = config.color.stimulationRaster
            this.ctx.fillRect(b*xStep,bars[b] * yStep,xStep,barsStim[b] * yStep)

            exportBars += `${(b * barWidth).toFixed(3)};${bars[b]+barsStim[b]};${bars[b]};${barsStim[b]}\n`
        }

        if(document.getElementById('checkboxExportStack').checked) download(`stack-${mcdFile.basename}-E${n}.csv`,exportBars)
    }
}

function exportRaster(){
    if(!controls.params.checkboxRaster) return console.error('Compute rasterplot first')
    const divSingleElectrode = document.getElementById('tile-Single')
    if(!divSingleElectrode.checkVisibility()) return console.error('Select electrode first')
    //MCD data consts
    const n = Number(divSingleElectrode.getAttribute('ax-electrode'))
    const size = Module.getRasteredSize()

    let points = []

    for(let i=0; i<size;i++){
        const v = Module.getRastered(n,i)
        if(v==false) continue
        points.push(i/20000)
    }

    download(`raster-${mcdFile.basename}-E${n}.csv`,points.join(';'))
}

function multipleStack(){
    if(!controls.params.checkboxRaster) return console.error('Compute rasterplot first')
    const electrodes = document.getElementById('inputMultipleElectrodeStack').value.split(' ')
    if(electrodes.length && electrodes[0] == '') return console.error('Empty list')
    document.getElementById('checkboxExportStack').checked = true
    // electrodes.forEach((electrode)=>{
    //     selectElectrode({target:document.querySelector(`[ax-electrode="${electrode}"]`)})
    //     rasterBarChart.plot()
    // })
    const laps = document.getElementById(`inputStackDownloadLaps`)
    for(let i = 0; i<electrodes.length; i++){
        setTimeout(()=>{
            selectElectrode({target:document.querySelector(`[ax-electrode="${electrodes[i]}"]`)})
            rasterBarChart.plot()
        },i*laps)
    }
}

function exportTotalSpikes(){
    const exclusionList = [128,255]
    const size = Module.getRasteredSize()
    const firstPoint = controls.params.inputOffset / 1000 * 20000
    const barStart = document.getElementById('inputBarStart').value
    const barEnd = document.getElementById('inputBarEnd').value

    let totalSpikes = new Array(257).fill(0)
    let totalSpikesOff = new Array(257).fill(0)
    let totalSpikesOn = new Array(257).fill(0)

    for(let electrode = 1; electrode <= 256;electrode++){
        if(exclusionList.includes(electrode)) continue
        for(let i=firstPoint;i<size;i++){
            if(i < barStart/1000 * 20000 || (barEnd !=0 && i > barEnd/1000 * 20000)) continue
            const v = Module.getRastered(electrode,i)
            if(v == true){
                if(!Module.getStimFromCache(electrode,i)) totalSpikesOff[electrode]++
                else totalSpikesOn[electrode]++
                totalSpikes[electrode]++
            }
        }
    }

    let exportSpikes = `electrode;spikes;spikesOFF;spikesON\n`
    let maxOn,maxOff,max
    maxOn = Math.max(...totalSpikesOn)
    maxOff = Math.max(...totalSpikesOff)
    max = Math.max(...totalSpikes)
    exportSpikes += `MAX;${max};${maxOff};${maxOn}\n`

    for(let electrode = 1; electrode <= 256; electrode++){
        exportSpikes += `${electrode};${totalSpikesOff[electrode]+totalSpikesOn[electrode]};${totalSpikesOff[electrode]};${totalSpikesOn[electrode]}\n`
    }

    download(`totalSpikes-${mcdFile.basename}.txt`,exportSpikes)

}