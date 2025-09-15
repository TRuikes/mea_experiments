class Grid{
    constructor(id, idSingle){
        this.div = document.getElementById(id)
        this.divSingle = document.getElementById(idSingle)
        this.divSingle.hidden = true
        this.tiles = []
        this.canvases = []
        this.plots = []
        this.currentRasterLine = 0
        this.rasterCached = false
        this.build()
    }

    build(){
        //MEA Grid
        for(let i=0;i<256;i++){
            const n = config.map_mea[i]

            let tile = document.createElement('div')
            let canvas = document.createElement('canvas')
            tile.appendChild(canvas)
            
            tile.classList.add("tile")
            tile.setAttribute('ax-electrode',n)
            tile.onclick = selectElectrode
            canvas.width = config.gridCanvas.w
            canvas.height = config.gridCanvas.h
            canvas.id = 'tile-'+n
            canvas.setAttribute('ax-electrode',n)
            this.div.appendChild(tile)
            this.tiles.push(tile)
            this.canvases.push(canvas)
            this.plots.push(new Plot(canvas))
        }

        //MEA Single Electrode
        let tile = document.createElement('div')
        let canvas = document.createElement('canvas')
        tile.appendChild(canvas)
        tile.classList.add("tileSingle")
        tile.onclick = closeElectrode
        canvas.width = config.gridCanvas.w * 16 + 15*8
        canvas.height = config.gridCanvas.h * 16 + 15*8
        canvas.id = 'tile-Single'
        canvas.onmouseover = infoElectrodeOver
        canvas.onmousemove = infoElectrodeMove
        canvas.onmouseout = infoElectrodeOut

        canvas.onmousedown = infoElectrodeDown
        canvas.onmouseup = infoElectrodeUp


        this.divSingle.appendChild(tile)
        this.tiles.push(tile)
        this.canvases.push(canvas)
        this.plots.push(new Plot(canvas))
    }

    plot(){
        if(controls.params.checkboxRaster) return this.plotRaster()
        clearInterval(this.intervalRaster)
        loading.reset(0)
        this.plots.forEach((plot)=>{
            setTimeout(()=> plot.plot(),1)
        })
    }

    plotSingle(){
        this.plots[this.plots.length-1].plot()
    }

    clear(){
        console.log("Clear Grid")
        this.plots.forEach((plot)=>{
            plot.clear()
        })
    }

    calculateRasterLines() {
        const framesCount = mcdFile.framesCount
        const timeWindowWidth100ms = controls.params.inputTimeWindowWidth * 10
        return Math.ceil(framesCount / timeWindowWidth100ms)
    }

    plotRaster(){
        this.clear()

        const rastered = Module.isRastered()
        this.rasterLines = this.calculateRasterLines()
        this.currentRasterLine = 0
        loading.reset(this.rasterLines)

        if(!rastered){
            this.intervalRaster = setInterval(()=>{
                this.loadLines()
                loading.add()
            },1)
            return
        }
        for(let n = 0; n < this.plots.length; n++){
            setTimeout(()=> this.plots[n].plotRaster(),1)
        }
    }

    loadLines(){
        const line = this.currentRasterLine++
        if(this.currentRasterLine > this.rasterLines){
            clearInterval(this.intervalRaster)
            Module.setRastered(true)
            this.plot()
            loading.reset(0)
            return
        }

        const timeWindowWidth100ms = controls.params.inputTimeWindowWidth * 10
        const firstFrame = line * timeWindowWidth100ms

        mcdFile.loadFrames(firstFrame,timeWindowWidth100ms)
        Module.loadRaster()
    }

    scale(factor){
        this.plots.forEach((plot)=>{
            plot.scale = factor
        })
    }

    toggleSingle(){
        this.div.hidden = true
        this.divSingle.hidden = false
    }

    toggleGrid(){
        this.div.hidden = false
        this.divSingle.hidden = true
    }

    highlightElectrode(nSelected){
        for(let n = 0; n < this.plots.length; n++){
            this.plots[n].highlight(nSelected)
        }
    }

}


class Plot{
    constructor(tile){
        this.canvas = tile
        this.ctx = this.canvas.getContext('2d')
        this.w = this.canvas.width
        this.h = this.canvas.height
        this.ctx.transform(1, 0, 0, -1, 0, this.h)
        this.scale = 1
    }

    plot(){
        if(!this.canvas.checkVisibility()) return
        const n = Number(this.canvas.getAttribute('ax-electrode'))
        const zero = 32768
        const amplitude = 0.1042
        this.clear()
        const size = Module.getSize(n)
        
        //Converters
        const Xpos = (i)=> zoom.getX(i/size) * this.w
        const Ypos = (v)=>{
            const yStep = this.h * this.scale
            return v*yStep/2+this.h/2
        }

        //Stim
        if(controls.params.checkboxStimulation){
            this.ctx.fillStyle = config.color.stimulation
            for(let i = 0; i < size; i++){
                const v = Module.getStim(128,i)
                if(v) this.ctx.fillRect(Xpos(i),0,1,this.h)
            }
        }
        
        //Raw
        if(controls.params.checkboxRaw){
            this.ctx.strokeStyle = config.color.raw
            this.ctx.beginPath()
            this.ctx.moveTo(0,this.h/2)
            for(let i = 0; i < size; i++){
                const v = (Module.get(n,i) - zero) * amplitude
                this.ctx.lineTo(Xpos(i),Ypos(v))
            }
            this.ctx.stroke()
        }

        //Filtered
        if(controls.params.checkboxFiltered){
            this.ctx.strokeStyle = config.color.filtered
            this.ctx.beginPath()
            this.ctx.moveTo(0,this.h/2)
            for(let i = 0; i < size; i++){
                const v = Module.getFiltered(n,i) * amplitude
                this.ctx.lineTo(Xpos(i),Ypos(v))
            }
            this.ctx.stroke()
        }

        //MAD
        if(this.canvas.id == 'tile-Single' && controls.params.checkboxMAD){
            this.ctx.strokeStyle = config.color.spikes
            const v = -1 * Module.getMAD(n) * amplitude * controls.params.inputThreshold
            const y = Ypos(v)
            // this.ctx.fillRect(0,y,this.w,1)
            this.ctx.beginPath()
            this.ctx.setLineDash([5, 10])
            this.ctx.moveTo(0, y)
            this.ctx.lineTo(this.w, y)
            this.ctx.stroke()
            this.ctx.setLineDash([])
        }

        //Spikes
        if(controls.params.checkboxSpikes){
            this.ctx.fillStyle = config.color.spikes
            for(let i = 0; i < size; i++){
                const v = Module.getSpikesorted(n,i)
                if(v == 1) this.ctx.fillRect(Xpos(i),this.h/4,1,this.h/2)
            }
        }

        //Scale
        if(this.canvas.id == 'tile-Single' && controls.params.checkboxScale){
            this.ctx.fillStyle = config.color.scale
            this.ctx.font = '0.8rem sans-serif'
            
            this.ctx.save()
            this.ctx.scale(1,-1)
            this.ctx.translate(0,-this.h/2)
            //Abscissa
            this.ctx.fillRect(0,0,this.w,1)
            this.ctx.fillRect(0,-5,1,10)
            const t0 = (controls.params.rangeFirstFrame + zoom.Xpos(0)) * controls.params.inputTimeWindowWidth * 1000
            this.ctx.fillText(t0.toFixed(0),2,-10)
            this.ctx.fillRect(this.w/2,-5,1,10)
            const t1 = ((controls.params.rangeFirstFrame + zoom.Xpos(0.5)) * controls.params.inputTimeWindowWidth * 1000)
            this.ctx.fillText(t1.toFixed(0),2+this.w/2,-10)
            this.ctx.fillText('ms',this.w - 30,10)
            //Ordinate
            const v = (-2*0.25 + 1) * Math.pow(10,-1*controls.params.inputScaleMagnitude)
            this.ctx.fillRect(0,-this.h/4,this.w,1)
            this.ctx.fillText(`${v.toFixed(0)} µV`,2,-this.h/4-10)
            this.ctx.fillRect(0,this.h/4,this.w,1)
            this.ctx.fillText(`-${v.toFixed(0)} µV`,2,this.h/4-10)
            
            this.ctx.restore()
        }  
    }

    plotRaster(){
        if(!this.canvas.checkVisibility()) return
        const totalLengthInFrames = mcdFile.framesCount
        const lineWidthInFrames = controls.params.inputTimeWindowWidth * 10
        const lineWidth = lineWidthInFrames * 2000
        const firstPoint = controls.params.inputOffset / 100 * 2000

        const xStep = this.w / lineWidth
        const yStep = this.h / Math.ceil(totalLengthInFrames/lineWidthInFrames)

        const n = Number(this.canvas.getAttribute('ax-electrode'))
        const size = Module.getRasteredSize()

        for(let i=firstPoint; i<size;i++){
            const v = Module.getRastered(n,i)
            if(v==false) continue
            const j = i - firstPoint
            const x = j % lineWidth
            const y = Math.floor(j / lineWidth)
            if(Module.getStimFromCache(n,i)) this.ctx.fillStyle = config.color.stimulationRaster
            else this.ctx.fillStyle = config.color.raster
            this.ctx.fillRect(x*xStep,y*yStep,config.rasterSpikesWidth,yStep)
        }
    }

    clear(){
        this.ctx.clearRect(0,0,this.w,this.h)
    }

    highlight(nSelected){
        const n = Number(this.canvas.getAttribute('ax-electrode'))
        const parent = this.canvas.parentNode
        if(n == nSelected) return parent.classList.add('selectedElectrode')
        parent.classList.remove('selectedElectrode')
    }
}

function selectElectrode(ev){
    const n = ev.target.getAttribute('ax-electrode')
    document.querySelector('#MEASingleElectrode > div').setAttribute('ax-electrode',n)
    document.querySelector('#MEASingleElectrode > div > canvas').setAttribute('ax-electrode',n)
    grid.toggleSingle()
    grid.plot()
}

function closeElectrode(ev){
    if(ev.layerX < 970 || ev.layerY > 30) return
    analysisMenu.closeAll()
    grid.toggleGrid()
    grid.plot()
}

//Electrode Legend
function infoElectrodeUpdate(ev){
    const canvas = grid.canvases[256]
    const t = zoom.Xpos(ev.layerX / canvas.width) * controls.params.inputTimeWindowWidth
    const v = (-2*ev.layerY / canvas.height + 1) * Math.pow(10,-1*controls.params.inputScaleMagnitude)
    infoElectrode.innerText = `${(t * 1000).toFixed(0)} ms` + ((controls.params.checkboxRaster) ? '' : ` / ${v.toFixed(1)} µV`)
    infoElectrode.style.left = ev.pageX + 'px'
    infoElectrode.style.top = ev.pageY + 'px'
}

function infoElectrodeOver(ev){
    infoElectrodeUpdate(ev)
    infoElectrode.hidden = false
}

//Zoom
let zoom = new Zoom()

function infoElectrodeMove(ev){
    infoElectrode.hidden = ev.ctrlKey
    if(ev.ctrlKey){
        zoomElectrode.style.width = (ev.layerX - zoom.x0*ev.target.width) + 'px'
        zoomElectrode.style.height = (ev.layerY - zoom.y0*ev.target.height) + 'px'
    }
    infoElectrodeUpdate(ev)
}

function infoElectrodeOut(ev){
    infoElectrode.hidden = true
}

function infoElectrodeDown(ev){
    if(!ev.ctrlKey) return
    //Show Zoom Area
    zoomElectrode.style.left = ev.pageX + 'px'
    zoomElectrode.style.top = ev.pageY + 'px'
    zoomElectrode.hidden = false
    //Update
    zoom.down(ev.layerX / ev.target.width,ev.layerY / ev.target.height)
}

function infoElectrodeUp(ev){
    if(!ev.ctrlKey) return
    zoomElectrode.hidden = true
    zoom.up(ev.layerX/ev.target.width,ev.layerY/ev.target.height)
    plot()
}

function resetZoom(){
    zoom.reset()
    plot()
}