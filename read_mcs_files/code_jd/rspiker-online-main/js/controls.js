class Control{
    constructor(){
        this.refresh = new Map()
        this.plot = new Map()
        this.raster = new Map()
    }

    addRefresh(id){
        this.refresh.set(id,document.getElementById(id))
    }

    addPlot(id){
        this.plot.set(id,document.getElementById(id))
    }

    addRange(id){
        this.addRefresh(id)
        this.range = this.refresh.get(id)
    }

    addRefreshRaster(id){
        this.raster.set(id,document.getElementById(id))
    }

    handlerRefresh(callback){
        this.refresh.forEach((domElement,id)=>{
            domElement.onchange = callback
        })
    }

    handlerRefreshRaster(callback){
        this.raster.forEach((domElement,id)=>{
            domElement.onchange = callback
        })
    }

    handlerPlot(callback){
        this.plot.forEach((domElement,id)=>{
            domElement.onchange = callback
        })
    }

    get params(){
        let obj = {}
        this.refresh.forEach((domElement,id)=>{
            if(domElement.type == 'checkbox') obj[id] = domElement.checked
            else obj[id] = Number(domElement.value)
        })
        this.raster.forEach((domElement,id)=>{
            if(domElement.type == 'checkbox') obj[id] = domElement.checked
            else obj[id] = Number(domElement.value)
        })
        this.plot.forEach((domElement,id)=>{
            if(domElement.type == 'checkbox') obj[id] = domElement.checked
            else obj[id] = Number(domElement.value)
        })
        return obj
    }
}